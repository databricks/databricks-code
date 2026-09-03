"""Interactive managed-config authoring behind `ucode configure` (admin path).

Workspace admins reach this through ``ucode configure`` to build the ``CodingAgentConfig`` their
developers will pull, then publish it with ``ucode publish`` (a separate, explicit command — authoring
only ever saves a draft, never publishes). The draft lives in the ``draft`` slot of
``~/.ucode/managed-state.json`` (owned by :mod:`ucode.managed_config`), kept separate from the
launch-fetched published snapshot.

:func:`author_managed_config` picks the agents and models; ``ucode configure spend-tiers``
(:func:`configure_spend_tiers_command`) edits the tiered spend policy. Authoring carries the spend
policy and tracing table forward untouched (:func:`_carry_forward_sections`). The managed config
carries agents/models/global policy/spend tiers only — MCP servers and skills are personal
configuration (`ucode mcp` / `ucode skills`), not part of it.

Serialization, validation, and the per-agent model catalogs live in :mod:`ucode.managed_setup`; this
module is the interaction layer on top of them.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import cast

from ucode import config_io
from ucode.agents import TOOL_SPECS, check_gateway_endpoint
from ucode.databricks import (
    ANTHROPIC_FAMILIES,
    all_users_can_use_schema,
    create_coding_agent_config,
    discover_claude_models_unbucketed,
    ensure_databricks_auth,
    get_databricks_token,
    has_cached_model_provider_services,
    is_model_provider_feature_unavailable,
    is_workspace_admin,
    list_model_provider_services,
    list_workspace_budgets,
    map_claude_family_models,
    model_service_exists,
    service_usable_for_tool,
    update_coding_agent_config,
)
from ucode.managed_config import (
    get_managed_config,
    load_draft_config,
    load_published_config,
    save_draft_config,
    save_published_config,
)
from ucode.managed_setup import (
    CLAUDE_SLOT_FOR_FAMILY,
    claude_family_candidates,
    claude_family_for_model,
    model_options_for_agent,
    supports_provider_service,
    validate_manifest,
)
from ucode.state import load_state
from ucode.ui import (
    console,
    format_usd,
    kv_line,
    print_err,
    print_heading,
    print_note,
    print_panel,
    print_section,
    print_success,
    print_warning,
    print_warning_panel,
    prompt_for_multi_selection,
    prompt_for_percentage,
    prompt_for_selection,
    prompt_for_text,
    prompt_for_tools,
    prompt_yes_no_default,
    spinner,
)

# Shown whenever the workspace's coding-agent-config APIs return FEATURE_DISABLED.
CODING_AGENT_CONFIGS_DISABLED_MESSAGE = (
    "Workspace-managed coding agent configuration is not available on this workspace. Use "
    "`ucode configure` to set up agents for individual users instead."
)

BUDGET_POLICY_BLURB = (
    "As the workspace spends more of a budget, a tiered spend policy automatically switches "
    "everyone's default agent and model to a cheaper one — for example Claude Code / Opus normally, "
    "Claude Code / Sonnet once spend passes 80%, OpenCode / Kimi past 100%.\n\n"
    "It only moves the default. Developers can still pick any model they have access to, and the "
    "budget's own hard block is what actually caps spend."
)

# Agents not offered in `ucode configure`'s managed picker, even when the workspace serves them.
# `ucode gemini` still works as a launch target; it's just not part of the managed config authored
# here. Serialize/validate keep supporting it, so a `--from-file` manifest can still name it.
SETUP_EXCLUDED_AGENTS = frozenset({"gemini"})


def _tracing_table_from_state(state: dict) -> str | None:
    """The UC table `ucode configure tracing` wired up, or None when tracing is off.

    ``configure tracing`` records the destination as ``uc_destination``; the managed config calls the
    same thing ``tracing.table``.
    """
    tracing = state.get("tracing")
    if not isinstance(tracing, dict) or not tracing.get("enabled"):
        return None
    destination = tracing.get("uc_destination")
    return destination if isinstance(destination, str) and destination else None


def provider_service_model_options(service: dict) -> list[str]:
    """Model ids an admin can pick from a provider service, or [] when they can't be enumerated.

    A service's ``config.targets`` names the provider-side models it exposes, which is exactly the
    vocabulary the manifest's ``default_model`` must use when ``model_provider_service`` is set. Two
    cases yield nothing to pick from, and the caller falls back to free-text:

    - ``allow_all_targets`` — the service passes through the provider's whole catalog, which ucode
      cannot enumerate (there is no list-models call for a provider service).
    - no targets at all — e.g. a relayed Anthropic subscription service, which routes by canonical
      model name rather than by an explicit target list.
    """
    if service.get("allow_all_targets"):
        return []
    targets = service.get("targets")
    if not isinstance(targets, list):
        return []
    return sorted({t for t in targets if isinstance(t, str) and t})


def _select_provider_service(tool: str, workspace: str, token: str) -> dict | None:
    """Offer Databricks-hosted vs an external Model Provider Service for ``tool``.

    Returns the chosen service dict (as :func:`list_model_provider_services` shapes it), or None to
    stay on Databricks-hosted models. The whole dict is returned rather than just the name so the
    model prompt can offer the service's ``targets`` instead of asking the admin to type a model id
    from memory.

    Only claude and codex can route through a provider service today; every other agent short-cuts to
    Databricks. Mirrors `cli._maybe_select_provider_service`, but returns the choice instead of
    persisting it — the wizard is authoring a manifest, not configuring this machine.
    """
    if not any(
        supports_provider_service(tool, provider_type)
        for provider_type in ("anthropic", "amazon_bedrock", "openai")
    ):
        return None

    display = TOOL_SPECS[tool]["display"]
    # The listing is memoized per workspace, so only the first agent's call does any I/O. That one
    # takes over a second and deserves a spinner; the rest are instant, and spinning once per agent
    # made the wizard look like it re-listed the services every time.
    if has_cached_model_provider_services(workspace):
        services, reason = list_model_provider_services(workspace, token)
    else:
        with spinner("Checking for model provider services..."):
            services, reason = list_model_provider_services(workspace, token)
    if reason is not None:
        # A workspace without the feature enabled is the common case and not worth a warning; any
        # other failure is worth surfacing, or the admin silently loses the MPS option and has no
        # idea why. Mirrors `cli._maybe_select_provider_service`.
        if not is_model_provider_feature_unavailable(reason):
            print_warning(f"Could not list model provider services: {reason}")
            print_note("Falling back to Databricks-hosted models.")
        return None

    usable = [service for service in services if service_usable_for_tool(tool, service)]
    if tool == "claude":
        # Claude subscription relays are not reliable enough for managed configurations yet.
        usable = [service for service in usable if not service.get("relayed")]
    if not usable:
        if services:
            # Services exist but none match this agent's dialect — say so, since "no picker appeared"
            # is otherwise indistinguishable from the feature being off.
            print_note(
                f"No model provider service matches {display}'s API dialect "
                f"({len(services)} found on this workspace); using Databricks-hosted models."
            )
        return None

    choice = prompt_for_selection(
        f"How should {display} get its models?",
        [
            ("databricks", "Databricks Hosted"),
            ("mps", "External Models (Model Provider Service)"),
        ],
    )
    if choice != "mps":
        return None
    selected = prompt_for_selection(
        f"Select the model provider service for {display}:",
        [(service["name"], service["name"]) for service in usable],
        searchable=True,
    )
    if not selected:
        return None
    service = next(service for service in usable if service["name"] == selected)
    _warn_if_mps_not_broadly_accessible(workspace, token, service["name"])
    return service


def _warn_if_mps_not_broadly_accessible(workspace: str, token: str, service_name: str) -> None:
    """Warn if the picked MPS's schema isn't granted to all workspace users.

    A developer who pulls a config routing through this MPS needs USE_SCHEMA on its schema, or they
    hit "User does not have USE_SCHEMA on Schema <catalog>.<schema>" at launch. This only warns
    (never blocks): access may instead come from a team group the check can't see, and an
    inconclusive check stays silent.
    """
    schema = ".".join(service_name.split(".")[:2])
    if schema.count(".") != 1:
        return
    with spinner("Checking who can use this service..."):
        accessible = all_users_can_use_schema(workspace, token, schema)
    if accessible is False:
        print_warning(
            f"All workspace users don't appear to have USE_SCHEMA on `{schema}`, so developers "
            f"who pull this config may not be able to use `{service_name}`. Grant USE_SCHEMA on "
            f"`{schema}` to the `account users` group (or the teams that need it) in Unity Catalog."
        )


def _prompt_models_for_agent(tool: str, state: dict, provider_service: dict | None) -> dict:
    """Build one agent's ``model_config``. Every agent ends up with a ``default_model``.

    Databricks-hosted agents pick from the workspace's discovered models, filtered to the families
    that agent can actually serve. Provider-service agents pick from the service's own ``targets``,
    falling back to free-text only when those can't be enumerated (``allow_all_targets``, or a
    relayed service that routes by canonical name).

    An empty selection is re-prompted rather than accepted: an agent with no ``default_model`` cannot
    be the config's ``default_agent`` (the server rejects it) and gives developers nothing to launch,
    so "none" is never a useful answer here. Ctrl-C still aborts the whole flow.

    Model ids are stored bare (e.g. ``system.ai.claude-opus-4-8``), not provider-prefixed: each
    agent's own writer adds whatever prefix its config format needs (see
    ``opencode._resolve_model_selector``), which keeps the manifest agent-neutral.

    Codex takes a single model (the harness selects one); Claude's picks are bucketed into
    ``ClaudeDefaultModels`` family slots; the rest keep a flat list plus their chosen default.
    """
    display = TOOL_SPECS[tool]["display"]
    model_config: dict = {}
    if provider_service:
        service_name = provider_service["name"]
        model_config["model_provider_service"] = service_name
        targets = provider_service_model_options(provider_service)
        if tool == "claude" and _pins_family_models(targets):
            # `targets` (not the raw service) publishes explicit Claude models, and `render_overlay`
            # pins each family to a chosen version from them — Bedrock slugs
            # (`us.anthropic.claude-opus-4-8-v1:0`) or canonical Anthropic ids (`claude-opus-4-8`).
            # So Claude needs a default *per family*, not one overall, mirroring the Databricks-hosted
            # path. (A service with no enumerable targets pins nothing and takes a single default —
            # handled below.) Keyed on `targets`, the same list the prompt consumes, so the decision
            # and the prompt can't disagree — `allow_all_targets` zeroes `targets`, so it correctly
            # falls through to the single-default branch even if the raw service also lists Claude.
            model_config.update(_prompt_claude_provider_family_models(targets, service_name))
        elif targets:
            model_config["default_model"] = _require_selection(
                f"Default model for {display} (from {service_name}):",
                [(target, target) for target in targets],
            )
        else:
            # No enumerable target list: the service either passes through the provider's whole
            # catalog or routes by canonical model name, so the admin has to name the model.
            print_note(
                f"{service_name} does not publish an explicit model list, so enter the model id "
                "as the provider names it (e.g. claude-sonnet-4-6)."
            )
            model_config["default_model"] = _require_text(f"Default model for {display}")
        return model_config

    if tool == "claude":
        return _prompt_claude_models(state)

    options = model_options_for_agent(tool, state)
    if not options:
        print_warning(f"No models were discovered for {display} on this workspace.")
        return {"default_model": _require_text(f"Default model for {display}")}

    custom: list[str] = []
    if tool in SINGLE_MODEL_AGENTS:
        model = _select_hosted_model(
            f"Select the default model for {display}:", options, state, custom
        )
        single: dict = {"default_model": model}
        if custom:
            single["custom_models"] = custom
        return single

    # Nothing pre-checked: the first option is whatever discovery sorted first, not a
    # recommendation — for pi it is a Claude model, for codex the oldest GPT. Pre-checking it made
    # "hit Enter" produce an arbitrary config. (A worthwhile follow-up is to pre-check the models
    # this workspace was configured with last time, which `load_draft_config` already loads for
    # the agent picker, so a re-run becomes an edit rather than a re-entry.)
    picked = _select_hosted_models_multi(f"Select models for {display}:", options, state, custom)
    if len(picked) == 1:
        model_config["default_model"] = picked[0]
    else:
        model_config["default_model"] = _require_selection(
            f"Default model for {display}:", [(model, model) for model in picked]
        )

    model_config["models"] = picked
    if custom:
        model_config["custom_models"] = list(dict.fromkeys(custom))
    return model_config


# Agents that get a single model rather than a multi-select. Codex's proto has no model list at all.
# Gemini and Copilot do declare `repeated string models`, but their config writers take one model
# (`gemini.write_tool_config(state, model)` / `copilot.write_tool_config(state, model)`) and write a
# single env var — so a published list would be read by nothing. Offering one keeps the manifest
# honest about what ucode can apply; widen this when those writers grow a picker.
SINGLE_MODEL_AGENTS = frozenset({"codex", "gemini", "copilot"})

# Skip sentinel for a Claude family prompt. Every `ClaudeDefaultModels` slot is optional, and an
# unset one falls back to `default_model`, so leaving a family out is a legitimate choice.
_SKIP_FAMILY = "__skip__"


def _confirm_agent(tool: str, agent_config: dict) -> None:
    """One consistent closing line per agent in step 2, whatever its model shape.

    Every agent — a single-model codex, a multi-model opencode, a family-slotted claude — ends its
    block with the same `✔ <agent> configured — <default> · <scope>` line, so the step reads as a
    uniform checklist rather than each agent's picker trailing off differently.
    """
    display = TOOL_SPECS.get(tool, {}).get("display", tool)
    model_config = agent_config.get("model_config") or {}
    detail = model_config.get("default_model") or "no model"
    provider = model_config.get("model_provider_service")
    if provider:
        detail = f"{detail} via {provider}"
    print_success(f"{display} configured — {detail}")


def _render_family_slots(slots: dict[str, str]) -> None:
    """Recap the Claude family → model slots just chosen, before the overall-default question.

    Same "form filling in" motif as :func:`_selected_recap`: the per-family answers scrolled by one at
    a time, so gathering them into one box makes "which of these is the overall default?" a choice
    over something the admin can see rather than recall.
    """
    lines = [
        kv_line(slot.removeprefix("default_").removesuffix("_model"), model)
        for slot, model in slots.items()
    ]
    print_panel("Claude Code models", lines)


def _prompt_claude_models(state: dict) -> dict:
    """Build Claude's ``model_config`` one family slot at a time.

    Claude Code addresses models by family alias, not from a list, so the config is a set of slots:
    `default_opus_model`, `default_sonnet_model`, `default_haiku_model`, `default_fable_model`. A flat
    multi-select can't express that — and because `state["claude_models"]` holds only the newest id
    per family, it could only ever offer one model per family anyway. Asking per family surfaces the
    alternatives (six opus versions on a typical workspace, not one) and matches the proto.

    Each family may be skipped; the overall `default_model` is then chosen from the slots that were
    filled, so it can never name a model the config doesn't carry.
    """
    display = TOOL_SPECS["claude"]["display"]
    # No spinner: the model-services listing is already cached by the time the flow reaches here
    # (`configure_shared_state` walked it up front), so this is a filter over data in hand, not a
    # fetch. Showing "Fetching Claude models..." made the wizard look like it listed the catalog
    # twice.
    candidates = _claude_candidates(state)
    if not candidates:
        print_warning(f"No Claude models were discovered for {display} on this workspace.")
        return {"default_model": _require_text(f"Default model for {display}")}

    print_note(
        "Claude Code picks a model by family, so set a default per family. Skip any family you "
        "don't want configured — it falls back to the overall default."
    )
    slots: dict[str, str] = {}
    custom: list[str] = []
    for family in ANTHROPIC_FAMILIES:
        family_models = candidates.get(family)
        if not family_models:
            continue
        rows = [(model, model) for model in family_models] + [
            (_CUSTOM_MODEL, _CUSTOM_MODEL_LABEL),
            (_SKIP_FAMILY, f"(skip {family})"),
        ]
        choice = prompt_for_selection(f"Default {family} model:", rows, searchable=True)
        if choice is None:
            raise KeyboardInterrupt
        if choice == _SKIP_FAMILY:
            continue
        if choice == _CUSTOM_MODEL:
            choice = _prompt_custom_model(state)
            custom.append(choice)
        slots[CLAUDE_SLOT_FOR_FAMILY[family]] = choice

    if not slots:
        # Every slot skipped is a legitimate, minimal config: the proto leaves `models` optional and
        # each unset slot falls back to `default_model`, so one model covers every family. Pick it
        # from the same candidates rather than asking the admin to type an id.
        print_note(f"No families configured, so {display} will use a single model for all of them.")
        every_model = list(dict.fromkeys(m for fm in candidates.values() for m in fm))
        fallback_custom: list[str] = []
        model = _select_hosted_model(
            f"Which model should {display} use?", every_model, state, fallback_custom
        )
        single: dict = {"default_model": model}
        if fallback_custom:
            single["custom_models"] = fallback_custom
        return single

    chosen = list(dict.fromkeys(slots.values()))
    model_config: dict = {"models": slots}
    if len(chosen) == 1:
        # A one-option prompt is a wasted keystroke, but skipping it silently reads as a dropped
        # step — say what was inferred so the admin knows the default is set, and to what.
        model_config["default_model"] = chosen[0]
        print_note(f"Only one model configured, so it's {display}'s overall default.")
    else:
        _render_family_slots(slots)
        model_config["default_model"] = _require_selection(
            f"Which of those is {display}'s overall default?", [(m, m) for m in chosen]
        )
    if custom:
        model_config["custom_models"] = list(dict.fromkeys(custom))
    return model_config


def _pins_family_models(targets: list[str]) -> bool:
    """True when Claude behind a service is pinned per family at launch.

    Keyed on the *behavior*, not the vendor: when a service publishes explicit Claude targets — a
    Bedrock provider-side slug like ``us.anthropic.claude-opus-4-8-v1:0`` *or* a canonical Anthropic
    id like ``claude-opus-4-8`` — ``render_overlay`` pins each ``ANTHROPIC_DEFAULT_<FAMILY>_MODEL`` to
    a chosen version, so the wizard prompts one model per family, mirroring the Databricks-hosted
    path. No Claude-family targets (``allow_all_targets`` zeroes the list, or a relayed subscription
    lists none) means nothing is pinned — Claude Code's canonical names route fine — so it takes a
    single default.

    Takes the *enumerated* targets (``provider_service_model_options`` output), the exact list the
    per-family prompt consumes, so the decision and the prompt can't disagree. Reading the raw
    ``service["targets"]`` here would diverge: an ``allow_all_targets`` service that still lists
    Claude models would test True but hand the prompt an empty list, aborting the wizard.
    """
    return bool(map_claude_family_models(targets))


def _prompt_claude_provider_family_models(targets: list[str], service_name: str) -> dict:
    """Claude family slots (and overall default) chosen from a service's own Claude target ids.

    The Databricks-hosted path (:func:`_prompt_claude_models`) prompts per family because Claude
    Code addresses models by family alias; the same holds behind a Model Provider Service, except
    the ids are the service's own — Bedrock provider-side slugs or canonical Anthropic ids rather
    than ``system.ai.*``. ``render_overlay`` pins each ``ANTHROPIC_DEFAULT_<FAMILY>_MODEL`` from
    them, so a single overall default would leave the other families unpinned.

    Targets are grouped by family via :func:`claude_family_for_model`, which matches the
    ``claude-<family>-`` segment in any spelling (``anthropic.claude-…`` Bedrock or bare
    ``claude-…`` canonical). A target that names no family is offered only as the overall default.
    Falls back to a single default when nothing maps to a family at all.
    """
    display = TOOL_SPECS["claude"]["display"]
    by_family: dict[str, list[str]] = {}
    for target in targets:
        family = claude_family_for_model(target)
        if family:
            by_family.setdefault(family, []).append(target)

    if not by_family:
        # No target maps to a Claude family (unusual for a Bedrock Claude service); the most this can
        # honestly ask for is one overall default.
        return {
            "default_model": _require_selection(
                f"Default model for {display} (from {service_name}):",
                [(t, t) for t in targets],
            )
        }

    # Quick setup: fill each family with the service's newest id (highest version, broadest region),
    # the same pick a developer's own `ucode configure` would make. The alternative is choosing a
    # specific id per family — e.g. to pin an older, validated version or a particular region.
    # map_claude_family_models covers opus/sonnet/haiku but not fable, so a fable-only service has
    # nothing to quick-fill — only offer quick setup when it would actually populate a slot.
    family_models = map_claude_family_models(targets)
    if family_models:
        print_note(
            "Quick setup fills each Claude family with the newest model this service offers. Answer "
            "no to choose a specific model per family instead (pin an older version, a region)."
        )
    if family_models and prompt_yes_no_default("Quick setup?", default=True):
        slots = {CLAUDE_SLOT_FOR_FAMILY[family]: model for family, model in family_models.items()}
        # Overall default = the highest-tier family the service offers, not whichever target happened
        # to sort first. Fable is last: it's the premium opt-in model, a poor default. `family_models`
        # is non-empty here, so `next` always finds one.
        default_family = next(
            fam for fam in ("opus", "sonnet", "haiku", "fable") if fam in family_models
        )
        model_config = {"models": slots, "default_model": family_models[default_family]}
        summary = ", ".join(
            f"{fam}={slots[CLAUDE_SLOT_FOR_FAMILY[fam]]}"
            for fam in ANTHROPIC_FAMILIES
            if CLAUDE_SLOT_FOR_FAMILY[fam] in slots
        )
        # A note, not a success line: the loop's `_confirm_agent` prints the single ✔ for the agent.
        print_note(f"Quick setup — {summary} (default: {default_family}).")
        return model_config

    print_note(
        f"Claude Code picks a model by family, so set a default per family from {service_name}. "
        "Skip any family you don't want configured — it falls back to the overall default."
    )
    slots: dict[str, str] = {}
    for family in ANTHROPIC_FAMILIES:
        family_targets = by_family.get(family)
        if not family_targets:
            continue
        choice = prompt_for_selection(
            f"Default {family} model:",
            [(t, t) for t in family_targets] + [(_SKIP_FAMILY, f"(skip {family})")],
            searchable=True,
        )
        if choice is None:
            raise KeyboardInterrupt
        if choice != _SKIP_FAMILY:
            slots[CLAUDE_SLOT_FOR_FAMILY[family]] = choice

    model_config: dict = {}
    if slots:
        model_config["models"] = slots
    chosen = list(dict.fromkeys(slots.values()))
    if len(chosen) == 1:
        model_config["default_model"] = chosen[0]
        print_note(f"Only one model configured, so it's {display}'s overall default.")
    else:
        if slots:
            _render_family_slots(slots)
        # Offered over every target, not just the slots: `default_model` needn't be a family model,
        # and a mixed-catalog service may expose one an admin wants as the overall default.
        options = chosen or list(targets)
        model_config["default_model"] = _require_selection(
            f"Which of those is {display}'s overall default?", [(m, m) for m in options]
        )
    return model_config


def _claude_candidates(state: dict) -> dict[str, list[str]]:
    """Claude models grouped by family. Degrades to the per-family picks if the listing fails.

    Caches the full listing on ``state["all_claude_models"]`` so `validate_manifest` recognizes the
    older versions these prompts offer — ``claude_models`` alone holds just the newest per family,
    and would reject a legitimately-picked ``claude-opus-4-8``.

    INVARIANT: whatever this returns must be recognizable by ``validate_manifest``, which reads
    ``all_claude_models`` (falling back to ``claude_models``) via ``_known_models``. The two paths
    below both satisfy it, for different reasons: the listing path widens the candidates *and* sets
    the cache, while the fallback path sets nothing but also narrows the candidates to
    ``claude_models``, which ``_known_models`` already covers. Widening the fallback without also
    populating the cache breaks the invariant, and the symptom is a confusing rejection at the very
    end of the flow ("claude: model 'system.ai.claude-opus-4-8' is not available on this
    workspace") rather than an error at the prompt that offered it.
    """
    cached = state.get("all_claude_models")
    if isinstance(cached, list) and cached:
        return claude_family_candidates([m for m in cached if isinstance(m, str)], state)

    workspace = state.get("workspace")
    all_claude: list[str] = []
    if workspace:
        try:
            token = get_databricks_token(workspace, state.get("profile"))
            all_claude, _ = discover_claude_models_unbucketed(workspace, token)
        except (RuntimeError, OSError):
            # OSError covers a missing `databricks` binary: `get_databricks_token` shells out, so a
            # machine without the CLI on PATH raises FileNotFoundError rather than RuntimeError.
            # Either way the per-family picks below are a usable fallback.
            all_claude = []
    if all_claude:
        state["all_claude_models"] = all_claude
    return claude_family_candidates(all_claude, state)


# Every picker in this flow chooses a model, a provider service, or a budget — lists that on a real
# workspace run to a dozen-plus entries (16 GPT models on the workspace this was built against), so
# they are all filterable by typing. That trades away j/k navigation, which questionary can't offer
# alongside search; arrow keys still work.
def _require_selection(prompt: str, options: list[tuple[str, str]]) -> str:
    """Single-select that won't take "nothing" for an answer.

    ``prompt_for_selection`` returns None for both Ctrl-C and an empty submission, and the two are
    genuinely indistinguishable here: questionary's ``Question.ask`` catches KeyboardInterrupt
    internally and returns None (v2.1.1, question.py), so nothing propagates for a caller to see.
    A None is therefore treated as an abort rather than re-asked — re-asking looped forever on
    Ctrl-C, printing the error once per keypress and never exiting.
    """
    answer = prompt_for_selection(prompt, options, searchable=True)
    if not answer:
        raise KeyboardInterrupt
    return answer


def _require_multi_selection(
    prompt: str, options: list[tuple[str, str]], preselected: list[str] | None = None
) -> list[str]:
    """Multi-select that requires at least one choice. None (Ctrl-C) still aborts."""
    while True:
        picked = prompt_for_multi_selection(
            prompt, options, preselected=preselected, searchable=True
        )
        if picked is None:
            raise KeyboardInterrupt
        if picked:
            return picked
        print_err("Select at least one model (space to toggle, enter to confirm).")


def _require_text(prompt: str) -> str:
    """Free-text prompt that requires a non-empty answer.

    ``required=True`` makes closed stdin abort instead of returning None. Without it a
    non-interactive run (piped stdin, CI) spun here forever: ``prompt_for_text`` returns its default
    on EOF, the default is None, and the loop re-asked an empty stream. Reachable whenever model
    discovery finds nothing, which is exactly when a run is most likely to be scripted.
    """
    while True:
        answer = prompt_for_text(prompt, required=True)
        if answer:
            return answer
        print_err("Please enter a model id.")


# Discovered model lists run long (a dozen-plus ids on a real workspace), so every hosted-model
# picker is searchable and scrolls (see `prompt_for_selection`); all discovered ids are offered, and
# an explicit "type your own" row still covers a custom model service outside `system.ai` that
# discovery never lists at all.
_CUSTOM_MODEL = "__custom_model__"
_CUSTOM_MODEL_LABEL = "✎ Enter a custom model…"


def _custom_option_rows(options: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """All discovered model rows plus a 'type your own' row, for a hosted-model picker."""
    return list(options) + [(_CUSTOM_MODEL, _CUSTOM_MODEL_LABEL)]


def _short_reason(reason: str | None) -> str:
    """A one-line reason fit for a prompt, without the raw JSON body the transport appends.

    HTTP failures come back as ``HTTP <code> <reason>: <body-excerpt>`` (the body is a JSON error
    blob for gateway/UC errors); an admin at a prompt wants the status, not the payload. Keeps the
    ``HTTP <code> <reason>`` head and drops a ``{...}`` body, leaving plain reasons (``network
    error: ...``) untouched.
    """
    if not reason:
        return "unknown error"
    return reason.split(": {", 1)[0].strip()


def _verify_custom_model(state: dict, model: str) -> tuple[bool | None, str | None]:
    """Whether ``model`` is a model service on the workspace; None when the check can't run.

    Mirrors how ``_claude_candidates`` reaches the workspace — a token from ``state`` — and turns a
    missing workspace or a failed token fetch into an inconclusive result rather than an error.
    """
    workspace = state.get("workspace")
    if not workspace:
        return None, "no workspace in local state"
    try:
        token = get_databricks_token(workspace, state.get("profile"))
    except (RuntimeError, OSError) as exc:
        # OSError covers a missing `databricks` binary (get_databricks_token shells out).
        return None, str(exc)
    return model_service_exists(workspace, token, model)


def _prompt_custom_model(state: dict) -> str:
    """Prompt for a custom model-service id, re-asking until it exists on the workspace.

    A typo shouldn't get baked into a published config, so the id is checked against the workspace's
    model services and a miss is re-prompted. An inconclusive check (no workspace/token, or a
    transient API error) is accepted with a warning rather than blocking a possibly-valid model.
    """
    while True:
        model = _require_text("Custom model (catalog.schema.model)")
        exists, reason = _verify_custom_model(state, model)
        if exists:
            return model
        if exists is None:
            print_warning(
                f"Couldn't verify '{model}' on this workspace ({_short_reason(reason)}); "
                "using it as typed."
            )
            return model
        print_err(
            f"'{model}' isn't a model service on this workspace. Check the name and try again "
            "(expected catalog.schema.model, e.g. main.default.claude-opus-4-5)."
        )


def _select_hosted_model(
    prompt: str, options: list[str], state: dict, custom_sink: list[str]
) -> str:
    """Single-select over all discovered ``options`` plus a custom-entry row.

    Records any custom id in ``custom_sink`` so the caller can mark it in
    ``model_config.custom_models``, which keeps validation from rejecting a model discovery didn't
    surface.
    """
    choice = _require_selection(prompt, _custom_option_rows([(m, m) for m in options]))
    if choice == _CUSTOM_MODEL:
        model = _prompt_custom_model(state)
        custom_sink.append(model)
        return model
    return choice


def _select_hosted_models_multi(
    prompt: str, options: list[str], state: dict, custom_sink: list[str]
) -> list[str]:
    """Multi-select over all discovered ``options`` plus a custom-entry row; requires one pick.

    Selecting the custom row prompts for custom model ids — as many as the admin wants, since a
    multi-select agent (opencode, pi) can carry a whole list — and folds them into the picks. Records
    each custom id in ``custom_sink`` (see :func:`_select_hosted_model`).
    """
    rows = _custom_option_rows([(m, m) for m in options])
    picked = _require_multi_selection(prompt, rows)
    models = [p for p in picked if p != _CUSTOM_MODEL]
    if _CUSTOM_MODEL in picked:
        while True:
            custom = _prompt_custom_model(state)
            if custom not in models:
                models.append(custom)
                custom_sink.append(custom)
            if not prompt_yes_no_default("Add another custom model?", default=False):
                break
    return models


def configured_models_for_agent(agent_config: dict) -> list[str]:
    """Models an agent was configured with, in the manifest's own vocabulary.

    ``model_config.models`` is a flat list for most agents but a family-slot dict for claude
    (``default_opus_model`` -> id), so both shapes collapse to a list here. The ``default_model`` is
    included because codex has no model list at all — it is the only model that agent has.
    """
    model_config = agent_config.get("model_config")
    if not isinstance(model_config, dict):
        return []
    models: list[str] = []
    raw = model_config.get("models")
    if isinstance(raw, dict):
        models.extend(v for v in raw.values() if isinstance(v, str) and v)
    elif isinstance(raw, list):
        models.extend(m for m in raw if isinstance(m, str) and m)
    default_model = model_config.get("default_model")
    if isinstance(default_model, str) and default_model:
        models.append(default_model)
    # dict.fromkeys de-duplicates while keeping the admin's preference order.
    return list(dict.fromkeys(models))


def _render_tier_ladder(tiers: list[dict], threshold: object, *, base_default: str = "") -> None:
    """Show the tiers built so far as spend ranges, so the fallback ladder reads at a glance.

    A tier activates once spend passes its percentage and the highest passed tier wins, so each
    tier really owns the range from its own percentage up to the next tier's. Rendering those ranges
    ("50–90%", "90%+") rather than bare thresholds ("at 50%", "at 90%") is what makes the ladder
    legible — the admin sees which agent a developer actually gets at any level of spend. Reprinted
    as the ladder grows, so the sequence forms in front of them instead of in their head.
    """
    ordered = sorted(tiers, key=lambda t: t["spending_percentage"])
    lines: list[str] = []
    if base_default:
        # Below the first tier the manifest's own default applies; naming it anchors the sequence.
        first = ordered[0]["spending_percentage"] * 100
        lines.append(kv_line(f"under {first:g}%", f"{base_default} (default)"))
    for i, tier in enumerate(ordered):
        low = tier["spending_percentage"] * 100
        agent = TOOL_SPECS.get(tier["default_agent"], {}).get("display", tier["default_agent"])
        if i + 1 < len(ordered):
            span = f"{low:g}–{ordered[i + 1]['spending_percentage'] * 100:g}%"
        else:
            span = f"{low:g}%+"
        lines.append(kv_line(span, f"{agent} / {tier['default_model']}"))
    print_panel("Budget tiers so far", lines)


def _prompt_budget_policy(
    workspace: str,
    token: str,
    enabled_agents: dict[str, dict],
    state: dict,
    *,
    base_default: str = "",
) -> dict | None:
    """Author a spend-routing ``budget_policy``, or None when the admin backs out or can't.

    Budgets themselves are created in the Databricks console (they're account-level objects), so the
    admin picks an existing one here. Tiers are prompted in percent and stored as fractions, which is
    what the API validates.

    ``enabled_agents`` is what the manifest gives each agent, so a tier's model choices come from
    that rather than the workspace catalog. Offering the catalog would let a tier point an agent at a
    model it wasn't given, which neither this validation nor the server's would reject: the tier would
    activate and hand the developer a model their agent doesn't have.

    Asks no "set up a tiered spend policy?" gate — running `ucode configure spend-tiers` is the answer to
    that question, the same way `ucode configure <thing>` needs no confirmation.
    """
    print_section("Tiered Spend Policy")

    # Check for attachable budgets before anything else: budgets are created in the Databricks
    # console, so if there are none (or none that can enforce routing) there is nothing to do here.
    # Bail with a boxed warning and skip the explanatory blurb — no point explaining a feature the
    # workspace can't use yet.
    with spinner("Listing workspace budgets..."):
        budgets, reason = list_workspace_budgets(workspace, token)
    if reason is not None or not budgets:
        print_warning_panel(
            "No AI Gateway budgets are visible for this workspace, so there is nothing to attach a "
            "policy to. Create a budget in the Databricks console first, then re-run "
            "`ucode configure spend-tiers`. Currently, only AI Gateway budgets with hard blocks are "
            "eligible to be associated with Tiered Spend Policies."
        )
        return None

    # Spend routing only works on a budget with a per-user threshold that hard-blocks: without a
    # per-user threshold the gateway reports no spend and every tier stays inert, and without a
    # BLOCK_USAGE action the policy is never enforced (an email-only alert does not gate spend). The
    # listing now exposes each alert's action, so hide the budgets that can't enforce routing.
    usable = [budget for budget in budgets if budget.get("has_per_user_block")]
    if not usable:
        print_warning_panel(
            "None of this workspace's AI Gateway budgets have a per-user threshold with a usage "
            "block configured, which spend routing enforces. Add a per-user alert threshold with a "
            "block action to a budget in the Databricks console, then re-run "
            "`ucode configure spend-tiers`."
        )
        return None

    # Budgets exist — now explain what a policy does, before asking the admin to pick one. Boxed so
    # the concept is read as a unit rather than skimmed as one more bullet.
    print_panel("What is a Tiered Spend Policy?", [BUDGET_POLICY_BLURB])
    print_note(
        "Showing only budgets with a per-user hard block configured, which spend routing enforces."
    )

    budget_id = prompt_for_selection(
        "Which budget should this policy track?",
        [
            (budget["id"], f"{budget['display_name'] or budget['id']} ({budget['id']})")
            for budget in usable
        ],
        searchable=True,
    )
    if not budget_id:
        return None

    policy: dict = {"budget_id": budget_id}
    # Remember the budget's own name so the summary can show it beside the policy name. It's a local
    # display aid only — `_budget_policy_payload` doesn't serialize it, so it never reaches the API.
    budget_display_name = next(
        (budget["display_name"] for budget in usable if budget["id"] == budget_id), ""
    )
    if budget_display_name:
        policy["budget_display_name"] = budget_display_name
    display_name = prompt_for_text("Policy name", default="coding-agents-tiered-routing")
    if display_name:
        policy["display_name"] = display_name

    # The per-user monthly cap the budget was created with. Tiers are picked as percentages of it, so
    # showing the dollar amount (and what each percentage works out to) tells the admin what the total
    # possible per-user spend even is. None when the listing couldn't read it — then we just skip the
    # dollar hints and prompt in percent as before.
    threshold = next(
        (budget.get("per_user_threshold") for budget in usable if budget["id"] == budget_id), None
    )
    if threshold is not None:
        print_note(f"This budget's per-user limit is {format_usd(threshold)} per month.")

    tiers: list[dict] = []
    seen_percentages: set[float] = set()
    seen_combos: set[tuple[str, str]] = set()
    print_note(
        "Add a tier for each step down: once spend passes the percentage you set, everyone's "
        "default switches to the cheaper agent and model you pick."
    )
    while True:
        index = len(tiers) + 1

        # Percentage first, in its own retry loop so a duplicate here re-asks only the percentage.
        while True:
            fraction = prompt_for_percentage(
                f"Tier {index}: switch once spend passes what % of the budget? Ex: 50%"
            )
            if fraction in seen_percentages:
                print_err("That percentage is already used by another tier; pick a different one.")
                continue
            break
        if threshold is not None:
            # Echo the dollars this percentage stands for, so the admin can sanity-check the tier
            # against the real per-user cap instead of reasoning about percentages in a vacuum.
            print_note(
                f"  {fraction * 100:g}% of {format_usd(threshold)} is "
                f"{format_usd(threshold * Decimal(str(fraction)))}."
            )

        # Agent + model in their own retry loop: a duplicate agent/model re-asks just these two, so
        # the admin doesn't have to retype the percentage they already entered for this tier.
        agent = model = None
        while True:
            agent = prompt_for_selection(
                f"Tier {index}: switch the default to which agent?",
                [(tool, TOOL_SPECS[tool]["display"]) for tool in enabled_agents],
            )
            if not agent:
                break
            # Only what this agent was actually configured with; the workspace catalog would offer
            # models the agent doesn't have.
            options = configured_models_for_agent(enabled_agents.get(agent) or {})
            if not options:
                options = model_options_for_agent(agent, state)
            if options:
                model = prompt_for_selection(
                    f"Tier {index}: using which model?",
                    [(m, m) for m in options],
                    searchable=True,
                )
            else:
                model = prompt_for_text(f"Tier {index}: using which model?")
            if not model:
                break
            if (agent, model) in seen_combos:
                # The highest crossed tier wins, so a second tier on the same agent+model never
                # changes what the lower one already selected — a step-down that doesn't step down.
                # Reject it rather than build a policy with a silently inert tier; only the agent and
                # model are re-asked, the percentage above is kept.
                print_err(
                    f"{TOOL_SPECS[agent]['display']} / {model} is already used by another tier; a "
                    "repeated agent/model makes this tier do nothing. Pick a different one."
                )
                continue
            break
        # Cancelling the agent or model picker abandons this tier and stops adding more.
        if not agent or not model:
            break

        seen_percentages.add(fraction)
        seen_combos.add((agent, model))
        tiers.append(
            {
                "spending_percentage": fraction,
                "default_agent": agent,
                "default_model": model,
            }
        )
        _render_tier_ladder(tiers, threshold, base_default=base_default)
        if not prompt_yes_no_default("Add another tier?", default=False):
            break

    if tiers:
        policy["tiers"] = tiers
    return policy


def _render_summary(workspace: str, manifest: dict) -> None:
    """Print the authored config in a box so an admin can eyeball it before publishing.

    Boxed rather than printed as loose lines: this is the one block an admin is meant to read as a
    whole and check against what they intended, and it lands after a long flow of prompts.
    """
    lines: list[str] = [kv_line("Workspace", workspace)]
    default_agent = manifest.get("default_agent")
    if isinstance(default_agent, str):
        lines.append(
            kv_line(
                "Default agent", TOOL_SPECS.get(default_agent, {}).get("display", default_agent)
            )
        )

    for tool, agent_config in (manifest.get("enabled_agents") or {}).items():
        display = TOOL_SPECS.get(tool, {}).get("display", tool)
        model_config = agent_config.get("model_config") or {}
        detail = model_config.get("default_model") or "no model"
        provider = model_config.get("model_provider_service")
        if provider:
            detail = f"{detail} via {provider}"
        lines.append(kv_line(display, detail))
        # Spell out the per-family slots and model lists: the one-line default alone doesn't show
        # which families an admin configured, which is most of what they chose for claude.
        models = model_config.get("models")
        if isinstance(models, dict):
            for slot, model in models.items():
                family = slot.removeprefix("default_").removesuffix("_model")
                lines.append(kv_line(f"  {family}", str(model)))
        elif isinstance(models, list) and len(models) > 1:
            lines.append(kv_line("  models", ", ".join(str(m) for m in models)))

    # Managed tracing isn't offered by the flow yet, so a "disabled" line is just noise. Only surface
    # it when a `--from-file` config actually set a table.
    if manifest.get("tracing_table"):
        lines.append(kv_line("Tracing", str(manifest["tracing_table"])))

    policy = manifest.get("budget_policy")
    if isinstance(policy, dict):
        tiers = policy.get("tiers") or []
        lines.append(
            kv_line("Budget", policy.get("budget_display_name") or policy.get("budget_id") or "set")
        )
        lines.append(kv_line("Policy name", policy.get("display_name") or "unnamed"))
        for tier in tiers:
            agent = tier.get("default_agent")
            display = TOOL_SPECS.get(agent, {}).get("display", agent)
            percent = float(tier.get("spending_percentage", 0)) * 100
            lines.append(kv_line(f"  at {percent:g}%", f"{display} / {tier.get('default_model')}"))
    else:
        lines.append(kv_line("Tiered Spend Policy", "none"))

    print_panel("Configuration summary", lines)


def _config_facts(manifest: dict) -> list[tuple[str, str, str]]:
    """Flatten a normalized config into ordered ``(key, label, value)`` facts, for diffing.

    Each fact is one thing an admin would think of as a single setting — the default agent, an agent's
    model, its settings scope, the tracing table, a budget tier. The ``key`` is
    a stable identity so the same setting lines up across two configs even when values differ; the
    ``label`` is what the admin reads. Deliberately mirrors what :func:`_render_summary` chooses to
    show, so the diff and the summary never disagree about what's in a config.
    """
    facts: list[tuple[str, str, str]] = []

    display_name = manifest.get("display_name")
    if isinstance(display_name, str) and display_name:
        facts.append(("display_name", "Display name", display_name))

    default_agent = manifest.get("default_agent")
    if isinstance(default_agent, str):
        display = TOOL_SPECS.get(default_agent, {}).get("display", default_agent)
        facts.append(("default_agent", "Default agent", display))

    for tool, agent_config in (manifest.get("enabled_agents") or {}).items():
        display = TOOL_SPECS.get(tool, {}).get("display", tool)
        model_config = agent_config.get("model_config") or {}
        detail = model_config.get("default_model") or "no model"
        provider = model_config.get("model_provider_service")
        if provider:
            detail = f"{detail} via {provider}"
        facts.append((f"agent:{tool}", display, detail))
        models = model_config.get("models")
        if isinstance(models, dict):
            for slot, model in models.items():
                family = slot.removeprefix("default_").removesuffix("_model")
                facts.append((f"agent:{tool}:model:{family}", f"{display} ({family})", str(model)))
        elif isinstance(models, list) and len(models) > 1:
            facts.append((f"agent:{tool}:models", f"{display} models", ", ".join(map(str, models))))

    tracing = manifest.get("tracing_table")
    if tracing:
        facts.append(("tracing_table", "Tracing table", str(tracing)))

    policy = manifest.get("budget_policy")
    if isinstance(policy, dict):
        facts.append(
            (
                "budget:id",
                "Budget",
                policy.get("budget_display_name") or policy.get("budget_id", ""),
            )
        )
        if policy.get("display_name"):
            facts.append(("budget:name", "Policy name", str(policy["display_name"])))
        for tier in policy.get("tiers") or []:
            agent = tier.get("default_agent")
            agent_display = TOOL_SPECS.get(agent, {}).get("display", agent)
            percent = float(tier.get("spending_percentage", 0)) * 100
            facts.append(
                (
                    f"budget:tier:{percent:g}",
                    f"Budget tier at {percent:g}%",
                    f"{agent_display} / {tier.get('default_model')}",
                )
            )

    return facts


def _render_config_diff(existing: dict | None, incoming: dict, workspace: str) -> bool:
    """Show what publishing ``incoming`` changes versus the ``existing`` published config.

    Returns True when there is a difference. Lists only what changes — labelled ADD, DELETE, or
    CHANGE (``old → new``) — since the full config was just printed by :func:`_render_summary` above;
    repeating the unchanged rows here would bury the actual delta. Both configs are in ucode's
    normalized shape (the caller round-trips the local manifest through serialize/normalize first),
    so the comparison is field-for-field with what the workspace holds.
    """
    old = {key: (label, value) for key, label, value in _config_facts(existing or {})}
    new = {key: (label, value) for key, label, value in _config_facts(incoming)}

    # Fixed-width verbs so the labels line up in a column and the eye can scan one kind of change.
    add = "[green]ADD   [/green]"
    delete = "[red]DELETE[/red]"
    change = "[yellow]CHANGE[/yellow]"

    # Incoming order first (added/changed read top-down like the summary), then removed keys.
    ordered = list(new) + [key for key in old if key not in new]
    rows: list[str] = []
    for key in ordered:
        if key in new and key not in old:
            label, value = new[key]
            rows.append(f"  {add}  {label}: {value}")
        elif key in old and key not in new:
            label, value = old[key]
            rows.append(f"  {delete}  {label}: {value}")
        elif old[key][1] != new[key][1]:
            label, old_value = old[key]
            rows.append(f"  {change}  {label}: {old_value} → {new[key][1]}")

    if not rows:
        return False
    print_heading(f"Changes to publish on {workspace}")
    for row in rows:
        console.print(row)
    return True


def _require_admin(workspace: str, token: str, *, strict: bool = False) -> None:
    """Stop unless the caller is a workspace admin.

    By default an unverifiable check (SCIM unreachable) warns and continues: the API enforces the same
    rule, so the worst case is a clear PERMISSION_DENIED at publish time rather than a false block. A
    ``strict`` gate instead refuses when admin status can't be determined — used by commands that must
    be admin-only up front (`ucode configure spend-tiers`) rather than relying on a later publish.
    """
    with spinner("Checking workspace admin permissions..."):
        admin = is_workspace_admin(workspace, token)
    if admin is False:
        raise RuntimeError(
            f"You are not an admin of {workspace}. Authoring the workspace-wide coding config is "
            "restricted to workspace admins."
        )
    if admin is None:
        if strict:
            raise RuntimeError(
                f"Could not verify that you are an admin of {workspace}, and this command is "
                "admin-only. Check your workspace access (the SCIM `Me` API must be reachable) and "
                "try again."
            )
        print_warning(
            "Could not verify workspace admin permissions. Continuing — `ucode publish` will fail "
            "if you lack them."
        )
    else:
        print_success("Admin permissions verified")


def configure_from_file(path: str) -> int:
    """Validate an admin-written manifest and save it as the local draft, skipping the wizard.

    The non-interactive path for CI and for admins who keep the JSON in version control (backing
    `ucode configure --from-file`). Reads ucode's own manifest shape (the same thing the wizard
    writes), not proto-JSON, and writes the draft slot only — nothing is published. Admin-only, since
    it authors the workspace's managed config; a non-admin gets an actionable error.
    """
    manifest_path = Path(path).expanduser()
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not read manifest file: {manifest_path}") from exc
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{manifest_path} is not valid JSON: {exc.msg} (line {exc.lineno})."
        ) from None
    if not isinstance(manifest, dict):
        raise RuntimeError(f"{manifest_path} must contain a JSON object.")

    state = load_state()
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError(
            "No workspace is configured. Run `ucode configure` first so ucode knows which "
            "workspace this manifest is for."
        )
    profile = state.get("profile")
    ensure_databricks_auth(workspace, profile)
    _require_admin(workspace, get_databricks_token(workspace, profile), strict=True)

    errors = validate_manifest(manifest, state)
    if errors:
        print_err(f"{manifest_path} is not a valid managed config:")
        for error in errors:
            print_note(error)
        return 1

    save_draft_config(workspace, manifest)
    _render_summary(workspace, manifest)
    print_success(f"Saved draft from {manifest_path.name} -> ~/.ucode/managed-state.json")
    print_managed_next_steps(manifest)
    return 0


SETUP_SECTIONS: list[tuple[str, str, Callable[[dict], bool]]] = [
    (
        "ucode configure spend-tiers",
        "Tiered Spend Policy",
        lambda m: isinstance(m.get("budget_policy"), dict),
    ),
]


def _command_line(command: str, description: str, *, marker: str = " ", width: int = 0) -> str:
    """A ``  <marker> <command>   <description>`` row, for command lists that read as a column."""
    return f"  {marker} [bold]{command.ljust(width)}[/bold]   {description}"


SETUP_STEP_TITLES = ["Coding agents", "Models & settings", "Default agent"]


def _step_banner(index: int, title: str, command_label: str = "ucode configure") -> None:
    """Announce one phase of the flow as `step N of M`, branded to the invoking command."""
    print_section(f"{command_label} · step {index} of {len(SETUP_STEP_TITLES)} · {title}")


def _selected_recap(workspace: str, enabled_agents: dict, default_agent: str | None) -> None:
    """A compact panel of what's chosen so far, reprinted as the flow advances.

    Turns the run of prompts into something that reads like a form filling in: each phase reprints
    the growing set of decisions before asking the next question. Agents still mid-configuration show
    a `…` placeholder for their model.
    """
    lines = [kv_line("Workspace", workspace)]
    for tool, config in enabled_agents.items():
        model = (config.get("model_config") or {}).get("default_model") or "…"
        lines.append(kv_line(TOOL_SPECS.get(tool, {}).get("display", tool), model))
    if default_agent:
        lines.append(
            kv_line("Default", TOOL_SPECS.get(default_agent, {}).get("display", default_agent))
        )
    print_panel("Selected so far", lines)


def _section_status_lines(manifest: dict, width: int = 0) -> list[str]:
    """One row per optional section: its command, what it covers, and whether it's configured."""
    width = width or max(len(command) for command, _, _ in SETUP_SECTIONS)
    lines: list[str] = []
    for command, label, configured in SETUP_SECTIONS:
        if configured(manifest):
            marker, state = "[green]✔[/green]", "[green]configured[/green]"
        else:
            marker, state = "[dim]○[/dim]", "[dim]not configured[/dim]"
        lines.append(_command_line(command, f"{label} — {state}", marker=marker, width=width))
    return lines


def print_managed_next_steps(manifest: dict) -> None:
    """Report that the draft is saved, list the optional spend-tiers command, then advise publishing.

    Printed rather than prompted: the admin flow only ever saves a draft — publishing is always an
    explicit, separate `ucode publish` step (never offered inline). Showing what is already configured
    keeps a re-run from looking like it lost a section — it didn't; authoring carries sections forward.
    """
    console.print()
    print_heading("Next steps")
    if config_io.is_dry_run():
        # Under --dry-run nothing was written, so the section command (which reads the saved draft)
        # and `publish` have nothing to act on. Say so rather than send the admin to commands that
        # would report "run `ucode configure` first".
        print_note("Dry run — nothing was saved. Re-run without --dry-run to author the config.")
        return
    print_note("[dim]Optional — configure this too, or skip straight to publishing:[/dim]")
    for line in _section_status_lines(manifest):
        console.print(line)
    print_panel(
        "All done?",
        ["Publish with [bold]ucode publish[/bold] so all developers use this configuration."],
    )


CARRIED_SECTIONS: list[tuple[str, str, str]] = [
    ("tracing_table", "Tracing table", "ucode configure --from-file"),
    ("budget_policy", "Tiered Spend Policy", "ucode configure spend-tiers"),
]


def _carry_forward_sections(previous: dict, manifest: dict) -> None:
    """Copy the sections authoring no longer prompts for out of a previously authored config.

    Authoring writes the whole manifest, so without this a re-run would silently clear the tracing
    table and budget policy an admin set with the other command — they'd have to redo them just to
    change a model.

    Each section is probe-validated before it's carried, and dropped with a warning if it no longer
    fits. Otherwise a carried section could make the manifest invalid and block the save outright,
    with no way out: the commands that could repair a section read the very manifest that can't be
    written. The live case is a budget-policy tier naming an agent the admin just de-selected, but
    hand-edited drafts and configs authored by an older ucode can trip the others the same way.
    """
    # Validating against no inventory keeps this to structural checks, which is all that's at stake
    # here: the models were just picked from the workspace's own catalog a few prompts ago.
    baseline = validate_manifest(manifest, None)
    for key, label, rebuild in CARRIED_SECTIONS:
        if key not in previous:
            continue
        candidate = previous[key]
        new_errors = [
            error
            for error in validate_manifest({**manifest, key: candidate}, None)
            if error not in baseline
        ]
        if not new_errors:
            manifest[key] = candidate
            continue
        print_warning(
            f"{label} from the existing config no longer fits what you just picked, so it was left "
            "out:"
        )
        for error in new_errors:
            print_note(error)
        print_note(f"Rebuild it with `{rebuild}`.")


def author_managed_config(
    *,
    workspace: str,
    profile: str | None,
    token: str,
    published: dict | None = None,
    command_label: str = "ucode configure",
) -> int | None:
    """Guided agent/model authoring for the workspace's managed config draft.

    The shared step-by-step picker behind ``ucode configure``'s admin path: choose the agents, then
    per-agent models / provider service / settings scope, then the default agent. Agents and models
    only — the tiered spend policy has its own command (`ucode configure spend-tiers`), and it (plus
    the tracing table) is carried forward untouched from the existing draft or the ``published``
    snapshot rather than cleared (:func:`_carry_forward_sections`).

    Saves the result to the local **draft** slot only — never publishes — then advises `ucode
    publish`. The caller (`ucode configure`) has already resolved and authenticated the workspace,
    determined admin access, and fetched the ``published`` snapshot, and hands them all in, so this
    never re-checks admin or re-fetches the workspace's config. ``token`` is reused for provider-
    service discovery; ``command_label`` brands the step banners.

    Returns a process exit code, or ``None`` when the admin selected no agents: nothing was saved, so
    the caller must not treat an earlier draft as the result of this run. Raises RuntimeError for
    actionable failures (no agents available) and KeyboardInterrupt when the admin aborts a picker; the
    CLI maps both.
    """
    # Imported here rather than at module scope: `cli` imports this module, so a top-level import
    # would be circular.
    from ucode.cli import configure_shared_state

    print_section(command_label)
    print_note("Choose the coding agents and models for this workspace's managed config.")
    print_note("Developers pull it automatically when they run ucode.")

    state = configure_shared_state(workspace, profile=profile, force_login=False)
    workspace = state.get("workspace") or workspace

    available = [
        tool
        for tool in TOOL_SPECS
        if tool not in SETUP_EXCLUDED_AGENTS and check_gateway_endpoint(state, tool)
    ]
    if not available:
        raise RuntimeError(
            f"No coding agents are available on {workspace}. Check that the workspace's AI Gateway "
            "serves models for at least one agent."
        )

    # The local draft is the carry-forward source, falling back to the passed-in published snapshot:
    # a fresh machine (or one after `ucode revert`) has no draft, and without the fallback the next
    # publish would silently wipe the workspace's tracing table and budget policy.
    previous = load_draft_config(workspace) or published or {}
    previously_enabled = [
        tool for tool in (previous.get("enabled_agents") or {}) if tool in available
    ]
    _step_banner(1, SETUP_STEP_TITLES[0], command_label)
    picked = prompt_for_tools(
        [(tool, TOOL_SPECS[tool]["display"]) for tool in available],
        preselected=previously_enabled or None,
    )
    if not picked:
        print_note("No coding agents selected — nothing to configure.")
        return None

    _step_banner(2, SETUP_STEP_TITLES[1], command_label)
    enabled_agents: dict[str, dict] = {}
    for index, tool in enumerate(picked, start=1):
        print_heading(f"{TOOL_SPECS[tool]['display']}  ({index} of {len(picked)})")
        provider_service = _select_provider_service(tool, workspace, token)
        # Always set: `_prompt_models_for_agent` re-prompts rather than returning empty, so every
        # enabled agent carries a default_model and any of them can be the default_agent.
        agent_config: dict = {
            "model_config": _prompt_models_for_agent(tool, state, provider_service)
        }
        enabled_agents[tool] = agent_config
        _confirm_agent(tool, agent_config)

    # Pick the default after configuring each agent, not before: by now the admin has seen every
    # agent's models go by, so "which is the default?" is a choice among things they've just set up
    # rather than a bare list up front. The recap reprints those picks so the choice is informed.
    _step_banner(3, SETUP_STEP_TITLES[2], command_label)
    default_agent = picked[0]
    if len(picked) > 1:
        _selected_recap(workspace, enabled_agents, default_agent=None)
        chosen = prompt_for_selection(
            "Which coding agent should be the default?",
            [(tool, TOOL_SPECS[tool]["display"]) for tool in picked],
        )
        if not chosen:
            raise KeyboardInterrupt
        default_agent = chosen
    print_success(f"Default agent set to {TOOL_SPECS[default_agent]['display']}")

    manifest: dict = {"default_agent": default_agent, "enabled_agents": enabled_agents}

    # Tracing is intentionally not prompted here: the managed-tracing path isn't working yet, so
    # asking would author a `tracing_table` the workspace can't honor. The manifest field and its
    # serialize/validate support stay in place, so a hand-written `--from-file` config can still set
    # it once the backend is ready.
    _carry_forward_sections(previous, manifest)

    errors = validate_manifest(manifest, state)
    if errors:
        # A validation failure here is a wizard bug, not admin error — the pickers only offer valid
        # choices. Surface it plainly rather than writing a manifest that `publish` would reject.
        print_err("The generated config is not valid:")
        for error in errors:
            print_note(error)
        return 1

    save_draft_config(workspace, manifest)
    _render_summary(workspace, manifest)
    console.print()
    print_success("Draft saved to ~/.ucode/managed-state.json")
    return 0


def _resolve_admin_workspace() -> tuple[str, str | None, str]:
    """Resolve the workspace a section command edits, authenticate, and strictly gate on admin.

    Returns ``(workspace, profile, token)``. Takes the workspace strictly from local state (set by
    ``ucode configure``) rather than prompting or falling back to the draft file's workspace, and
    gates with a **strict** admin check: a section command like ``ucode configure spend-tiers`` is
    admin-only up front, so an undetermined admin status is refused here rather than deferred to a
    later publish failure.
    """
    state = load_state()
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError(
            "No workspace is configured. Run `ucode configure` first, then re-run this command to "
            "author this workspace's managed config."
        )
    profile = state.get("profile")
    ensure_databricks_auth(workspace, profile)
    token = get_databricks_token(workspace, profile)
    _require_admin(workspace, token, strict=True)
    return workspace, profile, token


def _manifest_for_edit(workspace: str) -> dict:
    """The draft a section command edits, seeded from the published snapshot when no draft exists.

    Reads the local draft; on a fresh machine (no draft yet) it seeds from the workspace's published
    snapshot so an admin editing one section starts from the live config rather than an empty one.
    An empty ``enabled_agents`` counts as "no config": a launch records ``{}`` for a workspace with no
    managed config (see ``refresh_managed_config``), so a file existing is not proof of authored work.
    Requiring agents first also keeps the budget-policy tiers honest — they can only name agents the
    manifest enables.
    """
    manifest = load_draft_config(workspace) or load_published_config(workspace)
    if not (manifest or {}).get("enabled_agents"):
        raise RuntimeError(
            f"No managed config has been authored for {workspace} yet. Run `ucode configure` first "
            "to pick the agents and models, then re-run this command."
        )
    return cast(dict, manifest)


def _save_section_update(workspace: str, manifest: dict) -> int:
    """Validate the edited manifest structurally, save it as the draft, and advise publishing.

    Validated with no model inventory (``state=None``), so only structure is checked here — not model
    availability. That's deliberate: a section command doesn't touch agents or models, so re-checking
    them would only reject a legitimately pinned older Claude model (`load_state` keeps just the newest
    per family) or, worse, wrongly flag a codex/gemini model whenever the re-fetched inventory happens
    to be Claude-only. `ucode publish` runs the full model check against the live catalog before
    publishing, which is where it belongs.
    """
    errors = validate_manifest(manifest, None)
    if errors:
        print_err("The updated config is not valid:")
        for error in errors:
            print_note(error)
        return 1

    save_draft_config(workspace, manifest)
    _render_summary(workspace, manifest)
    console.print()
    print_success("Draft saved to ~/.ucode/managed-state.json")
    print_managed_next_steps(manifest)
    return 0


def configure_spend_tiers_command() -> int:
    """Author the managed config's tiered spend policy (`ucode configure spend-tiers`). Admin-only."""
    workspace, _, token = _resolve_admin_workspace()
    manifest = _manifest_for_edit(workspace)

    # The manifest's own default, shown as the "under the first rung" row of the fallback ladder so
    # the admin sees what developers get before any tier kicks in.
    default_agent = manifest.get("default_agent")
    default_config = (manifest.get("enabled_agents") or {}).get(default_agent) or {}
    default_model = (default_config.get("model_config") or {}).get("default_model")
    base_default = (
        f"{TOOL_SPECS.get(default_agent, {}).get('display', default_agent)} / {default_model}"
        if default_agent and default_model
        else ""
    )

    # `_prompt_budget_policy` returns None on an environmental dead end too (no budgets, no budget with
    # a per-user block, or the admin backing out of a picker), not only on an explicit decline. Leave
    # any existing policy untouched in every one of those cases — never pop it — so a transient budget
    # listing failure can't silently delete a policy the admin already published.
    policy = _prompt_budget_policy(
        workspace, token, manifest["enabled_agents"], load_state(), base_default=base_default
    )
    if not policy:
        print_note("The managed config's tiered spend policy is unchanged.")
        return 0
    manifest["budget_policy"] = policy
    return _save_section_update(workspace, manifest)


# Server-side failures an admin is actually likely to hit, mapped to something they can act on. The
# raw reasons are `HTTP <code> <reason>: <body>` strings from the transport, and the body carries the
# API's `error_code`, so matching on that is more robust than on status codes alone.
def _explain_publish_failure(reason: str) -> str:
    lowered = reason.lower()
    if "feature_disabled" in lowered:
        return CODING_AGENT_CONFIGS_DISABLED_MESSAGE
    if "permission_denied" in lowered or "http 403" in lowered:
        return (
            "Publishing a managed config requires workspace admin. Your account can read the "
            "workspace but not author its coding config."
        )
    if "already_exists" in lowered:
        return (
            "This workspace already has a managed config, but ucode couldn't read it to update in "
            "place. Run `ucode publish` again — if it keeps failing, the existing config may need to "
            "be deleted by hand."
        )
    if "invalid_parameter_value" in lowered:
        # The server names the offending field; passing it through beats paraphrasing.
        return f"The workspace rejected the config: {reason}"
    return f"Could not publish the managed config: {reason}"


def _with_claude_inventory(state: dict, workspace: str, profile: str | None) -> dict:
    """``state`` plus the full Claude listing, for validating a manifest against the workspace.

    ``state["claude_models"]`` holds only the newest id per family (the launch path pins one model
    per family alias), but authoring deliberately offers the older versions too — pinning
    ``default_opus_model`` to a known-good ``claude-opus-4-8`` is a normal thing for an admin to
    want. Validating against ``claude_models`` alone therefore rejected a model the wizard itself
    had just offered:

        claude: model 'system.ai.claude-opus-4-8' is not available on this workspace.

    The wizard stashes the full listing on ``state["all_claude_models"]`` mid-run, but that is never
    persisted — authoring saves the manifest, not the state — so a separate `ucode publish` process
    starts from a fresh ``load_state()`` without it. Re-fetching here makes the check independent of
    what the wizard happened to leave behind, which also covers a hand-edited or ``--from-file``
    manifest authored on another machine.

    Best-effort: a failed listing returns ``state`` untouched, leaving validation on the narrower
    inventory rather than blocking a publish on a transient API error.
    """
    if isinstance(state.get("all_claude_models"), list) and state["all_claude_models"]:
        return state
    try:
        token = get_databricks_token(workspace, profile)
        all_claude, _ = discover_claude_models_unbucketed(workspace, token)
    except (RuntimeError, OSError):
        # OSError covers a missing `databricks` binary: `get_databricks_token` shells out, so a
        # machine without the CLI on PATH raises FileNotFoundError rather than RuntimeError.
        return state
    if not all_claude:
        return state
    return {**state, "all_claude_models": all_claude}


def publish_command(*, file_path: str | None = None, yes: bool = False) -> int:
    """Publish a managed config to the workspace.

    With no ``file_path`` the locally authored manifest is published; with one, the config file
    (produced by ``ucode export``) is published instead. Both routes are validated against the
    configured workspace and canonicalized before anything is sent. Updates the existing config in
    place when there is one, rather than deleting and recreating it: a failed recreate would leave
    the workspace with no managed config at all, and every developer would silently fall back to
    their own settings. Returns a process exit code.

    Model-availability validation runs against the authored manifest for the no-file case, since the
    export round-trip that builds its payload drops the internal ``model_config.custom_models``
    exemption for hand-entered ids that discovery won't surface.
    """
    from ucode.cli import _prompt_for_configuration
    from ucode.managed_publish import load_publish_payload, parse_publish_payload

    print_section("ucode publish")

    state = load_state()
    workspace = state.get("workspace")
    profile = state.get("profile")
    if not workspace:
        workspace, profile = _prompt_for_configuration()

    manifest, api_payload = parse_publish_payload(load_publish_payload(file_path), workspace)

    # Auth first: validating a Claude manifest needs the workspace's full model listing, and that
    # listing needs a token. Nothing is written until well below this point.
    ensure_databricks_auth(workspace, profile)

    validation_manifest = load_draft_config(workspace) if file_path is None else manifest
    errors = validate_manifest(
        validation_manifest or manifest, _with_claude_inventory(state, workspace, profile)
    )
    if errors:
        print_err("The config is not valid, so it was not published:")
        for error in errors:
            print_note(error)
        if file_path is None:
            print_note("Re-run `ucode configure` to fix it, or edit ~/.ucode/managed-state.json.")
        else:
            print_note("Fix the config file and re-run `ucode publish -f`.")
        return 1

    token = get_databricks_token(workspace, profile)
    _require_admin(workspace, token)

    _render_summary(workspace, manifest)

    # Read before writing: the resource name tells us whether to create or update, and shows the
    # admin what they are about to overwrite.
    with spinner("Checking for an existing managed config..."):
        existing, reason = get_managed_config(workspace, token)
    if reason is not None:
        if "feature_disabled" in reason.lower():
            raise RuntimeError(CODING_AGENT_CONFIGS_DISABLED_MESSAGE)
        raise RuntimeError(
            f"Could not check whether {workspace} already has a managed config: {reason}. "
            "Refusing to publish without knowing, since that could overwrite a config silently."
        )

    existing_name = (existing or {}).get("name")
    if existing is not None and not isinstance(existing_name, str):
        raise RuntimeError(
            "This workspace has a managed config but the API didn't return its resource name, so "
            "ucode can't update it in place. Delete it in the workspace and re-run `ucode publish`."
        )

    console.print()
    if existing is None:
        print_note(f"This will create a new managed config on {workspace}.")
    else:
        # Diff against what's live, normalized the same way, so the admin sees exactly what changes
        # rather than a bare "this replaces the current config". Comparing the round-tripped payload
        # (not the raw manifest) shows the real post-publish state — any field serialization drops
        # won't appear as a phantom change.
        changed = _render_config_diff(existing, manifest, workspace)
        if not changed:
            print_success(f"{workspace}'s published config already matches this one.")
            print_note("Nothing to publish.")
            save_published_config(workspace, manifest)
            return 0
        console.print()
        print_warning("This takes effect for every developer on their next `ucode` run.")
    if not yes and not prompt_yes_no_default("Publish this config?", default=False):
        print_note("Nothing was published.")
        return 1

    if existing is None:
        with spinner("Publishing the managed config..."):
            published, publish_reason = create_coding_agent_config(workspace, token, api_payload)
    else:
        with spinner("Updating the managed config..."):
            published, publish_reason = update_coding_agent_config(
                workspace, token, cast("str", existing_name), api_payload
            )
    if publish_reason is not None:
        raise RuntimeError(_explain_publish_failure(publish_reason))

    save_published_config(workspace, manifest)

    name = (published or {}).get("name") or existing_name or "coding-agent-configs/?"
    print_success(f"Published {name} to {workspace}")
    print_note("Developers get it automatically the next time they run `ucode`.")
    return 0


__all__ = [
    "author_managed_config",
    "configure_from_file",
    "configure_spend_tiers_command",
    "print_managed_next_steps",
    "publish_command",
]
