"""Interactive `ucode setup`: author the workspace's managed coding-agent config.

Workspace admins run this to build the ``CodingAgentConfig`` their developers will pull. It walks
the admin through agents, per-agent models, tracing, MCP servers, skills, and a spend-routing budget
policy, then writes the manifest to ``~/.ucode/managed-state.json`` (the one local managed-config
file, owned by :mod:`ucode.managed_config`). An admin can try it with ``ucode --dry-run`` and then
publish it to the workspace with ``ucode apply`` (a separate command, so the file can be reviewed
first).

Serialization, validation, and the per-agent model catalogs live in :mod:`ucode.managed_setup`; this
module is the interaction layer on top of them. Sub-flows an admin already knows — tracing, MCP,
skills — are delegated to the existing ``ucode configure <thing>`` commands and their results read
back out of ``state.json``, so there is exactly one picker per concern in the codebase.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from ucode.agents import TOOL_SPECS, check_gateway_endpoint
from ucode.config_io import is_dry_run
from ucode.databricks import (
    ANTHROPIC_FAMILIES,
    all_users_can_use_schema,
    create_coding_agent_config,
    delete_coding_agent_config,
    discover_claude_models_unbucketed,
    ensure_databricks_auth,
    get_databricks_token,
    has_cached_model_provider_services,
    is_model_provider_feature_unavailable,
    is_workspace_admin,
    list_model_provider_services,
    list_workspace_budgets,
    map_claude_family_models,
    service_usable_for_tool,
    update_coding_agent_config,
)
from ucode.managed_config import (
    get_managed_config,
    load_managed_state,
    managed_state_workspace,
    save_managed_state,
)
from ucode.managed_setup import (
    CLAUDE_SLOT_FOR_FAMILY,
    claude_family_candidates,
    claude_family_for_model,
    model_options_for_agent,
    serialize_managed_config,
    supports_provider_service,
    validate_manifest,
)
from ucode.state import load_state
from ucode.ui import (
    console,
    kv_line,
    print_err,
    print_heading,
    print_note,
    print_panel,
    print_section,
    print_success,
    print_warning,
    prompt_for_multi_selection,
    prompt_for_percentage,
    prompt_for_selection,
    prompt_for_text,
    prompt_for_tools,
    prompt_yes_no_default,
    spinner,
)

# What `use_as_global_settings` actually does, in plain terms. Admins are choosing between a
# machine-wide managed settings file and a per-user one, which is not obvious from the field name.
GLOBAL_SETTINGS_BLURB = (
    "Write this agent's config to the machine's managed settings file, which applies to every "
    "user on the machine and cannot be overridden locally. Answer no to write the per-user "
    "settings file instead, which developers can still change."
)

BUDGET_POLICY_BLURB = (
    "A budget policy moves developers onto cheaper agents and models as the workspace spends "
    "against a budget — for example Claude Code on Opus by default, then Sonnet at 80%, then "
    "OpenCode on Kimi at 100%. It only changes the default; developers can still pick anything "
    "they have access to. Hard caps stay with the budget's own blocking threshold."
)


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


def _mcp_server_from_url(url: str) -> tuple[str, str] | None:
    """Derive a managed-config ``(name, type)`` entry from a registered server's resolved URL.

    ``state.json`` stores each MCP server's resolved URL but not its type, while the managed config
    stores ``{name, type}`` and lets the developer's ucode rebuild the URL. So map the URL back to the
    type *and* the identifier the ai-gateway ``McpServer.name`` field is meant to hold for that type
    (a UC name for a UC service, a Genie space id for a genie space, a `<catalog>.<schema>` for
    vector-search / uc-functions, a connection name for external). Deriving ``name`` from the URL —
    rather than reusing the local display slug — is what lets the developer's ucode reconstruct the
    URL on launch. Returns None for a URL that matches nothing reconstructable (e.g. an app's
    off-workspace host), so those are skipped rather than published unusably.
    """
    stripped = url.rstrip("/")
    marker = "/ai-gateway/mcp-services/"
    if marker in url:
        # `.../mcp-services/<catalog>.<schema>.<svc>` — store the dash form the launch path expects.
        service = url.split(marker, 1)[1].split("/", 1)[0]
        return service.replace(".", "-"), "mcp-service"
    for fragment, tag in (
        ("/api/2.0/mcp/external/", "external"),
        ("/api/2.0/mcp/genie/", "genie-space"),
    ):
        if fragment in url:
            # external -> connection name; genie -> space id. Both are the single trailing segment.
            return url.split(fragment, 1)[1].split("/", 1)[0], tag
    for fragment, tag in (
        ("/api/2.0/mcp/vector-search/", "vector-search"),
        ("/api/2.0/mcp/functions/", "uc-functions"),
    ):
        if fragment in url:
            # `.../<catalog>/<schema>` — store the `<catalog>.<schema>` the launch path splits back.
            rest = url.split(fragment, 1)[1].split("/")
            if len(rest) >= 2 and rest[0] and rest[1]:
                return f"{rest[0]}.{rest[1]}", tag
            return None
    if stripped.endswith("/api/2.0/mcp/sql"):
        return "databricks-sql", "sql"
    # Databricks apps are the residual case: an arbitrary app host with a /mcp suffix. Its host isn't
    # reconstructable from the workspace + an id, so it can't be published to the managed config yet.
    if stripped.endswith("/mcp"):
        return None
    return None


def _mcp_servers_from_state(state: dict) -> list[dict]:
    """The registered MCP servers, as managed-config ``{name, type}`` entries.

    Skips the skills registry connection: skills are published under the manifest's own ``skills``
    field, so including its MCP entry would configure it twice.
    """
    from ucode.mcp import SKILLS_MCP_KIND

    servers: list[dict] = []
    seen: set[str] = set()
    for entry in state.get("mcp_servers") or []:
        if not isinstance(entry, dict) or entry.get("kind") == SKILLS_MCP_KIND:
            continue
        name = entry.get("name")
        url = entry.get("url")
        if not isinstance(name, str) or not name or not isinstance(url, str):
            continue
        resolved = _mcp_server_from_url(url)
        if resolved is None:
            print_warning(
                f"Skipping MCP server '{name}': ucode can't publish it to a managed config "
                f"(unrecognized or app-hosted URL: {url})."
            )
            continue
        config_name, tag = resolved
        if config_name in seen:
            continue
        seen.add(config_name)
        servers.append({"name": config_name, "type": tag})
    return servers


def _skill_names_from_state(state: dict) -> list[str]:
    """Skill schemas registered on the skills MCP connection (``catalog.schema`` entries)."""
    from ucode.mcp import _skill_mcp_locations

    return [name for name in _skill_mcp_locations(state) if isinstance(name, str) and name]


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

    if tool in SINGLE_MODEL_AGENTS:
        return {
            "default_model": _require_selection(
                f"Select the model for {display}:", [(model, model) for model in options]
            )
        }

    # Nothing pre-checked: the first option is whatever discovery sorted first, not a
    # recommendation — for pi it is a Claude model, for codex the oldest GPT. Pre-checking it made
    # "hit Enter" produce an arbitrary config. (A worthwhile follow-up is to pre-check the models
    # this workspace was configured with last time, which `load_managed_state` already loads for
    # the agent picker, so a re-run becomes an edit rather than a re-entry.)
    picked = _require_multi_selection(
        f"Select models for {display}:",
        [(model, model) for model in options],
    )
    if len(picked) == 1:
        model_config["default_model"] = picked[0]
    else:
        model_config["default_model"] = _require_selection(
            f"Default model for {display}:", [(model, model) for model in picked]
        )

    model_config["models"] = picked
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
    for family in ANTHROPIC_FAMILIES:
        family_models = candidates.get(family)
        if not family_models:
            continue
        choice = prompt_for_selection(
            f"Default {family} model:",
            [(model, model) for model in family_models] + [(_SKIP_FAMILY, f"(skip {family})")],
            searchable=True,
        )
        if choice is None:
            raise KeyboardInterrupt
        if choice != _SKIP_FAMILY:
            slots[CLAUDE_SLOT_FOR_FAMILY[family]] = choice

    if not slots:
        # Every slot skipped is a legitimate, minimal config: the proto leaves `models` optional and
        # each unset slot falls back to `default_model`, so one model covers every family. Pick it
        # from the same candidates rather than asking the admin to type an id.
        print_note(f"No families configured, so {display} will use a single model for all of them.")
        every_model = [m for family_models in candidates.values() for m in family_models]
        return {
            "default_model": _require_selection(
                f"Which model should {display} use?",
                [(m, m) for m in dict.fromkeys(every_model)],
            )
        }

    chosen = list(dict.fromkeys(slots.values()))
    model_config: dict = {"models": slots}
    if len(chosen) == 1:
        # A one-option prompt is a wasted keystroke, but skipping it silently reads as a dropped
        # step — say what was inferred so the admin knows the default is set, and to what.
        model_config["default_model"] = chosen[0]
        print_success(f"Overall default for {display}: {chosen[0]} (the only model configured)")
    else:
        model_config["default_model"] = _require_selection(
            f"Which of those is {display}'s overall default?", [(m, m) for m in chosen]
        )
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
        print_success(f"{display}: {summary} (default: {default_family})")
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
        print_success(f"Overall default for {display}: {chosen[0]} (the only model configured)")
    else:
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


def _prompt_budget_policy(
    workspace: str, token: str, enabled_agents: dict[str, dict], state: dict
) -> dict | None:
    """Author a spend-routing ``budget_policy``, or None when the admin declines or can't.

    Budgets themselves are created in the Databricks console (they're account-level objects), so the
    admin picks an existing one here. Tiers are prompted in percent and stored as fractions, which is
    what the API validates.

    A tier's model choices come from what the admin configured for that agent earlier in this run —
    not the workspace catalog. Offering the catalog would let a tier point an agent at a model it
    wasn't given, which neither this validation nor the server's would reject: the tier would
    activate and hand the developer a model their agent doesn't have.
    """
    print_section("Budget policy")
    print_note(BUDGET_POLICY_BLURB)
    if not prompt_yes_no_default("Set up a budget policy for this workspace?", default=False):
        return None

    with spinner("Listing workspace budgets..."):
        budgets, reason = list_workspace_budgets(workspace, token)
    if reason is not None or not budgets:
        print_warning(
            "No AI Gateway budgets are visible for this workspace, so there is nothing to attach a "
            "policy to. Create a budget in the Databricks console first, then re-run `ucode setup`."
        )
        return None

    # Spend routing only works on a budget with a per-user threshold; without one the gateway reports
    # no spend and every tier stays inert. The listing can't reveal the alert's action, so this hides
    # the clearly-unusable budgets and the server rejects the rest on create.
    usable = [budget for budget in budgets if budget.get("has_per_user_alert")]
    if not usable:
        print_warning(
            "None of this workspace's AI Gateway budgets have a per-user threshold configured, which "
            "spend routing requires. Add a per-user alert threshold to a budget in the Databricks "
            "console, then re-run `ucode setup`."
        )
        return None
    print_note(
        "Showing only budgets with a per-user threshold configured, which spend routing needs."
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
    display_name = prompt_for_text("Policy name", default="coding-agents-tiered-routing")
    if display_name:
        policy["display_name"] = display_name

    tiers: list[dict] = []
    seen_percentages: set[float] = set()
    seen_combos: set[tuple[str, str]] = set()
    print_note(
        "Add one tier per step-down. Each tier activates once spend reaches its percentage, and "
        "the highest activated tier wins."
    )
    while True:
        index = len(tiers) + 1
        fraction = prompt_for_percentage(f"Tier {index}: activates at what percent of budget?")
        if fraction in seen_percentages:
            print_err("That percentage is already used by another tier; pick a different one.")
            continue
        agent = prompt_for_selection(
            f"Tier {index}: which agent becomes the default?",
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
                f"Tier {index}: which model?", [(m, m) for m in options], searchable=True
            )
        else:
            model = prompt_for_text(f"Tier {index}: which model?")
        if not model:
            break
        if (agent, model) in seen_combos:
            # The highest crossed tier wins, so a second tier on the same agent+model never changes
            # what the lower one already selected — it is a step-down that doesn't step down. Reject
            # it here rather than let the admin build a policy with a silently inert tier.
            print_err(
                f"{TOOL_SPECS[agent]['display']} / {model} is already used by another tier; a "
                "repeated agent/model makes this tier a no-op. Pick a different one."
            )
            continue
        seen_percentages.add(fraction)
        seen_combos.add((agent, model))
        tiers.append(
            {
                "spending_percentage": fraction,
                "default_agent": agent,
                "default_model": model,
            }
        )
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
        scope = "machine-wide" if agent_config.get("use_as_global_settings") else "per-user"
        lines.append(kv_line(display, f"{detail} ({scope})"))
        # Spell out the per-family slots and model lists: the one-line default alone doesn't show
        # which families an admin configured, which is most of what they chose for claude.
        models = model_config.get("models")
        if isinstance(models, dict):
            for slot, model in models.items():
                family = slot.removeprefix("default_").removesuffix("_model")
                lines.append(kv_line(f"  {family}", str(model)))
        elif isinstance(models, list) and len(models) > 1:
            lines.append(kv_line("  models", ", ".join(str(m) for m in models)))

    mcp_servers = manifest.get("mcp_servers") or []
    lines.append(
        kv_line(
            "MCP servers",
            ", ".join(str(server.get("name")) for server in mcp_servers) if mcp_servers else "none",
        )
    )
    skills = (manifest.get("skills") or {}).get("names") or []
    lines.append(kv_line("Skills", ", ".join(skills) if skills else "none"))
    lines.append(kv_line("Tracing", manifest.get("tracing_table") or "disabled"))

    policy = manifest.get("budget_policy")
    if isinstance(policy, dict):
        tiers = policy.get("tiers") or []
        lines.append(
            kv_line("Budget policy", policy.get("display_name") or policy.get("budget_id") or "set")
        )
        for tier in tiers:
            agent = tier.get("default_agent")
            display = TOOL_SPECS.get(agent, {}).get("display", agent)
            percent = float(tier.get("spending_percentage", 0)) * 100
            lines.append(kv_line(f"  at {percent:g}%", f"{display} / {tier.get('default_model')}"))
    else:
        lines.append(kv_line("Budget policy", "none"))

    print_panel("Configuration summary", lines)


def _require_admin(workspace: str, token: str) -> None:
    """Stop unless the caller is a workspace admin.

    An unverifiable check (SCIM unreachable) warns and continues: the API enforces the same rule, so
    the worst case is a clear PERMISSION_DENIED at publish time rather than a false block here.
    """
    with spinner("Checking workspace admin permissions..."):
        admin = is_workspace_admin(workspace, token)
    if admin is False:
        raise RuntimeError(
            f"You are not an admin of {workspace}. `ucode setup` authors the workspace-wide "
            "coding config, so it is restricted to workspace admins."
        )
    if admin is None:
        print_warning(
            "Could not verify workspace admin permissions. Continuing — `ucode apply` will fail "
            "if you lack them."
        )
    else:
        print_success("Admin permissions verified")


def _handle_existing_config(workspace: str, token: str) -> bool:
    """Decide what to do when the workspace already has a published config.

    Returns True to keep authoring a new config (the wizard continues; publishing later replaces the
    existing one) and False to stop (no config exists, the check failed, or the admin chose to delete
    the existing one instead of authoring a replacement).

    Deliberately doesn't itemize what the existing config holds. The admin doesn't need an inventory
    to act on this — the instruction is the same either way ("include everything you want to keep")
    — and `ucode setup show` prints the real thing for anyone who wants to compare.
    """
    with spinner("Checking for an existing managed config..."):
        existing, reason = get_managed_config(workspace, token)
    if reason is not None:
        print_note(f"Could not check for an existing config: {reason}")
        return True
    if existing is None:
        return True

    print_warning(
        "This workspace already has a managed configuration — one config covers every agent, MCP "
        "server, skill, tracing table, and budget policy for the whole workspace."
    )
    choice = prompt_for_selection(
        "What would you like to do?",
        [
            ("create", "Author a new config (replaces the existing one when you publish)"),
            ("delete", "Delete the existing config (removes it from the workspace, leaves none)"),
        ],
    )
    if choice is None:
        raise KeyboardInterrupt
    if choice == "create":
        print_note("Make sure this run includes everything you want to keep.")
        return True

    _delete_existing_config(workspace, token, existing)
    return False


def _delete_existing_config(workspace: str, token: str, existing: dict) -> None:
    """Delete the workspace's published config after confirming. Raises RuntimeError on failure.

    Deleting leaves the workspace with no managed config, so every developer falls back to their own
    settings on their next ucode run — confirm before doing it, and honor ``--dry-run``.
    """
    name = existing.get("name")
    if not isinstance(name, str):
        raise RuntimeError(
            "This workspace has a managed config but the API didn't return its resource name, so "
            "ucode can't delete it. Delete it in the workspace directly."
        )
    print_warning(
        "Deleting removes the managed config entirely. Every developer falls back to their own "
        "settings on their next ucode run."
    )
    if not prompt_yes_no_default("Delete the existing managed config?", default=False):
        print_note("Nothing was deleted.")
        return
    if is_dry_run():
        print_success("Dry run: the config was not deleted.")
        return
    with spinner("Deleting the managed config..."):
        delete_reason = delete_coding_agent_config(workspace, token, name)
    if delete_reason is not None:
        raise RuntimeError(f"Could not delete the managed config on {workspace}: {delete_reason}.")
    print_success(f"Deleted the managed config from {workspace}")


def setup_from_file(path: str) -> int:
    """Validate an admin-written manifest and save it, skipping the interactive flow.

    The non-interactive path for CI and for admins who'd rather keep the JSON in version control.
    Reads ucode's own manifest shape (the same thing the wizard writes), not proto-JSON.
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

    errors = validate_manifest(manifest, state)
    if errors:
        print_err(f"{manifest_path} is not a valid managed config:")
        for error in errors:
            print_note(error)
        return 1

    save_managed_state(workspace, manifest)
    _render_summary(workspace, manifest)
    print_success(f"Saved to {manifest_path.name} -> ~/.ucode/managed-state.json")
    _print_next_steps()
    return 0


def _print_next_steps() -> None:
    console.print()
    print_heading("Next steps")
    # The authored manifest is saved to the same local file a launch reads, so `ucode --dry-run`
    # previews this machine's agents *as configured by the manifest* without fetching or overwriting
    # it — a real local test of the config before it is published.
    print_note("Try it locally:               ucode --dry-run")
    print_note("Publish it to the workspace:  ucode apply")


def setup_command(from_file: str | None = None) -> int:
    """Author the workspace's managed coding-agent config interactively.

    Returns a process exit code. Raises RuntimeError for actionable failures (not an admin, no
    agents available) and KeyboardInterrupt when the admin aborts a picker; the CLI maps both.
    """
    if from_file is not None:
        return setup_from_file(from_file)

    # Imported here rather than at module scope: `cli` imports this module, so a top-level import
    # would be circular.
    from ucode.cli import _prompt_for_configuration, configure_shared_state

    print_section("ucode setup")
    print_note("Author the managed coding config for this workspace.")
    print_note("Developers pull it automatically when they run ucode.")

    workspace, profile = _prompt_for_configuration()
    # `configure_shared_state` below authenticates too and prints its own success line, so this one
    # stays quiet rather than reporting the same thing twice. It still has to run first: the admin
    # gate and the existing-config check both need a token before discovery.
    ensure_databricks_auth(workspace, profile, quiet=True)
    token = get_databricks_token(workspace, profile)

    _require_admin(workspace, token)
    if not _handle_existing_config(workspace, token):
        return 0

    # Discover the workspace's models and gateway URLs. This also logs in and persists local state,
    # which is what lets the admin dry-run the config on their own machine afterwards.
    state = configure_shared_state(workspace, profile=profile, force_login=False)
    workspace = state.get("workspace") or workspace
    profile = state.get("profile") or profile

    available = [tool for tool in TOOL_SPECS if check_gateway_endpoint(state, tool)]
    if not available:
        raise RuntimeError(
            f"No coding agents are available on {workspace}. Check that the workspace's AI Gateway "
            "serves models for at least one agent."
        )

    previous = load_managed_state(workspace) or {}
    previously_enabled = [
        tool for tool in (previous.get("enabled_agents") or {}) if tool in TOOL_SPECS
    ]
    picked = prompt_for_tools(
        [(tool, TOOL_SPECS[tool]["display"]) for tool in available],
        preselected=previously_enabled or None,
    )
    if not picked:
        print_note("No coding agents selected — nothing to configure.")
        return 0

    default_agent = picked[0]
    if len(picked) > 1:
        chosen = prompt_for_selection(
            "Which agent should launch when a developer runs `ucode`?",
            [(tool, TOOL_SPECS[tool]["display"]) for tool in picked],
        )
        if not chosen:
            raise KeyboardInterrupt
        default_agent = chosen
    print_success(f"Default agent set to {TOOL_SPECS[default_agent]['display']}")

    enabled_agents: dict[str, dict] = {}
    for tool in picked:
        print_heading(TOOL_SPECS[tool]["display"])
        provider_service = _select_provider_service(tool, workspace, token)
        # Always set: `_prompt_models_for_agent` re-prompts rather than returning empty, so every
        # enabled agent carries a default_model and any of them can be the default_agent.
        agent_config: dict = {
            "model_config": _prompt_models_for_agent(tool, state, provider_service)
        }
        agent_config["use_as_global_settings"] = prompt_yes_no_default(
            f"Apply {TOOL_SPECS[tool]['display']} config machine-wide? ({GLOBAL_SETTINGS_BLURB})",
            default=False,
        )
        enabled_agents[tool] = agent_config

    manifest: dict = {"default_agent": default_agent, "enabled_agents": enabled_agents}

    # Tracing is intentionally not prompted here: the managed-tracing path isn't working yet, so
    # asking would author a `tracing_table` the workspace can't honor. The manifest field and its
    # serialize/validate support stay in place, so a hand-written `--from-file` config can still set
    # it once the backend is ready. Re-add the section below when it is.

    print_section("MCP servers")
    if prompt_yes_no_default("Set up managed MCP servers for this workspace?", default=False):
        from ucode.mcp import configure_mcp_command

        # Managed configs can't carry a Databricks app (its host isn't reconstructable from the
        # workspace), so hide apps from the picker rather than let an admin pick one that is then
        # dropped from the published config.
        configure_mcp_command(exclude_sources={"apps"})
        mcp_servers = _mcp_servers_from_state(load_state())
        if mcp_servers:
            manifest["mcp_servers"] = mcp_servers
            print_success(f"{len(mcp_servers)} MCP server(s) added to the managed config")

    print_section("Skills")
    if prompt_yes_no_default("Set up managed skills for this workspace?", default=False):
        locations = prompt_for_text(
            "Skill schemas to publish, comma-separated `catalog.schema` (blank to skip)",
            default="",
        )
        parsed: list[str] = [item.strip() for item in (locations or "").split(",") if item.strip()]
        if parsed:
            from ucode.mcp import configure_skills_mcp_command

            configure_skills_mcp_command(parsed)
            skill_names = _skill_names_from_state(load_state()) or parsed
            manifest["skills"] = {"names": skill_names}
            print_success(f"{len(skill_names)} skill schema(s) added to the managed config")

    budget_policy = _prompt_budget_policy(workspace, token, enabled_agents, state)
    if budget_policy:
        manifest["budget_policy"] = budget_policy

    errors = validate_manifest(manifest, state)
    if errors:
        # A validation failure here is a wizard bug, not admin error — the pickers only offer valid
        # choices. Surface it plainly rather than writing a manifest that `apply` would reject.
        print_err("The generated config is not valid:")
        for error in errors:
            print_note(error)
        return 1

    save_managed_state(workspace, manifest)
    _render_summary(workspace, manifest)
    console.print()
    print_success("Saved to ~/.ucode/managed-state.json")
    _print_next_steps()
    return 0


def show_command() -> int:
    """Print the authored manifest and the proto-JSON `ucode apply` would publish."""
    # Fall back to the workspace the on-disk file was authored for, so `ucode setup --show` still
    # works before `ucode configure` has put a workspace in local state.
    workspace = load_state().get("workspace") or managed_state_workspace()
    manifest = load_managed_state(workspace)
    if manifest is None:
        print_note("No managed config has been authored yet. Run `ucode setup` to create one.")
        return 0
    _render_summary(workspace or "unknown", manifest)
    console.print()
    print_heading("Payload for `ucode apply`")
    console.print(json.dumps(serialize_managed_config(manifest), indent=2))
    return 0


# Server-side failures an admin is actually likely to hit, mapped to something they can act on. The
# raw reasons are `HTTP <code> <reason>: <body>` strings from the transport, and the body carries the
# API's `error_code`, so matching on that is more robust than on status codes alone.
def _explain_publish_failure(reason: str) -> str:
    lowered = reason.lower()
    if "feature_disabled" in lowered:
        return (
            "Managed coding-agent configs aren't enabled on this workspace yet. Ask your Databricks "
            "contact to enable the `codingAgentConfigCrudEnabled` flag for it, then re-run "
            "`ucode apply`."
        )
    if "permission_denied" in lowered or "http 403" in lowered:
        return (
            "Publishing a managed config requires workspace admin. Your account can read the "
            "workspace but not author its coding config."
        )
    if "already_exists" in lowered:
        return (
            "This workspace already has a managed config, but ucode couldn't read it to update in "
            "place. Run `ucode apply` again — if it keeps failing, the existing config may need to "
            "be deleted by hand."
        )
    if "invalid_parameter_value" in lowered:
        # The server names the offending field; passing it through beats paraphrasing.
        return f"The workspace rejected the config: {reason}"
    return f"Could not publish the managed config: {reason}"


def _with_claude_inventory(state: dict, workspace: str, profile: str | None) -> dict:
    """``state`` plus the full Claude listing, for validating a manifest against the workspace.

    ``state["claude_models"]`` holds only the newest id per family (the launch path pins one model
    per family alias), but `ucode setup` deliberately offers the older versions too — pinning
    ``default_opus_model`` to a known-good ``claude-opus-4-8`` is a normal thing for an admin to
    want. Validating against ``claude_models`` alone therefore rejected a model the wizard itself
    had just offered:

        claude: model 'system.ai.claude-opus-4-8' is not available on this workspace.

    The wizard stashes the full listing on ``state["all_claude_models"]`` mid-run, but that is never
    persisted — `setup` saves the manifest, not the state — so a separate `ucode apply` process
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


def apply_command(*, yes: bool = False) -> int:
    """Publish the authored manifest to the workspace.

    Updates the existing config in place when there is one, rather than deleting and recreating it:
    a failed recreate would leave the workspace with no managed config at all, and every developer
    would silently fall back to their own settings. Returns a process exit code.
    """
    from ucode.cli import _prompt_for_configuration

    print_section("ucode apply")

    state = load_state()
    workspace = state.get("workspace")
    profile = state.get("profile")
    if not workspace:
        workspace, profile = _prompt_for_configuration()

    manifest = load_managed_state(workspace)
    if manifest is None:
        raise RuntimeError(
            "No managed config has been authored for this workspace. Run `ucode setup` first "
            "(or `ucode setup --from-file <json>`)."
        )

    # Auth first: validating a Claude manifest needs the workspace's full model listing, and that
    # listing needs a token. Nothing is written until well below this point.
    ensure_databricks_auth(workspace, profile)

    errors = validate_manifest(manifest, _with_claude_inventory(state, workspace, profile))
    if errors:
        print_err("The authored config is not valid, so it was not published:")
        for error in errors:
            print_note(error)
        print_note("Re-run `ucode setup` to fix it, or edit ~/.ucode/managed-state.json.")
        return 1

    token = get_databricks_token(workspace, profile)
    _require_admin(workspace, token)

    payload = serialize_managed_config(manifest)
    _render_summary(workspace, manifest)

    # Read before writing: the resource name tells us whether to create or update, and shows the
    # admin what they are about to overwrite.
    with spinner("Checking for an existing managed config..."):
        existing, reason = get_managed_config(workspace, token)
    if reason is not None:
        raise RuntimeError(
            f"Could not check whether {workspace} already has a managed config: {reason}. "
            "Refusing to publish without knowing, since that could overwrite a config silently."
        )

    existing_name = (existing or {}).get("name")
    if existing is not None and not isinstance(existing_name, str):
        raise RuntimeError(
            "This workspace has a managed config but the API didn't return its resource name, so "
            "ucode can't update it in place. Delete it in the workspace and re-run `ucode apply`."
        )

    console.print()
    if existing is None:
        print_note(f"This will create a new managed config on {workspace}.")
    else:
        agents = ", ".join((existing.get("enabled_agents") or {}).keys()) or "no agents"
        print_warning(
            f"This will replace the config already published on {workspace} (currently: {agents}). "
            "Every developer picks the new one up on their next ucode run."
        )
    if not yes and not prompt_yes_no_default("Publish this config?", default=False):
        print_note("Nothing was published.")
        return 1

    if existing is None:
        with spinner("Publishing the managed config..."):
            published, publish_reason = create_coding_agent_config(workspace, token, payload)
    else:
        with spinner("Updating the managed config..."):
            published, publish_reason = update_coding_agent_config(
                workspace, token, cast("str", existing_name), payload
            )
    if publish_reason is not None:
        raise RuntimeError(_explain_publish_failure(publish_reason))

    name = (published or {}).get("name") or existing_name or "coding-agent-configs/?"
    print_success(f"Published {name} to {workspace}")
    print_note("Developers pick this up on their next ucode run.")
    return 0


__all__ = ["apply_command", "setup_command", "setup_from_file", "show_command"]
