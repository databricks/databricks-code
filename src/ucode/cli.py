#!/usr/bin/env python3
"""CLI entry point for ucode."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.panel import Panel

from ucode.agents import (
    TOOL_SPECS,
    check_gateway_endpoint,
    configure_selected_tools,
    configure_single_tool,
    configure_tool,
    ensure_bootstrap_dependencies,
    ensure_provider_state,
    install_tool_binary,
    normalize_tool,
    provider_permission_error,
    resolve_launch_model,
    resolve_provider_models,
    validate_all_tools,
    validate_tool,
)
from ucode.agents import (
    launch as launch_agent,
)
from ucode.agents.codex import revert_legacy_shared_config
from ucode.agents.pi import PI_SETTINGS_BACKUP_PATH, PI_SETTINGS_PATH
from ucode.config_io import restore_file, set_dry_run
from ucode.databricks import (
    apply_pat_environment,
    build_shared_base_urls,
    discover_claude_models,
    discover_codex_models,
    discover_gemini_models,
    discover_model_services,
    ensure_ai_gateway_v2,
    ensure_databricks_auth,
    ensure_pat_bearer,
    find_profile_name_for_host,
    get_databricks_profiles,
    get_databricks_token,
    install_databricks_cli,
    is_model_provider_feature_unavailable,
    list_profile_entries,
    list_tool_provider_services,
    normalize_workspace_url,
    resolve_pat_token,
    run_databricks_login,
)
from ucode.mcp import (
    MCP_CLIENTS,
    SKILLS_MCP_KIND,
    configure_mcp_command,
    configure_skills_mcp_command,
    purge_cross_workspace_mcp_residue,
    revert_mcp_configs,
)
from ucode.state import (
    STATE_PATH,
    clear_state,
    get_provider_service,
    load_full_state,
    load_state,
    save_state,
    set_provider_service,
)
from ucode.tracing import configure_tracing_command
from ucode.ui import (
    console,
    heading,
    print_err,
    print_heading,
    print_kv,
    print_note,
    print_section,
    print_success,
    print_warning,
    prompt_for_selection,
    prompt_for_tools,
    prompt_for_workspace,
    prompt_yes_no_default,
    set_verbosity,
    spinner,
    status_badge,
)
from ucode.usage import usage as usage_report

_DISCOVERY_CONSUMERS: dict[str, tuple[str, ...]] = {
    "claude": ("claude", "opencode", "copilot", "pi"),
    "codex": ("codex", "copilot", "pi"),
    "gemini": ("gemini", "opencode", "pi"),
    "oss": ("opencode",),
}


def _print_discovery_diagnostics(state: dict) -> None:
    """Surface per-source reasons after a failed discovery so the user knows
    which API call returned what — instead of the generic 'no agents' line."""
    reasons = state.get("_discovery_reasons") or {}
    if not reasons:
        return
    labels = {
        "claude": "Claude models",
        "codex": "Codex models",
        "gemini": "Gemini models",
        "oss": "OSS models",
    }
    for source, reason in reasons.items():
        consumers = ", ".join(_DISCOVERY_CONSUMERS.get(source, ()))
        label = labels.get(source, source)
        if reason:
            print_note(f"{label} (needed for: {consumers}): {reason}")
        else:
            print_note(f"{label} (needed for: {consumers}): no models returned")
    print_note("Re-run with `UCODE_DEBUG=1` to log raw discovery responses to ~/.ucode/debug.log.")


def _prompt_for_configuration(tool: str | None = None) -> tuple[str, str | None]:
    if tool is None:
        desc = "Configure your Databricks workspace"
    else:
        desc = f"Configure {TOOL_SPECS[tool]['display']} to use your Databricks endpoint."
    with spinner("Loading Databricks workspaces and profiles..."):
        profiles = get_databricks_profiles()
    return prompt_for_workspace(desc, profiles)


def _parse_agents_option(agents: str) -> list[str]:
    tools: list[str] = []
    for raw_tool in agents.split(","):
        raw_tool = raw_tool.strip()
        if not raw_tool:
            continue
        tool = normalize_tool(raw_tool)
        if tool not in tools:
            tools.append(tool)
    if not tools:
        raise RuntimeError(
            "No agents provided for --agents. Use a comma-separated list like `--agents claude,codex`."
        )
    return tools


def _parse_skill_locations(location: str) -> list[str]:
    """Parse a comma-separated `--location` into `<catalog>.<schema>` refs,
    dropping duplicates while preserving order."""
    locations: list[str] = []
    for raw in location.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(".")
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise RuntimeError(f"--location entries must be `<catalog>.<schema>`, got `{raw}`.")
        if raw not in locations:
            locations.append(raw)
    if not locations:
        raise RuntimeError(
            "No schemas provided for --location. Use `<catalog>.<schema>`, "
            "comma-separated for multiple."
        )
    return locations


def _parse_workspaces_option(workspaces: str) -> list[tuple[str, str | None]]:
    """Parse `--workspaces` into [(url, profile_name | None), ...].

    `--workspaces` supplies bare URLs; the matching profile (if any) is
    resolved later via `find_profile_name_for_host`.
    """
    workspace_entries: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for raw_workspace in workspaces.split(","):
        raw_workspace = raw_workspace.strip()
        if not raw_workspace:
            continue
        try:
            workspace = normalize_workspace_url(raw_workspace)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if workspace not in seen:
            seen.add(workspace)
            workspace_entries.append((workspace, None))
    if not workspace_entries:
        raise RuntimeError(
            "No workspaces provided for --workspaces. Use a comma-separated list like "
            "`--workspaces https://workspace.databricks.com`."
        )
    return workspace_entries


def _parse_profiles_option(profiles: str) -> list[tuple[str, str | None]]:
    """Parse `--profiles` into [(url, profile_name), ...].

    Each name must be an existing Databricks CLI profile; its host supplies
    the workspace URL. Auth behaves the same as `--workspaces`: OAuth login is
    forced unless `--use-pat` is also passed."""
    available = {str(p.get("name")): p for p in list_profile_entries() if p.get("name")}
    workspace_entries: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for raw_name in profiles.split(","):
        name = raw_name.strip()
        if not name:
            continue
        entry = available.get(name)
        if entry is None:
            known = ", ".join(sorted(available)) or "none"
            raise RuntimeError(
                f"Databricks CLI profile '{name}' was not found (available: {known}). "
                "Check `databricks auth profiles` or add the profile to ~/.databrickscfg."
            )
        host = str(entry.get("host") or "").strip()
        if not host:
            raise RuntimeError(
                f"Databricks CLI profile '{name}' has no host configured in ~/.databrickscfg."
            )
        try:
            workspace = normalize_workspace_url(host)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if workspace not in seen:
            seen.add(workspace)
            workspace_entries.append((workspace, name))
    if not workspace_entries:
        raise RuntimeError(
            "No profiles provided for --profiles. Use a comma-separated list like "
            "`--profiles DEFAULT`."
        )
    return workspace_entries


def configure_shared_state(
    workspace: str,
    profile: str | None = None,
    tools: list[str] | None = None,
    force_login: bool = False,
    use_pat: bool | None = None,
    skip_model_discovery: bool = False,
    skip_preflight: bool = False,
    fable_enabled: bool | None = None,
    databricks_ai_tools_enabled: bool | None = None,
) -> dict:
    """Log into Databricks, enforce AI Gateway v2, fetch model lists, persist state.

    If tools is provided, only fetch models for those tools. Otherwise fetch all.
    If force_login is True, always run databricks auth login (used by explicit configure).
    If use_pat is True (explicit `configure --profiles <name> --use-pat`), the
    profile's personal access token from ~/.databrickscfg is used instead of
    OAuth and no interactive login ever runs. ``None`` means "inherit": a
    launch re-run keeps the mode the workspace was configured with.
    ``profile`` is the Databricks CLI profile name to address — passed via
    ``--profile`` to every CLI invocation so ambiguous `~/.databrickscfg`
    entries (e.g. DEFAULT and a named profile both pointing at the same host)
    don't error out. If ``None``, we resolve it from the host after login.
    If skip_preflight is True, skip the entire preflight block below — auth
    validation, the AI Gateway probe, and model discovery — trusting a prior
    ``ucode configure``. The PAT/bearer is already exported (``apply_pat_environment``
    in ``_launch_tool``) and the gateway was verified by that earlier configure.
    Only the local profile resolution and the shared state assembly still run;
    the saved model lists are preserved.
    ``fable_enabled`` opts the premium Claude Fable family into Claude Code's
    ``ANTHROPIC_DEFAULT_FABLE_MODEL`` pin (default off). ``None`` means "inherit":
    a launch re-run keeps whatever the workspace was configured with; ``True``/
    ``False`` come from an explicit ``configure --enable-fable``/``--disable-fable``.
    """
    workspace = normalize_workspace_url(workspace)
    prior_state = load_state()
    previous_workspace = prior_state.get("workspace")
    if use_pat is None:
        use_pat = bool(prior_state.get("use_pat")) and previous_workspace == workspace
    if fable_enabled is None:
        fable_enabled = bool(prior_state.get("fable_enabled")) and previous_workspace == workspace
    if databricks_ai_tools_enabled is None:
        # Opt-out: on by default. With no flag, keep this workspace's prior
        # choice but don't inherit another workspace's opt-out.
        disabled = (
            prior_state.get("databricks_ai_tools_enabled") is False
            and previous_workspace == workspace
        )
        databricks_ai_tools_enabled = not disabled
    fetch_all = tools is None

    # Assemble the shared workspace state that doesn't depend on model discovery:
    # workspace, profile, auth mode, base URLs. `profile` may still be None here;
    # each path below resolves it once, where a host->profile lookup is reliable
    # (the skip branch trusts the prior configure; the preflight resolves after
    # login). --skip-preflight persists exactly this and returns, trusting a prior
    # `ucode configure` — it already validated auth + the AI Gateway and saved the
    # model lists (carried over by load_state, left untouched).
    state = load_state()
    state["workspace"] = workspace
    if profile:
        state["profile"] = profile
    else:
        state.pop("profile", None)
    # UC discovery is now always-on; drop any flag persisted by older versions.
    state.pop("uc_enabled", None)
    # Persist the auth mode so launches rebuild the same (PAT-based) agent
    # auth command; an explicit re-configure without --use-pat clears it.
    if use_pat:
        state["use_pat"] = True
    else:
        state.pop("use_pat", None)
    # Persist the Fable opt-in so launches keep pinning the family; an explicit
    # `configure --disable-fable` (fable_enabled=False) clears it.
    if fable_enabled:
        state["fable_enabled"] = True
    else:
        state.pop("fable_enabled", None)
    state["databricks_ai_tools_enabled"] = databricks_ai_tools_enabled
    state["base_urls"] = build_shared_base_urls(workspace)

    if skip_preflight:
        # A prior `ucode configure` created the profile; resolve it locally (no
        # login needed) and persist it so launches disambiguate.
        if profile is None:
            profile = find_profile_name_for_host(workspace)
            if profile:
                state["profile"] = profile
        save_state(state)
        # Scrub MCP entries ucode wrote for a previous workspace.
        if previous_workspace and previous_workspace != workspace:
            purge_cross_workspace_mcp_residue(state, workspace)
        # Diagnostic reasons are transient (attached after save_state so they
        # don't land on disk). No discovery ran, so there is nothing to report.
        state["_discovery_reasons"] = {"claude": None, "gemini": None, "codex": None, "oss": None}
        return state

    # ── Preflight (bypassed above under --skip-preflight): validate Databricks
    #    auth + the AI Gateway, then discover the available models. ──
    if use_pat:
        if not profile:
            raise RuntimeError(
                "--use-pat requires a Databricks CLI profile. Pass one via `--profiles <name>`."
            )
        pat = resolve_pat_token(profile)
        if not pat:
            raise RuntimeError(
                f"--use-pat: profile '{profile}' has no personal access token in "
                "~/.databrickscfg (its auth_type must be `pat`). Add a `token = <PAT>` "
                f"entry under [{profile}], or re-run without --use-pat to use OAuth."
            )
        # Export the PAT for this process and launched agent subprocesses so
        # every token fetch takes the static-bearer path. ensure_pat_bearer
        # keeps a non-empty pre-set bearer (CI escape hatch) but treats an
        # empty one as absent, so it never shadows the PAT. Pass the validated
        # token to avoid re-reading ~/.databrickscfg.
        ensure_pat_bearer(profile, pat)
        ensure_databricks_auth(workspace, profile)
    elif force_login:
        run_databricks_login(workspace, profile)
    else:
        ensure_databricks_auth(workspace, profile)
    # After login the profile exists in ~/.databrickscfg, so a host->profile
    # lookup is reliable even when it returned nothing above.
    if profile is None:
        profile = find_profile_name_for_host(workspace)
        if profile:
            state["profile"] = profile
    with spinner("Verifying Unity AI Gateway..."):
        token = get_databricks_token(workspace, profile)
        ensure_ai_gateway_v2(workspace, token)
    print_success("Unity AI Gateway detected")

    want_claude = (
        fetch_all or "claude" in tools or "opencode" in tools or "copilot" in tools or "pi" in tools
    )
    want_gemini = fetch_all or "gemini" in tools or "opencode" in tools or "pi" in tools
    want_codex = fetch_all or "codex" in tools or "copilot" in tools or "pi" in tools
    want_oss = fetch_all or "opencode" in tools

    claude_reason: str | None = None
    gemini_reason: str | None = None
    codex_reason: str | None = None
    oss_reason: str | None = None
    claude_models = {}
    gemini_models = []
    codex_models = []
    oss_models = []
    opencode_models: dict[str, list[str]] = {}
    web_search_model: str | None = None
    if skip_model_discovery:
        # Provider mode: the agent routes through a Model Provider Service and
        # pins no Databricks model, so the full family discovery is unused. Web
        # search (claude only) still needs one Responses-capable model, so fetch
        # just that with a single call.
        if want_claude:
            with spinner("Fetching web search model..."):
                ws_models, _ = discover_codex_models(workspace, token)
            if ws_models:
                web_search_model = ws_models[0]
    else:
        # UC-first, best-effort: one UC model-services call yields all families
        # as `system.ai.<model-name>` ids, bucketed by name. If a family comes
        # back empty (workspace without UC model-services, or the listing
        # failed), fall back to the per-family AI Gateway listing for that
        # family only.
        with spinner("Fetching available models..."):
            ms_claude, ms_codex, ms_gemini, ms_oss, ms_reason = discover_model_services(
                workspace, token
            )
            if want_claude:
                claude_models, claude_reason = ms_claude, ms_reason
                if not claude_models:
                    claude_models, claude_reason = discover_claude_models(workspace, token)
                # Fable is opt-in (`configure --enable-fable`). Unless enabled,
                # drop it from the discovered bundle entirely so it never becomes
                # part of any agent's config — not claude's family pins, nor the
                # opencode/pi/copilot model lists built from claude_models.
                if not fable_enabled:
                    claude_models.pop("fable", None)
            if want_gemini:
                gemini_models, gemini_reason = ms_gemini, ms_reason
                if not gemini_models:
                    gemini_models, gemini_reason = discover_gemini_models(workspace, token)
            if want_codex:
                codex_models, codex_reason = ms_codex, ms_reason
                if not codex_models:
                    codex_models, codex_reason = discover_codex_models(workspace, token)
            if want_oss:
                oss_models, oss_reason = ms_oss, ms_reason
        if claude_models:
            opencode_models["anthropic"] = list(claude_models.values())
        if gemini_models:
            opencode_models["gemini"] = gemini_models
        if oss_models:
            opencode_models["oss"] = oss_models

    if skip_model_discovery:
        # Don't clobber any previously-discovered Databricks model lists; provider
        # mode just doesn't refresh or use them. Persist the web-search model so
        # claude's web_search MCP keeps working through the normal gateway.
        if web_search_model:
            state["web_search_model"] = web_search_model
    else:
        if want_claude:
            state["claude_models"] = claude_models
        if want_gemini:
            state["gemini_models"] = gemini_models
        if want_codex:
            state["codex_models"] = codex_models
        if want_oss:
            state["oss_models"] = oss_models
        if fetch_all or "opencode" in tools:
            state["opencode_models"] = opencode_models
    save_state(state)
    # Scrub MCP entries that ucode wrote for the previous workspace so the new
    # workspace's agent configs aren't stale.
    if previous_workspace and previous_workspace != workspace:
        purge_cross_workspace_mcp_residue(state, workspace)
    # Diagnostic reasons are transient — attach after save_state so they don't
    # land on disk but are available to the caller for this run.
    state["_discovery_reasons"] = {
        "claude": claude_reason,
        "gemini": gemini_reason,
        "codex": codex_reason,
        "oss": oss_reason,
    }
    return state


def _configure_shared_workspace_states(
    workspaces: list[tuple[str, str | None]],
    tools: list[str] | None,
    *,
    force_login: bool,
    use_pat: bool = False,
    fable_enabled: bool | None = None,
    databricks_ai_tools_enabled: bool | None = None,
) -> list[dict]:
    if not workspaces:
        raise RuntimeError("At least one workspace must be provided.")
    states: list[dict] = []
    for workspace, profile in workspaces:
        states.append(
            configure_shared_state(
                workspace,
                profile=profile,
                tools=tools,
                force_login=force_login,
                use_pat=use_pat,
                fable_enabled=fable_enabled,
                databricks_ai_tools_enabled=databricks_ai_tools_enabled,
            )
        )
    return states


def _provider_summary(tool: str, state: dict) -> str:
    """Short label for the Configuration Complete box: 'Databricks' when no
    Model Provider Service is configured, otherwise the external provider type
    backing this tool (claude routes to Anthropic, codex to OpenAI)."""
    if not get_provider_service(state, tool):
        return "Databricks"
    return {"claude": "Anthropic", "codex": "OpenAI"}.get(tool, "Model Provider Service")


def _maybe_select_provider_service(tool: str, state: dict) -> dict:
    """Interactively let the user route claude/codex through a Model Provider
    Service instead of Databricks models, and persist (or clear) the choice.

    No-op for tools other than claude/codex. Falls back to Databricks when no
    matching provider services are found or the listing fails.
    """
    if tool not in ("claude", "codex"):
        return state
    display = TOOL_SPECS[tool]["display"]

    def _use_databricks() -> dict:
        new_state = set_provider_service(state, tool, None)
        save_state(new_state)
        return new_state

    # Probe first so we only offer the picker when it's actually usable. The
    # interactive path always reaches here, so explain any fallback rather than
    # silently dropping back to Databricks.
    token = get_databricks_token(state["workspace"], state.get("profile"))
    with spinner("Checking for model provider services..."):
        names, reason = list_tool_provider_services(tool, state["workspace"], token)
    if reason is not None:
        # Most workspaces don't have the feature enabled — that's the common case,
        # so fall back to Databricks silently. Only surface unexpected failures.
        if not is_model_provider_feature_unavailable(reason):
            print_warning(f"Could not list model provider services: {reason}")
            print_note("Falling back to Databricks models.")
        return _use_databricks()
    if not names:
        # Feature is on but no service matches this tool's provider type.
        print_note(f"No model provider services available for {display}; using Databricks models.")
        return _use_databricks()

    choice = prompt_for_selection(
        f"How should {display} be configured?",
        [
            ("databricks", "Databricks Hosted"),
            ("mps", "External Models"),
        ],
    )
    if choice is None:
        raise KeyboardInterrupt
    if choice == "databricks":
        return _use_databricks()

    selected = prompt_for_selection(
        "Select a model provider service:", [(name, name) for name in names]
    )
    if selected is None:
        raise KeyboardInterrupt
    state = set_provider_service(state, tool, selected)
    save_state(state)
    print_success(f"{display} will route through {selected}")
    return state


def configure_workspace_command(
    tool: str | None = None,
    selected_tools: list[str] | None = None,
    workspaces: list[tuple[str, str | None]] | None = None,
    *,
    prompt_optional_updates: bool = True,
    use_pat: bool = False,
    skip_validate: bool = False,
    fable_enabled: bool | None = None,
    databricks_ai_tools_enabled: bool | None = None,
) -> int:
    if tool is not None and selected_tools is not None:
        raise RuntimeError("Use either --agent or --agents, not both.")

    # The Databricks-vs-Model-Provider-Service picker is shown only on the fully
    # interactive path (`ucode configure` with no --agent/--agents). Naming agents
    # explicitly signals the non-interactive flow, which stays on Databricks.
    offer_provider = tool is None and selected_tools is None

    workspace_entries = workspaces or [_prompt_for_configuration(tool)]

    if tool is not None:
        states = _configure_shared_workspace_states(
            workspace_entries,
            [tool],
            force_login=True,
            use_pat=use_pat,
            fable_enabled=fable_enabled,
            databricks_ai_tools_enabled=databricks_ai_tools_enabled,
        )
        state = states[0]
        state = configure_single_tool(tool, state)
        spec = TOOL_SPECS[tool]
        console.print(
            Panel(
                f"[bold]Workspace:[/bold] [cyan]{state['workspace']}[/cyan]\n"
                f"[bold]{spec['display']}:[/bold] [green]configured[/green] "
                f"[dim](Provider: {_provider_summary(tool, state)})[/dim]",
                title="Configuration Complete",
                style="green",
                expand=False,
            )
        )
        if skip_validate:
            print_note(f"Skipping {spec['display']} validation (--skip-validate).")
            return 0
        with spinner(f"Validating {spec['display']}..."):
            ok, err = validate_tool(tool)
        if ok:
            print_success(f"{spec['display']} is working")
        else:
            print_err(f"{spec['display']}: {provider_permission_error(tool, state, err)}")
            managed = bool(state.get("managed_configs", {}).get(tool))
            restore_file(spec["config_path"], spec["backup_path"], managed)
            available_tools = [t for t in (state.get("available_tools") or []) if t != tool]
            state["available_tools"] = available_tools
            save_state(state)
            raise RuntimeError(f"{spec['display']} validation failed — config reverted.")
        return 0

    states = _configure_shared_workspace_states(
        workspace_entries,
        selected_tools,
        force_login=True,
        use_pat=use_pat,
        fable_enabled=fable_enabled,
        databricks_ai_tools_enabled=databricks_ai_tools_enabled,
    )
    state = states[0]
    save_state(state)

    available_on_workspace: list[str] = []
    tools_to_check = selected_tools or list(TOOL_SPECS)
    for tool_name in tools_to_check:
        with spinner(f"Checking {TOOL_SPECS[tool_name]['display']} availability..."):
            if check_gateway_endpoint(state, tool_name):
                available_on_workspace.append(tool_name)

    if not available_on_workspace:
        print_err("No coding agents are available on this workspace.")
        _print_discovery_diagnostics(state)
        return 1

    if selected_tools is None:
        picked = prompt_for_tools([(t, TOOL_SPECS[t]["display"]) for t in available_on_workspace])
    else:
        unavailable_tools = [
            tool_name for tool_name in selected_tools if tool_name not in available_on_workspace
        ]
        if unavailable_tools:
            _print_discovery_diagnostics(state)
            displays = ", ".join(
                TOOL_SPECS[tool_name]["display"] for tool_name in unavailable_tools
            )
            raise RuntimeError(f"Requested agent(s) not available on this workspace: {displays}.")
        picked = selected_tools

    if not picked:
        print_note("No coding agents selected — nothing to configure.")
        return 0

    for tool_name in picked:
        install_tool_binary(
            tool_name,
            strict=False,
            update_existing=True,
            prompt_optional_updates=prompt_optional_updates,
        )

    # Offer the provider picker for the chosen claude/codex tools only on the
    # interactive path (no --agents); otherwise stay on the Databricks path.
    if offer_provider:
        for tool_name in picked:
            state = _maybe_select_provider_service(tool_name, state)

    # Last question in the interactive flow: opt out of AI Tools. When a flag
    # already decided it, configure_shared_state persisted that; skip the prompt.
    # The default is the resolved prior choice, so Enter won't undo a past opt-out.
    if databricks_ai_tools_enabled is None and offer_provider:
        state["databricks_ai_tools_enabled"] = prompt_yes_no_default(
            "Install Databricks AI Tools for your coding agents? "
            "This adds Databricks skills and plugins.",
            default=state.get("databricks_ai_tools_enabled", True),
        )

    state = configure_selected_tools(state, picked)

    summary_lines = [f"[bold]Workspace:[/bold] [cyan]{state['workspace']}[/cyan]"]
    for tool_name in picked:
        spec = TOOL_SPECS[tool_name]
        summary_lines.append(
            f"[bold]{spec['display']}:[/bold] [green]configured[/green] "
            f"[dim](Provider: {_provider_summary(tool_name, state)})[/dim]"
        )
    console.print(
        Panel(
            "\n".join(summary_lines),
            title="Configuration Complete",
            style="green",
            expand=False,
        )
    )

    if skip_validate:
        print_note("Skipping agent validation (--skip-validate).")
        return 0
    # Limit validation to just-configured tools so we don't re-validate
    # previously-configured tools the user didn't touch this run.
    validate_state = {**state, "available_tools": picked}
    validate_all_tools(validate_state)
    return 0


def status() -> int:
    state = load_state()
    workspace = state.get("workspace")
    managed_configs = state.get("managed_configs") or {}
    mcp_servers = state.get("mcp_servers") or []
    configured_tools = set(state.get("available_tools") or managed_configs.keys())

    console.print(heading("ucode status"))
    console.print(
        f"  {status_badge('Configured', 'ok') if workspace else status_badge('Not Configured', 'warn')}"
    )

    print_heading("Provider")
    print_kv("Workspace URL", workspace or "not configured")
    profile = state.get("profile")
    if profile:
        print_kv("CLI profile", profile)

    print_heading("Coding Agents")
    for tool, spec in TOOL_SPECS.items():
        configured = tool in configured_tools
        base_url = (
            state.get("base_urls", {}).get(tool, "not configured")
            if configured
            else "not configured"
        )
        config_path = spec["config_path"]
        print_kv("Coding Agent", spec["display"])
        print_kv("Configured", "yes" if configured else "no")
        provider_service = get_provider_service(state, tool)
        if configured and provider_service:
            print_kv("Model Provider Service", provider_service)
        print_kv("Base URL", base_url)
        if configured and tool in MCP_CLIENTS:
            tool_mcp_servers = [
                str(server.get("name"))
                for server in mcp_servers
                if tool in (server.get("clients") or [])
                and server.get("name")
                and server.get("kind") != SKILLS_MCP_KIND
            ]
            print_kv("MCP list command", str(MCP_CLIENTS[tool]["list_command"]))
            print_kv(
                "MCP servers",
                ", ".join(tool_mcp_servers) if tool_mcp_servers else "none saved by ucode",
            )
        print_kv("Config file", str(config_path) if config_path.exists() else "missing")
        console.print()

    print_heading("Skills")
    skill_mcp_entry = next((s for s in mcp_servers if s.get("kind") == SKILLS_MCP_KIND), None)
    if not skill_mcp_entry:
        print_kv("Skills", "not configured")
    else:
        locations = skill_mcp_entry.get("skill_locations") or []
        print_kv(
            "Skill MCP Locations",
            ", ".join(locations) if locations else "none — utility tools only",
        )
        configured_agents = [
            str(MCP_CLIENTS[client]["display"])
            for client in (skill_mcp_entry.get("clients") or [])
            if client in MCP_CLIENTS
        ]
        print_kv("Configured", ", ".join(configured_agents) if configured_agents else "none")

    print_heading("Tracing")
    tracing = state.get("tracing") or {}
    if tracing.get("enabled"):
        print_kv("MLflow tracing", "enabled")
        print_kv("Tracking URI", str(tracing.get("tracking_uri") or "unknown"))
        print_kv(
            "Experiment",
            f"{tracing.get('experiment_name')} (id {tracing.get('experiment_id')})",
        )
        uc_destination = tracing.get("uc_destination")
        if uc_destination:
            print_kv("Unity Catalog", str(uc_destination))
        sql_warehouse_id = tracing.get("sql_warehouse_id")
        if sql_warehouse_id:
            print_kv("SQL warehouse", str(sql_warehouse_id))
    else:
        print_kv("MLflow tracing", "disabled")

    print_heading("State")
    print_kv("State file", str(STATE_PATH) if STATE_PATH.exists() else "missing")
    print_note("Use `ucode configure` to update workspace settings or configure new tools.")
    print_note(
        "Use `ucode configure mcp` to add Databricks MCP servers to configured coding tools."
    )
    print_note(
        "Use `ucode configure skills --location <catalog>.<schema> --mcp` to connect Unity "
        "Catalog Skills."
    )
    print_note("Use `ucode configure tracing` to log coding sessions to an MLflow experiment.")
    print_note("Use `ucode revert` to clear managed configs and restore prior files.")
    return 0


def revert() -> int:
    state = load_state()
    managed_configs = state.get("managed_configs") or {}
    mcp_results = revert_mcp_configs(state)

    results: dict[str, bool] = {
        tool: restore_file(
            spec["config_path"], spec["backup_path"], bool(managed_configs.get(tool))
        )
        for tool, spec in TOOL_SPECS.items()
    }
    pi_settings_restored = restore_file(
        PI_SETTINGS_PATH, PI_SETTINGS_BACKUP_PATH, bool(managed_configs.get("pi"))
    )
    # Older Codex (< 0.134.0) had ucode edit the shared ~/.codex/config.toml in
    # place; restoring the per-profile file above does not undo that.
    legacy_codex_stripped = revert_legacy_shared_config()
    clear_state()

    print_heading("Revert")
    print_kv("Workspace", state.get("workspace") or "none")
    for tool, spec in TOOL_SPECS.items():
        print_kv(f"{spec['display']} config", "restored" if results[tool] else "unchanged")
    if legacy_codex_stripped:
        print_kv("Codex shared config", "ucode entries removed")
    print_kv("Pi settings", "restored" if pi_settings_restored else "unchanged")
    for client, spec in MCP_CLIENTS.items():
        print_kv(
            f"{spec['display']} MCP config",
            "restored" if mcp_results.get(client) else "unchanged",
        )
    print_success("ucode state cleared")
    return 0


# ---------------------------------------------------------------------------
# typer app
# ---------------------------------------------------------------------------


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
configure_app = typer.Typer(add_completion=False, no_args_is_help=False)
app.add_typer(configure_app, name="configure", help="Configure workspace and tool settings.")
mcp_app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(mcp_app, name="mcp", help="MCP servers exposed by ucode.")


@mcp_app.command("web-search")
def mcp_web_search_cmd() -> None:
    """Run the web_search MCP server over stdio. Invoked as a subprocess by Claude Code."""
    from ucode.mcp_web_search import serve

    serve()


@app.command("auth-token", hidden=True)
def auth_token_cmd(
    host: Annotated[
        str | None, typer.Option("--host", help="Workspace URL. Defaults to the saved workspace.")
    ] = None,
    profile: Annotated[
        str | None, typer.Option("--profile", help="Databricks CLI profile.")
    ] = None,
    use_pat: Annotated[
        bool, typer.Option("--use-pat", help="Read the profile's static PAT instead of OAuth.")
    ] = False,
) -> None:
    """Print a Databricks bearer token to stdout, then exit.

    This is the cross-platform helper invoked by Claude Code's `apiKeyHelper`
    and Codex's auth command on every token refresh. It is not meant for
    interactive use. All token logic (DATABRICKS_BEARER short-circuit, PAT
    profiles, OAuth refresh) lives in `get_databricks_token`, so the same
    binary works on macOS, Linux, and Windows without any POSIX shell."""
    import sys

    state = load_state()
    workspace = host or state.get("workspace")
    if not workspace:
        print_err("No workspace configured. Run `ucode configure` first.")
        raise typer.Exit(1)
    profile = profile or state.get("profile")
    if use_pat or state.get("use_pat"):
        # --use-pat explicitly means "serve the profile's static PAT". Fail
        # closed if it can't be read rather than falling through to OAuth —
        # `auth token` cannot serve a PAT-only profile, so that path would
        # surface a misleading stale-login error instead of the real cause.
        if not ensure_pat_bearer(profile):
            print_err(
                f"--use-pat: no personal access token available for profile "
                f"'{profile or '<none>'}'. Add a `token = <PAT>` entry under "
                f"[{profile or 'your-profile'}] in ~/.databrickscfg, or re-run "
                "`ucode configure` without --use-pat to use OAuth."
            )
            raise typer.Exit(1)
    try:
        token = get_databricks_token(workspace, profile)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    # Write the bare token (with trailing newline) to stdout — nothing else may
    # land on stdout or the consuming agent will treat it as part of the token.
    sys.stdout.write(token + "\n")


def _auto_configure_tool(tool: str) -> None:
    """First-time setup for a single tool — mirrors configure_workspace_command."""
    existing = load_state()
    workspace = existing.get("workspace")
    profile = existing.get("profile")
    if not workspace:
        workspace, profile = _prompt_for_configuration(tool)
    state = configure_shared_state(workspace, profile=profile, tools=[tool])

    state = configure_single_tool(tool, state)

    spec = TOOL_SPECS[tool]
    console.print(
        Panel(
            f"[bold]Workspace:[/bold] [cyan]{state['workspace']}[/cyan]\n"
            f"[bold]{spec['display']}:[/bold] [green]configured[/green] "
            f"[dim](Provider: {_provider_summary(tool, state)})[/dim]",
            title="Configuration Complete",
            style="green",
            expand=False,
        )
    )

    with spinner(f"Validating {spec['display']}..."):
        ok, err = validate_tool(tool)
    if ok:
        print_success(f"{spec['display']} is working")
    else:
        print_err(f"{spec['display']}: {provider_permission_error(tool, state, err)}")
        managed = bool(state.get("managed_configs", {}).get(tool))
        restore_file(spec["config_path"], spec["backup_path"], managed)
        available_tools = [t for t in (state.get("available_tools") or []) if t != tool]
        state["available_tools"] = available_tools
        save_state(state)
        raise RuntimeError(f"{spec['display']} validation failed — config reverted.")


def _launch_tool(
    tool_name: str,
    ctx: typer.Context,
    provider: str | None = None,
    skip_preflight: bool = False,
) -> None:
    try:
        tool = normalize_tool(tool_name)
        existing = load_state()
        # Workspaces configured with --use-pat export the profile's PAT as
        # DATABRICKS_BEARER up front so every auth check below (and the
        # launched agent itself) uses the static token instead of OAuth.
        apply_pat_environment(existing)
        needs_auto_configure = not existing.get("workspace") or tool not in (
            existing.get("available_tools") or []
        )
        ensure_bootstrap_dependencies(tool, update_existing=needs_auto_configure)
        if needs_auto_configure:
            _auto_configure_tool(tool)
        state = ensure_provider_state(tool)
        # An explicit --provider overrides the persisted choice; otherwise fall
        # back to whatever `ucode configure` saved for this tool.
        provider = provider or get_provider_service(state, tool)
        # Validate the provider service before launching — it must exist, be a
        # provider type this tool can route to (e.g. claude can't use an OpenAI
        # or Foundry service), and, for Bedrock, expose Claude models to pin.
        # Surfaces a clear error up front instead of a cryptic gateway failure
        # mid-session. For a Bedrock service this also returns the model ids.
        provider_models = None
        if provider:
            provider_models, error = resolve_provider_models(tool, state, provider)
            if error:
                raise RuntimeError(error)
        # Re-fetch model lists on every launch so newly-added Databricks
        # endpoints show up without a manual `ucode configure` (and so that
        # tools like pi which read multiple model bundles never run on
        # stale state from before a tool added a new bundle). Under a provider
        # this heavy discovery is skipped (only a web-search model is fetched).
        state = configure_shared_state(
            state["workspace"],
            profile=state.get("profile"),
            tools=[tool],
            skip_model_discovery=bool(provider),
            skip_preflight=skip_preflight,
        )
        if provider:
            # Routing through a Model Provider Service pins no Databricks model;
            # the agent uses its own canonical model names (header selects the
            # provider). Skip model resolution, which would otherwise fail when
            # the workspace has no matching Databricks models.
            resolved_model = None
        else:
            state, resolved_model = resolve_launch_model(tool, state, None)
        state = configure_tool(
            tool, state, resolved_model, provider=provider, provider_models=provider_models
        )
        print_section(f"ucode with {TOOL_SPECS[tool]['display']}")
        if provider:
            print_kv("Provider", provider)
        elif resolved_model:
            print_kv("Model", resolved_model)
        if tool in ("gemini", "opencode", "copilot", "pi"):
            print_note(
                f"{TOOL_SPECS[tool]['display']} token refresh is managed automatically "
                f"every 30 minutes while the session is running."
            )
        print_success(f"Starting {TOOL_SPECS[tool]['display']}")
        launch_agent(tool, state, ctx.args)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


# Launch-only escape hatch for managed/headless launchers (e.g. omnigent) that
# have already run `ucode configure`: skip the ~5-10s per-launch auth + AI
# Gateway re-validation. Distinct from the configure-only `--skip-validate`,
# which skips the model smoke test.
SkipPreflightOption = Annotated[
    bool,
    typer.Option(
        "--skip-preflight",
        help="Skip the per-launch Databricks auth + AI Gateway re-validation, trusting a "
        "prior `ucode configure`.",
    ),
]


@app.command("codex", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def codex_cmd(
    ctx: typer.Context,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help="Route through a Unity Catalog Model Provider Service "
            "(<catalog>.<schema>.<name>). Skips Databricks model pinning; pass "
            "before any `--` separator.",
        ),
    ] = None,
    skip_preflight: SkipPreflightOption = False,
) -> None:
    """Launch Codex via Databricks."""
    _launch_tool("codex", ctx, provider=provider, skip_preflight=skip_preflight)


@app.command("claude", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def claude_cmd(
    ctx: typer.Context,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            help="Route through a Unity Catalog Model Provider Service "
            "(<catalog>.<schema>.<name>). Skips Databricks model pinning; pass "
            "before any `--` separator.",
        ),
    ] = None,
    skip_preflight: SkipPreflightOption = False,
) -> None:
    """Launch Claude Code via Databricks."""
    _launch_tool("claude", ctx, provider=provider, skip_preflight=skip_preflight)


@app.command("gemini", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def gemini_cmd(ctx: typer.Context, skip_preflight: SkipPreflightOption = False) -> None:
    """Launch Gemini CLI via Databricks."""
    _launch_tool("gemini", ctx, skip_preflight=skip_preflight)


@app.command(
    "opencode", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def opencode_cmd(ctx: typer.Context, skip_preflight: SkipPreflightOption = False) -> None:
    """Launch OpenCode via Databricks."""
    _launch_tool("opencode", ctx, skip_preflight=skip_preflight)


@app.command("copilot", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def copilot_cmd(ctx: typer.Context, skip_preflight: SkipPreflightOption = False) -> None:
    """Launch GitHub Copilot CLI via Databricks."""
    _launch_tool("copilot", ctx, skip_preflight=skip_preflight)


@app.command("pi", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def pi_cmd(ctx: typer.Context, skip_preflight: SkipPreflightOption = False) -> None:
    """Launch Pi coding agent via Databricks."""
    _launch_tool("pi", ctx, skip_preflight=skip_preflight)


@configure_app.callback(invoke_without_command=True)
def configure(
    ctx: typer.Context,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print config files without writing them.")
    ] = False,
    agent: Annotated[
        str | None,
        typer.Option(
            "--agent",
            help="Configure only the named agent (e.g. claude, codex, gemini, opencode, copilot, pi).",
        ),
    ] = None,
    agents: Annotated[
        str | None,
        typer.Option(
            "--agents",
            help="Configure a comma-separated list of agents without prompting (e.g. claude,codex).",
        ),
    ] = None,
    workspaces: Annotated[
        str | None,
        typer.Option(
            "--workspaces",
            help="Configure a comma-separated list of workspaces without prompting.",
        ),
    ] = None,
    profiles: Annotated[
        str | None,
        typer.Option(
            "--profiles",
            help="Configure a comma-separated list of existing Databricks CLI profiles "
            "without the workspace prompt. Each profile's host from ~/.databrickscfg "
            "supplies the workspace URL. Auth behaves like --workspaces: OAuth login "
            "is forced unless --use-pat is also passed.",
        ),
    ] = None,
    use_pat: Annotated[
        bool,
        typer.Option(
            "--use-pat",
            help="Authenticate with the personal access token stored in "
            "~/.databrickscfg for the selected profile(s) instead of OAuth. "
            "Requires --profiles; no interactive login is run. Intended for "
            "CI / headless environments.",
        ),
    ] = False,
    skip_validate: Annotated[
        bool,
        typer.Option(
            "--skip-validate",
            help="Skip the post-configure validation step that sends a quick test "
            "message through each agent. Config files are still written with the "
            "freshly discovered models.",
        ),
    ] = False,
    enable_fable: Annotated[
        bool | None,
        typer.Option(
            "--enable-fable/--disable-fable",
            help="Pin the premium Claude Fable family via ANTHROPIC_DEFAULT_FABLE_MODEL "
            "for Claude Code (opt-in; off by default). Only takes effect when the "
            "workspace's AI Gateway actually advertises a Claude Fable model. "
            "--disable-fable clears a prior opt-in. Omitting both keeps the "
            "workspace's existing setting. Passed on its own (no --agent/--agents), "
            "it configures Claude Code directly since Fable is Claude-only.",
        ),
    ] = None,
    enable_databricks_ai_tools: Annotated[
        bool | None,
        typer.Option(
            "--enable-databricks-ai-tools/--disable-databricks-ai-tools",
            help="Install Databricks AI Tools (skills + plugins that teach agents to use "
            "Databricks) for the configured agents. Installed by default; pass "
            "--disable-databricks-ai-tools to opt out.",
        ),
    ] = None,
    mcp: Annotated[
        str | None,
        typer.Option(
            "--mcp",
            help="Also register the given Databricks MCP service(s) for the configured "
            "coding agents, in one command. Pass a comma-separated list of fully-qualified "
            "names like `system.ai.slack`. Combine with --agents to set up an agent and its "
            "MCP servers together (e.g. `--agents claude --mcp system.ai.slack`); use without "
            "--agents for MCP-only clients such as Cursor.",
        ),
    ] = None,
    tracing: Annotated[
        bool,
        typer.Option(
            "--tracing",
            help="Also enable MLflow tracing for the configured workspace(s).",
        ),
    ] = False,
    skip_upgrade: Annotated[
        bool,
        typer.Option(
            "--skip-upgrade",
            help="Don't prompt to upgrade already-installed agent CLIs to a newer version. "
            "Required updates (when an agent is below its minimum supported version) are "
            "still applied.",
        ),
    ] = False,
    verbose: Annotated[
        str,
        typer.Option(
            "--verbose",
            help="Output verbosity: 'normal' (default) renders decorative panels; "
            "'low' prints terse single-line status instead.",
        ),
    ] = "normal",
) -> None:
    """Configure workspace URL and AI Gateway."""
    if ctx.invoked_subcommand is not None:
        return
    if verbose not in ("normal", "low"):
        print_err("--verbose must be one of: normal, low.")
        raise typer.Exit(2)
    set_dry_run(dry_run)
    set_verbosity(verbose)
    prompt_optional_updates = not skip_upgrade
    try:
        install_databricks_cli()
        if agent is not None and agents is not None:
            raise RuntimeError("Use either --agent or --agents, not both.")
        if workspaces is not None and profiles is not None:
            raise RuntimeError("Use either --workspaces or --profiles, not both.")
        if use_pat and profiles is None:
            raise RuntimeError(
                "--use-pat requires --profiles. Pass the PAT-backed Databricks CLI "
                "profile(s) explicitly, e.g. `ucode configure --profiles DEFAULT --use-pat`."
            )
        workspace_entries = _parse_workspaces_option(workspaces) if workspaces is not None else None
        if profiles is not None:
            workspace_entries = _parse_profiles_option(profiles)
        # Only forward the opt-in flags when set so existing call expectations
        # (and defaults) stay unchanged for the common interactive path.
        skip_kwargs: dict = {}
        if use_pat:
            skip_kwargs["use_pat"] = True
        if skip_validate:
            skip_kwargs["skip_validate"] = True
        # Only forward the Fable opt-in when the user passed the flag; `None`
        # (neither flag given) lets configure_shared_state inherit the prior
        # workspace setting instead of clobbering it.
        if enable_fable is not None:
            skip_kwargs["fable_enabled"] = enable_fable
        # Fable is a Claude-only model family, so `--enable-fable`/`--disable-fable`
        # only makes sense for Claude Code. When passed on its own, implicitly
        # target claude instead of dropping into the interactive agent picker.
        if enable_fable is not None and agent is None and agents is None:
            agent = "claude"
        if enable_databricks_ai_tools is not None:
            skip_kwargs["databricks_ai_tools_enabled"] = enable_databricks_ai_tools
        if agent is not None:
            tool = normalize_tool(agent)
            install_tool_binary(
                tool,
                strict=True,
                update_existing=True,
                prompt_optional_updates=prompt_optional_updates,
            )
            if workspace_entries is None:
                configure_workspace_command(tool, **skip_kwargs)
            else:
                configure_workspace_command(
                    tool,
                    workspaces=workspace_entries,
                    **skip_kwargs,
                )
        elif agents is not None:
            selected_tools = _parse_agents_option(agents)
            if workspace_entries is None:
                configure_workspace_command(
                    selected_tools=selected_tools,
                    prompt_optional_updates=prompt_optional_updates,
                    **skip_kwargs,
                )
            else:
                configure_workspace_command(
                    selected_tools=selected_tools,
                    workspaces=workspace_entries,
                    prompt_optional_updates=prompt_optional_updates,
                    **skip_kwargs,
                )
        elif mcp is not None:
            # MCP-only: `--mcp` without --agent(s) (e.g. Cursor, which isn't a
            # model agent, or adding MCP servers to an already-configured setup).
            # Configure just the workspace — no interactive agent picker — so the
            # `--mcp` registration below has a current workspace to target.
            if workspace_entries is None:
                workspace_entries = [_prompt_for_configuration(None)]
            _configure_shared_workspace_states(
                workspace_entries,
                tools=[],
                force_login=not use_pat,
                use_pat=use_pat,
            )
        else:
            # Tool binaries are installed after the user picks which agents
            # they want, in configure_workspace_command.
            if workspace_entries is None:
                configure_workspace_command(
                    prompt_optional_updates=prompt_optional_updates,
                    **skip_kwargs,
                )
            else:
                configure_workspace_command(
                    workspaces=workspace_entries,
                    prompt_optional_updates=prompt_optional_updates,
                    **skip_kwargs,
                )
        if tracing:
            # The workspaces were just configured, so enable tracing for them
            # directly instead of re-prompting. Fall back to the workspace that
            # `configure_workspace_command` made current (the interactive pick).
            tracing_workspaces = workspace_entries
            if tracing_workspaces is None:
                current = load_full_state().get("current_workspace")
                tracing_workspaces = [(current, None)] if current else None
            if tracing_workspaces:
                configure_tracing_command(workspaces=tracing_workspaces)
        if mcp is not None:
            # The workspace + agents were just configured above, so the current
            # workspace state now lists the agents whose MCP configs we should
            # write. `--mcp` takes fully-qualified service names, which
            # `configure_mcp_command` locates and registers without a picker
            # (bare short names would need --location, which we don't accept here).
            services = {name.strip() for name in mcp.split(",") if name.strip()}
            if not services:
                raise RuntimeError(
                    "--mcp needs at least one fully-qualified MCP service name, e.g. "
                    "`--mcp system.ai.slack`."
                )
            bare = sorted(name for name in services if name.count(".") < 2)
            if bare:
                raise RuntimeError(
                    "--mcp names must be fully qualified `<catalog>.<schema>.<name>` "
                    f"(got: {', '.join(bare)}). Use `ucode configure mcp` for the "
                    "interactive picker."
                )
            configure_mcp_command(services=services)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@configure_app.command("mcp")
def configure_mcp(
    location: Annotated[
        str | None,
        typer.Option(
            "--location",
            help="Non-interactive: replace registered MCPs with exactly the services "
            "in the given Unity Catalog `<catalog>.<schema>` (e.g. `system.ai`) and "
            "exit without showing the picker. Any previously-registered MCPs outside "
            "this location are removed.",
        ),
    ] = None,
    services: Annotated[
        str | None,
        typer.Option(
            "--services",
            help="Configure exactly this comma-separated subset of MCP services (adding and "
            "removing to match) instead of a whole schema. Full names like `system.ai.github` "
            "work on their own; bare short names like `github` need --location to locate them. "
            "Omit --services to configure the whole --location schema; pass an empty string "
            "(with --location) to remove all.",
        ),
    ] = None,
) -> None:
    """Add Databricks MCP servers to installed coding tools."""
    # `--services` absent -> None (whole schema); present (even empty) -> the
    # explicit subset, so `--services ""` deselects everything.
    selected = None if services is None else {s.strip() for s in services.split(",") if s.strip()}
    try:
        configure_mcp_command(location=location, services=selected)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@configure_app.command("skills")
def configure_skills(
    location: Annotated[
        str,
        typer.Option("--location", help="Comma-separated `<catalog>.<schema>` skill scopes."),
    ],
    mcp: Annotated[
        bool,
        typer.Option(
            "--mcp", help="Manage the skills MCP connection (required until download mode lands)."
        ),
    ] = False,
) -> None:
    """Configure Databricks Skills for your coding tools.

    ``--location`` sets the skills MCP connection's scope to exactly the listed
    schemas, replacing any previous set.
    """
    try:
        if not mcp:
            raise RuntimeError("Download mode is not available yet; pass --mcp for now.")
        configure_skills_mcp_command(_parse_skill_locations(location))
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@configure_app.command("tracing")
def configure_tracing(
    disable: Annotated[
        bool, typer.Option("--disable", help="Turn off MLflow tracing for configured agents.")
    ] = False,
) -> None:
    """Send coding-session traces to an MLflow experiment in your workspace."""
    try:
        install_databricks_cli()
        configure_tracing_command(disable=disable)
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        print_err("Interrupted.")
        raise typer.Exit(130) from None


@app.command("status")
def status_cmd() -> None:
    """Show current workspace, tool configs, and saved model selections."""
    try:
        status()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None


@app.command("revert")
def revert_cmd() -> None:
    """Clear ucode state and restore backed-up agent config files."""
    try:
        revert()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None


@app.command("usage")
def usage_cmd() -> None:
    """Show Databricks AI Gateway usage summary (last 7 days)."""
    try:
        install_databricks_cli()
        usage_report()
    except RuntimeError as exc:
        print_err(str(exc))
        raise typer.Exit(1) from None


@app.command("upgrade")
def upgrade_cmd() -> None:
    """Upgrade ucode to the latest version from GitHub."""
    import subprocess

    git_url = "git+https://github.com/databricks/ucode"
    print_section("Upgrade")
    print_kv("Source", git_url)
    try:
        subprocess.run(
            ["uv", "tool", "install", "--reinstall", git_url],
            check=True,
        )
    except FileNotFoundError:
        print_err("`uv` was not found on PATH. Install uv to upgrade ucode.")
        raise typer.Exit(1) from None
    except subprocess.CalledProcessError as exc:
        print_err(f"Upgrade failed (exit code {exc.returncode}).")
        raise typer.Exit(1) from None
    print_success("ucode upgraded")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
