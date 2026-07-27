"""MCP (Model Context Protocol) server registration for coding tools."""

from __future__ import annotations

import json
import os
import shutil
import string
import subprocess
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import questionary
from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition, IsDone
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.shortcuts import PromptSession
from questionary.prompts.common import InquirerControl
from questionary.question import Question
from questionary.styles import merge_styles_default

from ucode.agents import copilot, gemini, opencode
from ucode.config_io import restore_file
from ucode.databricks import (
    apply_pat_environment,
    build_mcp_service_url,
    build_skills_mcp_url,
    ensure_databricks_auth,
    get_databricks_token,
    list_databricks_apps,
    list_databricks_connections,
    list_genie_spaces,
    list_mcp_services,
    list_uc_functions_catalog_schemas,
    list_vector_search_catalog_schemas,
    workspace_hostname,
)
from ucode.state import load_full_state, load_state, save_state
from ucode.ui import (
    print_note,
    print_section,
    print_success,
    print_warning,
    spinner,
)

MCP_AUTH_TOKEN_ENV_VAR = "OAUTH_TOKEN"
MCP_USER_SCOPE = "user"
MCP_CLEANUP_SCOPES = ("local", "project", MCP_USER_SCOPE)
MCP_PICKER_VISIBLE_ROWS = 10
MCP_CLIENTS = {
    "claude": {
        "binary": "claude",
        "display": "Claude Code",
        "list_command": "claude mcp list",
    },
    "codex": {
        "binary": "codex",
        "display": "Codex",
        "list_command": "codex mcp list",
    },
    "gemini": {
        "binary": "gemini",
        "display": "Gemini CLI",
        "list_command": "gemini mcp list",
    },
    "opencode": {
        "binary": "opencode",
        "display": "OpenCode",
        "list_command": "opencode mcp list",
    },
    "copilot": {
        "binary": "copilot",
        "display": "GitHub Copilot CLI",
        "list_command": "copilot mcp list",
    },
}
SKILLS_MCP_KIND = "skills"
SKILLS_MCP_SERVER_NAME = "databricks-skill-registry"
EXTERNAL_MCP_SELECTION_PREFIX = "external:"
SQL_MCP_VALUE = "managed:sql"
GENIE_SPACE_SELECTION_PREFIX = "genie-space:"
APP_MCP_SELECTION_PREFIX = "app:"
MCP_SERVICE_SELECTION_PREFIX = "mcp-service:"
VECTOR_SEARCH_SELECTION_PREFIX = "vector-search:"
UC_FUNCTIONS_SELECTION_PREFIX = "uc-functions:"
MCP_ADD_PREFIX = "add:"
MCP_CONNECTION_MARKERS = (
    "is_mcp",
    "is_mcp_connection",
    "mcp",
    "mcp_enabled",
    "enable_mcp",
)


def build_mcp_http_entry(url: str, *, always_load: bool = False) -> dict:
    entry: dict[str, Any] = {
        "type": "http",
        "url": url,
        "headers": {
            "Authorization": f"Bearer ${{{MCP_AUTH_TOKEN_ENV_VAR}}}",
        },
    }
    if always_load:
        entry["alwaysLoad"] = True
    return entry


def add_claude_mcp_server(name: str, entry: dict, scope: str = MCP_USER_SCOPE) -> None:
    try:
        subprocess.run(
            ["claude", "mcp", "add-json", name, json.dumps(entry), "-s", scope],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to add MCP server '{name}' via claude CLI.") from exc


def _is_missing_mcp_server_output(output: str) -> bool:
    normalized = output.lower()
    return (
        "not found" in normalized
        or "no mcp server" in normalized
        or "no server named" in normalized
        or ("mcp server found with name" in normalized and "no " in normalized)
    )


def remove_claude_mcp_server(name: str, scope: str) -> bool:
    try:
        subprocess.run(
            ["claude", "mcp", "remove", name, "-s", scope],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return True
    except subprocess.CalledProcessError as exc:
        output = f"{exc.stderr or ''}\n{exc.stdout or ''}"
        if _is_missing_mcp_server_output(output):
            return False
        raise RuntimeError(f"Failed to remove MCP server '{name}' via claude CLI.") from exc


def add_codex_mcp_server(name: str, url: str) -> None:
    try:
        subprocess.run(
            [
                "codex",
                "mcp",
                "add",
                name,
                "--url",
                url,
                "--bearer-token-env-var",
                MCP_AUTH_TOKEN_ENV_VAR,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to add MCP server '{name}' via codex CLI.") from exc


def remove_codex_mcp_server(name: str) -> bool:
    try:
        result = subprocess.run(
            ["codex", "mcp", "remove", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timed out removing MCP server '{name}' via codex CLI.") from exc

    output = f"{result.stderr or ''}\n{result.stdout or ''}"
    if _is_missing_mcp_server_output(output):
        return False
    if result.returncode != 0:
        raise RuntimeError(f"Failed to remove MCP server '{name}' via codex CLI.")
    return True


def _gemini_cli_env() -> dict[str, str]:
    # Pin GEMINI_CLI_HOME to the same directory the launcher.
    env = os.environ.copy()
    env["GEMINI_CLI_HOME"] = str(gemini.GEMINI_HOME_DIR)
    return env


def add_gemini_mcp_server(name: str, url: str) -> None:
    try:
        subprocess.run(
            [
                "gemini",
                "mcp",
                "add",
                name,
                url,
                "--type",
                "http",
                "--scope",
                MCP_USER_SCOPE,
                "--header",
                f"Authorization: Bearer ${{{MCP_AUTH_TOKEN_ENV_VAR}}}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=_gemini_cli_env(),
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to add MCP server '{name}' via gemini CLI.") from exc


def remove_gemini_mcp_server(name: str) -> bool:
    try:
        result = subprocess.run(
            ["gemini", "mcp", "remove", name, "--scope", MCP_USER_SCOPE],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=_gemini_cli_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timed out removing MCP server '{name}' via gemini CLI.") from exc

    output = f"{result.stderr or ''}\n{result.stdout or ''}"
    if _is_missing_mcp_server_output(output):
        return False
    if result.returncode != 0:
        raise RuntimeError(f"Failed to remove MCP server '{name}' via gemini CLI.")
    return True


def available_mcp_clients() -> list[str]:
    return [client for client, spec in MCP_CLIENTS.items() if shutil.which(str(spec["binary"]))]


def configured_mcp_clients(state: dict, installed_clients: list[str]) -> list[str]:
    configured_tools = state.get("available_tools") or []
    if not isinstance(configured_tools, list):
        configured_tools = []
    configured = set(configured_tools)
    return [
        client for client in MCP_CLIENTS if client in configured and client in installed_clients
    ]


def configure_client_mcp_server(client: str, name: str, url: str, entry: dict) -> list[str]:
    if client == "claude":
        removed_scopes = [
            scope for scope in MCP_CLEANUP_SCOPES if remove_claude_mcp_server(name, scope)
        ]
        add_claude_mcp_server(name, entry, MCP_USER_SCOPE)
        return removed_scopes
    if client == "codex":
        removed = remove_codex_mcp_server(name)
        add_codex_mcp_server(name, url)
        return [MCP_USER_SCOPE] if removed else []
    if client == "gemini":
        removed = remove_gemini_mcp_server(name)
        add_gemini_mcp_server(name, url)
        return [MCP_USER_SCOPE] if removed else []
    if client == "opencode":
        removed = opencode.write_mcp_server_config(name, url)
        return [MCP_USER_SCOPE] if removed else []
    if client == "copilot":
        removed = copilot.write_mcp_server_config(name, url)
        return [MCP_USER_SCOPE] if removed else []
    raise RuntimeError(f"Unsupported MCP client '{client}'.")


def remove_client_mcp_server(client: str, name: str) -> list[str]:
    if client == "claude":
        return [scope for scope in MCP_CLEANUP_SCOPES if remove_claude_mcp_server(name, scope)]
    if client == "codex":
        return [MCP_USER_SCOPE] if remove_codex_mcp_server(name) else []
    if client == "gemini":
        return [MCP_USER_SCOPE] if remove_gemini_mcp_server(name) else []
    if client == "opencode":
        return [MCP_USER_SCOPE] if opencode.remove_mcp_server_config(name) else []
    if client == "copilot":
        return [MCP_USER_SCOPE] if copilot.remove_mcp_server_config(name) else []
    raise RuntimeError(f"Unsupported MCP client '{client}'.")


def revert_mcp_configs(state: dict) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for server in state.get("mcp_servers") or []:
        name = server.get("name")
        if not isinstance(name, str) or not name:
            continue
        for client in server.get("clients") or []:
            if client not in MCP_CLIENTS:
                continue
            removed_scopes = remove_client_mcp_server(client, name)
            results[client] = bool(removed_scopes) or results.get(client, False)

    # OpenCode MCP entries live in the normal OpenCode config and are restored
    # by the main agent config revert. Copilot stores MCP servers separately,
    # so restore its original MCP file after removing per-server entries above.
    results["copilot"] = restore_file(
        copilot.COPILOT_MCP_CONFIG_PATH,
        copilot.COPILOT_MCP_BACKUP_PATH,
        any(
            "copilot" in (server.get("clients") or []) for server in state.get("mcp_servers") or []
        ),
    ) or results.get("copilot", False)
    return results


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return None


def _mcp_marker_value(connection: dict) -> bool | None:
    containers = [connection]
    options = connection.get("options")
    if isinstance(options, dict):
        containers.append(options)

    for container in containers:
        for marker in MCP_CONNECTION_MARKERS:
            if marker in container:
                value = _coerce_bool(container.get(marker))
                if value is not None:
                    return value
    return None


def is_external_mcp_connection(connection: dict) -> bool:
    connection_type = connection.get("connection_type")
    if not isinstance(connection_type, str) or connection_type.upper() != "HTTP":
        return False

    marker_value = _mcp_marker_value(connection)
    if marker_value is False:
        return False
    return True


def external_mcp_connection_names(connections: list[dict]) -> list[str]:
    names: set[str] = set()
    for connection in connections:
        if not is_external_mcp_connection(connection):
            continue
        name = connection.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return sorted(names)


def discover_external_mcp_connection_names(workspace: str, profile: str | None = None) -> list[str]:
    return external_mcp_connection_names(list_databricks_connections(workspace, profile))


def discover_mcp_service_names(workspace: str, profile: str | None = None) -> list[str]:
    """Curated `system.ai.*` MCP services. Empty list if discovery fails so
    callers can fall back to legacy connection discovery without surfacing
    every error to the picker."""
    token = get_databricks_token(workspace, profile)
    names, _reason = list_mcp_services(workspace, token)
    return names


def _normalize_workspace_title(text: str) -> str:
    """Collapse a Databricks workspace title to lowercase alphanumerics joined
    by single hyphens, trimmed at the edges. Output is safe to use as an MCP
    server-name token across every supported agent CLI."""
    chars: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            chars.append(ch)
        elif chars and chars[-1] != "-":
            chars.append("-")
    return "".join(chars).strip("-")


def _genie_server_name(title: str, space_id: str, taken: set[str]) -> str:
    """Prefer a friendly name derived from the Genie space title; fall back to
    the raw space_id when there is no title or the derived name collides with
    one we already emitted."""
    slug = _normalize_workspace_title(title) if title else ""
    if slug:
        candidate = f"databricks-genie-{slug}"
        if candidate not in taken:
            return candidate
    return f"databricks-genie-{space_id}"


def genie_mcp_servers(spaces: list[dict], workspace: str) -> list[dict]:
    servers: list[dict] = []
    seen_names: set[str] = set()
    for space in spaces:
        space_id = space.get("space_id")
        if not isinstance(space_id, str) or not space_id.strip():
            continue
        space_id = space_id.strip()
        raw_title = space.get("title")
        title = raw_title.strip() if isinstance(raw_title, str) and raw_title.strip() else ""
        server_name = _genie_server_name(title, space_id, seen_names)
        if server_name in seen_names:
            continue
        seen_names.add(server_name)
        servers.append(
            {
                "name": server_name,
                "title": title or space_id,
                "url": f"{workspace}/api/2.0/mcp/genie/{space_id}",
            }
        )
    return sorted(servers, key=lambda server: str(server["title"]).lower())


def discover_genie_mcp_servers(workspace: str, profile: str | None = None) -> list[dict]:
    return genie_mcp_servers(list_genie_spaces(workspace, profile), workspace)


def app_mcp_servers(apps: list[dict]) -> list[dict]:
    servers: list[dict] = []
    seen_names: set[str] = set()
    for app in apps:
        app_name = app.get("name")
        app_url = app.get("url")
        if not isinstance(app_name, str) or not app_name.strip():
            continue
        if not app_name.strip().startswith("mcp-"):
            continue
        if not isinstance(app_url, str) or not app_url.strip():
            continue
        name = app_name.strip()
        server_name = f"databricks-app-{name}"
        if server_name in seen_names:
            continue
        seen_names.add(server_name)
        servers.append(
            {
                "name": server_name,
                "title": name,
                "url": f"{app_url.strip().rstrip('/')}/mcp",
            }
        )
    return sorted(servers, key=lambda server: str(server["title"]).lower())


def discover_app_mcp_servers(workspace: str, profile: str | None = None) -> list[dict]:
    return app_mcp_servers(list_databricks_apps(workspace, profile))


def _catalog_schema_server_name(prefix: str, catalog: str, schema: str, taken: set[str]) -> str:
    """Stable server name for a per-(catalog, schema) managed MCP entry.

    Prefers the lowercase alphanumeric slug; falls back to a numeric suffix on
    collision so two schemas that slug to the same value still both render."""
    slug = f"{_normalize_workspace_title(catalog)}-{_normalize_workspace_title(schema)}".strip("-")
    candidate = f"{prefix}-{slug}" if slug else prefix
    if candidate not in taken:
        return candidate
    counter = 2
    while f"{candidate}-{counter}" in taken:
        counter += 1
    return f"{candidate}-{counter}"


def vector_search_mcp_servers(pairs: list[tuple[str, str]], workspace: str) -> list[dict]:
    servers: list[dict] = []
    seen_names: set[str] = set()
    for catalog, schema in pairs:
        if not catalog or not schema:
            continue
        name = _catalog_schema_server_name("databricks-vector-search", catalog, schema, seen_names)
        seen_names.add(name)
        servers.append(
            {
                "name": name,
                "title": f"{catalog}.{schema}",
                "catalog": catalog,
                "schema": schema,
                "url": f"{workspace}/api/2.0/mcp/vector-search/{catalog}/{schema}",
            }
        )
    return sorted(servers, key=lambda server: str(server["title"]).lower())


def discover_vector_search_mcp_servers(workspace: str, profile: str | None = None) -> list[dict]:
    token = get_databricks_token(workspace, profile)
    pairs, _reason = list_vector_search_catalog_schemas(workspace, token)
    return vector_search_mcp_servers(pairs, workspace)


def uc_functions_mcp_servers(pairs: list[tuple[str, str]], workspace: str) -> list[dict]:
    servers: list[dict] = []
    seen_names: set[str] = set()
    for catalog, schema in pairs:
        if not catalog or not schema:
            continue
        name = _catalog_schema_server_name("databricks-functions", catalog, schema, seen_names)
        seen_names.add(name)
        servers.append(
            {
                "name": name,
                "title": f"{catalog}.{schema}",
                "catalog": catalog,
                "schema": schema,
                "url": f"{workspace}/api/2.0/mcp/functions/{catalog}/{schema}",
            }
        )
    return sorted(servers, key=lambda server: str(server["title"]).lower())


def discover_uc_functions_mcp_servers(workspace: str, profile: str | None = None) -> list[dict]:
    token = get_databricks_token(workspace, profile)
    pairs, _reason = list_uc_functions_catalog_schemas(workspace, token)
    return uc_functions_mcp_servers(pairs, workspace)


def _picker_style() -> questionary.Style:
    return questionary.Style(
        [
            ("pointer", "fg:cyan bold"),
            ("highlighted", "noinherit"),
            ("selected", "noinherit"),
            ("answer", "fg:cyan"),
        ]
    )


def _server_name(server: dict) -> str | None:
    name = server.get("name")
    return name if isinstance(name, str) and name else None


def _servers_by_name(mcp_servers: list[dict]) -> dict[str, dict]:
    servers: dict[str, dict] = {}
    for server in mcp_servers:
        name = _server_name(server)
        if name:
            servers[name] = server
    return servers


def _mcp_entry_url_host(entry: dict) -> str | None:
    """Return the host of an MCP entry's URL, or ``None`` if missing/malformed."""
    url = entry.get("url")
    if not isinstance(url, str) or not url:
        return None
    try:
        return urlparse(url).hostname
    except ValueError:
        return None


def _partition_mcp_entries_by_workspace(
    entries: list[dict], workspace: str
) -> tuple[list[dict], list[dict]]:
    """Split MCP entries into ones that belong to ``workspace`` and ones that don't."""
    workspace_host = workspace_hostname(workspace)
    current: list[dict] = []
    foreign: list[dict] = []
    for entry in entries:
        if _mcp_entry_url_host(entry) == workspace_host:
            current.append(entry)
        else:
            foreign.append(entry)
    return current, foreign


def _mcp_entries_only_in_other_workspaces(current_workspace: str) -> dict[str, set[str]]:
    """Return ``{name: {client, ...}}`` for MCPs ucode tracks only in workspaces other than ``current_workspace``."""
    full_state = load_full_state()
    workspaces = full_state.get("workspaces")
    if not isinstance(workspaces, dict):
        return {}

    current_names: set[str] = set()
    current_bucket = workspaces.get(current_workspace)
    if isinstance(current_bucket, dict):
        for entry in current_bucket.get("mcp_servers") or []:
            name = _server_name(entry)
            if name:
                current_names.add(name)

    external_entries: dict[str, set[str]] = {}
    for ws, bucket in workspaces.items():
        if ws == current_workspace or not isinstance(bucket, dict):
            continue
        for entry in bucket.get("mcp_servers") or []:
            name = _server_name(entry)
            if not name or name in current_names:
                continue
            client_set = external_entries.setdefault(name, set())
            for client in entry.get("clients") or []:
                client_set.add(client)
    return external_entries


def _server_choice(name: str, checked: bool, title: str | None = None) -> questionary.Choice:
    return questionary.Choice(
        title=title or name,
        value=name,
        checked=checked,
    )


def _add_choice(selection: str, title: str) -> questionary.Choice:
    return questionary.Choice(title=title, value=f"{MCP_ADD_PREFIX}{selection}")


def _scrolling_checkbox(
    message: str,
    choices: list[questionary.Choice | questionary.Separator],
    instruction: str,
    style: questionary.Style,
) -> Question:
    merged_style = merge_styles_default(
        [
            questionary.Style([("bottom-toolbar", "noreverse")]),
            style,
        ]
    )
    control = InquirerControl(
        choices,
        pointer="›",
        show_description=False,
    )

    def get_prompt_tokens() -> list[tuple[str, str]]:
        tokens = [("class:qmark", ""), ("class:question", f" {message} ")]
        if control.is_answered:
            selected_count = len(control.selected_options)
            answer = "done" if selected_count == 0 else f"done ({selected_count} selections)"
            tokens.append(("class:answer", answer))
        else:
            tokens.append(("class:instruction", instruction))
        return tokens

    def get_selected_values() -> list[Any]:
        return [choice.value for choice in control.get_selected_values()]

    def perform_validation() -> bool:
        control.error_message = None
        return True

    prompt_session: PromptSession = PromptSession(get_prompt_tokens, reserve_space_for_menu=0)
    visible_rows = min(MCP_PICKER_VISIBLE_ROWS, max(1, len(choices)))
    has_more_choices = len(choices) > MCP_PICKER_VISIBLE_ROWS

    @Condition
    def has_search_string() -> bool:
        return control.get_search_string_tokens() is not None

    validation_prompt: PromptSession = PromptSession(bottom_toolbar=lambda: control.error_message)
    layout = Layout(
        HSplit(
            [
                prompt_session.layout.container,
                ConditionalContainer(
                    Window(control, height=Dimension(preferred=visible_rows, max=visible_rows)),
                    filter=~IsDone(),
                ),
                ConditionalContainer(
                    Window(
                        height=Dimension.exact(1),
                        content=FormattedTextControl(
                            lambda: [("class:instruction", "  ↑/↓ scroll for more")]
                        ),
                    ),
                    filter=Condition(lambda: has_more_choices) & ~IsDone(),
                ),
                ConditionalContainer(
                    Window(
                        height=Dimension.exact(2),
                        content=FormattedTextControl(control.get_search_string_tokens),
                    ),
                    filter=has_search_string & ~IsDone(),
                ),
                ConditionalContainer(
                    validation_prompt.layout.container,
                    filter=Condition(lambda: control.error_message is not None),
                ),
            ]
        )
    )

    bindings = KeyBindings()

    @bindings.add(Keys.ControlQ, eager=True)
    @bindings.add(Keys.ControlC, eager=True)
    def _(event: Any) -> None:
        event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    @bindings.add(" ", eager=True)
    def _(_event: Any) -> None:
        pointed_choice = control.get_pointed_at().value
        if pointed_choice in control.selected_options:
            control.selected_options.remove(pointed_choice)
        else:
            control.selected_options.append(pointed_choice)
        perform_validation()

    def move_cursor_down(event: Any) -> None:
        control.select_next()
        while not control.is_selection_valid():
            control.select_next()

    def move_cursor_up(event: Any) -> None:
        control.select_previous()
        while not control.is_selection_valid():
            control.select_previous()

    def search_filter(event: Any) -> None:
        control.add_search_character(event.key_sequence[0].key)

    for character in string.printable:
        if character in string.whitespace:
            continue
        bindings.add(character, eager=True)(search_filter)
    bindings.add(Keys.Backspace, eager=True)(search_filter)

    bindings.add(Keys.Down, eager=True)(move_cursor_down)
    bindings.add(Keys.Up, eager=True)(move_cursor_up)
    bindings.add(Keys.ControlN, eager=True)(move_cursor_down)
    bindings.add(Keys.ControlP, eager=True)(move_cursor_up)

    @bindings.add(Keys.ControlM, eager=True)
    def _(event: Any) -> None:
        control.submission_attempted = True
        if perform_validation():
            control.is_answered = True
            event.app.exit(result=get_selected_values())

    @bindings.add(Keys.Any)
    def _(_event: Any) -> None:
        """Ignore other text input."""

    return Question(
        Application(
            layout=layout,
            key_bindings=bindings,
            style=merged_style,
        )
    )


def build_mcp_picker_choices(
    available_external_names: list[str],
    available_genie_servers: list[dict],
    available_app_servers: list[dict],
    original_servers: list[dict],
    available_mcp_service_names: list[str] | None = None,
    available_vector_search_servers: list[dict] | None = None,
    available_uc_functions_servers: list[dict] | None = None,
) -> list[questionary.Choice | questionary.Separator]:
    original_by_name = _servers_by_name(original_servers)
    known_names = set(original_by_name)

    choices: list[questionary.Choice | questionary.Separator] = []
    displayed_names: set[str] = set()

    if "databricks-sql" in known_names:
        choices.append(_server_choice("databricks-sql", True, "Databricks SQL"))
    else:
        choices.append(_add_choice(SQL_MCP_VALUE, "Databricks SQL"))
    displayed_names.add("databricks-sql")

    for name in available_mcp_service_names or []:
        # Picker shows the dotted UC name; state/agents store the dashed form
        # (see resolver). Compare against the dashed form when checking what's
        # already registered.
        registered_as = name.replace(".", "-")
        if registered_as in known_names:
            choices.append(_server_choice(registered_as, True, name))
        else:
            choices.append(_add_choice(f"{MCP_SERVICE_SELECTION_PREFIX}{name}", name))
        displayed_names.add(registered_as)

    for name in available_external_names:
        if name in known_names:
            choices.append(_server_choice(name, True, name))
        else:
            choices.append(_add_choice(f"{EXTERNAL_MCP_SELECTION_PREFIX}{name}", name))
        displayed_names.add(name)

    for server in available_genie_servers:
        name = _server_name(server)
        title = server.get("title")
        if not name:
            continue
        display_title = f"Genie: {title}" if isinstance(title, str) and title else name
        if name in known_names:
            choices.append(_server_choice(name, True, display_title))
        else:
            choices.append(
                _add_choice(
                    f"{GENIE_SPACE_SELECTION_PREFIX}{name.removeprefix('databricks-genie-')}",
                    display_title,
                )
            )
        displayed_names.add(name)

    for server in available_app_servers:
        name = _server_name(server)
        title = server.get("title")
        if not name:
            continue
        display_title = f"App: {title}" if isinstance(title, str) and title else name
        if name in known_names:
            choices.append(_server_choice(name, True, display_title))
        else:
            choices.append(
                _add_choice(
                    f"{APP_MCP_SELECTION_PREFIX}{name.removeprefix('databricks-app-')}",
                    display_title,
                )
            )
        displayed_names.add(name)

    for server in available_vector_search_servers or []:
        name = _server_name(server)
        catalog = server.get("catalog")
        schema = server.get("schema")
        if not name or not isinstance(catalog, str) or not isinstance(schema, str):
            continue
        display_title = f"Vector Search: {catalog}.{schema}"
        if name in known_names:
            choices.append(_server_choice(name, True, display_title))
        else:
            choices.append(
                _add_choice(
                    f"{VECTOR_SEARCH_SELECTION_PREFIX}{catalog}.{schema}",
                    display_title,
                )
            )
        displayed_names.add(name)

    for server in available_uc_functions_servers or []:
        name = _server_name(server)
        catalog = server.get("catalog")
        schema = server.get("schema")
        if not name or not isinstance(catalog, str) or not isinstance(schema, str):
            continue
        display_title = f"UC Functions: {catalog}.{schema}"
        if name in known_names:
            choices.append(_server_choice(name, True, display_title))
        else:
            choices.append(
                _add_choice(
                    f"{UC_FUNCTIONS_SELECTION_PREFIX}{catalog}.{schema}",
                    display_title,
                )
            )
        displayed_names.add(name)

    for name in sorted(known_names - displayed_names):
        choices.append(_server_choice(name, True))
    return choices


def prompt_for_mcp_server_choices(
    available_external_names: list[str],
    available_genie_servers: list[dict],
    available_app_servers: list[dict],
    original_servers: list[dict],
    available_mcp_service_names: list[str] | None = None,
    available_vector_search_servers: list[dict] | None = None,
    available_uc_functions_servers: list[dict] | None = None,
) -> list[str] | None:
    selection = _scrolling_checkbox(
        "MCP:",
        choices=build_mcp_picker_choices(
            available_external_names,
            available_genie_servers,
            available_app_servers,
            original_servers,
            available_mcp_service_names,
            available_vector_search_servers,
            available_uc_functions_servers,
        ),
        style=_picker_style(),
        instruction="(space to toggle, enter to save, type to filter)",
    ).ask()
    if selection is None:
        return None
    return [str(value) for value in selection]


def _mcp_server_clients(server: dict) -> list[str]:
    return [client for client in (server.get("clients") or []) if client in MCP_CLIENTS]


def _resolve_mcp_selection(
    selection: str,
    workspace: str,
    available_app_servers: list[dict] | None = None,
    available_genie_servers: list[dict] | None = None,
    available_vector_search_servers: list[dict] | None = None,
    available_uc_functions_servers: list[dict] | None = None,
) -> tuple[str, str]:
    if selection.startswith(APP_MCP_SELECTION_PREFIX):
        app_name = selection.removeprefix(APP_MCP_SELECTION_PREFIX)
        if not app_name:
            raise RuntimeError("missing Databricks app name")
        server = _servers_by_name(available_app_servers or []).get(f"databricks-app-{app_name}")
        if not server:
            raise RuntimeError(f"Databricks app `{app_name}` was not in the discovered app list")
        url = server.get("url")
        if not isinstance(url, str) or not url:
            raise RuntimeError(f"Databricks app `{app_name}` has no MCP URL")
        return f"databricks-app-{app_name}", url

    if selection.startswith(GENIE_SPACE_SELECTION_PREFIX):
        suffix = selection.removeprefix(GENIE_SPACE_SELECTION_PREFIX)
        if not suffix:
            raise RuntimeError("missing Genie space id")
        server_name = f"databricks-genie-{suffix}"
        server = _servers_by_name(available_genie_servers or []).get(server_name)
        if server:
            url = server.get("url")
            if isinstance(url, str) and url:
                return server_name, url
        # Fallback for legacy picker values that carried the raw space_id.
        return server_name, f"{workspace}/api/2.0/mcp/genie/{suffix}"

    if selection.startswith(EXTERNAL_MCP_SELECTION_PREFIX):
        server_name = selection.removeprefix(EXTERNAL_MCP_SELECTION_PREFIX)
        if not server_name:
            raise RuntimeError("missing external connection name")
        return server_name, f"{workspace}/api/2.0/mcp/external/{server_name}"

    if selection.startswith(MCP_SERVICE_SELECTION_PREFIX):
        full_name = selection.removeprefix(MCP_SERVICE_SELECTION_PREFIX)
        if not full_name:
            raise RuntimeError("missing MCP service name")
        # Agent CLIs (claude/codex/gemini) reject dots in registered names.
        # URL keeps the UC `<cat>.<schema>.<id>` form; entry name uses dashes.
        return full_name.replace(".", "-"), build_mcp_service_url(workspace, full_name)

    if selection.startswith(VECTOR_SEARCH_SELECTION_PREFIX):
        return _resolve_catalog_schema_selection(
            selection.removeprefix(VECTOR_SEARCH_SELECTION_PREFIX),
            kind="vector search",
            url_path="vector-search",
            name_prefix="databricks-vector-search",
            workspace=workspace,
            available_servers=available_vector_search_servers,
        )

    if selection.startswith(UC_FUNCTIONS_SELECTION_PREFIX):
        return _resolve_catalog_schema_selection(
            selection.removeprefix(UC_FUNCTIONS_SELECTION_PREFIX),
            kind="UC functions",
            url_path="functions",
            name_prefix="databricks-functions",
            workspace=workspace,
            available_servers=available_uc_functions_servers,
        )

    if selection == SQL_MCP_VALUE:
        return "databricks-sql", f"{workspace}/api/2.0/mcp/sql"

    raise RuntimeError(f"unrecognized selection prefix in `{selection}`")


def _resolve_catalog_schema_selection(
    payload: str,
    *,
    kind: str,
    url_path: str,
    name_prefix: str,
    workspace: str,
    available_servers: list[dict] | None,
) -> tuple[str, str]:
    """Map a `catalog.schema` picker value back to the discovered server's name
    and URL, falling back to a deterministic slug when discovery has been lost
    (e.g. picker reopened on a stale workspace)."""
    if not payload or "." not in payload:
        raise RuntimeError(f"missing catalog.schema for {kind}")
    catalog, _, schema = payload.partition(".")
    if not catalog or not schema:
        raise RuntimeError(f"missing catalog.schema for {kind}")
    for server in available_servers or []:
        if server.get("catalog") == catalog and server.get("schema") == schema:
            name = _server_name(server)
            url = server.get("url")
            if name and isinstance(url, str) and url:
                return name, url
    name = _catalog_schema_server_name(name_prefix, catalog, schema, set())
    return name, f"{workspace}/api/2.0/mcp/{url_path}/{catalog}/{schema}"


def _discover_mcp_source(label: str, discover: Callable[[], list[Any]]) -> list[Any]:
    try:
        with spinner(f"Discovering {label}..."):
            return discover()
    except RuntimeError:
        print_warning(f"Skipped {label}.")
        return []


def apply_mcp_server_changes(
    original_servers: list[dict],
    working_servers: list[dict],
    clients: list[str],
) -> bool:
    original_by_name = _servers_by_name(original_servers)
    working_by_name = _servers_by_name(working_servers)
    changed = False

    for name, server in original_by_name.items():
        if name not in working_by_name:
            for client in _mcp_server_clients(server):
                remove_client_mcp_server(client, name)
            changed = True

    for name, server in working_by_name.items():
        original = original_by_name.get(name)
        if original == server:
            continue
        url = server.get("url")
        if not isinstance(url, str) or not url:
            continue
        # alwaysLoad (Claude-only) keeps the skills registry's utility tools
        # discoverable without an explicit mention; other clients ignore it.
        entry = build_mcp_http_entry(url, always_load=server.get("kind") == SKILLS_MCP_KIND)
        for client in clients:
            configure_client_mcp_server(client, name, url, entry)
        changed = True

    return changed


def purge_cross_workspace_mcp_residue(state: dict, workspace: str) -> None:
    installed = set(available_mcp_clients())

    raw_mcp_servers = list(state.get("mcp_servers") or [])
    current_mcp_servers, foreign_mcp_servers = _partition_mcp_entries_by_workspace(
        raw_mcp_servers, workspace
    )
    if foreign_mcp_servers:
        foreign_names = ", ".join(
            (_server_name(server) or "(unnamed)") for server in foreign_mcp_servers
        )
        noun = "entry" if len(foreign_mcp_servers) == 1 else "entries"
        print_warning(
            f"Dropping {len(foreign_mcp_servers)} stale MCP {noun} "
            f"not bound to this workspace: {foreign_names}."
        )
        for server in foreign_mcp_servers:
            name = _server_name(server)
            if not name:
                continue
            for client in server.get("clients") or []:
                if client not in installed or client not in MCP_CLIENTS:
                    continue
                try:
                    remove_client_mcp_server(client, name)
                except RuntimeError as exc:
                    print_warning(
                        f"Failed to remove `{name}` from {MCP_CLIENTS[client]['display']}: {exc}"
                    )
        state["mcp_servers"] = current_mcp_servers
        save_state(state)

    other_ws_mcps = _mcp_entries_only_in_other_workspaces(workspace)
    actually_removed: list[str] = []
    for name in sorted(other_ws_mcps):
        any_removed = False
        for client in other_ws_mcps[name]:
            if client not in installed or client not in MCP_CLIENTS:
                continue
            try:
                removed_scopes = remove_client_mcp_server(client, name)
            except RuntimeError as exc:
                print_warning(
                    f"Failed to remove `{name}` from {MCP_CLIENTS[client]['display']}: {exc}"
                )
                continue
            if removed_scopes:
                any_removed = True
        if any_removed:
            actually_removed.append(name)
    if actually_removed:
        noun = "entry" if len(actually_removed) == 1 else "entries"
        print_warning(
            f"Removed {len(actually_removed)} MCP {noun} left over from "
            f"previously-configured workspaces: {', '.join(actually_removed)}."
        )


def _skills_entries(servers: list[dict]) -> list[dict]:
    return [s for s in servers if s.get("kind") == SKILLS_MCP_KIND]


def _resolve_location_mcp_servers(
    workspace: str,
    profile: str | None,
    clients: list[str],
    location: str,
    original_servers: list[dict],
    services: set[str] | None = None,
) -> list[dict]:
    """Build the desired MCP server list for ``--location <cat>.<schema>``.

    Strict replacement for mcp-services: the returned list is exactly the ones
    discovered at ``location`` (any previously-registered mcp-service outside it
    is removed by ``apply_mcp_server_changes``), plus any existing skills
    connection, preserved untouched. Raises ``RuntimeError`` for an invalid
    location (HTTP 404 from the listing API) or any other listing failure.

    When ``services`` is given, the discovered set is narrowed to exactly that
    subset (matched by full name like ``system.ai.github`` or bare short name
    like ``github``); names not found at ``location`` are skipped with a
    warning rather than failing, so a saved selection that references a
    since-removed service still configures the rest. An empty set selects
    nothing (every previously-registered service in the location is removed).
    ``None`` keeps the whole schema."""
    if location.count(".") != 1 or not all(part.strip() for part in location.split(".")):
        raise RuntimeError(f"--location must be `<catalog>.<schema>`, got `{location}`.")

    token = get_databricks_token(workspace, profile)
    with spinner(f"Discovering MCP services in {location}..."):
        names, reason = list_mcp_services(workspace, token, parent=location)

    if reason and reason.startswith("HTTP 404"):
        raise RuntimeError(
            f"Invalid location: `{location}` is not a valid Unity Catalog schema "
            "in this workspace (or you lack USE permission on it)."
        )
    if reason:
        raise RuntimeError(f"Failed to list MCP services at `{location}`: {reason}")
    if not names:
        print_note(f"No MCP services exist at `{location}`.")

    if services is not None:
        discovered_full = set(names)
        discovered_short = {full_name.split(".")[-1] for full_name in names}
        unknown = services - discovered_full - discovered_short
        if unknown:
            print_warning(
                f"Ignoring requested MCP services not found in `{location}`: "
                f"{', '.join(sorted(unknown))}."
            )
        names = [
            full_name
            for full_name in names
            if full_name in services or full_name.split(".")[-1] in services
        ]

    original_by_name = _servers_by_name(original_servers)
    working_servers: list[dict] = []
    for full_name in names:
        entry_name = full_name.replace(".", "-")
        original = original_by_name.get(entry_name)
        original_clients = list((original or {}).get("clients") or [])
        merged_clients = original_clients + [c for c in clients if c not in original_clients]
        candidate = {
            "name": entry_name,
            "url": build_mcp_service_url(workspace, full_name),
            "auth": f"env:{MCP_AUTH_TOKEN_ENV_VAR}",
            "clients": merged_clients,
        }
        if original is not None and original == candidate:
            working_servers.append(original.copy())
        else:
            working_servers.append(candidate)
    return [*working_servers, *_skills_entries(original_servers)]


def setup_mcp_clients(state: dict, section: str) -> tuple[str, str | None, list[str]]:
    """Validate the workspace, resolve configured MCP clients, and prepare auth.

    Returns ``(workspace, profile, clients)`` and prints the section header, the
    "Configuring for" note, and a warning per configured-but-uninstalled client.
    """
    workspace = state.get("workspace")
    if not workspace:
        raise RuntimeError("Workspace is not configured. Run `ucode configure` first.")

    purge_cross_workspace_mcp_residue(state, workspace)

    installed_clients = available_mcp_clients()
    if not installed_clients:
        raise RuntimeError(
            "No supported MCP clients are installed. Install Claude, Codex, Gemini, OpenCode, "
            "or GitHub Copilot CLI."
        )
    clients = configured_mcp_clients(state, installed_clients)
    if not clients:
        raise RuntimeError(
            "No configured MCP-capable coding agents are installed. Run `ucode configure` "
            "for Codex, Claude, Gemini, OpenCode, or GitHub Copilot CLI first."
        )
    configured_tools = set(state.get("available_tools") or [])
    missing_clients = [
        client for client in MCP_CLIENTS if client in configured_tools and client not in clients
    ]

    profile = state.get("profile")
    apply_pat_environment(state)
    ensure_databricks_auth(workspace, profile)

    print_section(section)
    client_names = ", ".join(str(MCP_CLIENTS[client]["display"]) for client in clients)
    print_note(f"Configuring for: {client_names}")
    for client in missing_clients:
        print_warning(
            f"{MCP_CLIENTS[client]['display']} is configured in ucode but not installed; "
            "skipping MCP config."
        )
    return workspace, profile, clients


def configure_mcp_command(location: str | None = None, services: set[str] | None = None) -> int:
    if services is not None and location is None:
        # `--services` works standalone with full names (`system.ai.github`): the
        # `<catalog>.<schema>` to configure is derived from them. Bare short names
        # (`github`) can't be located without `--location`.
        schemas = {".".join(s.split(".")[:2]) for s in services if s.count(".") >= 2}
        bare = sorted(s for s in services if s.count(".") < 2)
        if bare:
            raise RuntimeError(
                "--services short names need --location (or pass full names like "
                f"`system.ai.<name>`): {', '.join(bare)}"
            )
        if len(schemas) != 1:
            raise RuntimeError(
                "--services without --location must all share one `<catalog>.<schema>` "
                f"(got: {', '.join(sorted(schemas)) or 'none'}); pass --location instead."
            )
        location = next(iter(schemas))
    state = load_state()
    workspace, profile, clients = setup_mcp_clients(state, "MCP Servers")

    original_mcp_servers_for_location: list[dict] = list(state.get("mcp_servers") or [])
    if location is not None:
        working_mcp_servers = _resolve_location_mcp_servers(
            workspace, profile, clients, location, original_mcp_servers_for_location, services
        )
        changed = apply_mcp_server_changes(
            original_mcp_servers_for_location, working_mcp_servers, clients
        )
        if changed or original_mcp_servers_for_location != working_mcp_servers:
            state["mcp_servers"] = working_mcp_servers
            save_state(state)
            print_success("Saved")
        return 0

    available_external_mcp_names = _discover_mcp_source(
        "external connections",
        lambda: discover_external_mcp_connection_names(workspace, profile),
    )
    available_genie_mcp_servers = _discover_mcp_source(
        "Genie spaces",
        lambda: discover_genie_mcp_servers(workspace, profile),
    )
    available_app_mcp_servers = _discover_mcp_source(
        "Databricks apps",
        lambda: discover_app_mcp_servers(workspace, profile),
    )
    # Curated `system.ai.*` MCP services live behind a separate UC API. Like
    # the other sources this is best-effort — `_discover_mcp_source` swallows
    # failures and returns [] so workspaces without them just see nothing extra.
    available_mcp_service_names = _discover_mcp_source(
        "MCP services",
        lambda: discover_mcp_service_names(workspace, profile),
    )
    # Per-(catalog, schema) managed MCP servers (Vector Search + UC Functions).
    available_vector_search_servers = _discover_mcp_source(
        "Vector Search",
        lambda: discover_vector_search_mcp_servers(workspace, profile),
    )
    available_uc_functions_servers = _discover_mcp_source(
        "UC functions",
        lambda: discover_uc_functions_mcp_servers(workspace, profile),
    )

    original_mcp_servers: list[dict] = list(state.get("mcp_servers") or [])
    # Skills connections are managed by `configure skills`, so keep them out of
    # the picker and carry them through untouched.
    skills_servers = _skills_entries(original_mcp_servers)
    picker_servers = [s for s in original_mcp_servers if s.get("kind") != SKILLS_MCP_KIND]
    original_by_name = _servers_by_name(picker_servers)
    selections = prompt_for_mcp_server_choices(
        available_external_mcp_names,
        available_genie_mcp_servers,
        available_app_mcp_servers,
        picker_servers,
        available_mcp_service_names,
        available_vector_search_servers,
        available_uc_functions_servers,
    )
    if selections is None:
        return 0

    working_mcp_servers: list[dict] = list(skills_servers)
    working_names: set[str] = set()
    add_selections: list[str] = []
    for selection in selections:
        if selection.startswith(MCP_ADD_PREFIX):
            add_selections.append(selection.removeprefix(MCP_ADD_PREFIX))
            continue
        original = original_by_name.get(selection)
        if original and selection not in working_names:
            working_mcp_servers.append(original.copy())
            working_names.add(selection)

    for selection in add_selections:
        try:
            entry_name, url = _resolve_mcp_selection(
                selection,
                workspace,
                available_app_mcp_servers,
                available_genie_mcp_servers,
                available_vector_search_servers,
                available_uc_functions_servers,
            )
        except RuntimeError as exc:
            print_warning(f"Skipped MCP selection `{selection}`: {exc}.")
            continue
        if entry_name in working_names:
            continue
        working_mcp_servers.append(
            {
                "name": entry_name,
                "url": url,
                "auth": f"env:{MCP_AUTH_TOKEN_ENV_VAR}",
                "clients": clients,
            }
        )
        working_names.add(entry_name)

    changed = apply_mcp_server_changes(original_mcp_servers, working_mcp_servers, clients)
    if changed or original_mcp_servers != working_mcp_servers:
        state["mcp_servers"] = working_mcp_servers
        save_state(state)
        print_success("Saved")
    elif not selections and not original_mcp_servers:
        # User submitted the picker without toggling anything --> make it clear nothing was selected
        print_note("No MCP servers selected. Press space to toggle an item, then enter to save.")
    return 0


def _merge_clients(prior: list[str] | None, new: list[str]) -> list[str]:
    """Order-preserving union of a prior client list with newly-configured ones."""
    prior = list(prior or [])
    return prior + [c for c in new if c not in prior]


def _build_skills_entry(workspace: str, locations: list[str], clients: list[str]) -> dict:
    """Canonical single skills-registry entry. ``skill_locations`` is the source
    of truth; the URL is always derived from it, never parsed back."""
    return {
        "name": SKILLS_MCP_SERVER_NAME,
        "kind": SKILLS_MCP_KIND,
        "skill_locations": list(locations),
        "url": build_skills_mcp_url(workspace, locations),
        "auth": f"env:{MCP_AUTH_TOKEN_ENV_VAR}",
        "clients": clients,
    }


def _resolve_skills_mcp_servers(
    workspace: str,
    clients: list[str],
    locations: list[str],
    original_servers: list[dict],
) -> list[dict]:
    """Rebuild the MCP server list around exactly one skills entry.

    Drops every prior ``kind=="skills"`` entry and any entry named
    ``SKILLS_MCP_SERVER_NAME`` (single-connection invariant; also sweeps up a
    stray old-named entry via ``apply_mcp_server_changes``), keeps everything
    else, and appends one rebuilt entry whose clients merge the prior skills
    entry's clients with ``clients``.
    """
    prior = next((s for s in original_servers if s.get("kind") == SKILLS_MCP_KIND), None)
    merged = _merge_clients((prior or {}).get("clients"), clients)
    kept = [
        s
        for s in original_servers
        if s.get("kind") != SKILLS_MCP_KIND and _server_name(s) != SKILLS_MCP_SERVER_NAME
    ]
    return [*kept, _build_skills_entry(workspace, locations, merged)]


def _update_skills_mcp(
    state: dict, workspace: str, clients: list[str], locations: list[str]
) -> None:
    """Rebuild the single skills connection for ``locations`` and persist it."""
    original = list(state.get("mcp_servers") or [])
    working = _resolve_skills_mcp_servers(workspace, clients, locations, original)
    changed = apply_mcp_server_changes(original, working, clients)
    if changed or original != working:
        state["mcp_servers"] = working
        save_state(state)
        print_success("Saved")


def configure_skills_mcp_command(locations: list[str]) -> int:
    """Set the skills MCP connection's ``skill_locations`` to exactly ``locations``,
    replacing any previous set."""
    state = load_state()
    workspace, _profile, clients = setup_mcp_clients(state, "Skills MCP")
    _update_skills_mcp(state, workspace, clients, locations)
    return 0


def _skill_mcp_locations(state: dict) -> list[str]:
    """The skills MCP connection's ``skill_locations``, or ``[]`` if none exists."""
    entry = next(iter(_skills_entries(list(state.get("mcp_servers") or []))), None)
    return list((entry or {}).get("skill_locations") or [])


def register_schemaless_skills_connection(state: dict, workspace: str, clients: list[str]) -> None:
    """Register/keep the skills MCP connection without changing its schema set.

    Download mode calls this after writing files: it preserves any prior
    ``--mcp`` ``skill_locations`` and otherwise registers the bare schema-less
    route (utility tools only)."""
    _update_skills_mcp(state, workspace, clients, _skill_mcp_locations(state))
