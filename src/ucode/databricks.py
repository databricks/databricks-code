"""Databricks workspace integration: CLI auth, token retrieval, model
discovery, AI Gateway v2 enforcement, SQL warehouse discovery, URL builders."""

from __future__ import annotations

import configparser
import functools
import json
import logging
import logging.handlers
import os
import platform
import re
import shlex
import shutil
import subprocess
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from concurrent.futures import (
    TimeoutError as FutureTimeoutError,
)
from pathlib import Path
from typing import Literal, cast, overload
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlencode, urlparse

from databricks.sql.exc import ServerOperationError

from ucode.config_io import APP_DIR
from ucode.ui import (
    err_console,
    normalize_workspace_url,
    print_kv,
    print_note,
    print_section,
    print_success,
    print_warning,
    spinner,
)

UNIX_DATABRICKS_INSTALL_URL = (
    "https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh"
)
WINDOWS_DATABRICKS_INSTALL_URL = (
    "https://raw.githubusercontent.com/databricks/setup-cli/main/install.ps1"
)
AI_GATEWAY_V2_DOCS_URL = "https://docs.databricks.com/aws/en/ai-gateway/overview-beta"
# v1.0.0 is the release that ships `databricks aitools`.
MIN_DATABRICKS_CLI_VERSION = (1, 0, 0)
TOKEN_REFRESH_INTERVAL_SECONDS = 1800


def _debug_enabled() -> bool:
    return os.environ.get("UCODE_DEBUG") == "1"


_DEBUG_LOGGER: logging.Logger | None = None


def _get_debug_logger() -> logging.Logger | None:
    """Lazily configure a rotating file logger when UCODE_DEBUG=1.

    Returns the logger on first call (and caches it), or None if debug is
    disabled or the log file could not be opened. A one-time breadcrumb is
    printed to stderr so the user knows where to tail."""
    global _DEBUG_LOGGER
    if _DEBUG_LOGGER is not None or not _debug_enabled():
        return _DEBUG_LOGGER

    log_path = APP_DIR / "debug.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
        )
    except OSError:
        return None

    logger = logging.getLogger("ucode.debug")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    _DEBUG_LOGGER = logger
    err_console.print(f"[dim]\\[ucode debug] logging to {log_path}[/dim]")
    return _DEBUG_LOGGER


def _debug(label: str, detail: str) -> None:
    """When UCODE_DEBUG=1, append a timestamped entry to ~/.ucode/debug.log."""
    logger = _get_debug_logger()
    if logger is not None:
        logger.debug("%s: %s", label, detail)


_SECRET_KEY_PATTERN = re.compile(r"(token|secret|password|bearer|api_key|apikey)", re.IGNORECASE)


def _format_subprocess_result(
    result: subprocess.CompletedProcess[str],
) -> str:
    """Format a CompletedProcess for the debug log without leaking tokens.

    On success, stdout is suppressed (it often contains the access token).
    On failure, stdout/stderr are included truncated."""
    stderr = (result.stderr or "").strip()[:500]
    if result.returncode == 0:
        return f"rc=0 stderr={stderr!r}"
    stdout = (result.stdout or "").strip()[:500]
    return f"rc={result.returncode} stdout={stdout!r} stderr={stderr!r}"


def _scrub_databrickscfg(text: str) -> str:
    """Redact value of any INI key that looks secret-bearing."""
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if "=" in stripped and not stripped.startswith(("#", ";")):
            key = stripped.split("=", 1)[0].strip()
            if _SECRET_KEY_PATTERN.search(key):
                indent = line[: len(line) - len(stripped)]
                out.append(f"{indent}{key} = <redacted>")
                continue
        out.append(line)
    return "\n".join(out)


def _scrub_json(value: object) -> object:
    if isinstance(value, dict):
        return {
            k: (
                "<redacted>"
                if isinstance(k, str) and _SECRET_KEY_PATTERN.search(k)
                else _scrub_json(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_scrub_json(v) for v in value]
    return value


@functools.cache
def _log_auth_diagnostics() -> None:
    """Dump CLI version, profiles, and ~/.databrickscfg (scrubbed) to the debug log.

    No-op unless UCODE_DEBUG=1; cached so it runs at most once per process."""
    if not _debug_enabled():
        return

    try:
        version_result = subprocess.run(
            ["databricks", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = (version_result.stdout or version_result.stderr or "").strip()
        _debug("databricks --version", version[:200])
    except (OSError, subprocess.TimeoutExpired) as exc:
        _debug("databricks --version", f"exception: {type(exc).__name__}: {exc}")

    try:
        profiles_result = subprocess.run(
            ["databricks", "auth", "profiles", "--output", "json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        _debug(
            "databricks auth profiles",
            f"rc={profiles_result.returncode} "
            f"stderr={(profiles_result.stderr or '').strip()[:300]!r}",
        )
        if profiles_result.returncode == 0 and profiles_result.stdout:
            try:
                payload = json.loads(profiles_result.stdout)
                _debug("profiles json", json.dumps(_scrub_json(payload))[:2000])
            except json.JSONDecodeError as exc:
                _debug("profiles json", f"decode error: {exc}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        _debug("databricks auth profiles", f"exception: {type(exc).__name__}: {exc}")

    cfg_path = Path(os.environ.get("DATABRICKS_CONFIG_FILE") or "~/.databrickscfg").expanduser()
    try:
        if cfg_path.is_file():
            raw = cfg_path.read_text(encoding="utf-8", errors="replace")
            _debug(f"databrickscfg ({cfg_path})", _scrub_databrickscfg(raw)[:4000])
        else:
            _debug(f"databrickscfg ({cfg_path})", "not present")
    except OSError as exc:
        _debug(f"databrickscfg ({cfg_path})", f"read error: {exc}")


def _http_get_json(
    url: str, token: str, *, timeout: int = 10
) -> tuple[dict | list | None, str | None]:
    """GET a JSON endpoint. Returns (payload, None) on success, (None, reason) on failure.

    Honors UCODE_DEBUG=1 to append status + truncated body to ~/.ucode/debug.log.
    """
    request = urllib_request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
        _debug(f"GET {url}", f"HTTP 200, {len(body)} bytes")
        if _debug_enabled():
            _debug("body", body[:4000])
        try:
            return json.loads(body), None
        except json.JSONDecodeError as exc:
            return None, f"response was not valid JSON ({exc.msg})"
    except urllib_error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        except Exception:
            body = ""
        _debug(f"GET {url}", f"HTTP {exc.code} {exc.reason}")
        if _debug_enabled() and body:
            _debug("body", body[:4000])
        reason = f"HTTP {exc.code} {exc.reason}"
        # Surface the response body too — gateway auth failures return 400
        # with body `Invalid Token`, which is invisible without this.
        body_excerpt = body.strip()[:200]
        if body_excerpt:
            reason = f"{reason}: {body_excerpt}"
        return None, reason
    except urllib_error.URLError as exc:
        _debug(f"GET {url}", f"URLError: {exc.reason}")
        return None, f"network error: {exc.reason}"


def _http_post_json(
    url: str, token: str, payload: dict, *, timeout: int = 10
) -> tuple[dict | list | None, str | None]:
    """POST a JSON body to an endpoint. Returns (payload, None) on success,
    (None, reason) on failure. Mirrors `_http_get_json`."""
    body_bytes = json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        url,
        data=body_bytes,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
        _debug(f"POST {url}", f"HTTP {response.status}, {len(body)} bytes")
        if _debug_enabled():
            _debug("body", body[:4000])
        try:
            return json.loads(body), None
        except json.JSONDecodeError as exc:
            return None, f"response was not valid JSON ({exc.msg})"
    except urllib_error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        except Exception:
            body = ""
        _debug(f"POST {url}", f"HTTP {exc.code} {exc.reason}")
        if _debug_enabled() and body:
            _debug("body", body[:4000])
        reason = f"HTTP {exc.code} {exc.reason}"
        body_excerpt = body.strip()[:200]
        if body_excerpt:
            reason = f"{reason}: {body_excerpt}"
        return None, reason
    except urllib_error.URLError as exc:
        _debug(f"POST {url}", f"URLError: {exc.reason}")
        return None, f"network error: {exc.reason}"


def get_current_user_name(workspace: str, token: str) -> str | None:
    """Return the current user's login (email) via SCIM `Me`, or None on failure.

    Databricks puts the workspace login in `userName`; fall back to the first
    `emails` entry for workspaces that diverge."""
    hostname = workspace_hostname(workspace)
    payload, _ = _http_get_json(f"https://{hostname}/api/2.0/preview/scim/v2/Me", token)
    if not isinstance(payload, dict):
        return None
    user_name = payload.get("userName")
    if isinstance(user_name, str) and user_name.strip():
        return user_name.strip()
    emails = payload.get("emails")
    if isinstance(emails, list):
        for entry in emails:
            if isinstance(entry, dict) and isinstance(entry.get("value"), str):
                return entry["value"].strip()
    return None


# Experiment tag Databricks sets when an experiment's traces are written to a
# Unity Catalog table. Its value is the UC destination, e.g.
# "my_catalog.my_schema.my_table". A plain (file/DBFS-backed) experiment does
# not carry this tag, so its presence is our signal that traces land in UC.
UC_TRACE_DESTINATION_TAG = "mlflow.experiment.databricksTraceDestinationPath"


def _experiment_tags(experiment: dict) -> dict[str, str | None]:
    """Flatten an experiment's ``tags`` list ([{key, value}, ...]) into a dict."""
    out: dict[str, str | None] = {}
    tags = experiment.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, dict) and isinstance(tag.get("key"), str):
                out[tag["key"]] = tag.get("value")
    return out


def _uc_trace_destination(experiment: dict) -> str | None:
    """The Unity Catalog destination (``catalog.schema.table``) an experiment
    logs traces to, or None when it isn't UC-backed. Any three-part UC name
    qualifies — the specific catalog/schema/table is not constrained."""
    value = _experiment_tags(experiment).get(UC_TRACE_DESTINATION_TAG)
    if isinstance(value, str):
        parts = value.split(".")
        if len(parts) == 3 and all(parts):
            return value
    return None


def find_uc_backed_experiment(
    workspace: str, token: str, leaf_name: str
) -> tuple[dict | None, str | None]:
    """Find an existing experiment whose final path segment is ``leaf_name`` and
    whose traces are backed by Unity Catalog.

    Returns (experiment, reason). On success ``experiment`` is
    ``{"experiment_id", "experiment_name", "uc_destination"}`` and reason is
    None. On failure ``experiment`` is None and reason explains why (no such
    experiment, or it exists but isn't UC-backed) so the caller can tell the
    user to create one."""
    hostname = workspace_hostname(workspace)
    # Leaf-match in the filter (anything ending in the name), then confirm the
    # exact leaf segment in Python so "/Users/<me>/ucode-traces" matches but
    # "team-ucode-traces" does not.
    safe_leaf = leaf_name.replace("'", "")
    payload, reason = _http_post_json(
        f"https://{hostname}/api/2.0/mlflow/experiments/search",
        token,
        {"filter": f"name LIKE '%{safe_leaf}'", "max_results": 1000},
    )
    if not isinstance(payload, dict):
        return None, reason or "could not search MLflow experiments"

    experiments = payload.get("experiments")
    named = [
        exp
        for exp in (experiments if isinstance(experiments, list) else [])
        if isinstance(exp, dict)
        and str(exp.get("name") or "").rsplit("/", 1)[-1] == leaf_name
        and exp.get("experiment_id")
    ]
    if not named:
        return None, f"no experiment named '{leaf_name}' exists on this workspace"

    for exp in named:
        dest = _uc_trace_destination(exp)
        if dest:
            return {
                "experiment_id": str(exp["experiment_id"]),
                "experiment_name": str(exp.get("name") or leaf_name),
                "uc_destination": dest,
            }, None

    return (
        None,
        f"experiment '{leaf_name}' exists but its traces are not backed by Unity Catalog",
    )


def resolve_sql_warehouse_id(workspace: str, token: str) -> tuple[str | None, str | None]:
    """Pick a SQL warehouse for writing traces to a UC-backed experiment.

    Writing traces to a Unity Catalog table requires a SQL warehouse
    (``MLFLOW_TRACING_SQL_WAREHOUSE_ID``); without one the MLflow exporter
    silently drops them. We prefer a RUNNING warehouse so the first trace isn't
    blocked on a cold start, falling back to any existing warehouse (a stopped
    one auto-starts on first query). Returns (warehouse_id, reason); reason is
    None on success, else explains why none could be resolved."""
    hostname = workspace_hostname(workspace)
    payload, reason = _http_get_json(f"https://{hostname}/api/2.0/sql/warehouses", token)
    if not isinstance(payload, dict):
        return None, reason or "could not list SQL warehouses"

    warehouses = payload.get("warehouses")
    warehouses = (
        [w for w in warehouses if isinstance(w, dict) and w.get("id")]
        if isinstance(warehouses, list)
        else []
    )
    if not warehouses:
        return None, "no SQL warehouse exists on this workspace"

    running = next((w for w in warehouses if str(w.get("state")).upper() == "RUNNING"), None)
    chosen = running or warehouses[0]
    return str(chosen["id"]), None


@overload
def run(
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    text: Literal[True],
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]: ...


@overload
def run(
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    text: Literal[False] = False,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[bytes]: ...


def run(
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    text: bool = False,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        args,
        check=check,
        capture_output=capture_output,
        text=text,
        env=env,
        timeout=timeout,
    )


def build_databricks_cli_env(workspace: str, profile: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["DATABRICKS_HOST"] = workspace
    if profile is None:
        env.pop("DATABRICKS_CONFIG_PROFILE", None)
    return env


def workspace_hostname(workspace: str) -> str:
    parsed = urlparse(normalize_workspace_url(workspace))
    if not parsed.hostname:
        raise RuntimeError(f"Unable to derive hostname from workspace URL: {workspace}")
    return parsed.hostname


def _parse_databricks_cli_version(output: str) -> tuple[int, int, int] | None:
    # Example output: "Databricks CLI v0.299.2"
    match = re.search(r"v?(\d+)\.(\d+)\.(\d+)", output)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _run_databricks_cli_installer(brew_subcommand: str = "install") -> None:
    system = platform.system()
    try:
        if system == "Windows":
            run(
                ["powershell", "-Command", f"irm {WINDOWS_DATABRICKS_INSTALL_URL} | iex"],
                timeout=240,
            )
        elif system == "Darwin" and shutil.which("brew"):
            run(["brew", brew_subcommand, "databricks/tap/databricks"], timeout=240)
        elif shutil.which("curl"):
            run(["sh", "-c", f"curl -fsSL {UNIX_DATABRICKS_INSTALL_URL} | sudo sh"], timeout=240)
        elif shutil.which("wget"):
            run(["sh", "-c", f"wget -qO- {UNIX_DATABRICKS_INSTALL_URL} | sudo sh"], timeout=240)
        else:
            raise RuntimeError("Neither curl nor wget is available.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as exc:
        raise RuntimeError("Failed to install/upgrade Databricks CLI automatically.") from exc


def ensure_databricks_cli_version() -> None:
    try:
        result = run(
            ["databricks", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Failed to read Databricks CLI version.") from exc

    raw = result.stdout or result.stderr or ""
    output = (raw if isinstance(raw, str) else raw.decode(errors="replace")).strip()
    version = _parse_databricks_cli_version(output)
    if version is None:
        raise RuntimeError(
            f"Could not parse Databricks CLI version from `databricks --version` output: {output!r}"
        )
    if version < MIN_DATABRICKS_CLI_VERSION:
        current = ".".join(str(n) for n in version)
        required = ".".join(str(n) for n in MIN_DATABRICKS_CLI_VERSION)
        print_warning(
            f"Databricks CLI v{current} is too old (need v{required} or newer). Upgrading..."
        )
        _run_databricks_cli_installer(brew_subcommand="upgrade")
        ensure_databricks_cli_version()


def install_databricks_cli() -> None:
    if shutil.which("databricks"):
        ensure_databricks_cli_version()
        return

    print_section("Bootstrap")
    print_warning("`databricks` was not found. Installing Databricks CLI...")
    _run_databricks_cli_installer(brew_subcommand="install")

    if not shutil.which("databricks"):
        raise RuntimeError(
            "Databricks CLI install completed, but `databricks` is still not on PATH."
        )
    ensure_databricks_cli_version()


def install_ai_tools(agent_tokens: list[str], profile: str | None = None) -> None:
    """Install Databricks AI Tools for the given agents (e.g. ``claude-code``).

    Databricks AI Tools is the set of skills and plugins that teach coding
    agents how to work with Databricks (installed via ``databricks aitools``).
    Idempotent and best-effort: any failure only warns (surfacing the CLI's
    own error), since AI Tools aren't required to launch an agent."""
    if not agent_tokens:
        return

    agents_arg = ",".join(agent_tokens)
    try:
        with spinner(f"Installing Databricks AI Tools for {agents_arg}..."):
            run(
                ["databricks", "aitools", "install", "--agents", agents_arg, "--scope", "global"]
                + _profile_args(profile),
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        # The CLI version is already guaranteed by ensure_databricks_cli_version,
        # so any failure here is something else (e.g. an agent binary missing
        # from PATH). Surface the CLI's own error rather than guessing a cause.
        detail = getattr(exc, "stderr", None) or ""
        if isinstance(detail, bytes):  # TimeoutExpired.stderr is bytes even with text=True
            detail = detail.decode(errors="replace")
        detail = detail.strip()
        reason = detail.splitlines()[-1] if detail else str(exc)
        print_warning(f"Could not install Databricks AI Tools: {reason}")
    else:
        print_success("Databricks AI Tools installed")


def _profile_args(profile: str | None) -> list[str]:
    """Return ``["--profile", profile]`` when set, otherwise an empty list.

    Centralizing this keeps every `databricks` CLI invocation in this module
    consistent when a workspace's `~/.databrickscfg` has more than one profile
    pointing at the same host."""
    return ["--profile", profile] if profile else []


def has_valid_databricks_auth(workspace: str, profile: str | None = None) -> bool:
    # Honor the CI short-circuit (see ``get_databricks_token``): if a
    # pre-fetched bearer is available, treat auth as valid and skip the
    # `databricks auth token` shell-out (which only knows user-OAuth).
    if os.environ.get("DATABRICKS_BEARER", "").strip():
        return True
    _log_auth_diagnostics()
    # Mirror run_databricks_login: when ~/.databrickscfg has multiple
    # profiles for the same host, `databricks auth token --host …` refuses
    # to disambiguate without --profile, so resolve it from the host here.
    profile = profile or find_profile_name_for_host(workspace)
    try:
        env = build_databricks_cli_env(workspace, profile)
        result = run(
            [
                "databricks",
                "auth",
                "token",
                "--host",
                workspace,
                *_profile_args(profile),
                "--output",
                "json",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )
        _debug(
            "has_valid_databricks_auth",
            _format_subprocess_result(result),
        )
        if result.returncode != 0:
            return False
        data = json.loads(result.stdout or "{}")
        return bool(data.get("access_token"))
    except (json.JSONDecodeError, OSError, subprocess.TimeoutExpired) as exc:
        _debug("has_valid_databricks_auth", f"exception: {type(exc).__name__}: {exc}")
        return False


def list_profile_entries() -> list[dict]:
    """Return raw profile dicts ({"name", "host", "auth_type", ...}) from
    `databricks auth profiles`.

    Returns ``[]`` on any failure (CLI missing, timeout, non-zero exit, JSON
    decode error). When ``UCODE_DEBUG=1`` each dropout path logs *why* the
    result was empty so a silently-disappearing workspace picker is
    diagnosable from ``~/.ucode/debug.log``.
    """
    try:
        result = run(
            ["databricks", "auth", "profiles", "--output", "json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _debug("list_profile_entries", f"subprocess error: {type(exc).__name__}: {exc}")
        return []
    if result.returncode != 0:
        _debug("list_profile_entries", _format_subprocess_result(result))
        return []
    try:
        profiles = json.loads(result.stdout or "{}").get("profiles") or []
    except json.JSONDecodeError as exc:
        _debug("list_profile_entries", f"json decode error: {exc.msg}")
        return []
    return [p for p in profiles if isinstance(p, dict)]


def get_databricks_profiles() -> list[tuple[str, str]]:
    """Return [(host_url, profile_name), ...] from Databricks CLI profiles."""
    profiles = list_profile_entries()

    # dict dedupes by host (first non-PAT profile wins).
    out: dict[str, str] = {}
    pat = 0
    for p in profiles:
        host = (p.get("host") or "").rstrip("/")
        name = p.get("name")
        if not host or not name:
            continue
        if p.get("auth_type") == "pat":
            pat += 1
            continue
        out.setdefault(host, name)

    _debug(
        "get_databricks_profiles",
        f"returned={len(out)} total={len(profiles)} pat={pat}",
    )
    return list(out.items())


def find_profile_name_for_host(workspace: str) -> str | None:
    """Find the Databricks CLI profile name matching a workspace URL."""
    normalized = workspace.rstrip("/")
    for host, name in get_databricks_profiles():
        if host == normalized:
            return name
    return None


def profile_auth_type(profile: str) -> str | None:
    """Return the auth_type of a Databricks CLI profile (e.g. "pat"), or None."""
    for p in list_profile_entries():
        if p.get("name") == profile:
            auth_type = p.get("auth_type")
            return auth_type if isinstance(auth_type, str) else None
    return None


def _read_databrickscfg_token(profile: str) -> str | None:
    """Read the static ``token`` value for a profile from ``~/.databrickscfg``.

    `databricks auth token` only knows OAuth caches; for PAT profiles the PAT
    itself is the credential, stored in the config file. The parser's default
    section is pointed at a name that never appears in the file so a token in
    ``[DEFAULT]`` does not leak into every named profile."""
    cfg_path = Path(os.environ.get("DATABRICKS_CONFIG_FILE") or "~/.databrickscfg").expanduser()
    parser = configparser.ConfigParser(default_section="@ucode-no-defaults@", interpolation=None)
    try:
        if not parser.read(cfg_path, encoding="utf-8"):
            return None
    except (configparser.Error, OSError):
        return None
    if not parser.has_section(profile):
        return None
    token = (parser.get(profile, "token", fallback="") or "").strip()
    return token or None


def resolve_pat_token(profile: str | None) -> str | None:
    """Return the static PAT of a PAT-type Databricks CLI profile, or None.

    Only consulted when the user explicitly opted in via
    ``ucode configure --profiles <name> --use-pat`` — ucode never picks up a
    PAT implicitly."""
    if profile and profile_auth_type(profile) == "pat":
        return _read_databrickscfg_token(profile)
    return None


def ensure_pat_bearer(profile: str | None, pat: str | None = None) -> bool:
    """Ensure ``DATABRICKS_BEARER`` holds a usable token for a ``--use-pat`` profile.

    If a non-empty bearer is already in the environment it wins (the CI escape
    hatch). Otherwise the profile's static PAT is exported — callers that have
    already resolved it (e.g. ``configure_shared_state``) pass it via ``pat`` to
    skip a redundant ``~/.databrickscfg`` read; everyone else lets this resolve
    it. An exported-but-*empty* ``DATABRICKS_BEARER`` is treated as absent —
    matching ``get_databricks_token``'s own ``.strip()`` check — so a stray
    ``export DATABRICKS_BEARER=`` does not shadow the PAT and silently force the
    OAuth path (which fails for PAT-only profiles).

    Returns ``True`` iff a usable bearer is now present in the environment."""
    if os.environ.get("DATABRICKS_BEARER", "").strip():
        return True
    pat = pat or resolve_pat_token(profile)
    if pat:
        os.environ["DATABRICKS_BEARER"] = pat
        return True
    return False


def apply_pat_environment(state: dict) -> None:
    """Export the configured profile's PAT as ``DATABRICKS_BEARER`` when the
    workspace was configured with ``--use-pat``.

    Every token fetch in this process (and in launched agent subprocesses,
    which inherit the environment) then takes the existing static-bearer
    short-circuit instead of the OAuth-only `databricks auth token` path.
    A non-empty bearer already present in the environment is left untouched."""
    if not state.get("use_pat"):
        return
    ensure_pat_bearer(state.get("profile"))


def run_databricks_login(workspace: str, profile: str | None = None) -> None:
    """Run databricks auth login unconditionally.

    When ``profile`` is provided, it is passed via ``--profile``. Otherwise we
    fall back to looking up an existing profile by host so a stored session is
    refreshed in place rather than overwriting another profile's tokens."""
    print_section("Databricks Login")
    print_kv("Workspace", workspace)
    print_note("A browser may open for `databricks auth login`.")
    try:
        profile_name = profile or find_profile_name_for_host(workspace)
        cmd = [
            "databricks",
            "auth",
            "login",
            "--host",
            workspace,
            *_profile_args(profile_name),
        ]
        run(cmd, env=build_databricks_cli_env(workspace, profile_name), timeout=300)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("`databricks auth login` failed.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("`databricks auth login` timed out.") from exc
    print_success("Databricks authentication complete")


def ensure_databricks_auth(workspace: str, profile: str | None = None) -> None:
    """Check auth and login only if needed (used by launch path)."""
    with spinner("Checking Databricks auth..."):
        auth_is_valid = has_valid_databricks_auth(workspace, profile)
    if auth_is_valid:
        print_success(f"Databricks auth already available for {workspace}")
        return
    run_databricks_login(workspace, profile)


def get_databricks_token(
    workspace: str,
    profile: str | None = None,
    *,
    force_refresh: bool = False,
) -> str:
    # ``DATABRICKS_BEARER`` is the CI escape hatch: when set, skip the
    # `databricks auth token` subprocess entirely and return the pre-fetched
    # bearer directly. Used by the e2e job, where the protected runner has
    # no `databricks auth login` cache and `databricks auth token` only knows
    # how to read user-OAuth caches (not M2M client_credentials). Mirrors the
    # same short-circuit baked into ``build_auth_shell_command``.
    bearer = os.environ.get("DATABRICKS_BEARER", "").strip()
    if bearer:
        _debug("get_databricks_token", "using DATABRICKS_BEARER env var")
        return bearer

    _log_auth_diagnostics()
    # See has_valid_databricks_auth: resolve the profile from the host when
    # the caller didn't supply one, so duplicate-host cfgs don't break us.
    profile = profile or find_profile_name_for_host(workspace)
    env = build_databricks_cli_env(workspace, profile)
    cmd = [
        "databricks",
        "auth",
        "token",
        "--host",
        workspace,
        *_profile_args(profile),
        "--output",
        "json",
    ]
    if force_refresh:
        cmd.append("--force-refresh")

    _debug(
        "get_databricks_token.env",
        "set="
        + ",".join(sorted(k for k in env if k.startswith("DATABRICKS_") or k in {"BUNDLE_PROFILE"}))
        + f" profile={profile or '<none>'}",
    )

    def _fetch() -> str:
        try:
            result = run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=15,
            )
            _debug("auth token", _format_subprocess_result(result))
            if result.returncode == 0:
                return json.loads(result.stdout or "{}").get("access_token", "")
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            _debug("auth token", f"exception: {type(exc).__name__}: {exc}")
        return ""

    token = _fetch()
    if not token:
        # Session may have expired — attempt non-interactive re-auth and retry once.
        _debug("auth token", "empty on first fetch; attempting auth login --no-browser")
        try:
            reauth = run(
                [
                    "databricks",
                    "auth",
                    "login",
                    "--host",
                    workspace,
                    *_profile_args(profile),
                    "--no-browser",
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            _debug("auth login", _format_subprocess_result(reauth))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            _debug("auth login", f"exception: {type(exc).__name__}: {exc}")
        token = _fetch()

    if not token:
        profile_name = profile or find_profile_name_for_host(workspace)
        stale_profile_hint = ""
        if profile_name:
            stale_profile_hint = (
                " The saved Databricks CLI profile may be stale or invalid. Try:\n"
                f"  databricks auth logout --profile {profile_name}\n"
                f"  databricks auth login --host {workspace} --profile {profile_name}"
            )
        raise RuntimeError(
            f"Databricks CLI returned no access token for {workspace}. "
            "Run `databricks auth login` to re-authenticate."
            f"{stale_profile_hint}"
        )
    return token


def _extract_connection_page(payload: object) -> tuple[list[dict], str | None]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)], None
    if not isinstance(payload, dict):
        raise RuntimeError("Databricks connections listing returned invalid JSON.")

    payload_dict = cast(dict[str, object], payload)
    raw_connections = payload_dict.get("connections") or []
    if not isinstance(raw_connections, list):
        raise RuntimeError("Databricks connections listing returned invalid JSON.")

    next_page_token = payload_dict.get("next_page_token")
    if next_page_token is not None and not isinstance(next_page_token, str):
        raise RuntimeError("Databricks connections listing returned invalid JSON.")

    return [item for item in raw_connections if isinstance(item, dict)], next_page_token


def list_databricks_connections(workspace: str, profile: str | None = None) -> list[dict]:
    env = build_databricks_cli_env(workspace)
    connections: list[dict] = []
    page_token: str | None = None
    seen_page_tokens: set[str] = set()

    try:
        while True:
            cmd = [
                "databricks",
                "connections",
                "list",
                *_profile_args(profile),
                "--max-results",
                "0",
                "--output",
                "json",
            ]
            if page_token:
                cmd.extend(["--page-token", page_token])

            result = run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            payload = json.loads(result.stdout or "{}")
            page_connections, page_token = _extract_connection_page(payload)
            connections.extend(page_connections)

            if not page_token:
                return connections
            if page_token in seen_page_tokens:
                raise RuntimeError("Databricks connections listing returned a repeated page token.")
            seen_page_tokens.add(page_token)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to list Databricks connections via `databricks connections list`."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timed out while listing Databricks connections.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Databricks connections listing returned invalid JSON.") from exc


def _extract_genie_spaces_page(payload: object) -> tuple[list[dict], str | None]:
    if not isinstance(payload, dict):
        raise RuntimeError("Databricks Genie spaces listing returned invalid JSON.")

    payload_dict = cast(dict[str, object], payload)
    raw_spaces = payload_dict.get("spaces") or []
    if not isinstance(raw_spaces, list):
        raise RuntimeError("Databricks Genie spaces listing returned invalid JSON.")

    next_page_token = payload_dict.get("next_page_token")
    if next_page_token is not None and not isinstance(next_page_token, str):
        raise RuntimeError("Databricks Genie spaces listing returned invalid JSON.")

    return [item for item in raw_spaces if isinstance(item, dict)], next_page_token


def list_genie_spaces(workspace: str, profile: str | None = None) -> list[dict]:
    env = build_databricks_cli_env(workspace)
    spaces: list[dict] = []
    page_token: str | None = None
    seen_page_tokens: set[str] = set()

    try:
        while True:
            cmd = [
                "databricks",
                "genie",
                "list-spaces",
                *_profile_args(profile),
                "--page-size",
                "100",
                "--output",
                "json",
            ]
            if page_token:
                cmd.extend(["--page-token", page_token])

            result = run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            payload = json.loads(result.stdout or "{}")
            page_spaces, page_token = _extract_genie_spaces_page(payload)
            spaces.extend(page_spaces)

            if not page_token:
                return spaces
            if page_token in seen_page_tokens:
                raise RuntimeError(
                    "Databricks Genie spaces listing returned a repeated page token."
                )
            seen_page_tokens.add(page_token)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to list Databricks Genie spaces via `databricks genie list-spaces`."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timed out while listing Databricks Genie spaces.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Databricks Genie spaces listing returned invalid JSON.") from exc


def _extract_apps_payload(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        payload_dict = cast(dict[str, object], payload)
        raw_apps = payload_dict.get("apps") or []
        if isinstance(raw_apps, list):
            return [item for item in raw_apps if isinstance(item, dict)]
    raise RuntimeError("Databricks apps listing returned invalid JSON.")


def list_databricks_apps(workspace: str, profile: str | None = None) -> list[dict]:
    env = build_databricks_cli_env(workspace)
    try:
        result = run(
            [
                "databricks",
                "apps",
                "list",
                *_profile_args(profile),
                "--limit",
                "1000",
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        return _extract_apps_payload(json.loads(result.stdout or "[]"))
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Failed to list Databricks apps via `databricks apps list`.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timed out while listing Databricks apps.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Databricks apps listing returned invalid JSON.") from exc


def _ucode_binary() -> str:
    """Resolve the absolute path to the running `ucode` executable.

    Agents persist the auth command into config files and re-run it on every
    token refresh, possibly from launchers without a full PATH (desktop GUIs).
    An absolute path keeps the helper working regardless of PATH. Falls back to
    the bare name when resolution fails."""
    return shutil.which("ucode") or "ucode"


def build_auth_token_argv(
    workspace: str, profile: str | None = None, *, use_pat: bool = False
) -> list[str]:
    """Argv for the cross-platform token helper: `ucode auth-token ...`.

    Unlike the previous POSIX `databricks ... | jq` pipeline, this is a single
    executable with plain arguments — no `sh`, no `jq`, no shell quoting — so it
    runs identically on macOS, Linux, and Windows (issue #116). The DATABRICKS_BEARER
    short-circuit and the PAT path both live inside `auth-token` itself."""
    argv = [_ucode_binary(), "auth-token", "--host", workspace.rstrip("/")]
    if profile:
        argv += ["--profile", profile]
    if use_pat:
        argv.append("--use-pat")
    return argv


def build_auth_shell_command(
    workspace: str, profile: str | None = None, *, use_pat: bool = False
) -> str:
    """Single-line, shell-quoted form of :func:`build_auth_token_argv`.

    Used where a tool wants the helper as one command *string* (Claude Code's
    `apiKeyHelper`). On every platform this resolves to the `ucode auth-token`
    executable rather than a POSIX shell pipeline, so no `sh`/`jq` is required."""
    argv = build_auth_token_argv(workspace, profile, use_pat=use_pat)
    if platform.system() == "Windows":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


# A model-service's `name` is `model-services/system.ai.<model-name>`; the
# part after the prefix is exactly the model string agents send (no
# `databricks-` infix — that only appears on the inner destination name).
_MODEL_SERVICE_NAME_PREFIX = "model-services/"
# The metastore-scope listing returns services from EVERY schema (e.g.
# `main.user.foo`, `temp.*`, internal DLT schemas). We only want the
# Databricks-managed foundation models under `system.ai`.
_MODEL_SERVICE_REQUIRED_PREFIX = "system.ai."

# Supported OSS chat families, matched by name substring. Add an entry to
# support a new family.
_OSS_MODEL_FAMILIES = ("kimi-", "glm-")

# Claude model families ucode buckets, newest tier first. Each maps to a
# Claude Code family alias (ANTHROPIC_DEFAULT_<FAMILY>_MODEL). Add an entry to
# support a new family in both discovery paths (`claude-<family>-*` via the
# model-services listing and `databricks-claude-<family>-*` via the AI Gateway).
ANTHROPIC_FAMILIES = ("fable", "opus", "sonnet", "haiku")

# Per-family token limits (context window + max output tokens). These are a
# property of the model + its `/ai-gateway/mlflow/v1` route (the gateway rejects
# requests whose output exceeds the cap), not of any one agent — so every agent
# that serves OSS models reads this single table and translates it into its own
# config dialect. Both fields are provided because agents like OpenCode require
# context and output together. Keyed by family substring; add an entry to bound
# a new model.
_MODEL_TOKEN_LIMITS: dict[str, dict[str, int]] = {
    # GLM-4.6: 200k context, but the gateway caps output well below the model's
    # native 128k — pin 25k so requests aren't rejected.
    "glm": {"context": 200_000, "output": 25_000},
}


def model_token_limits(model_id: str) -> dict[str, int] | None:
    """Return ``{"context": ..., "output": ...}`` limits for ``model_id``, or None.

    Matches by family substring (e.g. any ``*glm*`` id). None means the model
    has no known limits and the agent should not pin any."""
    for family, limits in _MODEL_TOKEN_LIMITS.items():
        if family in model_id:
            return dict(limits)
    return None


def _model_service_id(service: dict) -> str | None:
    """Extract the `system.ai.<model-name>` id from one model-service entry.

    Returns None for services in any other schema, so user/internal model
    services don't leak into the family buckets."""
    name = service.get("name")
    if not isinstance(name, str):
        return None
    name = name.strip()
    if name.startswith(_MODEL_SERVICE_NAME_PREFIX):
        name = name[len(_MODEL_SERVICE_NAME_PREFIX) :]
    if not name.startswith(_MODEL_SERVICE_REQUIRED_PREFIX):
        return None
    return name or None


# The model-services metastore listing REQUIRES a bounded `page_size`:
# unparameterized or large-page requests (verified against
# eng-ml-agent-platform.staging 2026-06-14) return `HTTP 499` with an empty
# body, while pages of 10–100 come back reliably. A page can still 499
# intermittently under load, so each gets a few retries before we give up.
_MODEL_SERVICES_PAGE_SIZE = 100
_MODEL_SERVICES_PAGE_RETRIES = 4


def _get_model_services_page(
    url: str, token: str, *, retries: int = _MODEL_SERVICES_PAGE_RETRIES
) -> tuple[dict | list | None, str | None]:
    """GET one model-services page, retrying on failure.

    The endpoint intermittently 499/504s under load; a retry usually succeeds.
    Returns the same (payload, reason) shape as ``_http_get_json`` — the last
    attempt's result when all retries are exhausted."""
    payload: dict | list | None = None
    reason: str | None = None
    for attempt in range(retries):
        payload, reason = _http_get_json(url, token, timeout=30)
        if payload is not None:
            return payload, None
        _debug("model-services page", f"attempt {attempt + 1}/{retries} failed: {reason}")
    return payload, reason


def list_model_services(
    workspace: str,
    token: str,
    *,
    page_size: int = _MODEL_SERVICES_PAGE_SIZE,
    max_pages: int = 100,
) -> tuple[list[str], str | None]:
    """List all `system.ai.*` model ids via the UC model-services API.

    Pages through ``/api/2.1/unity-catalog/model-services`` (metastore scope)
    with a bounded ``page_size`` (the endpoint 499s without one) and returns the
    de-duplicated, sorted list of ``system.ai.<model-name>`` ids. Returns
    (ids, reason); reason is None on success, otherwise it describes why the
    list is empty (HTTP/network error or no services).
    """
    hostname = workspace_hostname(workspace)
    ids: list[str] = []
    page_token: str | None = None
    seen_tokens: set[str] = set()
    last_reason: str | None = None
    for _ in range(max_pages):
        params: dict[str, str] = {"page_size": str(page_size)}
        if page_token:
            params["page_token"] = page_token
        url = f"https://{hostname}/api/2.1/unity-catalog/model-services?{urlencode(params)}"
        payload, reason = _get_model_services_page(url, token)
        if payload is None:
            # Surface the failure only if we have nothing yet; a mid-pagination
            # blip still returns whatever we collected.
            last_reason = reason
            break
        data = cast(dict, payload) if isinstance(payload, dict) else {}
        for service in data.get("model_services", []):
            if isinstance(service, dict):
                model_id = _model_service_id(service)
                if model_id:
                    ids.append(model_id)
        page_token = data.get("next_page_token") or None
        if not page_token:
            last_reason = None
            break
        if page_token in seen_tokens:
            break
        seen_tokens.add(page_token)

    deduped = sorted(set(ids))
    if deduped:
        return deduped, None
    return [], last_reason or "model-services listing returned no models"


def discover_model_services(
    workspace: str, token: str
) -> tuple[dict[str, str], list[str], list[str], list[str], str | None]:
    """Discover models via UC model-services and bucket them by family name.

    Returns (claude_models, codex_models, gemini_models, oss_models, reason):

    - ``claude_models`` maps ``fable``/``opus``/``sonnet``/``haiku`` to the
      newest matching ``system.ai.claude-*`` id (mirrors
      ``discover_claude_models``).
    - ``codex_models`` is the list of ``system.ai.*gpt-*`` ids.
    - ``gemini_models`` is the list of ``system.ai.*gemini-*`` ids, newest first.
    - ``oss_models`` is the list of OSS-model ``system.ai.*`` ids.

    ``reason`` is None on success, else explains why nothing was found. Family
    bucketing is by name substring because the model-services API does not
    expose per-model API dialects.
    """
    ids, reason = list_model_services(workspace, token)
    if not ids:
        return {}, [], [], [], reason

    claude_models: dict[str, str] = {}
    for family in ANTHROPIC_FAMILIES:
        candidates = sorted(
            [m for m in ids if f"claude-{family}-" in m],
            reverse=True,
        )
        if candidates:
            claude_models[family] = candidates[0]

    codex_models = [m for m in ids if "gpt-" in m]
    gemini_models = sorted([m for m in ids if "gemini-" in m], key=model_version_sort_key)

    oss_models = [m for m in ids if any(family in m for family in _OSS_MODEL_FAMILIES)]

    if not (claude_models or codex_models or gemini_models or oss_models):
        sample = ", ".join(ids[:5])
        return (
            {},
            [],
            [],
            [],
            (
                "model-services returned model ids but none matched "
                f"claude/gpt/gemini/oss families (got: {sample})"
            ),
        )
    return claude_models, codex_models, gemini_models, oss_models, None


# --- MCP services (parallel to model services) -----------------------------


_MCP_SERVICE_NAME_PREFIX = "mcp-services/"


def _mcp_service_full_name(service: dict, required_prefix: str) -> str | None:
    """Extract the full UC name from one mcp-service entry, or None if it
    doesn't live under ``required_prefix`` or isn't ACTIVE."""
    name = service.get("name")
    if not isinstance(name, str):
        return None
    name = name.strip().removeprefix(_MCP_SERVICE_NAME_PREFIX)
    if not name.startswith(required_prefix):
        return None
    status = ((service.get("config") or {}).get("connection") or {}).get("status")
    if status is not None and status != "ACTIVE":
        return None
    return name


def list_mcp_services(
    workspace: str, token: str, parent: str = "system.ai"
) -> tuple[list[str], str | None]:
    """List UC MCP services under ``parent`` (a ``<catalog>.<schema>`` ref).

    A non-None string indicates the listing call itself failed. Callers can inspect
    ``error`` for ``HTTP 404`` to distinguish "invalid location" from other failures.
    """
    hostname = workspace_hostname(workspace)
    url = (
        f"https://{hostname}/api/2.1/unity-catalog/mcp-services"
        f"?{urlencode({'parent': f'schemas/{parent}'})}"
    )
    payload, reason = _http_get_json(url, token, timeout=30)
    if payload is None:
        return [], reason
    expected_prefix = parent + "."
    data = cast(dict, payload) if isinstance(payload, dict) else {}
    names: list[str] = []
    for service in data.get("mcp_services") or []:
        if not isinstance(service, dict):
            continue
        full_name = _mcp_service_full_name(service, expected_prefix)
        if full_name:
            names.append(full_name)
    return sorted(set(names)), None


def build_mcp_service_url(workspace: str, full_name: str) -> str:
    return f"{workspace}/ai-gateway/mcp-services/{full_name}"


def build_skills_mcp_url(workspace: str, locations: list[str]) -> str:
    """Skills route with one ``?schema=`` scope per location. The trailing slash
    is required by the Envoy prefix even with no query params.

        []                        -> ``.../ai-gateway/skills/``
        ["main.default", "ml.a"]  -> ``.../ai-gateway/skills/?schema=main.default&schema=ml.a``
    """
    base = f"{workspace}/ai-gateway/skills/"
    if not locations:
        return base
    return base + "?" + urlencode([("schema", loc) for loc in locations])


# Maps the gateway routing dialect a coding tool speaks to the Model Provider
# Service `provider_type`s it can be backed by. claude speaks Anthropic's API,
# which both the `anthropic` and `amazon_bedrock` provider types serve (Bedrock
# just exposes different model ids); codex speaks OpenAI's. Tags are the short
# form produced by `_provider_type_tag` (e.g. `amazon_bedrock`).
_TOOL_PROVIDER_TYPES: dict[str, tuple[str, ...]] = {
    "claude": ("anthropic", "amazon_bedrock"),
    "codex": ("openai",),
}

# Provider types that expose Bedrock-style model ids (e.g.
# `us.anthropic.claude-sonnet-4-6`) instead of the agent's canonical model
# names, so ucode must pin them explicitly.
BEDROCK_PROVIDER_TYPES: tuple[str, ...] = ("amazon_bedrock",)


def tool_supports_provider_type(tool: str, provider_type: str) -> bool:
    """True when ``tool``'s API dialect can be backed by ``provider_type``."""
    return provider_type in _TOOL_PROVIDER_TYPES.get(tool, ())


def _provider_type_tag(provider_type: str | None) -> str:
    """Shorten `EXTERNAL_MODEL_PROVIDER_TYPE_ANTHROPIC` to `anthropic`."""
    if not isinstance(provider_type, str):
        return ""
    prefix = "EXTERNAL_MODEL_PROVIDER_TYPE_"
    tag = provider_type[len(prefix) :] if provider_type.startswith(prefix) else provider_type
    return tag.lower()


def list_model_provider_services(workspace: str, token: str) -> tuple[list[dict], str | None]:
    """List Unity Catalog Model Provider Services on the workspace.

    Returns ``(services, reason)`` where each service is
    ``{"name": "<catalog>.<schema>.<service>", "provider_type": "anthropic"|...,
    "targets": [model_id, ...], "allow_all_targets": bool}``. ``targets`` is the
    provider-side model ids the service exposes (used to pin Bedrock model
    names). A non-None ``reason`` means the listing call itself failed.
    """
    hostname = workspace_hostname(workspace)
    url = f"https://{hostname}/api/2.1/unity-catalog/model-provider-services"
    payload, reason = _http_get_json(url, token, timeout=30)
    if payload is None:
        return [], reason
    data = cast(dict, payload) if isinstance(payload, dict) else {}
    services: list[dict] = []
    for service in data.get("model_provider_services") or []:
        if not isinstance(service, dict):
            continue
        raw_name = service.get("name")
        if not isinstance(raw_name, str) or not raw_name:
            continue
        # The API returns `model-provider-services/<catalog>.<schema>.<name>`.
        full_name = raw_name.split("/", 1)[1] if "/" in raw_name else raw_name
        config = service.get("config") if isinstance(service.get("config"), dict) else {}
        targets = []
        for target in config.get("targets") or []:
            model_id = target.get("model") if isinstance(target, dict) else None
            if isinstance(model_id, str) and model_id:
                targets.append(model_id)
        services.append(
            {
                "name": full_name,
                "provider_type": _provider_type_tag(config.get("provider_type")),
                "targets": targets,
                "allow_all_targets": bool(config.get("allow_all_targets")),
            }
        )
    services.sort(key=lambda s: s["name"])
    return services, None


def is_model_provider_feature_unavailable(reason: str | None) -> bool:
    """True when a model-provider-services API failure means the workspace
    simply hasn't enabled the feature (HTTP 400 "feature is not available"),
    as opposed to a transient or auth error. Callers use this to fall back to
    Databricks models silently rather than surfacing a scary error.
    """
    return bool(reason) and "feature is not available" in reason.lower()


def list_tool_provider_services(
    tool: str, workspace: str, token: str
) -> tuple[list[str], str | None]:
    """Provider-service names whose provider type matches ``tool``'s API dialect.

    Returns ``(names, reason)``; ``reason`` is non-None when the listing failed.
    """
    services, reason = list_model_provider_services(workspace, token)
    if reason is not None:
        return [], reason
    names = [s["name"] for s in services if service_usable_for_tool(tool, s)]
    return names, None


def service_usable_for_tool(tool: str, service: dict) -> bool:
    """True when ``tool`` can actually route through ``service``.

    Beyond the provider-type match, a Bedrock service is only usable for claude
    if it exposes at least one Claude model in its targets — otherwise there's no
    routable model id to pin. (Anthropic services use canonical names, so any
    match is usable.)
    """
    provider_type = service.get("provider_type", "")
    if not tool_supports_provider_type(tool, provider_type):
        return False
    if provider_type in BEDROCK_PROVIDER_TYPES:
        return bool(map_bedrock_claude_models(service.get("targets") or []))
    return True


def resolve_provider_service(
    tool: str, service_name: str, workspace: str, token: str
) -> tuple[dict | None, str | None]:
    """Validate that ``service_name`` exists and is usable by ``tool``.

    Returns ``(service, error)``. On success ``service`` is the full service
    dict (``name``/``provider_type``/``targets``/``allow_all_targets``) and
    ``error`` is None. On failure ``service`` is None and ``error`` is an
    actionable message: the feature is off, the listing failed, the service
    doesn't exist, or its provider type isn't one ``tool`` can route to (e.g.
    pointing claude at an OpenAI service).
    """
    services, reason = list_model_provider_services(workspace, token)
    if is_model_provider_feature_unavailable(reason):
        return None, "Model Provider Service feature is not available yet for this workspace."
    if reason is not None:
        return None, f"Could not list model provider services: {reason}"
    match = next((s for s in services if s["name"] == service_name), None)
    if match is None:
        usable = [
            s["name"] for s in services if tool_supports_provider_type(tool, s["provider_type"])
        ]
        suffix = f" Available for {tool}: {', '.join(usable)}." if usable else ""
        return None, f"Model provider service '{service_name}' was not found.{suffix}"
    provider_type = match["provider_type"]
    if not tool_supports_provider_type(tool, provider_type):
        supported = ", ".join(_TOOL_PROVIDER_TYPES.get(tool, ())) or "none"
        return None, (
            f"Model provider service '{service_name}' is a '{provider_type}' provider, "
            f"which {tool} can't route to (supported: {supported})."
        )
    if provider_type in BEDROCK_PROVIDER_TYPES and not map_bedrock_claude_models(
        match.get("targets") or []
    ):
        return None, (
            f"Model provider service '{service_name}' exposes no Claude models — "
            f"add Claude targets to it or pick a different service."
        )
    return match, None


# Bedrock exposes Claude under provider-side ids like
# `us.anthropic.claude-sonnet-4-6`, `global.anthropic.claude-opus-4-8`, or the
# region-less `anthropic.claude-opus-4-8`. We map each service target to a
# Claude family and keep the best id per family. Claude Code only takes one
# default per family; users switch to any other listed region profile at runtime
# with `/model <full-id>` or `--model`.
_BEDROCK_CLAUDE_FAMILIES = ("opus", "sonnet", "haiku")
# When the same model/version is offered under several cross-region inference
# profiles, prefer the broadest-routing one as the pinned default.
_BEDROCK_REGION_RANK = {"global": 5, "us": 4, "eu": 3, "apac": 2, "": 1}


def _bedrock_target_family(model_id: str) -> str | None:
    lowered = model_id.lower()
    if "claude" not in lowered:
        return None
    return next((fam for fam in _BEDROCK_CLAUDE_FAMILIES if fam in lowered), None)


def _bedrock_region_rank(model_id: str) -> int:
    """Rank a target's cross-region inference profile (`us.`/`eu.`/`global.`/
    region-less) so ties on model version resolve deterministically."""
    head = model_id.lower().split("anthropic.", 1)[0].rstrip(".")
    return _BEDROCK_REGION_RANK.get(head, 0)


def _bedrock_sort_key(model_id: str) -> tuple:
    """Order targets best-first: highest model version, then preferred region."""
    version = tuple(int(n) for n in re.findall(r"\d+", model_id))
    return (version, _bedrock_region_rank(model_id))


def map_bedrock_claude_models(targets: list[str]) -> dict[str, str]:
    """Map Bedrock service targets to ``{family: model_id}`` for opus/sonnet/
    haiku, choosing the highest-versioned id per family and, on a version tie,
    the broadest-routing region profile. Targets that don't name a Claude family
    are ignored."""
    best_key: dict[str, tuple] = {}
    result: dict[str, str] = {}
    for model_id in targets:
        family = _bedrock_target_family(model_id)
        if not family:
            continue
        key = _bedrock_sort_key(model_id)
        if family not in best_key or key > best_key[family]:
            best_key[family] = key
            result[family] = model_id
    return result


# `list_vector_search_catalog_schemas` walks Vector Search endpoints+indexes.
# `list_uc_functions_catalog_schemas` walks UC catalogs+schemas in parallel and
# keeps only schemas with at least one user function.

_UC_LIST_PAGE_SIZE = 200
_UC_LIST_MAX_PAGES = 50
_UC_FUNCTION_PROBE_WORKERS = 16
_UC_LIST_HTTP_TIMEOUT = 10
_UC_FUNCTION_PROBE_TIMEOUT = 5
_VECTOR_SEARCH_DEADLINE_SECONDS = 15.0
_UC_FUNCTIONS_DEADLINE_SECONDS = 20.0
# Skip UC catalogs whose schemas almost never carry user-callable functions
# you'd want to expose as agent tools.
_UC_FUNCTIONS_SKIP_CATALOGS = frozenset(
    {"__databricks_internal", "hive_metastore", "samples", "system"}
)


def _drain_with_deadline(futures: dict, deadline: float, on_result) -> None:
    """Iterate `futures` via `as_completed`, calling `on_result(value, key)` per
    completed future, until either all are done or `deadline` passes. Per-task
    exceptions are swallowed so one failure doesn't stop the rest."""
    remaining = max(0.0, deadline - time.monotonic())
    try:
        for future in as_completed(futures, timeout=remaining):
            try:
                value = future.result()
            except Exception:  # noqa: BLE001
                continue
            on_result(value, futures[future])
            if time.monotonic() > deadline:
                break
    except FutureTimeoutError:
        pass


def _paginated_json_items(
    base_url: str,
    token: str,
    *,
    items_key: str,
    extra_params: dict[str, str] | None = None,
    page_size: int = _UC_LIST_PAGE_SIZE,
    max_pages: int = _UC_LIST_MAX_PAGES,
    timeout: int = 30,
) -> tuple[list[dict], str | None]:
    """Walk a Databricks `next_page_token` listing and return all items.

    Returns (items, reason). Items are dicts; reason is None on success or a
    short description of why the walk stopped early.
    """
    items: list[dict] = []
    page_token: str | None = None
    seen_tokens: set[str] = set()
    last_reason: str | None = None
    for _ in range(max_pages):
        params: dict[str, str] = {"max_results": str(page_size)}
        if extra_params:
            params.update(extra_params)
        if page_token:
            params["page_token"] = page_token
        url = f"{base_url}?{urlencode(params)}"
        payload, reason = _http_get_json(url, token, timeout=timeout)
        if payload is None:
            last_reason = reason
            break
        data = cast(dict, payload) if isinstance(payload, dict) else {}
        raw = data.get(items_key) or []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    items.append(item)
        page_token = data.get("next_page_token") or None
        if not page_token or page_token in seen_tokens:
            break
        seen_tokens.add(page_token)
    return items, last_reason


def _vector_index_catalog_schema(index: dict) -> tuple[str, str] | None:
    """Pull (catalog, schema) from one vector-search index entry."""
    catalog = index.get("catalog_name")
    schema = index.get("schema_name")
    if isinstance(catalog, str) and isinstance(schema, str) and catalog and schema:
        return catalog, schema
    # Fallback: `name` is the fully-qualified UC name `catalog.schema.index`.
    name = index.get("name")
    if isinstance(name, str):
        parts = name.split(".")
        if len(parts) >= 3 and parts[0] and parts[1]:
            return parts[0], parts[1]
    return None


def list_vector_search_catalog_schemas(
    workspace: str,
    token: str,
    *,
    deadline_seconds: float = _VECTOR_SEARCH_DEADLINE_SECONDS,
) -> tuple[list[tuple[str, str]], str | None]:
    """Return sorted unique `(catalog, schema)` pairs that contain at least
    one Databricks Vector Search index. Walks the per-endpoint index listings
    in parallel under a wall-clock budget; returns partial results once
    `deadline_seconds` is exceeded."""
    hostname = workspace_hostname(workspace)
    deadline = time.monotonic() + deadline_seconds
    endpoints, reason = _paginated_json_items(
        f"https://{hostname}/api/2.0/vector-search/endpoints",
        token,
        items_key="endpoints",
        timeout=_UC_LIST_HTTP_TIMEOUT,
    )
    if not endpoints:
        return [], reason or "no vector search endpoints found"

    endpoint_names = [e["name"] for e in endpoints if isinstance(e.get("name"), str) and e["name"]]
    if not endpoint_names:
        return [], "no vector search endpoints with names"

    pairs: set[tuple[str, str]] = set()
    workers = max(1, min(_UC_FUNCTION_PROBE_WORKERS, len(endpoint_names)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _paginated_json_items,
                f"https://{hostname}/api/2.0/vector-search/indexes",
                token,
                items_key="vector_indexes",
                extra_params={"endpoint_name": name},
                timeout=_UC_LIST_HTTP_TIMEOUT,
            ): name
            for name in endpoint_names
        }

        def collect(result, _endpoint):
            indexes, _ = result
            for index in indexes:
                pair = _vector_index_catalog_schema(index)
                if pair:
                    pairs.add(pair)

        _drain_with_deadline(futures, deadline, collect)
        pool.shutdown(wait=False, cancel_futures=True)

    if not pairs:
        return [], "no vector search indexes found"
    return sorted(pairs), None


def _schema_has_user_function(hostname: str, token: str, catalog: str, schema: str) -> bool:
    """One-shot probe: does `{catalog}.{schema}` expose any UC function?"""
    url = (
        f"https://{hostname}/api/2.1/unity-catalog/functions"
        f"?{urlencode({'catalog_name': catalog, 'schema_name': schema, 'max_results': '1'})}"
    )
    payload, _reason = _http_get_json(url, token, timeout=_UC_FUNCTION_PROBE_TIMEOUT)
    if not isinstance(payload, dict):
        return False
    functions = payload.get("functions") or []
    return isinstance(functions, list) and any(isinstance(item, dict) for item in functions)


def list_uc_functions_catalog_schemas(
    workspace: str,
    token: str,
    *,
    deadline_seconds: float = _UC_FUNCTIONS_DEADLINE_SECONDS,
) -> tuple[list[tuple[str, str]], str | None]:
    """Return sorted unique `(catalog, schema)` pairs containing at least one
    user-defined UC function."""
    hostname = workspace_hostname(workspace)
    deadline = time.monotonic() + deadline_seconds

    catalogs, catalogs_reason = _paginated_json_items(
        f"https://{hostname}/api/2.1/unity-catalog/catalogs",
        token,
        items_key="catalogs",
        timeout=_UC_LIST_HTTP_TIMEOUT,
    )
    if not catalogs:
        return [], catalogs_reason or "no UC catalogs found"

    catalog_names = [
        c["name"]
        for c in catalogs
        if isinstance(c.get("name"), str)
        and c["name"]
        and c["name"] not in _UC_FUNCTIONS_SKIP_CATALOGS
    ]
    if not catalog_names:
        return [], "no user UC catalogs found"
    if time.monotonic() > deadline:
        return [], "deadline exceeded while listing UC catalogs"

    # Parallel per-catalog schema listing.
    candidate_pairs: list[tuple[str, str]] = []
    schema_workers = max(1, min(_UC_FUNCTION_PROBE_WORKERS, len(catalog_names)))
    with ThreadPoolExecutor(max_workers=schema_workers) as pool:
        schema_futures = {
            pool.submit(
                _paginated_json_items,
                f"https://{hostname}/api/2.1/unity-catalog/schemas",
                token,
                items_key="schemas",
                extra_params={"catalog_name": cat},
                timeout=_UC_LIST_HTTP_TIMEOUT,
            ): cat
            for cat in catalog_names
        }

        def collect_schemas(result, catalog):
            schemas, _ = result
            for schema in schemas:
                schema_name = schema.get("name")
                # `information_schema` is auto-attached to every catalog and
                # never holds user functions.
                if (
                    isinstance(schema_name, str)
                    and schema_name
                    and schema_name != "information_schema"
                ):
                    candidate_pairs.append((catalog, schema_name))

        _drain_with_deadline(schema_futures, deadline, collect_schemas)
        pool.shutdown(wait=False, cancel_futures=True)

    if not candidate_pairs:
        if time.monotonic() > deadline:
            return [], "deadline exceeded while listing UC schemas"
        return [], "no UC schemas found"

    # Parallel function-existence probes.
    pairs: set[tuple[str, str]] = set()
    with ThreadPoolExecutor(max_workers=_UC_FUNCTION_PROBE_WORKERS) as pool:
        probe_futures = {
            pool.submit(_schema_has_user_function, hostname, token, cat, schema): (cat, schema)
            for cat, schema in candidate_pairs
        }

        def collect_pair(has_fn, pair):
            if has_fn:
                pairs.add(pair)

        _drain_with_deadline(probe_futures, deadline, collect_pair)
        pool.shutdown(wait=False, cancel_futures=True)

    if not pairs:
        if time.monotonic() > deadline:
            return [], "deadline exceeded probing UC schemas for functions"
        return [], "no UC schemas with user functions found"
    return sorted(pairs), None


def discover_claude_models(workspace: str, token: str) -> tuple[dict[str, str], str | None]:
    """Discover Claude families on this workspace's AI Gateway.

    Returns (models_by_family, reason). reason is None on success; otherwise it
    describes why the dict is empty (HTTP error, network error, or no models
    matching the expected naming convention).
    """
    hostname = workspace_hostname(workspace)
    payload, reason = _http_get_json(f"https://{hostname}/ai-gateway/anthropic/v1/models", token)
    if payload is None:
        return {}, reason

    data = cast(dict, payload) if isinstance(payload, dict) else {}
    raw_ids = [
        m["id"]
        for m in data.get("data", [])
        if isinstance(m.get("id"), str) and not m["id"].endswith("-anthropic")
    ]

    result: dict[str, str] = {}
    for family in ANTHROPIC_FAMILIES:
        candidates = sorted(
            [m for m in raw_ids if f"databricks-claude-{family}-" in m],
            reverse=True,
        )
        if candidates:
            result[family] = candidates[0]
    if result:
        return result, None
    if not raw_ids:
        return {}, "AI Gateway returned no Claude model ids"
    sample = ", ".join(raw_ids[:5])
    families = ",".join(ANTHROPIC_FAMILIES)
    return {}, (
        "AI Gateway returned model ids but none matched "
        f"`databricks-claude-{{{families}}}-*` (got: {sample})"
    )


def fetch_ai_gateway_claude_models(workspace: str, token: str) -> dict[str, str]:
    """Backwards-compatible wrapper that discards the diagnostic reason."""
    models, _ = discover_claude_models(workspace, token)
    return models


def model_version_sort_key(name: str) -> tuple:
    """Sort endpoint names so newer model versions come first.

    Endpoint names embed a dotted version as dash-separated digits, e.g.
    `databricks-gemini-3-5-flash` (3.5) or `databricks-gemini-3-flash` (3.0).
    Plain alphabetical sorting buries `3-5-flash` below `2-5-flash`; this key
    groups by the non-numeric prefix, orders by version descending, then falls
    back to the remaining text so ties stay stable and deterministic.
    """
    tokens = name.split("-")
    start = next((i for i, tok in enumerate(tokens) if tok.isdigit()), None)
    if start is None:
        # No version segment — sort these after versioned ones, alphabetically.
        # The leading 1 keeps the whole group below every versioned name (0).
        return (1, name, (), "")
    end = start
    while end < len(tokens) and tokens[end].isdigit():
        end += 1
    version = tuple(int(tok) for tok in tokens[start:end])
    # Pad to a fixed width so (3,) compares as (3, 0) — i.e. 3.0 < 3.5.
    padded = (version + (0, 0, 0))[:3]
    prefix = "-".join(tokens[:start])
    suffix = "-".join(tokens[end:])
    # Negate version components for descending order within a prefix group.
    return (0, prefix, tuple(-v for v in padded), suffix)


def discover_endpoints_with_api_type(
    workspace: str,
    token: str,
    api_type: str,
    *,
    sort_key=None,
) -> tuple[list[str], str | None]:
    """List endpoint names whose served_entities expose api_type with v2 support.

    Returns (endpoints, reason). reason is None on success; otherwise it
    describes why the list is empty. `sort_key` overrides the default
    alphabetical ordering of the returned names.
    """
    hostname = workspace_hostname(workspace)
    payload, reason = _http_get_json(
        f"https://{hostname}/api/2.0/serving-endpoints:foundation-models", token
    )
    if payload is None:
        return [], reason

    data = cast(dict, payload) if isinstance(payload, dict) else {}
    endpoints = data.get("endpoints", [])
    out: list[str] = []
    saw_endpoint_without_v2 = False
    for ep in endpoints:
        name = ep.get("name", "")
        entities = ep.get("config", {}).get("served_entities", [])
        api_types: set[str] = set()
        any_v2 = False
        for se in entities:
            fm = se.get("foundation_model", {})
            if fm.get("ai_gateway_v2_supported") is True:
                any_v2 = True
                api_types.update(fm.get("api_types", []))
        if not any_v2 and entities:
            saw_endpoint_without_v2 = True
        if api_type in api_types:
            out.append(name)
    if out:
        return sorted(out, key=sort_key), None
    if not endpoints:
        return [], "foundation-models listing returned no endpoints"
    if saw_endpoint_without_v2:
        return [], (
            f"no endpoint exposes api_type `{api_type}` with "
            "`ai_gateway_v2_supported=true` (workspace has v1-only endpoints)"
        )
    return [], f"no endpoint exposes api_type `{api_type}`"


def _fetch_endpoints_with_api_type(workspace: str, token: str, api_type: str) -> list[str]:
    """Backwards-compatible wrapper that discards the diagnostic reason."""
    endpoints, _ = discover_endpoints_with_api_type(workspace, token, api_type)
    return endpoints


def discover_gemini_models(workspace: str, token: str) -> tuple[list[str], str | None]:
    # Order newest model version first so `default_model()` (which picks the
    # first entry) launches e.g. gemini-3.5-flash rather than gemini-2.5-flash.
    return discover_endpoints_with_api_type(
        workspace, token, "gemini/v1/generateContent", sort_key=model_version_sort_key
    )


def discover_codex_models(workspace: str, token: str) -> tuple[list[str], str | None]:
    return discover_endpoints_with_api_type(workspace, token, "openai/v1/responses")


def fetch_gemini_models(workspace: str, token: str) -> list[str]:
    models, _ = discover_gemini_models(workspace, token)
    return models


def fetch_codex_models(workspace: str, token: str) -> list[str]:
    models, _ = discover_codex_models(workspace, token)
    return models


def ensure_ai_gateway_v2(workspace: str, token: str) -> None:
    """Probe AI Gateway v2 and raise if unavailable.

    Uses the dedicated v2 listing endpoint `GET /api/ai-gateway/v2/endpoints`:
    a 200 response (even with an empty list) means v2 is wired up on this
    workspace — a "no endpoints provisioned" case will surface naturally in
    downstream discovery. Failure branches:

    - 401 / 403 / 400 with `Invalid Token`: the token is bad for *this*
      workspace.
    - 404: AI Gateway V2 is not enabled on this workspace — point at the docs.
    - other (5xx, network errors): surface the reason verbatim.
    """
    hostname = workspace_hostname(workspace)
    url = f"https://{hostname}/api/ai-gateway/v2/endpoints?page_size=1"
    payload, reason = _http_get_json(url, token)
    if payload is not None:
        return
    reason_str = reason or "unknown error"
    if _looks_like_auth_failure(reason_str):
        raise RuntimeError(
            f"Databricks rejected the access token for {workspace} ({reason_str}). "
            f"Try:\n"
            f"  databricks auth logout --host {workspace}\n"
            f"  databricks auth login --host {workspace}"
        )
    if "HTTP 404" in reason_str:
        raise RuntimeError(
            "Databricks Unity AI Gateway is not enabled on this workspace "
            f"({reason_str}). See {AI_GATEWAY_V2_DOCS_URL}"
        )
    raise RuntimeError(
        "Databricks Unity AI Gateway probe failed on this workspace "
        f"({reason_str}). See {AI_GATEWAY_V2_DOCS_URL}"
    )


def _looks_like_auth_failure(reason: str) -> bool:
    """True when the gateway response signals the token is not accepted.

    Covers 401/403 directly and the gateway's 400 + `Invalid Token` body
    (which happens when the bearer is valid but issued for a different
    workspace)."""
    if "HTTP 401" in reason or "HTTP 403" in reason:
        return True
    if "HTTP 400" in reason and "invalid token" in reason.lower():
        return True
    return False


def discover_sql_warehouse_http_path(
    workspace: str,
    token: str,
    *,
    quiet: bool = False,
) -> str:
    hostname = workspace_hostname(workspace)
    request = urllib_request.Request(
        f"https://{hostname}/api/2.0/sql/warehouses",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )

    try:
        with urllib_request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        detail = body.strip() or f"HTTP {exc.code}"
        raise RuntimeError(f"Failed to list SQL warehouses: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Could not reach workspace hostname {hostname}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Databricks warehouse discovery returned invalid JSON.") from exc

    warehouses = payload.get("warehouses")
    if not isinstance(warehouses, list) or not warehouses:
        raise RuntimeError(
            "No SQL warehouses found in this workspace. Create one or pass `--http-path`."
        )

    running = [w for w in warehouses if isinstance(w, dict) and w.get("state") == "RUNNING"]
    chosen = (
        running[0]
        if running
        else next(
            (w for w in warehouses if isinstance(w, dict) and w.get("id")),
            None,
        )
    )
    if not chosen:
        raise RuntimeError("No usable SQL warehouse was returned by Databricks.")

    warehouse_id = chosen.get("id")
    if not isinstance(warehouse_id, str) or not warehouse_id.strip():
        raise RuntimeError("Databricks returned a warehouse without an ID.")

    warehouse_name = chosen.get("name")
    warehouse_state = chosen.get("state", "UNKNOWN")
    label_value = (
        warehouse_name if isinstance(warehouse_name, str) and warehouse_name else warehouse_id
    )
    if not quiet:
        print_note(f"Using SQL warehouse `{label_value}` ({warehouse_state}).")
    return f"/sql/1.0/warehouses/{warehouse_id}"


def run_usage_query(
    workspace: str,
    http_path: str,
    token: str,
    query: str,
) -> tuple[list[str], list[tuple]]:
    try:
        logging.getLogger("databricks.sql").setLevel(logging.ERROR)
        from databricks import sql
    except ImportError as exc:
        raise RuntimeError(
            "`databricks-sql-connector` is not installed. "
            "Install it with `pip install databricks-sql-connector`."
        ) from exc

    try:
        with sql.connect(
            server_hostname=workspace_hostname(workspace),
            http_path=http_path,
            access_token=token,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                columns = [desc[0] for desc in (cursor.description or [])]
                rows = cast(list[tuple], cursor.fetchall())
    except ServerOperationError as exc:
        if _is_usage_table_access_error(exc):
            raise RuntimeError(
                "Unable to read `system.ai_gateway.usage`. Ask your workspace admin "
                "to enable READ access to `system.ai_gateway.usage` for your account."
            ) from exc
        raise RuntimeError(f"Usage query failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Usage query failed: {exc}") from exc

    return columns, rows


def _is_usage_table_access_error(exc: BaseException) -> bool:
    """Return True when a `ServerOperationError` blocks reads of
    `system.ai_gateway.usage` — gated on one of the bracketed error codes
    `INSUFFICIENT_PERMISSIONS` plus a `system.ai_gateway` substring (identifier quoting
    stripped first)."""
    normalized = str(exc).lower().translate(str.maketrans("", "", """`[]"'"""))
    if "system.ai_gateway" not in normalized:
        return False
    return "insufficient_permissions" in normalized


# ---------------------------------------------------------------------------
# URL builders (AI Gateway v2 only — no fallback to /serving-endpoints)
# ---------------------------------------------------------------------------


def build_tool_base_url(tool: str, workspace: str) -> str:
    if tool == "codex":
        return f"{workspace}/ai-gateway/codex/v1"
    if tool == "claude":
        return f"{workspace}/ai-gateway/anthropic"
    if tool == "gemini":
        return f"{workspace}/ai-gateway/gemini"
    if tool == "opencode":
        raise RuntimeError(
            "OpenCode has multiple base URLs — use build_opencode_base_urls() instead."
        )
    if tool == "copilot":
        raise RuntimeError(
            "Copilot has multiple base URLs — use build_copilot_base_urls() instead."
        )
    if tool == "pi":
        raise RuntimeError("Pi has multiple base URLs — use build_pi_base_urls() instead.")
    raise RuntimeError(f"Unsupported tool '{tool}'.")


def build_opencode_base_urls(workspace: str) -> dict[str, str]:
    return {
        "anthropic": build_tool_base_url("claude", workspace) + "/v1",
        "gemini": build_tool_base_url("gemini", workspace) + "/v1beta",
        "oss": f"{workspace}/ai-gateway/mlflow/v1",
    }


def build_pi_base_urls(workspace: str) -> dict[str, str]:
    # Pi speaks each model family's native API dialect to its dedicated gateway
    # path (verified end-to-end). Each `api` type appends its own path suffix:
    #
    # - anthropic-messages       appends `/v1/messages`
    # - openai-responses         appends `/responses`
    # - google-generative-ai     appends `/v1beta/models/{id}:streamGenerateContent`
    # - openai-completions       appends `/chat/completions`
    #
    # So the baseUrls below stop just before the suffix Pi will tack on.
    # Compat flags applied per-provider in agents/pi.py; required for `oss`
    # only (MLflow rejects `store` and `tools[].function.strict`).
    return {
        "claude": build_tool_base_url("claude", workspace),
        "openai": build_tool_base_url("codex", workspace),
        "gemini": build_tool_base_url("gemini", workspace) + "/v1beta",
    }


def build_copilot_base_url(workspace: str) -> str:
    # Copilot CLI's `openai` provider appends `/chat/completions` to the
    # configured base URL. The Databricks MLflow chat-completions gateway is
    # OpenAI-compatible and serves Claude, codex (gpt-5), and gemini models
    # behind one URL.
    return f"{workspace}/ai-gateway/mlflow/v1"


def build_shared_base_urls(workspace: str) -> dict[str, str | dict[str, str]]:
    urls: dict[str, str | dict[str, str]] = {
        "codex": build_tool_base_url("codex", workspace),
        "claude": build_tool_base_url("claude", workspace),
        "gemini": build_tool_base_url("gemini", workspace),
        "opencode": build_opencode_base_urls(workspace),
        "copilot": build_copilot_base_url(workspace),
        "pi": build_pi_base_urls(workspace),
    }
    return urls
