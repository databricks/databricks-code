"""Cross-repository contract test against a real Hermes checkout."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from ucode.agents import hermes

WORKSPACE = "https://e2e-test.cloud.databricks.com"
MODEL = "system.ai.gpt-5-6"


def _hermes_binary() -> Path:
    configured = os.environ.get("HERMES_TEST_BINARY")
    if configured:
        return Path(configured).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2] / "hermes"
    venv_binary = checkout / ".venv" / "bin" / "hermes"
    return (venv_binary if venv_binary.is_file() else checkout / "hermes").resolve()


def _hermes_runtime(binary: Path) -> tuple[Path, Path]:
    """Return a Python with Hermes deps and the corresponding source root."""
    configured_root = os.environ.get("HERMES_TEST_ROOT")
    candidates = [Path(configured_root).expanduser()] if configured_root else []
    candidates.extend((binary.parent, *binary.parents))
    root = next((path for path in candidates if (path / "hermes_cli").is_dir()), None)
    if root is None:
        pytest.skip("Hermes source root unavailable; set HERMES_TEST_ROOT")
    sibling_python = binary.parent / "python"
    python = sibling_python if sibling_python.is_file() else Path(sys.executable)
    return python, root


def _state() -> dict:
    return {
        "workspace": WORKSPACE,
        "profile": "e2e profile",
        "codex_models": [MODEL],
        "claude_models": {},
        "gemini_models": [],
        "oss_models": [],
    }


def _runtime_resolution(binary: Path, env: dict[str, str]) -> dict:
    python, root = _hermes_runtime(binary)
    script = """
import json
from hermes_cli.config import load_config
from hermes_cli.runtime_provider import resolve_runtime_provider

config = load_config()
provider_id = config["model"]["provider"]
provider = config["providers"][provider_id]
runtime = resolve_runtime_provider(
    requested=provider_id,
    target_model=config["model"]["default"],
)
print(json.dumps({
    "provider_id": provider_id,
    "model": config["model"]["default"],
    "key_cmd": provider["key_cmd"],
    "transport": runtime["api_mode"],
    "base_url": runtime["base_url"],
    "extra_headers": runtime.get("extra_headers"),
    "dynamic_key": callable(runtime["api_key"]),
}))
"""
    completed = subprocess.run(
        [str(python), "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _loaded_config(binary: Path, env: dict[str, str]) -> dict:
    python, root = _hermes_runtime(binary)
    completed = subprocess.run(
        [
            str(python),
            "-c",
            "import json; from hermes_cli.config import load_config; "
            "print(json.dumps(load_config()))",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_real_hermes_apply_runtime_resolution_and_surgical_unconfigure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = _hermes_binary()
    if not binary.is_file():
        pytest.skip("Hermes checkout unavailable; set HERMES_TEST_BINARY to its executable")
    if binary.name != "hermes":
        pytest.skip("HERMES_TEST_BINARY must point to the Hermes CLI executable")

    home = tmp_path / "home"
    hermes_home = tmp_path / "hermes-home"
    home.mkdir()
    hermes_home.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("PATH", f"{binary.parent}{os.pathsep}{os.environ.get('PATH', '')}")

    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        "user_setting: keep\n"
        "providers:\n"
        "  personal-provider:\n"
        "    api: https://personal.example/v1\n"
        "    transport: openai_chat\n"
        "    default_model: personal-model\n"
    )

    configured_state = hermes.write_tool_config(
        {**_state(), "gemini_models": ["system.ai.gemini-3-flash"]},
        model=MODEL,
        hermes_home=hermes_home,
    )
    assert hermes.GEMINI_PROVIDER_ID in _loaded_config(binary, os.environ.copy())["providers"]

    configured_state = hermes.write_tool_config(
        {**_state(), "managed_configs": configured_state["managed_configs"]},
        model=MODEL,
        hermes_home=hermes_home,
    )
    assert hermes.GEMINI_PROVIDER_ID not in _loaded_config(binary, os.environ.copy())["providers"]

    env = os.environ.copy()
    resolved = _runtime_resolution(binary, env)
    assert resolved["provider_id"] == hermes.PROVIDER_ID
    assert resolved["model"] == MODEL
    key_cmd = shlex.split(resolved["key_cmd"])
    assert Path(key_cmd[0]).name == "ucode"
    assert key_cmd[1:] == [
        "auth-token",
        "--host",
        WORKSPACE,
        "--profile",
        "e2e profile",
    ]
    assert resolved["transport"] == "codex_responses"
    assert resolved["base_url"] == f"{WORKSPACE}/ai-gateway/codex/v1"
    assert resolved["extra_headers"] in (None, {})
    assert resolved["dynamic_key"] is True

    hermes.apply_config_patch(
        {
            "set": {
                f"providers.{hermes.PROVIDER_ID}": {
                    "api": "https://user-replacement.example/v1",
                    "transport": "openai_chat",
                    "default_model": "replacement-model",
                    "models": {"replacement-model": {}},
                },
                "model.provider": "personal-provider",
                "model.default": "personal-model",
            },
            "unset": [],
        },
        hermes_home=hermes_home,
    )
    ownership = configured_state["managed_configs"]["hermes"]
    receipt = hermes.unconfigure(
        hermes_home=ownership["hermes_home"],
        owned_model=ownership["active_model"],
        owned_provider_fingerprints=ownership["provider_fingerprints"],
    )
    assert receipt["status"] == "applied"
    remaining = _loaded_config(binary, env)
    assert remaining["user_setting"] == "keep"
    assert remaining["model"] == {
        "provider": "personal-provider",
        "default": "personal-model",
    }
    assert remaining["providers"] == {
        "personal-provider": {
            "api": "https://personal.example/v1",
            "transport": "openai_chat",
            "default_model": "personal-model",
        },
        hermes.PROVIDER_ID: {
            "api": "https://user-replacement.example/v1",
            "transport": "openai_chat",
            "default_model": "replacement-model",
            "models": {"replacement-model": {}},
        },
    }

    proxy_argv = ["ucode", "mcp-proxy", "https://managed.example/mcp"]
    managed_value = hermes.mcp_value_for_argv(proxy_argv)
    fingerprint = hermes.mcp_server_fingerprint(managed_value)
    hermes.apply_config_patch(
        {
            "set": {
                "mcp_servers.user-sibling": {"command": "user", "args": ["--keep"]},
            },
            "unset": [],
        },
        hermes_home=hermes_home,
    )
    hermes.write_mcp_server_config("managed", proxy_argv, hermes_home=hermes_home)
    assert (
        hermes.remove_mcp_server_config(
            "managed",
            hermes_home=hermes_home,
            expected_fingerprint=fingerprint,
        )
        is True
    )
    remaining = _loaded_config(binary, env)
    assert remaining["mcp_servers"] == {"user-sibling": {"command": "user", "args": ["--keep"]}}

    hermes.write_mcp_server_config("managed", proxy_argv, hermes_home=hermes_home)
    replacement = {"command": "user-replacement", "args": ["--keep"]}
    hermes.apply_config_patch(
        {"set": {"mcp_servers.managed": replacement}, "unset": []},
        hermes_home=hermes_home,
    )
    assert (
        hermes.remove_mcp_server_config(
            "managed",
            hermes_home=hermes_home,
            expected_fingerprint=fingerprint,
        )
        is False
    )
    assert _loaded_config(binary, env)["mcp_servers"]["managed"] == replacement
    with pytest.raises(RuntimeError, match="already exists"):
        hermes.write_mcp_server_config("managed", proxy_argv, hermes_home=hermes_home)
    assert _loaded_config(binary, env)["mcp_servers"]["managed"] == replacement
