from __future__ import annotations

import json
import sys

import pytest

from scripts import repro_claude_relayed as repro


def test_build_prompt_adds_requested_large_input_payload():
    prompt = repro.build_prompt(response_lines=10, attempt=1, input_kib=128)

    payload = prompt.split("INPUT_PAYLOAD_BEGIN\n", 1)[1].split("\nINPUT_PAYLOAD_END", 1)[0]
    assert len(payload) == 128 * 1024
    assert "Produce exactly 10 lines" in prompt


def test_build_artifact_prompt_targets_write_tool_path(tmp_path):
    artifact = tmp_path / "design.md"
    prompt = repro.build_artifact_prompt(artifact, sections=25, attempt=2)

    assert str(artifact) in prompt
    assert "Write tool" in prompt
    assert "exactly 25 numbered sections" in prompt
    assert "ARTIFACT_TEST_DONE" in prompt


def test_select_ucode_profile_persists_explicit_matching_choice(monkeypatch):
    state = {"available_tools": ["claude"]}
    selected = []
    saved = []
    monkeypatch.setattr(
        repro,
        "get_databricks_profiles",
        lambda: [("https://workspace.example.com", "chosen")],
    )
    monkeypatch.setattr(repro, "set_current_workspace", selected.append)
    monkeypatch.setattr(repro, "load_state", lambda: state)
    monkeypatch.setattr(repro, "save_state", lambda value: saved.append(dict(value)))

    workspace = repro.select_ucode_profile("https://workspace.example.com/", "chosen")

    assert workspace == "https://workspace.example.com"
    assert selected == ["https://workspace.example.com"]
    assert saved[0]["profile"] == "chosen"
    assert saved[0]["available_tools"] == ["claude"]


def test_select_ucode_profile_rejects_host_mismatch(monkeypatch):
    monkeypatch.setattr(
        repro,
        "get_databricks_profiles",
        lambda: [("https://other.example.com", "wrong")],
    )

    with pytest.raises(RuntimeError, match="targets https://other.example.com"):
        repro.select_ucode_profile("https://workspace.example.com", "wrong")


def test_build_command_exercises_stream_json_through_ucode(tmp_path):
    debug = tmp_path / "debug.log"
    command = repro.build_command(
        ["uv", "run", "ucode"],
        workspace="https://workspace.example.com",
        provider="catalog.schema.provider",
        prompt="long response please",
        debug_file=debug,
    )

    assert command[:4] == ["uv", "run", "ucode", "claude"]
    assert command[command.index("--provider") + 1] == "catalog.schema.provider"
    assert command[command.index("--workspace") + 1] == "https://workspace.example.com"
    assert "--skip-preflight" in command
    assert command[command.index("-p") + 1] == "long response please"
    assert command[command.index("--output-format") + 1] == "stream-json"
    assert "--include-partial-messages" in command
    assert command[command.index("--debug-file") + 1] == str(debug)
    assert command[-2:] == ["--tools", ""]


def test_build_command_enables_write_tool_for_artifact(tmp_path):
    command = repro.build_command(
        ["ucode"],
        workspace="https://workspace.example.com",
        provider="catalog.schema.provider",
        prompt="write the artifact",
        tools="Write",
        add_dir=tmp_path,
    )

    assert command[command.index("--permission-mode") + 1] == "acceptEdits"
    assert command[command.index("--add-dir") + 1] == str(tmp_path)
    assert command[-2:] == ["--tools", "Write"]


def test_classify_output_finds_recovered_claude_retry_and_proxy_failure():
    stdout = json.dumps({"type": "result", "is_error": True}) + "\n"
    stderr = "\n".join(
        [
            "API Error: The response stopped arriving. The response above may be incomplete.",
            "API error · Retrying in 0s · attempt 1/10",
            '[ucode-relay] {"event":"upstream_headers","status":200}',
            '[ucode-relay] {"event":"upstream_stream_error","error_type":"ReadError"}',
            '[ucode-relay] {"event":"upstream_headers","status":503}',
        ]
    )

    markers, events, statuses = repro.classify_output(stdout, stderr)

    assert markers["claude_response_stopped"] == 1
    assert markers["claude_api_error"] == 2
    assert markers["claude_retry"] == 1
    assert markers["claude_error_result"] == 1
    assert markers["proxy_upstream_stream_error"] == 1
    assert markers["proxy_non_2xx"] == 1
    assert events == {"upstream_headers": 2, "upstream_stream_error": 1}
    assert statuses == [200, 503]


def test_classify_output_ignores_expected_hello_probe_401():
    stderr = "\n".join(
        [
            '[ucode-relay] {"event":"request_start","request_id":"hello","path":"/api/hello"}',
            '[ucode-relay] {"event":"upstream_headers","request_id":"hello","status":401}',
            '[ucode-relay] {"event":"request_start","request_id":"turn","path":"/v1/messages"}',
            '[ucode-relay] {"event":"upstream_headers","request_id":"turn","status":200}',
        ]
    )

    markers, events, statuses = repro.classify_output("", stderr)

    assert markers == {}
    assert events == {"request_start": 2, "upstream_headers": 2}
    assert statuses == [401, 200]


def test_classify_output_flags_long_zero_byte_message_disconnect():
    stderr = "\n".join(
        [
            '[ucode-relay] {"event":"request_start","request_id":"turn","path":"/v1/messages"}',
            '[ucode-relay] {"event":"upstream_headers","request_id":"turn","status":200}',
            '[ucode-relay] {"event":"client_disconnect","request_id":"turn",'
            '"phase":"response","bytes":0,"elapsed_ms":284107}',
        ]
    )

    markers, _, _ = repro.classify_output("", stderr)

    assert markers == {"proxy_zero_byte_client_disconnect": 1}


def test_run_attempt_captures_diagnostics_and_exit_zero(tmp_path):
    fake = tmp_path / "fake.py"
    fake.write_text(
        "import json, os, sys\n"
        "assert os.environ['UCODE_RELAYED_PROXY_DIAGNOSTICS'] == '1'\n"
        "print(json.dumps({'type': 'result', 'is_error': False}))\n"
        "print('[ucode-relay] ' + json.dumps({'event': 'response_complete', "
        "'status': 200}), file=sys.stderr)\n",
        encoding="utf-8",
    )

    result = repro.run_attempt(
        [sys.executable, str(fake)], attempt=1, artifact_dir=tmp_path, timeout=10
    )

    assert result["returncode"] == 0
    assert result["markers"] == {}
    assert result["proxy_events"] == {"response_complete": 1}
    assert (tmp_path / "attempt-0001.stdout.jsonl").is_file()
    assert (tmp_path / "attempt-0001.stderr.log").is_file()
