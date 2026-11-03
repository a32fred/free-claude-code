"""Unit tests for the opt-in live Claude CLI tool capability probe."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).parents[2] / "scripts" / "check-model-tools.py"
SPEC = importlib.util.spec_from_file_location("model_tool_probe", SCRIPT)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _events(cwd: Path, nonce: str, *, failed_tool: str | None = None) -> str:
    expected = f"TOOLS_OK:{nonce}\n"
    (cwd / "output.txt").write_text(expected, encoding="utf-8")
    (cwd / "receipt.json").write_text(
        json.dumps(
            {
                "nonce": nonce,
                "sha256": hashlib.sha256(expected.encode()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    calls = [
        ("Read", {"file_path": str(cwd / "input.txt")}),
        ("Write", {"file_path": str(cwd / "output.txt"), "content": expected}),
        ("Bash", {"command": "python verify.py"}),
    ]
    lines = []
    for index, (name, tool_input) in enumerate(calls):
        tool_id = f"tool-{index}"
        lines.extend(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": tool_id,
                                    "name": name,
                                    "input": tool_input,
                                }
                            ]
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "is_error": name == failed_tool,
                                    "content": "ok",
                                }
                            ]
                        },
                    }
                ),
            ]
        )
    lines.append(
        json.dumps({"type": "result", "subtype": "success", "is_error": False})
    )
    return "\n".join(lines) + "\n"


def _run_mocked(
    *,
    stream: str | None = None,
    returncode: int = 0,
    timeout=False,
    mutate=None,
    result_error=False,
    malformed_mode=None,
    interrupt=False,
    result_subtype="success",
):
    observed = {}

    def fake_popen(argv, **kwargs):
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        cwd = Path(kwargs["cwd"])
        source = (cwd / "input.txt").read_text(encoding="utf-8")
        value = source.removeprefix("probe:").strip()
        if stream is not None:
            stdout = stream
        else:
            stdout = _events(
                cwd,
                value,
                failed_tool=mutate if mutate in {"Read", "Write", "Bash"} else None,
            )
            if result_error:
                stdout = stdout.replace(
                    '"subtype": "success", "is_error": false',
                    '"subtype": "error", "is_error": true',
                )
            if result_subtype != "success":
                stdout = stdout.replace(
                    '"subtype": "success"', f'"subtype": "{result_subtype}"'
                )
            if malformed_mode:
                events = [json.loads(line) for line in stdout.splitlines()]
                if malformed_mode == "trailing_event":
                    events.append({"type": "assistant", "message": {"content": []}})
                elif malformed_mode == "orphan_result":
                    events.append(
                        {
                            "type": "user",
                            "message": {
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": "unknown",
                                        "content": "ok",
                                    }
                                ]
                            },
                        }
                    )
                elif malformed_mode == "missing_content":
                    next(event for event in events if event.get("type") == "user")[
                        "message"
                    ]["content"][0].pop("content")
                stdout = "\n".join(json.dumps(event) for event in events) + "\n"
            if mutate == "output":
                (cwd / "output.txt").write_text("wrong\n", encoding="utf-8")
            elif mutate == "receipt":
                (cwd / "receipt.json").write_text("{}", encoding="utf-8")

        class Process:
            pid = 12345

            def __init__(self):
                self.returncode = returncode
                self.calls = 0
                self.timeouts = []
                self.killed = False

            def communicate(self, timeout=None):
                self.calls += 1
                self.timeouts.append(timeout)
                if globals_timeout:
                    globals_timeout.clear()
                    if interrupt:
                        raise KeyboardInterrupt
                    raise subprocess.TimeoutExpired(argv, float(timeout or 1))
                return stdout, "private stderr"

            def kill(self):
                self.killed = True

        globals_timeout = [True] if timeout or interrupt else []
        observed["process"] = Process()
        return observed["process"]

    output = io.StringIO()
    with (
        patch.object(probe.subprocess, "Popen", fake_popen),
        contextlib.redirect_stdout(output),
    ):
        status = probe.main(["--model", "example-model"])
    return status, json.loads(output.getvalue()), observed


def test_live_probe_requires_explicit_model():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = probe.main([])
    assert status != 0
    assert json.loads(output.getvalue())["status"] == "invalid_arguments"


def test_probe_passes_only_when_tool_events_and_fresh_artifacts_match():
    status, result, observed = _run_mocked()
    assert status == 0
    assert result["status"] == "route_capability_pass"
    assert result["model_approval"] is False
    assert result["requested_model"] == "example-model"
    assert "private stderr" not in json.dumps(result)
    argv = observed["argv"]
    assert "--model" in argv and argv[argv.index("--model") + 1] == "example-model"
    assert "--restricted" in argv
    assert "--dangerously-skip-permissions" not in argv
    assert observed["process"].timeouts[0] <= probe.MAX_TIMEOUT_SECONDS
    assert argv[argv.index("--max-turns") + 1] == str(probe.MAX_TURNS)
    assert argv.count("-p") == 1


def test_text_only_success_is_not_a_pass():
    status, result, _ = _run_mocked(
        stream=json.dumps({"type": "result", "result": "done"})
    )
    assert status != 0
    assert result["status"] == "tool_evidence_missing"


def test_partial_and_malformed_stream_is_not_a_pass():
    status, result, _ = _run_mocked(stream='{"type":"assistant"}\nnot-json\n')
    assert status != 0
    assert result["status"] == "malformed_stream"


def test_timeout_is_reported_without_cli_output():
    status, result, observed = _run_mocked(timeout=True)
    assert status != 0
    assert result["status"] == "timeout"
    assert "private" not in json.dumps(result)
    assert observed["process"].killed
    assert observed["process"].timeouts == [90, 5]


def test_interrupt_cleans_up_process_with_bounded_wait():
    status, result, observed = _run_mocked(interrupt=True)
    assert status != 0
    assert result["status"] == "interrupted"
    assert observed["process"].killed
    assert observed["process"].timeouts == [90, 5]


def test_nonzero_cli_exit_fails():
    status, result, _ = _run_mocked(returncode=1)
    assert status != 0
    assert result["status"] == "cli_failed"
    assert result["diagnostics"]["final_subtype"] == "success"
    assert result["diagnostics"]["tools"]["Read"] == {
        "completed": 1,
        "failed": 0,
        "unresolved": 0,
    }
    assert result["diagnostics"]["tools"]["Write"] == {
        "completed": 1,
        "failed": 0,
        "unresolved": 0,
    }
    assert result["diagnostics"]["tools"]["Bash"] == {
        "completed": 1,
        "failed": 0,
        "unresolved": 0,
    }
    assert result["diagnostics"]["artifacts_match"] is True


def test_error_max_turns_diagnostic_and_unknown_values_are_sanitized():
    status, result, _ = _run_mocked(returncode=1, result_subtype="error_max_turns")
    assert status != 0
    assert result["diagnostics"]["final_subtype"] == "error_max_turns"
    assert "prompt" not in json.dumps(result).lower()
    assert "private stderr" not in json.dumps(result)

    _, result, _ = _run_mocked(returncode=1, result_subtype="secret model text")
    assert result["diagnostics"]["final_subtype"] == "unknown"


def test_failed_tool_result_fails():
    status, result, _ = _run_mocked(mutate="Bash")
    assert status != 0
    assert result["status"] == "tool_failed"


def test_invalid_output_and_receipt_fail():
    status, result, _ = _run_mocked(mutate="output")
    assert status != 0
    assert result["status"] == "artifact_mismatch"

    status, result, _ = _run_mocked(mutate="receipt")
    assert status != 0
    assert result["status"] == "artifact_mismatch"


def test_result_error_fails_even_after_successful_tool_calls():
    status, result, _ = _run_mocked(result_error=True)
    assert status != 0
    assert result["status"] == "tool_failed"


def test_parser_rejects_nonterminal_result_and_malformed_tool_results():
    for mode in ("trailing_event", "orphan_result", "missing_content"):
        status, result, _ = _run_mocked(malformed_mode=mode)
        assert status != 0
        assert result["status"] in {"tool_evidence_missing", "malformed_stream"}


def test_unresolved_extra_tool_use_rejects_otherwise_successful_stream(tmp_path):
    events = _events(tmp_path, "abc").splitlines()
    events.insert(
        -1,
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "unresolved",
                            "name": "Read",
                            "input": {"file_path": str(tmp_path / "input.txt")},
                        }
                    ]
                },
            }
        ),
    )
    required, failed, malformed = probe._has_required_use_events(
        "\n".join(events), tmp_path
    )
    assert malformed or failed or not required


def test_diagnostics_tolerates_non_string_untrusted_names_and_subtypes():
    stream = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "tool_use", "name": [], "id": "x"}]
                    },
                }
            ),
            json.dumps({"type": "result", "subtype": {"unexpected": "value"}}),
        ]
    )
    report = probe._diagnostics(stream, False)
    assert report["final_subtype"] == "unknown"
    assert all(
        count == 0 for tool in report["tools"].values() for count in tool.values()
    )


def test_probe_prompt_explicitly_requires_final_newline():
    _, _, observed = _run_mocked()
    argv = observed["argv"]
    prompt = argv[argv.index("-p") + 1]
    assert "one LF newline" in prompt


def test_generated_verifier_accepts_exact_output_and_rejects_missing_newline(tmp_path):
    import sys

    source, expected, verifier = probe._fixture("abc123")
    (tmp_path / "input.txt").write_text(source)
    (tmp_path / "verify.py").write_text(verifier)
    (tmp_path / "output.txt").write_text(expected)
    good = subprocess.run(
        [sys.executable, "verify.py"], cwd=tmp_path, capture_output=True
    )
    assert good.returncode == 0
    assert probe._artifacts_match(tmp_path, "abc123", expected)
    (tmp_path / "output.txt").write_text(expected.rstrip("\n"))
    bad = subprocess.run(
        [sys.executable, "verify.py"], cwd=tmp_path, capture_output=True
    )
    assert bad.returncode != 0
    assert not probe._artifacts_match(tmp_path, "abc123", expected)
