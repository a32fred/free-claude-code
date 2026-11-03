#!/usr/bin/env python3
"""Opt-in live check that fcc-claude can use file and shell tools on a route."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

MAX_TIMEOUT_SECONDS = 180
MAX_TURNS = 5
DEFAULT_TIMEOUT_SECONDS = 90


def _emit(status: str, *, model: str | None = None) -> int:
    report: dict[str, Any] = {"status": status}
    if model:
        report.update(
            requested_model=model,
            model_approval=False,
            note="Route capability only; verify provider completion logs for actual model identity.",
        )
    print(json.dumps(report, separators=(",", ":")))
    return 0 if status == "route_capability_pass" else 1


def _resolved(root: Path, path: object) -> Path | None:
    if not isinstance(path, str):
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root.resolve())
        return resolved
    except OSError, ValueError:
        return None


def _inside(root: Path, path: object) -> bool:
    return _resolved(root, path) is not None


def _has_required_use_events(stdout: str, root: Path) -> tuple[bool, bool, bool]:
    uses: dict[str, tuple[str, dict[str, Any]]] = {}
    results: dict[str, bool] = {}
    failed = malformed = False
    final_events = 0
    final_ok = True
    last_event_type = None
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed = True
            last_event_type = None
            continue
        if not isinstance(event, dict):
            malformed = True
            last_event_type = None
            continue
        last_event_type = event.get("type")
        if event.get("is_error") is True:
            failed = True
        if last_event_type == "result":
            final_events += 1
            final_ok &= (
                event.get("subtype") == "success" and event.get("is_error") is not True
            )
        message = event.get("message")
        blocks = message.get("content") if isinstance(message, dict) else None
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                malformed = True
                continue
            if block.get("type") == "tool_use" and last_event_type == "assistant":
                tool_id, name, data = (
                    block.get("id"),
                    block.get("name"),
                    block.get("input"),
                )
                if not (
                    isinstance(tool_id, str)
                    and isinstance(name, str)
                    and isinstance(data, dict)
                ):
                    malformed = True
                    continue
                if tool_id in uses:
                    malformed = True
                uses[tool_id] = (name, data)
            elif block.get("type") == "tool_result" and last_event_type == "user":
                tool_id = block.get("tool_use_id")
                content = block.get("content")
                if not isinstance(tool_id, str) or tool_id not in uses:
                    malformed = True
                    continue
                if not isinstance(content, (str, list)):
                    malformed = True
                    continue
                if tool_id in results:
                    malformed = True
                success = block.get("is_error") is not True
                results[tool_id] = success
                failed |= not success

    malformed |= set(uses) != set(results)

    def success_for(name: str, path: Path | None = None) -> bool:
        return any(
            tool_name == name
            and results.get(tool_id) is True
            and (
                path is None
                or (
                    _inside(root, data.get("file_path"))
                    and _resolved(root, data.get("file_path")) == path.resolve()
                )
            )
            for tool_id, (tool_name, data) in uses.items()
        )

    read_ok = success_for("Read", root / "input.txt")
    write_ok = success_for("Write", root / "output.txt") or success_for(
        "Edit", root / "output.txt"
    )
    bash_ok = any(
        name == "Bash"
        and data.get("command") == "python verify.py"
        and results.get(tool_id) is True
        for tool_id, (name, data) in uses.items()
    )
    return (
        read_ok
        and write_ok
        and bash_ok
        and final_events == 1
        and last_event_type == "result"
        and final_ok,
        failed,
        malformed,
    )


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            process.kill()
    elif os.name == "nt":
        try:
            result = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode != 0:
                process.kill()
        except OSError, subprocess.TimeoutExpired:
            process.kill()
    else:
        process.kill()

    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)


def _fixture(nonce: str) -> tuple[str, str, str]:
    source = f"probe:{nonce}\n"
    expected = f"TOOLS_OK:{nonce}\n"
    verifier = """import hashlib, json, pathlib
root = pathlib.Path.cwd()
source = (root / "input.txt").read_text(encoding="utf-8")
if not source.startswith("probe:") or not source.endswith("\\n"):
    raise SystemExit(1)
nonce = source[len("probe:"):-1]
if not nonce or any(char not in "0123456789abcdef" for char in nonce):
    raise SystemExit(1)
expected = f"TOOLS_OK:{nonce}\\n"
output = (root / "output.txt").read_text(encoding="utf-8")
if output != expected:
    raise SystemExit(1)
receipt = {"nonce": nonce, "sha256": hashlib.sha256(output.encode("utf-8")).hexdigest()}
(root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
"""
    return source, expected, verifier


def _artifacts_match(root: Path, nonce: str, expected: str) -> bool:
    try:
        if (root / "output.txt").read_text(encoding="utf-8") != expected:
            return False
        receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    except OSError, UnicodeError, json.JSONDecodeError:
        return False
    return receipt == {
        "nonce": nonce,
        "sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="explicit model or route to probe")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    if not args.model or not args.model.strip() or args.timeout < 1:
        return _emit("invalid_arguments")
    timeout = min(args.timeout, MAX_TIMEOUT_SECONDS)

    nonce = secrets.token_hex(16)
    source, expected, verifier = _fixture(nonce)
    with tempfile.TemporaryDirectory(prefix="fcc-model-tool-probe-") as directory:
        root = Path(directory).resolve()
        (root / "input.txt").write_text(source, encoding="utf-8")
        (root / "verify.py").write_text(verifier, encoding="utf-8")
        prompt = (
            f"Read {root / 'input.txt'}. Transform its content by replacing the initial "
            "probe: with TOOLS_OK: and write the result to output.txt in this workspace. "
            "Then use Bash to run python verify.py. Do not claim success without using those tools."
        )
        command = [
            "fcc-claude",
            "--restricted",
            "--model",
            args.model,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--max-turns",
            str(MAX_TURNS),
            "--permission-mode",
            "dontAsk",
            "--permission-prompts",
            "none",
            "--tools",
            "Read,Write,Edit,Bash",
            "--allowedTools",
            "Read",
            "Write",
            "Edit",
            "Bash(python verify.py)",
        ]
        process = None
        try:
            process = subprocess.Popen(
                command,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=(os.name == "posix"),
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                ),
            )
            stdout, _stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if process is not None:
                _terminate_process_tree(process)
            return _emit("timeout", model=args.model)
        except KeyboardInterrupt:
            if process is not None:
                _terminate_process_tree(process)
            return _emit("interrupted", model=args.model)
        except OSError:
            return _emit("cli_unavailable", model=args.model)

        if process.returncode != 0:
            return _emit("cli_failed", model=args.model)
        required, failed, malformed = _has_required_use_events(stdout, root)
        if malformed:
            return _emit("malformed_stream", model=args.model)
        if failed:
            return _emit("tool_failed", model=args.model)
        if not required:
            return _emit("tool_evidence_missing", model=args.model)
        if not _artifacts_match(root, nonce, expected):
            return _emit("artifact_mismatch", model=args.model)
        return _emit("route_capability_pass", model=args.model)


if __name__ == "__main__":
    sys.exit(main())
