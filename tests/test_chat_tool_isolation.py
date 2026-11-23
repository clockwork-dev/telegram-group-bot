"""Unit tests for per-turn chat & zero-tool PreToolUse isolation and Adversarial Judge hardening."""

from __future__ import annotations

import io
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from PIL import Image

from bot.antigravity import (
    AntigravityCLIClient,
    AntigravityConfig,
    AntigravityNativeToolAbort,
    AntigravityProvider,
    write_chat_pre_tool_hook,
)
from bot.config.manager import ConfigManager
from bot.core.models import MediaAttachment, ProviderResponse
from bot.evolution.manager import SelfEvolutionManager
from bot.memory.db import MemoryStore
from bot.quota.manager import QuotaBudgetManager


def _run_hook(gate_script: Path, payload: str) -> dict[str, Any]:
    res = subprocess.run(
        [sys.executable, str(gate_script)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return json.loads(res.stdout.strip())


class _DummyPopen:
    def __init__(self, stdout_lines: list[str] | None = None, returncode: int = 0) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO("\n".join(stdout_lines or []) + ("\n" if stdout_lines else ""))
        self.stderr = io.StringIO("")
        self.pid = 7777
        self.returncode = returncode
        self._terminated = False

    def poll(self) -> int | None:
        return self.returncode if self._terminated else None

    def terminate(self) -> None:
        self._terminated = True

    def kill(self) -> None:
        self._terminated = True

    def wait(self, timeout: float | None = None) -> int:
        self._terminated = True
        return self.returncode


class _ScriptedAcpPopen:
    def __init__(self, script_fn: Any) -> None:
        self._script_fn = script_fn
        self.sent_requests: list[dict[str, Any]] = []
        self._responses: list[str] = []
        self._resp_idx = 0
        self.stderr = io.StringIO("")
        self.pid = 8888
        self.returncode = 0
        self._terminated = False
        outer = self

        class _Stdin:
            def write(self, data: str) -> int:
                for line in data.splitlines():
                    if line.strip():
                        outer._on_line(line)
                return len(data)

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        class _Stdout:
            def readline(self) -> str:
                if outer._resp_idx < len(outer._responses):
                    line = outer._responses[outer._resp_idx]
                    outer._resp_idx += 1
                    return line
                return ""

        self.stdin = _Stdin()
        self.stdout = _Stdout()

    def _on_line(self, line: str) -> None:
        msg = json.loads(line)
        self.sent_requests.append(msg)
        for reply in self._script_fn(msg):
            self._responses.append(json.dumps(reply) + "\n")

    def poll(self) -> int | None:
        return self.returncode if self._terminated else None

    def terminate(self) -> None:
        self._terminated = True

    def kill(self) -> None:
        self._terminated = True

    def wait(self, timeout: float | None = None) -> int:
        self._terminated = True
        return self.returncode


def _make_evo_config(tmp_path: Path) -> ConfigManager:
    bot_yaml = tmp_path / "bot.yaml"
    bot_yaml.write_text(
        yaml.dump(
            {
                "telegram": {
                    "mode": "bot_api",
                    "bot_token": "123456:AAFakeToken1234567890abcdefghij",
                    "api_id": 12345,
                    "api_hash": "0123456789abcdef0123456789abcdef",
                    "allowed_chat_ids": [-100111222333],
                    "allowed_dm_user_ids": [1001],
                    "admin_user_ids": [1001],
                },
                "behavior": {"bot_names": ["Jeff"], "typing_simulation": False},
                "media": {"temp_dir": str(tmp_path / "media_tmp")},
                "self_evolution": {
                    "enabled": True,
                    "staging_dir": str(tmp_path / "staging"),
                    "trigger_dir": str(tmp_path / "trigger"),
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "system_prompt.md").write_text("System prompt", encoding="utf-8")
    (tmp_path / "friends.yaml").write_text("friends: {}\n", encoding="utf-8")
    return ConfigManager(config_path=bot_yaml)


def test_chat_pre_tool_hook_matrix_and_zero_tool_mode(tmp_path: Path) -> None:
    chat_dir = tmp_path / "chat_sandbox"
    hooks_json, gate_script = write_chat_pre_tool_hook(chat_dir)
    assert hooks_json.is_file() and gate_script.is_file()
    hooks_data = json.loads(hooks_json.read_text(encoding="utf-8"))
    assert "bot-chat-firewall" in hooks_data
    assert hooks_data["bot-chat-firewall"]["PreToolUse"][0]["matcher"] == "*"

    for allowed_tool in ("search_web", "read_url_content", "SEARCH_WEB"):
        res = _run_hook(gate_script, json.dumps({"toolCall": {"name": allowed_tool, "args": {}}}))
        assert res["decision"] == "allow"

    for denied_tool in (
        "read_file",
        "view_file",
        "grep_search",
        "find_by_name",
        "list_dir",
        "run_command",
        "write_file",
        "write_to_file",
        "edit_file",
        "invoke_subagent",
        "send_message",
        "call_mcp_tool",
        "browser_subagent",
        "generate_image",
        "swarm",
    ):
        res = _run_hook(gate_script, json.dumps({"toolCall": {"name": denied_tool, "args": {}}}))
        assert res["decision"] == "deny"

    # Malformed / missing payloads must fail closed
    for bad_payload in ("{not-json", "[]", "{}", json.dumps({"toolCall": {}})):
        assert _run_hook(gate_script, bad_payload)["decision"] == "deny"

    # Zero-tool mode (Adversarial Judge): denies EVERYTHING including search_web & read_url_content
    zero_dir = tmp_path / "zero_sandbox"
    _, zero_gate = write_chat_pre_tool_hook(zero_dir, allowed_tools=frozenset())
    for tool_name in ("search_web", "read_url_content", "read_file", "run_command", "invoke_subagent"):
        res = _run_hook(zero_gate, json.dumps({"toolCall": {"name": tool_name, "args": {}}}))
        assert res["decision"] == "deny"


@pytest.mark.asyncio
async def test_cli_client_chat_profile_isolates_cwd_hook_and_cleans_up(tmp_path: Path) -> None:
    data_dir = tmp_path / "agy_home"
    ops_workspace = tmp_path / "ops_workspace"
    ops_workspace.mkdir()
    extra_dir = tmp_path / "extra_dir"
    extra_dir.mkdir()

    cli = AntigravityCLIClient(data_dir=data_dir, binary="/mock/bin/agy")
    captured: list[dict[str, Any]] = []

    def fake_popen(cmd: list[str], *, cwd: str | Path, env: dict[str, str]) -> _DummyPopen:
        cwd_path = Path(cwd)
        gate_path = cwd_path / ".agents" / "chat_gate.py"
        hooks_path = cwd_path / ".agents" / "hooks.json"
        captured.append({
            "cmd": list(cmd),
            "cwd": cwd_path,
            "mode": stat.S_IMODE(os.stat(cwd_path).st_mode),
            "hooks_exists": hooks_path.is_file(),
            "gate_exists": gate_path.is_file(),
            "search_web_decision": _run_hook(
                gate_path, json.dumps({"toolCall": {"name": "search_web", "args": {}}})
            )["decision"]
            if gate_path.is_file()
            else None,
            "read_file_decision": _run_hook(
                gate_path, json.dumps({"toolCall": {"name": "read_file", "args": {"Path": ".env"}}})
            )["decision"]
            if gate_path.is_file()
            else None,
            "run_command_decision": _run_hook(
                gate_path, json.dumps({"toolCall": {"name": "run_command", "args": {"CommandLine": "id"}}})
            )["decision"]
            if gate_path.is_file()
            else None,
            "subagent_decision": _run_hook(
                gate_path, json.dumps({"toolCall": {"name": "invoke_subagent", "args": {"TypeName": "self"}}})
            )["decision"]
            if gate_path.is_file()
            else None,
        })
        return _DummyPopen([
            json.dumps({"event": "init", "conversation_id": "conv_chat_iso"}),
            json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": "ok"}}),
        ])

    try:
        with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_popen):
            resp = await cli.generate_response(
                prompt="hello",
                tool_profile="chat",
                mode="auto",
                cwd=ops_workspace,
                add_dirs=[extra_dir],
            )
        assert resp.text == "ok"
        assert len(captured) == 1
        snap = captured[0]
        assert snap["cwd"].name.startswith("bot_chat_")
        assert snap["cwd"] != ops_workspace.resolve()
        assert snap["cwd"] != data_dir.resolve()
        assert snap["mode"] == 0o700
        assert snap["hooks_exists"] is True
        assert snap["gate_exists"] is True
        assert snap["search_web_decision"] == "allow"
        assert snap["read_file_decision"] == "deny"
        assert snap["run_command_decision"] == "deny"
        assert snap["subagent_decision"] == "deny"
        assert "--add-dir" not in snap["cmd"]
        assert "--dangerously-skip-permissions" not in snap["cmd"]
        assert "--mode" not in snap["cmd"]

        # Sandbox cleaned up after turn, and neither ops_workspace nor data_dir was polluted
        assert not snap["cwd"].exists()
        assert not (ops_workspace / ".agents").exists()
        assert not (data_dir / ".agents").exists()

        # Failure path (AntigravityNativeToolAbort) also cleans up sandbox cwd
        def fake_abort_popen(cmd: list[str], *, cwd: str | Path, env: dict[str, str]) -> _DummyPopen:
            captured.append({"cwd": Path(cwd)})
            return _DummyPopen([
                json.dumps({"event": "init", "conversation_id": "conv_chat_abort"}),
                json.dumps({
                    "event": "step_update",
                    "step_update": {"step_type": "tool", "tool_name": "read_file"},
                }),
            ])

        with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_abort_popen):
            with pytest.raises(AntigravityNativeToolAbort):
                await cli.generate_response(prompt="read secret", tool_profile="chat")
        assert not captured[-1]["cwd"].exists()
    finally:
        cli.close()


@pytest.mark.asyncio
async def test_cli_client_zero_tool_mode_denies_web_tools_at_hook_and_parser(tmp_path: Path) -> None:
    cli = AntigravityCLIClient(data_dir=tmp_path / "agy_home", binary="/mock/bin/agy")
    captured: list[dict[str, Any]] = []

    def fake_popen(cmd: list[str], *, cwd: str | Path, env: dict[str, str]) -> _DummyPopen:
        cwd_path = Path(cwd)
        gate_path = cwd_path / ".agents" / "chat_gate.py"
        captured.append({
            "cmd": list(cmd),
            "cwd": cwd_path,
            "search_web": _run_hook(
                gate_path, json.dumps({"toolCall": {"name": "search_web", "args": {}}})
            )["decision"],
            "read_url": _run_hook(
                gate_path, json.dumps({"toolCall": {"name": "read_url_content", "args": {}}})
            )["decision"],
        })
        return _DummyPopen([
            json.dumps({"event": "init", "conversation_id": "conv_zero_1"}),
            json.dumps({
                "event": "step_update",
                "step_update": {"step_type": "tool", "tool_name": "search_web"},
            }),
        ])

    try:
        with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_popen):
            with pytest.raises(AntigravityNativeToolAbort):
                await cli.generate_response(
                    prompt="judge this",
                    tool_profile="chat",
                    allowed_tools=frozenset(),
                )
        assert captured[0]["search_web"] == "deny"
        assert captured[0]["read_url"] == "deny"
        assert not captured[0]["cwd"].exists()
    finally:
        cli.close()


@pytest.mark.asyncio
async def test_cli_client_acp_and_worker_apply_chat_and_zero_tool_isolation(tmp_path: Path) -> None:
    data_dir = tmp_path / "agy_home"
    ops_workspace = tmp_path / "ops_ws"
    ops_workspace.mkdir()
    img = tmp_path / "pic.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(img, format="JPEG")

    cli = AntigravityCLIClient(data_dir=data_dir, binary="/mock/bin/agy")
    captured: list[dict[str, Any]] = []

    def script_ok(msg: dict[str, Any]) -> list[dict[str, Any]]:
        m, mid = msg.get("method"), msg.get("id")
        if m == "initialize":
            return [{"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": 1}}]
        if m == "session/new":
            return [{"jsonrpc": "2.0", "id": mid, "result": {"sessionId": "acp_iso_1"}}]
        if m == "session/prompt":
            return [
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "ACP ok"},
                        }
                    },
                },
                {"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "end_turn"}},
            ]
        return []

    def fake_acp_popen(cmd: list[str], *, cwd: str | Path, env: dict[str, str]) -> _ScriptedAcpPopen:
        cwd_path = Path(cwd)
        gate_path = cwd_path / ".agents" / "chat_gate.py"
        proc = _ScriptedAcpPopen(script_ok)
        captured.append({
            "cmd": list(cmd),
            "cwd": cwd_path,
            "hooks_exists": (cwd_path / ".agents" / "hooks.json").is_file(),
            "search_web": _run_hook(
                gate_path, json.dumps({"toolCall": {"name": "search_web", "args": {}}})
            )["decision"],
            "read_file": _run_hook(
                gate_path, json.dumps({"toolCall": {"name": "read_file", "args": {}}})
            )["decision"],
            "proc": proc,
        })
        return proc

    try:
        # 1. ACP chat turn uses isolated sandbox cwd and cleans up
        with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_acp_popen):
            resp = await cli.generate_response(
                prompt="What is here?",
                media_attachments=[MediaAttachment(file_path=img, mime_type="image/jpeg")],
                tool_profile="chat",
                cwd=ops_workspace,
                add_dirs=[tmp_path],
            )
        assert resp.text == "ACP ok"
        assert captured[0]["cwd"].name.startswith("bot_chat_")
        assert captured[0]["hooks_exists"] is True
        assert captured[0]["search_web"] == "allow"
        assert captured[0]["read_file"] == "deny"
        assert "--add-dir" not in captured[0]["cmd"]
        new_reqs = [r for r in captured[0]["proc"].sent_requests if r.get("method") == "session/new"]
        assert Path(new_reqs[0]["params"]["cwd"]) == captured[0]["cwd"]
        assert not captured[0]["cwd"].exists()
        assert not (ops_workspace / ".agents").exists()

        # 2. ACP zero-tool turn aborts on search_web and cleans up
        def script_web_tool(msg: dict[str, Any]) -> list[dict[str, Any]]:
            m, mid = msg.get("method"), msg.get("id")
            if m == "initialize":
                return [{"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": 1}}]
            if m == "session/new":
                return [{"jsonrpc": "2.0", "id": mid, "result": {"sessionId": "acp_iso_2"}}]
            if m == "session/prompt":
                return [
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "update": {
                                "sessionUpdate": "tool_call",
                                "toolCallId": "tc1",
                                "name": "search_web",
                            }
                        },
                    }
                ]
            return []

        acp_zero_cwds: list[Path] = []

        def fake_acp_zero(cmd: list[str], *, cwd: str | Path, env: dict[str, str]) -> _ScriptedAcpPopen:
            cwd_path = Path(cwd)
            acp_zero_cwds.append(cwd_path)
            return _ScriptedAcpPopen(script_web_tool)

        with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_acp_zero):
            with pytest.raises(AntigravityNativeToolAbort):
                await cli.generate_response(
                    prompt="No tools",
                    media_attachments=[MediaAttachment(file_path=img, mime_type="image/jpeg")],
                    tool_profile="chat",
                    allowed_tools=frozenset(),
                )
        assert not acp_zero_cwds[0].exists()

        # 3. _ensure_worker also isolates chat worker cwd and cleans up on _kill_worker
        worker_cwds: list[Path] = []

        def fake_worker_popen(cmd: list[str], *, cwd: str | Path, env: dict[str, str]) -> _DummyPopen:
            worker_cwds.append(Path(cwd))
            return _DummyPopen([])

        with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_worker_popen):
            cli._ensure_worker("gemini-4-hydrogen", "low", cwd=ops_workspace, tool_profile="chat")
            assert worker_cwds[0].name.startswith("bot_chat_")
            assert (worker_cwds[0] / ".agents" / "hooks.json").is_file()
            assert (worker_cwds[0] / ".agents" / "chat_gate.py").is_file()
            assert not (ops_workspace / ".agents").exists()
        cli._kill_worker()
        assert not worker_cwds[0].exists()
    finally:
        cli.close()


@pytest.mark.asyncio
async def test_provider_zero_tool_isolation_bypasses_sdk_and_forwards_to_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-for-sdk")
    provider = AntigravityProvider(
        AntigravityConfig(data_dir=tmp_path / "agy", enable_sdk=True, enable_cli_fallback=True)
    )
    try:
        if provider._sdk_client is not None:
            provider._sdk_client.generate_response = AsyncMock(
                side_effect=AssertionError("SDK must not be used for explicit zero-tool isolation")
            )
        assert provider._cli_client is not None
        provider._cli_client.generate_response = AsyncMock(
            return_value=ProviderResponse(text='{"verdict": "NO_CHANGE", "reason": "ok"}')
        )
        resp = await provider.generate_response(
            prompt="evaluate",
            tool_profile="chat",
            allowed_tools=frozenset(),
        )
        assert "NO_CHANGE" in resp.text
        provider._cli_client.generate_response.assert_awaited_once()
        assert provider._cli_client.generate_response.call_args.kwargs["allowed_tools"] == frozenset()
    finally:
        provider.close()


@pytest.mark.asyncio
async def test_adversarial_judge_disables_tools_and_supports_legacy_mock_providers(
    tmp_path: Path,
) -> None:
    cfg_mgr = _make_evo_config(tmp_path)
    store = MemoryStore(db_path=str(tmp_path / "evo.sqlite"))
    await store.init_db()
    qm = QuotaBudgetManager(memory_store=store, config=cfg_mgr.bot)

    try:
        # 1. Modern provider receives explicit no-tool system_instructions + allowed_tools=frozenset()
        provider = MagicMock()
        provider.generate_response = AsyncMock(
            return_value=ProviderResponse(
                text='{"verdict": "NO_CHANGE", "reason": "transient"}',
                model="gemini-4-hydrogen",
            )
        )
        mgr = SelfEvolutionManager(
            config_mgr=cfg_mgr,
            memory_store=store,
            quota_manager=qm,
            provider=provider,
            notify_admin=AsyncMock(),
        )
        verdict, reason, _ = await mgr._run_adversarial_judge("runtime_error", "sig_1", "boom")
        assert verdict == "NO_CHANGE"
        assert reason == "transient"
        kw = provider.generate_response.call_args.kwargs
        assert kw["tool_profile"] == "chat"
        assert kw["allowed_tools"] == frozenset()
        sys_instr = kw["system_instructions"].lower()
        assert "do not invoke any tools" in sys_instr
        assert "file reads" in sys_instr

        # 2. Legacy mock provider without `allowed_tools` param still works via TypeError fallback
        legacy_calls: list[dict[str, Any]] = []

        class _LegacyProvider:
            async def generate_response(
                self,
                prompt: str,
                system_instructions: str | None = None,
                model: str | None = None,
                effort: str | None = None,
                tool_profile: str = "chat",
            ) -> ProviderResponse:
                legacy_calls.append({
                    "prompt": prompt,
                    "system_instructions": system_instructions,
                    "tool_profile": tool_profile,
                })
                return ProviderResponse(
                    text='{"verdict": "CONFIG_ONLY", "reason": "tune yaml"}',
                    model=model,
                )

        mgr_legacy = SelfEvolutionManager(
            config_mgr=cfg_mgr,
            memory_store=store,
            quota_manager=qm,
            provider=_LegacyProvider(),
            notify_admin=AsyncMock(),
        )
        verdict2, reason2, _ = await mgr_legacy._run_adversarial_judge("admin", "sig_2", "spam")
        assert verdict2 == "CONFIG_ONLY"
        assert reason2 == "tune yaml"
        assert len(legacy_calls) == 1
        assert legacy_calls[0]["tool_profile"] == "chat"
        assert "do not invoke any tools" in str(legacy_calls[0]["system_instructions"]).lower()
    finally:
        await store.close()
