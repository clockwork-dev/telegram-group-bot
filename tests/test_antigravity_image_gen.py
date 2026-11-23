"""Unit tests for Antigravity native image generation adapter, stream gating, and PreToolUse firewall."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from bot.antigravity import (
    ALWAYS_FORBIDDEN_TOOLS,
    CHAT_ALLOWED_TOOLS,
    IMAGE_ALLOWED_SUBAGENT_TYPES,
    OPS_ALLOWED_TOOLS,
    SELF_DEV_ALLOWED_TOOLS,
    AntigravityBillingError,
    AntigravityCLIClient,
    AntigravityConfig,
    AntigravityNativeToolAbort,
    AntigravityProvider,
    AntigravityRateLimitError,
    AntigravityStreamParser,
    ErrorCategory,
    GeneratedImageResult,
    ImageGenerationError,
    ImagePolicyRefusalError,
    ImageTimeoutError,
    ImageValidationError,
    is_allowed_image_subagent_step,
    resolve_allowed_tools,
    write_image_pre_tool_hook,
)
from bot.core.models import MediaAttachment


def _make_image_bytes(fmt: str = "JPEG", size: tuple[int, int] = (32, 24)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, (220, 40, 40))
    img.save(buf, format=fmt)
    return buf.getvalue()


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


class _FakeStreamJsonPopen:
    """Minimal mock Popen emitting scripted NDJSON stdout lines."""

    def __init__(
        self,
        lines: list[str],
        *,
        on_start: Any = None,
        block_until_killed: bool = False,
        returncode: int = 0,
    ) -> None:
        self._lines = list(lines)
        self._idx = 0
        self._on_start = on_start
        self._block_until_killed = block_until_killed
        self._killed_event = threading.Event()
        self.pid = 999999
        self.returncode = returncode
        self.terminated = False
        self.stdin_chunks: list[str] = []
        outer = self

        class _Stdin:
            def write(self, data: str) -> int:
                outer.stdin_chunks.append(data)
                return len(data)

            def flush(self) -> None:
                pass

            def close(self) -> None:
                if outer._on_start is not None:
                    outer._on_start()

        class _Stdout:
            def readline(self) -> str:
                if outer._block_until_killed:
                    outer._killed_event.wait(timeout=5.0)
                    return ""
                if outer._idx < len(outer._lines):
                    line = outer._lines[outer._idx]
                    outer._idx += 1
                    return line
                return ""

        class _Stderr:
            def readline(self) -> str:
                return ""

        self.stdin = _Stdin()
        self.stdout = _Stdout()
        self.stderr = _Stderr()

    def poll(self) -> int | None:
        return self.returncode if self.terminated else None

    def terminate(self) -> None:
        self.terminated = True
        self._killed_event.set()

    def kill(self) -> None:
        self.terminated = True
        self._killed_event.set()

    def wait(self, timeout: float | None = None) -> int:
        self.terminated = True
        self._killed_event.set()
        return self.returncode


# ---------------------------------------------------------------------------
# 1. stream.py tool-profile & subagent interception tests
# ---------------------------------------------------------------------------


def test_existing_tool_profile_invariants_preserved():
    assert CHAT_ALLOWED_TOOLS == frozenset({"search_web", "read_url_content"})
    assert len(OPS_ALLOWED_TOOLS) == 23
    assert SELF_DEV_ALLOWED_TOOLS == OPS_ALLOWED_TOOLS
    assert IMAGE_ALLOWED_SUBAGENT_TYPES == frozenset({"image-generator"})
    for forbidden in ("browser_subagent", "call_mcp_tool", "swarm", "generate_image"):
        assert forbidden in ALWAYS_FORBIDDEN_TOOLS
    assert resolve_allowed_tools("image") == frozenset()
    with pytest.raises(ValueError):
        resolve_allowed_tools("invalid_profile")  # type: ignore[arg-type]


@pytest.mark.parametrize("profile", ["chat"])
@pytest.mark.parametrize("subagent_type", ["image-generator", "self", "research"])
def test_stream_parser_blocks_step_type_subagent_in_non_image_profiles(profile: str, subagent_type: str):
    """Regression test: step_type='subagent' must NEVER bypass chat profile restrictions."""
    intercepted = False

    def on_intercept() -> None:
        nonlocal intercepted
        intercepted = True

    parser = AntigravityStreamParser(tool_profile=profile, on_native_tool_intercept=on_intercept)  # type: ignore[arg-type]
    event = {
        "event": "step_update",
        "step_update": {
            "conversation_id": "conv-123",
            "step_index": 2,
            "state": "ACTIVE",
            "step_type": "subagent",
            "tool_name": "invoke_subagent",
            "subagent_info": {
                "subagents": [{"type_name": subagent_type, "role": "Helper"}],
            },
        },
    }
    with pytest.raises(AntigravityNativeToolAbort):
        parser.feed_line(json.dumps(event))
    assert intercepted is True
    assert parser.native_tool_intercepted is True


def test_stream_parser_image_profile_allows_only_image_generator():
    parser = AntigravityStreamParser(tool_profile="image")

    init_line = json.dumps(
        {"event": "init", "conversation_id": "conv-img-1", "init": {"model": "gemini-4-hydrogen"}}
    )
    parser.feed_line(init_line)
    assert parser.conversation_id == "conv-img-1"

    # Allowed: invoke_subagent exclusively targeting image-generator (case-insensitive)
    subagent_active = {
        "event": "step_update",
        "step_update": {
            "conversation_id": "conv-img-1",
            "step_index": 2,
            "state": "ACTIVE",
            "step_type": "subagent",
            "tool_name": "invoke_subagent",
            "subagent_info": {
                "subagents": [{"type_name": "Image-Generator", "role": "Image Generator"}],
            },
        },
    }
    assert is_allowed_image_subagent_step(subagent_active["step_update"]) is True
    parser.feed_line(json.dumps(subagent_active))

    # Allowed: system_message and agent_response text_delta
    parser.feed_line(
        json.dumps(
            {
                "event": "step_update",
                "step_update": {"step_index": 4, "state": "DONE", "step_type": "system_message"},
            }
        )
    )
    chunks = parser.feed_line(
        json.dumps(
            {
                "event": "step_update",
                "step_update": {
                    "step_index": 5,
                    "state": "DONE",
                    "step_type": "agent_response",
                    "text_delta": "Generated cat.jpg",
                },
            }
        )
    )
    assert any(c.text == "Generated cat.jpg" for c in chunks)

    parser.feed_line(
        json.dumps(
            {
                "event": "result",
                "result": {
                    "conversation_id": "conv-img-1",
                    "status": "SUCCESS",
                    "response": "Generated cat.jpg",
                },
            }
        )
    )
    assert parser.saw_result_event is True
    assert parser.conversation_id == "conv-img-1"


@pytest.mark.parametrize(
    "bad_step",
    [
        # Forbidden subagent types
        {
            "step_type": "subagent",
            "tool_name": "invoke_subagent",
            "subagent_info": {"subagents": [{"type_name": "self"}]},
        },
        {
            "step_type": "subagent",
            "tool_name": "invoke_subagent",
            "subagent_info": {"subagents": [{"type_name": "research"}]},
        },
        # Mixed list (one allowed + one forbidden)
        {
            "step_type": "subagent",
            "tool_name": "invoke_subagent",
            "subagent_info": {"subagents": [{"type_name": "image-generator"}, {"type_name": "self"}]},
        },
        # Empty / missing subagents list
        {
            "step_type": "subagent",
            "tool_name": "invoke_subagent",
            "subagent_info": {"subagents": []},
        },
        {
            "step_type": "subagent",
            "tool_name": "invoke_subagent",
        },
        # Wrong tool_name on subagent step
        {
            "step_type": "subagent",
            "tool_name": "browser_subagent",
            "subagent_info": {"subagents": [{"type_name": "image-generator"}]},
        },
        # Any direct tool call
        {"step_type": "tool", "tool_name": "run_command"},
        {"step_type": "tool", "tool_name": "read_file"},
        {"step_type": "tool", "tool_name": "call_mcp_tool"},
        {"step_type": "tool", "tool_name": "search_web"},
        {"step_type": "tool", "tool_name": "generate_image"},
        # Unnamed tool call
        {"step_type": "tool"},
    ],
)
def test_stream_parser_image_profile_aborts_unsafe_steps(bad_step: dict[str, Any]):
    parser = AntigravityStreamParser(tool_profile="image")
    with pytest.raises(AntigravityNativeToolAbort):
        parser.feed_line(json.dumps({"event": "step_update", "step_update": bad_step}))


# ---------------------------------------------------------------------------
# 2. PreToolUse hook firewall direct execution tests
# ---------------------------------------------------------------------------


def test_pre_tool_hook_firewall_fail_closed_matrix(tmp_path: Path):
    req_cwd = tmp_path / "req"
    inputs_dir = req_cwd / "inputs"
    inputs_dir.mkdir(parents=True)

    staged_img = inputs_dir / "ref_0.jpg"
    staged_img.write_bytes(_make_image_bytes("JPEG"))

    outside_secret = tmp_path / "secret.env"
    outside_secret.write_text("SECRET=1", encoding="utf-8")

    sym_escape = inputs_dir / "escape.jpg"
    sym_escape.symlink_to(outside_secret)

    sym_internal = inputs_dir / "alias.jpg"
    sym_internal.symlink_to(staged_img)

    # A) Without staged inputs allowed
    hooks_no_in, gate_no_in = write_image_pre_tool_hook(req_cwd, allow_staged_inputs=False)
    assert hooks_no_in.is_file() and gate_no_in.is_file()

    assert _run_hook(
        gate_no_in,
        json.dumps({"toolCall": {"name": "invoke_subagent", "args": {"TypeName": "image-generator"}}}),
    )["decision"] == "allow"
    assert _run_hook(
        gate_no_in,
        json.dumps({"toolCall": {"name": "invoke_subagent", "args": {"TypeName": "self"}}}),
    )["decision"] == "deny"
    assert _run_hook(
        gate_no_in,
        json.dumps({"toolCall": {"name": "invoke_subagent", "args": {"TypeName": "research"}}}),
    )["decision"] == "deny"
    assert _run_hook(
        gate_no_in,
        json.dumps({"toolCall": {"name": "read_file", "args": {"Path": "/opt/telegram-group-bot/.env"}}}),
    )["decision"] == "deny"
    assert _run_hook(
        gate_no_in,
        json.dumps({"toolCall": {"name": "read_file", "args": {"Path": str(staged_img)}}}),
    )["decision"] == "deny"
    assert _run_hook(
        gate_no_in,
        json.dumps({"toolCall": {"name": "generate_image", "args": {"Prompt": "cat", "ImagePaths": []}}}),
    )["decision"] == "allow"
    assert _run_hook(
        gate_no_in,
        json.dumps({"toolCall": {"name": "generate_image", "args": {"Prompt": "cat"}}}),
    )["decision"] == "allow"
    assert _run_hook(
        gate_no_in,
        json.dumps({"toolCall": {"name": "generate_image", "args": {"ImagePaths": ["/etc/passwd"]}}}),
    )["decision"] == "deny"
    assert _run_hook(
        gate_no_in,
        json.dumps({"toolCall": {"name": "generate_image", "args": {"ImagePaths": [str(staged_img)]}}}),
    )["decision"] == "deny"
    assert _run_hook(
        gate_no_in,
        json.dumps({"toolCall": {"name": "send_message", "args": {"Message": "done"}}}),
    )["decision"] == "allow"
    for blocked_tool in ("call_mcp_tool", "manage_task", "list_resources", "read_resource", "run_command", "write_to_file"):
        assert _run_hook(
            gate_no_in,
            json.dumps({"toolCall": {"name": blocked_tool, "args": {}}}),
        )["decision"] == "deny"
    # Malformed / non-object JSON must fail closed
    assert _run_hook(gate_no_in, "{not-valid-json")["decision"] == "deny"
    assert _run_hook(gate_no_in, "[]")["decision"] == "deny"
    assert _run_hook(gate_no_in, "{}")["decision"] == "deny"

    # B) With staged inputs allowed
    _, gate_with_in = write_image_pre_tool_hook(req_cwd, allow_staged_inputs=True)
    assert _run_hook(
        gate_with_in,
        json.dumps({"toolCall": {"name": "read_file", "args": {"Path": str(staged_img)}}}),
    )["decision"] == "allow"
    assert _run_hook(
        gate_with_in,
        json.dumps({"toolCall": {"name": "generate_image", "args": {"ImagePaths": [str(staged_img)]}}}),
    )["decision"] == "allow"
    # Symlink escapes, internal symlinks, traversal, and outside paths remain DENIED
    assert _run_hook(
        gate_with_in,
        json.dumps({"toolCall": {"name": "read_file", "args": {"Path": str(sym_escape)}}}),
    )["decision"] == "deny"
    assert _run_hook(
        gate_with_in,
        json.dumps({"toolCall": {"name": "read_file", "args": {"Path": str(sym_internal)}}}),
    )["decision"] == "deny"
    assert _run_hook(
        gate_with_in,
        json.dumps({"toolCall": {"name": "read_file", "args": {"Path": str(inputs_dir / ".." / ".agents" / "hooks.json")}}}),
    )["decision"] == "deny"
    assert _run_hook(
        gate_with_in,
        json.dumps({"toolCall": {"name": "generate_image", "args": {"ImagePaths": [str(sym_escape)]}}}),
    )["decision"] == "deny"
    assert _run_hook(
        gate_with_in,
        json.dumps({"toolCall": {"name": "read_file", "args": {"Path": str(outside_secret)}}}),
    )["decision"] == "deny"


# ---------------------------------------------------------------------------
# 3. AntigravityCLIClient.generate_image end-to-end mocked subprocess tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cli_client_generate_image_success_and_cleanup(tmp_path: Path):
    data_dir = tmp_path / "agy_home"
    out_dir = tmp_path / "img_tmp"
    conv_id = "conv-img-success-001"
    jpg_bytes = _make_image_bytes("JPEG", (64, 48))

    captured: dict[str, Any] = {}
    ref_src = tmp_path / "ref_input.png"
    ref_src.write_bytes(_make_image_bytes("PNG", (20, 20)))

    def fake_popen(cmd: list[str], *, cwd: str | Path, env: dict[str, str]):
        req_cwd = Path(cwd)
        captured["cmd"] = list(cmd)
        captured["cwd"] = req_cwd
        captured["env"] = dict(env)
        captured["hooks_existed"] = (req_cwd / ".agents" / "hooks.json").is_file()
        captured["gate_existed"] = (req_cwd / ".agents" / "image_gate.py").is_file()
        captured["staged_inputs"] = list((req_cwd / "inputs").glob("*"))

        brain_dir = Path(env["HOME"]) / ".gemini" / "antigravity-cli" / "brain" / conv_id
        brain_dir.mkdir(parents=True, exist_ok=True)
        out_img = brain_dir / "cat_space_1795085829454.jpg"
        captured["brain_img"] = out_img

        def _produce() -> None:
            out_img.write_bytes(jpg_bytes)

        lines = [
            json.dumps({"event": "init", "conversation_id": conv_id, "init": {"model": "gemini-4-hydrogen"}}) + "\n",
            json.dumps(
                {
                    "event": "step_update",
                    "step_update": {
                        "conversation_id": conv_id,
                        "step_index": 2,
                        "state": "ACTIVE",
                        "step_type": "subagent",
                        "tool_name": "invoke_subagent",
                        "subagent_info": {"subagents": [{"type_name": "image-generator"}]},
                    },
                }
            )
            + "\n",
            json.dumps(
                {
                    "event": "result",
                    "result": {
                        "conversation_id": conv_id,
                        "status": "SUCCESS",
                        "response": "Saved cat_space_1795085829454.jpg",
                        "usage": {"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 20},
                    },
                }
            )
            + "\n",
        ]
        return _FakeStreamJsonPopen(lines, on_start=_produce)

    cli = AntigravityCLIClient(data_dir=data_dir, binary="/mock/bin/agy")
    with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_popen):
        result = await cli.generate_image(
            "cat in space",
            aspect_ratio="16:9",
            reference_images=[MediaAttachment(file_path=ref_src, mime_type="image/png")],
            output_dir=out_dir,
            cleanup_artifacts=True,
        )

    assert isinstance(result, GeneratedImageResult)
    assert result.image_bytes == jpg_bytes
    assert result.mime_type == "image/jpeg"
    assert result.filename == "cat_space_1795085829454.jpg"
    assert (result.width, result.height) == (64, 48)
    assert result.conversation_id == conv_id
    assert result.prompt_used == "cat in space"
    assert result.model_used == "gemini-4-hydrogen"
    assert result.usage["input_tokens"] == 100
    assert result.usage["reasoning_tokens"] == 20

    # Transport & confinement assertions (cwd is passed directly to popen without duplicate --add-dir)
    assert "--input-format" in captured["cmd"] and "stream-json" in captured["cmd"]
    assert "acp" not in captured["cmd"]
    add_dir_indices = [i for i, tok in enumerate(captured["cmd"]) if tok == "--add-dir"]
    assert len(add_dir_indices) == 0
    assert captured["cwd"].name.startswith("bot_img_")
    assert captured["hooks_existed"] is True
    assert captured["gate_existed"] is True
    assert len(captured["staged_inputs"]) == 1
    assert not captured["staged_inputs"][0].is_symlink()

    # Guaranteed cleanup in finally
    assert not captured["cwd"].exists()
    assert not captured["brain_img"].exists()


@pytest.mark.asyncio
async def test_cli_client_generate_image_timeout_kills_process_and_cleans_up(tmp_path: Path):
    data_dir = tmp_path / "agy_home"
    out_dir = tmp_path / "img_tmp"
    captured: dict[str, Any] = {}

    def fake_popen(cmd: list[str], *, cwd: str | Path, env: dict[str, str]):
        proc = _FakeStreamJsonPopen([], block_until_killed=True)
        captured["cwd"] = Path(cwd)
        captured["proc"] = proc
        return proc

    cli = AntigravityCLIClient(data_dir=data_dir, binary="/mock/bin/agy")
    with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_popen):
        with pytest.raises(ImageTimeoutError):
            await cli.generate_image(
                "slow image",
                timeout_seconds=0.15,
                output_dir=out_dir,
            )

    assert captured["proc"].terminated is True
    assert not captured["cwd"].exists()


@pytest.mark.asyncio
async def test_cli_client_generate_image_policy_refusal(tmp_path: Path):
    data_dir = tmp_path / "agy_home"
    out_dir = tmp_path / "img_tmp"
    conv_id = "conv-refusal-001"

    def fake_popen(cmd: list[str], *, cwd: str | Path, env: dict[str, str]):
        lines = [
            json.dumps({"event": "init", "conversation_id": conv_id}) + "\n",
            json.dumps(
                {
                    "event": "result",
                    "result": {
                        "conversation_id": conv_id,
                        "status": "SUCCESS",
                        "response": "I cannot generate this image because it violates safety policy guidelines.",
                    },
                }
            )
            + "\n",
        ]
        return _FakeStreamJsonPopen(lines)

    cli = AntigravityCLIClient(data_dir=data_dir, binary="/mock/bin/agy")
    with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_popen):
        with pytest.raises(ImagePolicyRefusalError):
            await cli.generate_image("unsafe request", output_dir=out_dir)


@pytest.mark.asyncio
async def test_cli_client_generate_image_rejects_symlink_escape_and_corrupt_bytes(tmp_path: Path):
    data_dir = tmp_path / "agy_home"
    out_dir = tmp_path / "img_tmp"
    outside_img = tmp_path / "outside.png"
    outside_img.write_bytes(_make_image_bytes("PNG"))

    # Case 1: Symlink inside brain/<cid> pointing outside -> ImageValidationError
    conv_sym = "conv-symlink-001"

    def fake_popen_symlink(cmd: list[str], *, cwd: str | Path, env: dict[str, str]):
        brain_dir = Path(env["HOME"]) / ".gemini" / "antigravity-cli" / "brain" / conv_sym
        brain_dir.mkdir(parents=True, exist_ok=True)
        sym = brain_dir / "evil.png"
        if sym.exists() or sym.is_symlink():
            sym.unlink()
        sym.symlink_to(outside_img)
        lines = [
            json.dumps({"event": "init", "conversation_id": conv_sym}) + "\n",
            json.dumps({"event": "result", "result": {"conversation_id": conv_sym, "status": "SUCCESS", "response": "evil.png"}}) + "\n",
        ]
        return _FakeStreamJsonPopen(lines)

    cli = AntigravityCLIClient(data_dir=data_dir, binary="/mock/bin/agy")
    with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_popen_symlink):
        with pytest.raises(ImageValidationError):
            await cli.generate_image("test symlink", output_dir=out_dir)

    # Case 2: Corrupt image bytes inside brain/<cid> -> ImageValidationError
    conv_bad = "conv-corrupt-001"

    def fake_popen_corrupt(cmd: list[str], *, cwd: str | Path, env: dict[str, str]):
        brain_dir = Path(env["HOME"]) / ".gemini" / "antigravity-cli" / "brain" / conv_bad
        brain_dir.mkdir(parents=True, exist_ok=True)
        (brain_dir / "broken.jpg").write_bytes(b"NOT_A_REAL_JPEG_AT_ALL")
        lines = [
            json.dumps({"event": "init", "conversation_id": conv_bad}) + "\n",
            json.dumps({"event": "result", "result": {"conversation_id": conv_bad, "status": "SUCCESS", "response": "broken.jpg"}}) + "\n",
        ]
        return _FakeStreamJsonPopen(lines)

    with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_popen_corrupt):
        with pytest.raises(ImageValidationError):
            await cli.generate_image("test corrupt", output_dir=out_dir)


@pytest.mark.asyncio
async def test_cli_client_generate_image_aborts_unauthorized_subagent(tmp_path: Path):
    data_dir = tmp_path / "agy_home"
    out_dir = tmp_path / "img_tmp"
    captured: dict[str, Any] = {}

    def fake_popen(cmd: list[str], *, cwd: str | Path, env: dict[str, str]):
        lines = [
            json.dumps({"event": "init", "conversation_id": "conv-bad-sub"}) + "\n",
            json.dumps(
                {
                    "event": "step_update",
                    "step_update": {
                        "conversation_id": "conv-bad-sub",
                        "step_index": 2,
                        "state": "ACTIVE",
                        "step_type": "subagent",
                        "tool_name": "invoke_subagent",
                        "subagent_info": {"subagents": [{"type_name": "self"}]},
                    },
                }
            )
            + "\n",
        ]
        proc = _FakeStreamJsonPopen(lines)
        captured["proc"] = proc
        captured["cwd"] = Path(cwd)
        return proc

    cli = AntigravityCLIClient(data_dir=data_dir, binary="/mock/bin/agy")
    with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_popen):
        with pytest.raises(AntigravityNativeToolAbort):
            await cli.generate_image("prompt injection trying self subagent", output_dir=out_dir)

    assert captured["proc"].terminated is True
    assert not captured["cwd"].exists()


# ---------------------------------------------------------------------------
# 4. AntigravityProvider.generate_image circuit-breaker & routing tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_generate_image_uses_cli_and_honors_circuit_breaker(tmp_path: Path):
    provider = AntigravityProvider(AntigravityConfig(data_dir=tmp_path / "agy", enable_sdk=False))
    expected = GeneratedImageResult(
        image_bytes=_make_image_bytes("PNG"),
        mime_type="image/png",
        filename="ok.png",
        width=32,
        height=24,
        prompt_used="sun",
        model_used="gemini-4-hydrogen",
    )
    provider._cli_client = MagicMock()
    provider._cli_client.generate_image = AsyncMock(return_value=expected)
    # Even if an SDK client exists, generate_image must route strictly to CLI
    provider._sdk_client = MagicMock()
    provider._sdk_client.generate_response = AsyncMock(side_effect=AssertionError("SDK must not be called"))

    res = await provider.generate_image("sun", aspect_ratio="1:1")
    assert res is expected
    provider._cli_client.generate_image.assert_awaited_once()

    # Rate limit failure trips per-model cooldown
    provider._cli_client.generate_image = AsyncMock(
        side_effect=AntigravityRateLimitError("429 Too Many Requests; retry after 60s")
    )
    with pytest.raises(AntigravityRateLimitError):
        await provider.generate_image("moon", model="gemini-4-hydrogen")
    assert provider.is_rate_limited("gemini-4-hydrogen") is True

    # Subsequent call fails fast without hitting CLI
    provider._cli_client.generate_image.reset_mock()
    with pytest.raises(AntigravityRateLimitError):
        await provider.generate_image("stars", model="gemini-4-hydrogen")
    provider._cli_client.generate_image.assert_not_awaited()

    # Billing cooldown fails fast with AntigravityBillingError
    provider.reset_cooldown()
    provider.trigger_cooldown(120.0, reason="credits exhausted", category=ErrorCategory.BILLING, model="gemini-4-hydrogen")
    with pytest.raises(AntigravityBillingError):
        await provider.generate_image("planet", model="gemini-4-hydrogen")

    # Missing CLI client raises ImageGenerationError
    provider.reset_cooldown()
    provider._cli_client = None
    with pytest.raises(ImageGenerationError):
        await provider.generate_image("cloud")
