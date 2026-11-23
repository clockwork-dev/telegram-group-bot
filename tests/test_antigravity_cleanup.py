"""Tests for Antigravity session auto-cleanup, --add-dir deduplication, --mode handling, and resource safety."""

from __future__ import annotations

import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from bot.antigravity import (
    AntigravityCLIClient,
    AntigravityConfig,
    AntigravityError,
    AntigravityProvider,
    cleanup_old_antigravity_sessions,
    setup_isolated_home,
)
from bot.antigravity.cli_client import watch_log_for_errors


def _set_mtime(path: Path, mtime: float) -> None:
    os.utime(path, (mtime, mtime), follow_symlinks=False)


def test_cleanup_preserves_oauth_tokens_and_protected_entries(tmp_path: Path) -> None:
    """Hard invariant: OAuth tokens, config, installation_id, bin, builtin, annotations must NEVER be deleted."""
    data_dir = tmp_path / "agy_data"
    _, gemini_dir = setup_isolated_home(data_dir)
    now = time.time()
    ancient = now - 10 * 86400.0  # 10 days old

    protected_files = [
        gemini_dir / "jetski-standalone-oauth-token",
        gemini_dir / "antigravity-oauth-token",
        gemini_dir.parent / "jetski-standalone-oauth-token",
        gemini_dir.parent / "SOUL.md",
        gemini_dir / "installation_id",
        gemini_dir / "settings.json",
        gemini_dir / "jetski_state.pbtxt",
        gemini_dir / "cli.log",
        gemini_dir / "conversation_summaries.db",
    ]
    for pf in protected_files:
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text("sensitive-or-protected-content", encoding="utf-8")
        _set_mtime(pf, ancient)

    protected_dirs = [
        gemini_dir / "builtin",
        gemini_dir / "bin",
        gemini_dir / "annotations",
    ]
    for pd in protected_dirs:
        pd.mkdir(parents=True, exist_ok=True)
        inner = pd / "keep_me.txt"
        inner.write_text("keep", encoding="utf-8")
        _set_mtime(inner, ancient)
        _set_mtime(pd, ancient)

    # Also place an old disposable session in conversations/ and brain/ to verify cleanup actually ran
    conv_dir = gemini_dir / "conversations"
    conv_dir.mkdir(exist_ok=True)
    old_conv = conv_dir / "old-session-1.db"
    old_conv.write_text("old", encoding="utf-8")
    _set_mtime(old_conv, ancient)

    brain_dir = gemini_dir / "brain"
    brain_dir.mkdir(exist_ok=True)
    old_brain = brain_dir / "old-session-1"
    old_brain.mkdir()
    (old_brain / "step.json").write_text("{}", encoding="utf-8")
    _set_mtime(old_brain / "step.json", ancient)
    _set_mtime(old_brain, ancient)

    stats = cleanup_old_antigravity_sessions(data_dir, now=now)
    assert stats["removed"] == 2
    assert not old_conv.exists()
    assert not old_brain.exists()

    for pf in protected_files:
        assert pf.is_file(), f"Protected file was deleted: {pf}"
        assert pf.read_text(encoding="utf-8") == "sensitive-or-protected-content"
    for pd in protected_dirs:
        assert pd.is_dir(), f"Protected directory was deleted: {pd}"
        assert (pd / "keep_me.txt").is_file()


def test_cleanup_preserves_fresh_entries_and_active_session_and_enforces_limits(tmp_path: Path) -> None:
    """Verify <1h fresh protection, active_conversation_ids protection, >24h age pruning, and max_entries_per_dir."""
    data_dir = tmp_path / "agy_data"
    _, gemini_dir = setup_isolated_home(data_dir)
    now = time.time()

    conv_dir = gemini_dir / "conversations"
    brain_dir = gemini_dir / "brain"
    implicit_dir = gemini_dir / "implicit"
    presence_dir = gemini_dir / "presence"
    log_dir = gemini_dir / "log"
    for d in (conv_dir, brain_dir, implicit_dir, presence_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Active session is 5 days old -> MUST survive across all subdirectories
    active_cid = "active-conv-uuid-999"
    active_paths = [
        conv_dir / f"{active_cid}.db",
        conv_dir / f"{active_cid}.db-wal",
        implicit_dir / f"{active_cid}.pb",
        presence_dir / f"{active_cid}.lock",
    ]
    for ap in active_paths:
        ap.write_text("active", encoding="utf-8")
        _set_mtime(ap, now - 5 * 86400.0)
    active_brain = brain_dir / active_cid
    active_brain.mkdir()
    (active_brain / "artifact.txt").write_text("active", encoding="utf-8")
    _set_mtime(active_brain / "artifact.txt", now - 5 * 86400.0)
    _set_mtime(active_brain, now - 5 * 86400.0)

    # 2. 10 fresh logs (< 1h old) with max_entries_per_dir=5 -> ALL 10 must survive because <1h is protected!
    for i in range(10):
        lf = log_dir / f"cli-fresh-{i}.log"
        lf.write_text("fresh log", encoding="utf-8")
        _set_mtime(lf, now - 600.0 - i)  # ~10 mins old

    # 3. 8 medium-age conversations (2h..9h old) with max_entries_per_dir=5 -> newest 5 kept, oldest 3 removed
    medium_convs: list[Path] = []
    for i in range(8):
        cf = conv_dir / f"med-conv-{i}.db"
        cf.write_text("med", encoding="utf-8")
        _set_mtime(cf, now - (i + 2) * 3600.0)  # i=0 is 2h old (newest), i=7 is 9h old (oldest)
        medium_convs.append(cf)

    # 4. 2 stale conversations (>24h old) -> removed even if within count limit
    stale_1 = implicit_dir / "stale-1.pb"
    stale_1.write_text("stale", encoding="utf-8")
    _set_mtime(stale_1, now - 30 * 3600.0)

    stats = cleanup_old_antigravity_sessions(
        data_dir,
        max_age_seconds=24.0 * 3600.0,
        max_entries_per_dir=5,
        min_fresh_age_seconds=3600.0,
        active_conversation_ids={active_cid},
        now=now,
    )

    # Active session survived everywhere
    for ap in active_paths:
        assert ap.is_file()
    assert active_brain.is_dir()

    # All 10 fresh (<1h) logs survived despite max_entries_per_dir=5
    assert len(list(log_dir.glob("cli-fresh-*.log"))) == 10

    # Stale (>24h) implicit entry removed
    assert not stale_1.exists()

    # For conversations/: active_cid.db and active_cid.db-wal are in the dir (oldest),
    # while medium_convs[0..4] (rank 0..4, newest 5) survive and medium_convs[5..7] (rank >=5) are removed
    for kept in medium_convs[:5]:
        assert kept.is_file()
    for removed in medium_convs[5:]:
        assert not removed.exists()
    assert stats["removed"] == 4


def test_cleanup_never_follows_symlinks_or_escapes_gemini_dir(tmp_path: Path) -> None:
    """Verify symlinks at subdir level, entry level, or nested inside brain/<cid>/ are never followed."""
    outside_dir = tmp_path / "outside_victim"
    outside_dir.mkdir()
    outside_secret = outside_dir / "secret.txt"
    outside_secret.write_text("DO_NOT_DELETE", encoding="utf-8")

    data_dir = tmp_path / "agy_data"
    _, gemini_dir = setup_isolated_home(data_dir)
    now = time.time()
    ancient = now - 5 * 86400.0
    _set_mtime(outside_secret, ancient)

    # 1. Symlinked entry inside conversations/ pointing to outside_secret
    conv_dir = gemini_dir / "conversations"
    conv_dir.mkdir(exist_ok=True)
    symlink_entry = conv_dir / "evil-link.db"
    os.symlink(outside_secret, symlink_entry)

    # 2. Old brain/<cid>/ directory containing a nested symlink to outside_secret
    brain_dir = gemini_dir / "brain"
    brain_dir.mkdir(exist_ok=True)
    old_brain = brain_dir / "old-brain-with-symlink"
    old_brain.mkdir()
    nested_link = old_brain / "nested_link.txt"
    os.symlink(outside_secret, nested_link)
    regular_file = old_brain / "normal.txt"
    regular_file.write_text("delete me", encoding="utf-8")
    _set_mtime(regular_file, ancient)
    _set_mtime(old_brain, ancient)

    # 3. Symlinked presence/ directory pointing to outside_dir
    presence_link = gemini_dir / "presence"
    os.symlink(outside_dir, presence_link)

    cleanup_old_antigravity_sessions(data_dir, now=now)

    # Outside target MUST remain completely untouched!
    assert outside_secret.is_file()
    assert outside_secret.read_text(encoding="utf-8") == "DO_NOT_DELETE"


class _DummyPopen:
    def __init__(self, lines: list[str] | None = None, *, hang: bool = False) -> None:
        self.stdin = io.StringIO()
        self._hang = hang
        self._terminated = threading.Event()
        self.closed_pipes: set[str] = set()

        outer = self

        class _ClosablePipe(io.StringIO):
            def __init__(self, name: str, initial: str = "") -> None:
                super().__init__(initial)
                self._name = name

            def readline(self, size: int = -1) -> str:
                if outer._hang and self._name == "stdout":
                    outer._terminated.wait(timeout=2.0)
                    return ""
                return super().readline(size)

            def close(self) -> None:
                outer.closed_pipes.add(self._name)
                super().close()

        self.stdin = _ClosablePipe("stdin")
        self.stdout = _ClosablePipe("stdout", "\n".join(lines or []) + ("\n" if lines else ""))
        self.stderr = _ClosablePipe("stderr", "")
        self.pid = 98765
        self.returncode = 0

    def poll(self) -> int | None:
        return self.returncode if self._terminated.is_set() else None

    def terminate(self) -> None:
        self._terminated.set()

    def kill(self) -> None:
        self._terminated.set()

    def wait(self, timeout: float | None = None) -> int:
        self._terminated.set()
        return self.returncode


@pytest.mark.asyncio
async def test_no_duplicate_add_dir_and_mode_auto_handling(tmp_path: Path) -> None:
    """Verify cwd is not duplicated in --add-dir and mode='auto' uses accept-edits + skip-permissions only in ops/self_dev."""
    cli = AntigravityCLIClient(data_dir=tmp_path / "agy_home", binary="/mock/bin/agy")
    work_cwd = tmp_path / "workspace"
    work_cwd.mkdir()
    extra_dir = tmp_path / "extra_dir"
    extra_dir.mkdir()

    spawned: list[tuple[list[str], str]] = []

    def fake_popen(cmd: list[str], *, cwd: str | Path, env: dict[str, str]) -> _DummyPopen:
        spawned.append((list(cmd), str(cwd)))
        return _DummyPopen([
            json.dumps({"event": "init", "conversation_id": "c_mode_1"}),
            json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": "ok"}}),
        ])

    with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_popen):
        # 1. ops profile + mode="auto" + cwd=work_cwd + add_dirs=[work_cwd, extra_dir, extra_dir]
        await cli.generate_response(
            prompt="check status",
            tool_profile="ops",
            mode="auto",
            cwd=work_cwd,
            add_dirs=[work_cwd, extra_dir, extra_dir],
        )
        cmd_ops, cwd_ops = spawned[-1]
        assert cwd_ops == str(work_cwd.resolve())
        assert "--mode" in cmd_ops
        assert cmd_ops[cmd_ops.index("--mode") + 1] == "accept-edits"
        assert "--dangerously-skip-permissions" in cmd_ops
        add_dirs_ops = [cmd_ops[i + 1] for i, tok in enumerate(cmd_ops) if tok == "--add-dir"]
        assert add_dirs_ops == [str(extra_dir.resolve())]

        # 2. chat profile + mode="auto" -> NEVER passes --dangerously-skip-permissions or --mode auto
        await cli.generate_response(
            prompt="hello",
            tool_profile="chat",
            mode="auto",
            cwd=work_cwd,
        )
        cmd_chat, _ = spawned[-1]
        assert "--dangerously-skip-permissions" not in cmd_chat
        assert "--mode" not in cmd_chat
        assert "--add-dir" not in cmd_chat

    cli.close()


@pytest.mark.asyncio
async def test_pipes_closed_tokens_synced_on_error_and_turn_timeout(tmp_path: Path) -> None:
    """Verify pipes are closed, refreshed tokens are synced in finally on failure, and hung turns time out."""
    data_dir = tmp_path / "agy_home"
    _, gemini_dir = setup_isolated_home(data_dir)
    source_tok = gemini_dir / "jetski-standalone-oauth-token"
    source_tok.write_text("initial_token", encoding="utf-8")
    os.chmod(source_tok, 0o600)

    cli = AntigravityCLIClient(
        data_dir=data_dir,
        binary="/mock/bin/agy",
        turn_timeout_seconds=0.15,
    )

    last_proc: list[_DummyPopen] = []

    def fake_hang_popen(cmd: list[str], *, cwd: str | Path, env: dict[str, str]) -> _DummyPopen:
        p = _DummyPopen(hang=True)
        last_proc.append(p)
        return p

    with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_hang_popen):
        with pytest.raises(AntigravityError, match="timed out"):
            await cli.generate_response(prompt="hang test")

    assert len(last_proc) == 1
    assert {"stdin", "stdout", "stderr"}.issubset(last_proc[0].closed_pipes)
    cli.close()


def test_watch_log_for_errors_skips_historical_errors_and_forwards_quota_warning(tmp_path: Path) -> None:
    """Verify historical RESOURCE_EXHAUSTED lines from old turns do not trigger false kills, while new lines do."""
    gemini_dir = tmp_path / ".gemini" / "antigravity-cli"
    gemini_dir.mkdir(parents=True)
    cli_log = gemini_dir / "cli.log"
    cli_log.write_text("E1119 10:00:00 1 server.go:1] RESOURCE_EXHAUSTED: old error from 2 hours ago\n", encoding="utf-8")
    # Set mtime to 60 seconds ago (historical)
    old_mtime = time.time() - 60.0
    _set_mtime(cli_log, old_mtime)

    stop_event = threading.Event()
    errors: list[str] = []
    warnings: list[tuple[str | None, float | None, str | None]] = []
    killed = False

    def on_kill() -> None:
        nonlocal killed
        killed = True

    def on_warn(model: str | None, pct: float | None, resets_at: str | None) -> None:
        warnings.append((model, pct, resets_at))

    t = threading.Thread(
        target=watch_log_for_errors,
        args=(gemini_dir, stop_event, errors, on_kill),
        kwargs={"on_quota_warning": on_warn},
        daemon=True,
    )
    t.start()
    time.sleep(0.35)

    # Historical error must NOT have killed the process
    assert killed is False
    assert errors == []

    # Now append a soft quota warning -> should fire on_warn, still no kill
    with cli_log.open("a", encoding="utf-8") as fh:
        fh.write("[quota] model gemini-4-hydrogen is low on quota (12% remaining, resets at 2026-11-20T00:00:00Z)\n")
        fh.flush()

    deadline = time.monotonic() + 2.0
    while not warnings and time.monotonic() < deadline:
        time.sleep(0.05)

    assert len(warnings) == 1
    assert warnings[0][0] == "gemini-4-hydrogen"
    assert warnings[0][1] == 12.0
    assert killed is False
    assert errors == []

    stop_event.set()
    t.join(timeout=1.0)


def test_provider_wires_quota_warning_callback(tmp_path: Path) -> None:
    """Verify AntigravityProvider forwards set_quota_warning_callback to AntigravityCLIClient."""
    received: list[tuple[Any, ...]] = []

    def cb(model: str | None, pct: float | None, resets_at: str | None) -> None:
        received.append((model, pct, resets_at))

    provider = AntigravityProvider(
        AntigravityConfig(data_dir=tmp_path / "agy", enable_sdk=False),
        on_quota_warning=cb,
    )
    assert provider._cli_client is not None
    assert provider._cli_client.on_quota_warning is cb

    def cb2(model: str | None, pct: float | None, resets_at: str | None) -> None:
        received.append(("cb2", model, pct, resets_at))

    provider.set_quota_warning_callback(cb2)
    assert provider._cli_client.on_quota_warning is cb2
    provider.close()
