#!/usr/bin/env bash
set -euo pipefail

BOT_PROD_DIR="${BOT_PROD_DIR:-/opt/telegram-group-bot}"
BOT_STAGING_DIR="${BOT_STAGING_DIR:-/opt/telegram-group-bot-staging}"
BOT_DEPLOY_REQUEST_FILE="${BOT_DEPLOY_REQUEST_FILE:-$BOT_PROD_DIR/data/self_deploy/request.json}"
BOT_DEPLOY_RESULT_FILE="${BOT_DEPLOY_RESULT_FILE:-$BOT_PROD_DIR/data/self_deploy/result.json}"
BOT_SERVICE_NAME="${BOT_SERVICE_NAME:-telegram-userbot.service}"
BOT_CANARY_SECONDS="${BOT_CANARY_SECONDS:-18}"
BOT_MAX_DIFF_LINES="${BOT_MAX_DIFF_LINES:-400}"

git_safe() {
  git -c "safe.directory=$BOT_PROD_DIR" -c "safe.directory=$BOT_STAGING_DIR" "$@"
}

if [ ! -f "$BOT_DEPLOY_REQUEST_FILE" ]; then
  exit 0
fi

write_result() {
  local status="$1"
  local req_id="$2"
  local prev_c="$3"
  local new_c="$4"
  local err_msg="$5"
  mkdir -p "$(dirname "$BOT_DEPLOY_RESULT_FILE")"
  local tmp_res="${BOT_DEPLOY_RESULT_FILE}.tmp.$$"
  python3 - "$tmp_res" "$status" "$req_id" "$prev_c" "$new_c" "$err_msg" <<'PY'
import json, sys
from datetime import datetime, timezone
tmp_path, status, req_id, prev_c, new_c, err_msg = sys.argv[1:7]
with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "request_id": req_id,
            "status": status,
            "prev_commit": prev_c,
            "new_commit": new_c,
            "error": err_msg,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
        f,
        ensure_ascii=False,
        indent=2,
    )
PY
  mv -f "$tmp_res" "$BOT_DEPLOY_RESULT_FILE"
  if [ "$(id -u)" -eq 0 ] && id -u botops >/dev/null 2>&1; then
    chown botops:botops "$BOT_DEPLOY_RESULT_FILE" || true
    chmod 0640 "$BOT_DEPLOY_RESULT_FILE" || true
  fi
}

PREV_COMMIT=$(git_safe -C "$BOT_PROD_DIR" rev-parse HEAD)

REQ_PARSED=$(python3 - "$BOT_DEPLOY_REQUEST_FILE" <<'PY' || true
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError("not a JSON object")
    print("OK")
    print(str(d.get("request_id", "")))
    print(str(d.get("branch", "")))
    print(str(d.get("commit_sha", "")))
except Exception as exc:
    print("ERR")
    print(str(exc))
PY
)
REQ_STATUS=$(echo "$REQ_PARSED" | sed -n '1p')
rm -f "$BOT_DEPLOY_REQUEST_FILE"

if [ "$REQ_STATUS" != "OK" ]; then
  REQ_ERR=$(echo "$REQ_PARSED" | sed -n '2p')
  write_result "rejected" "" "$PREV_COMMIT" "" "Malformed request.json: $REQ_ERR"
  exit 0
fi

REQ_ID=$(echo "$REQ_PARSED" | sed -n '2p')
REQ_BRANCH=$(echo "$REQ_PARSED" | sed -n '3p')
COMMIT_SHA=$(echo "$REQ_PARSED" | sed -n '4p')

if ! [[ "$COMMIT_SHA" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
  write_result "rejected" "$REQ_ID" "$PREV_COMMIT" "$COMMIT_SHA" "Invalid commit_sha format"
  exit 0
fi

if ! git_safe -C "$BOT_STAGING_DIR" cat-file -e "${COMMIT_SHA}^{commit}" 2>/dev/null; then
  write_result "rejected" "$REQ_ID" "$PREV_COMMIT" "$COMMIT_SHA" "Commit not found in staging"
  exit 0
fi

# Protected paths check
CHANGED_FILES=$(git_safe -C "$BOT_STAGING_DIR" diff --name-only "$PREV_COMMIT" "$COMMIT_SHA")
if [ -z "$CHANGED_FILES" ]; then
  write_result "rejected" "$REQ_ID" "$PREV_COMMIT" "$COMMIT_SHA" "Empty diff"
  exit 0
fi

while IFS= read -r rel_file; do
  [ -z "$rel_file" ] && continue
  case "$rel_file" in
    .env|.env.*|data/*|deploy/bot-self-deploy.sh|systemd/*)
      write_result "rejected" "$REQ_ID" "$PREV_COMMIT" "$COMMIT_SHA" "Protected path modified: $rel_file"
      exit 0
      ;;
  esac
done <<< "$CHANGED_FILES"

# Reject symlinks pointing outside staging or to protected paths
SYMLINKS=$(git_safe -C "$BOT_STAGING_DIR" ls-tree -r "$COMMIT_SHA" | awk '$1=="120000" {print $4}')
if [ -n "$SYMLINKS" ]; then
  write_result "rejected" "$REQ_ID" "$PREV_COMMIT" "$COMMIT_SHA" "Symlinks are forbidden in self-edit commits: $SYMLINKS"
  exit 0
fi

# Reject diffs touching or referencing the isolated MTProto userbot directory
if git_safe -C "$BOT_STAGING_DIR" diff "$PREV_COMMIT" "$COMMIT_SHA" | grep -qE "^\+.*telegram-mtproto-userbot"; then
  write_result "rejected" "$REQ_ID" "$PREV_COMMIT" "$COMMIT_SHA" "Forbidden reference to telegram-mtproto-userbot in diff"
  exit 0
fi

# Diff line budget check
TOTAL_DIFF_LINES=$(git_safe -C "$BOT_STAGING_DIR" diff --numstat "$PREV_COMMIT" "$COMMIT_SHA" | awk '{a+=$1; d+=$2} END {print (a>d?a:d)+0}')
if [ "$TOTAL_DIFF_LINES" -gt "$BOT_MAX_DIFF_LINES" ]; then
  write_result "rejected" "$REQ_ID" "$PREV_COMMIT" "$COMMIT_SHA" "Diff lines ($TOTAL_DIFF_LINES) exceed max ($BOT_MAX_DIFF_LINES)"
  exit 0
fi

# Run validation suite in staging
RUFF_BIN="$BOT_STAGING_DIR/.venv/bin/ruff"
[ -x "$RUFF_BIN" ] || RUFF_BIN="$BOT_PROD_DIR/.venv/bin/ruff"
PYTEST_BIN="$BOT_STAGING_DIR/.venv/bin/pytest"
[ -x "$PYTEST_BIN" ] || PYTEST_BIN="$BOT_PROD_DIR/.venv/bin/pytest"
CLI_BIN="$BOT_STAGING_DIR/.venv/bin/telegram-group-bot"
[ -x "$CLI_BIN" ] || CLI_BIN="$BOT_PROD_DIR/.venv/bin/telegram-group-bot"

if ! (
  cd "$BOT_STAGING_DIR"
  "$RUFF_BIN" check . &&
  PYTHONPATH="$BOT_STAGING_DIR/src" "$PYTEST_BIN" -q --ignore=tests/e2e &&
  DRY_RUN_CFG="$BOT_STAGING_DIR/config/bot.yaml" &&
  { [ -f "$DRY_RUN_CFG" ] || DRY_RUN_CFG="$BOT_STAGING_DIR/config/bot.example.yaml"; } &&
  "$CLI_BIN" run --config "$DRY_RUN_CFG" --dry-run >/dev/null
); then
  write_result "tests_failed" "$REQ_ID" "$PREV_COMMIT" "$COMMIT_SHA" "Staging ruff/pytest/dry-run check failed"
  exit 0
fi

# Apply commit to prod
if ! git_safe -C "$BOT_PROD_DIR" cat-file -e "${COMMIT_SHA}^{commit}" 2>/dev/null; then
  git_safe -C "$BOT_PROD_DIR" fetch "$BOT_STAGING_DIR" "$COMMIT_SHA" >/dev/null 2>&1 || true
fi
git_safe -C "$BOT_PROD_DIR" reset --hard "$COMMIT_SHA" >/dev/null

if [ "$(id -u)" -eq 0 ] && id -u botops >/dev/null 2>&1; then
  chown -R botops:botops "$BOT_PROD_DIR/src" "$BOT_PROD_DIR/config" "$BOT_PROD_DIR/tests" 2>/dev/null || true
fi

systemctl restart "$BOT_SERVICE_NAME"

if [ "$BOT_CANARY_SECONDS" -gt 0 ] 2>/dev/null; then
  sleep "$BOT_CANARY_SECONDS"
fi

CANARY_ERR=""
if ! systemctl is-active --quiet "$BOT_SERVICE_NAME"; then
  CANARY_ERR="Service is not active after restart"
else
  JLOG=$(journalctl -u "$BOT_SERVICE_NAME" -n 50 --no-pager 2>/dev/null || true)
  if echo "$JLOG" | grep -qE "CRITICAL|Traceback \(most recent call last\)|SyntaxError|ImportError"; then
    CANARY_ERR="Critical traceback detected in journalctl"
  fi
fi

if [ -n "$CANARY_ERR" ]; then
  git_safe -C "$BOT_PROD_DIR" reset --hard "$PREV_COMMIT" >/dev/null
  if [ "$(id -u)" -eq 0 ] && id -u botops >/dev/null 2>&1; then
    chown -R botops:botops "$BOT_PROD_DIR/src" "$BOT_PROD_DIR/config" "$BOT_PROD_DIR/tests" 2>/dev/null || true
  fi
  systemctl restart "$BOT_SERVICE_NAME" || true
  write_result "rolled_back" "$REQ_ID" "$PREV_COMMIT" "$COMMIT_SHA" "$CANARY_ERR"
  exit 0
fi

write_result "applied" "$REQ_ID" "$PREV_COMMIT" "$COMMIT_SHA" ""
exit 0
