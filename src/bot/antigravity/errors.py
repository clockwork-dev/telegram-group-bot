"""Error classification, exceptions, and backoff policies for Antigravity providers."""

from __future__ import annotations

import asyncio
import logging
import random
import re
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRY_PREFIX_RE = re.compile(
    r"(?:retry[_\s-]*after|retry\s+in|reset[s]?\s+in|try\s+again\s+in)\s*[:=]?\s*",
    re.IGNORECASE,
)
_DURATION_PART_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)(?![a-z])",
    re.IGNORECASE,
)
_BARE_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)(?![\d:.-])")
_RESETS_AT_RE = re.compile(
    r"reset[s]?\s+at\s*[:=]?\s*(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)",
    re.IGNORECASE,
)
_QUOTA_WARNING_RE = re.compile(r"^\s*(?:\S+\s+)?\[quota\]", re.IGNORECASE)
_QUOTA_PCT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_QUOTA_MODEL_RE = re.compile(r"model\s+([A-Za-z0-9._:/-]+)", re.IGNORECASE)
_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0}


def _exception_text(error: Exception | str) -> str:
    if isinstance(error, BaseException):
        parts: list[str] = []
        curr: BaseException | None = error
        seen: set[int] = set()
        while curr is not None and id(curr) not in seen:
            seen.add(id(curr))
            parts.append(str(curr))
            curr = curr.__cause__ or curr.__context__
        return " | ".join(p for p in parts if p)
    return str(error)


def _parse_iso_utc(value: str) -> datetime | None:
    raw = value.strip().replace(" ", "T")
    if raw.endswith(("Z", "z")):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def extract_retry_after(error: Exception | str) -> float | None:
    """Extract a retry-after duration in seconds (compound ``2h15m30s`` or ISO ``resets at``)."""
    if isinstance(error, BaseException):
        attr_val = getattr(error, "retry_after", None)
        if isinstance(attr_val, (int, float)) and attr_val > 0:
            return float(attr_val)
    text = _exception_text(error)

    prefix = _RETRY_PREFIX_RE.search(text)
    if prefix is not None:
        tail = text[prefix.end():]
        # Consume a contiguous run of "<num><unit>" tokens (e.g. "2h15m30s", "1h 30m").
        pos = 0
        total = 0.0
        found = False
        while True:
            m = _DURATION_PART_RE.match(tail, pos)
            if m is None:
                break
            total += float(m.group(1)) * _UNIT_SECONDS[m.group(2)[0].lower()]
            found = True
            pos = m.end()
            while pos < len(tail) and tail[pos] in " ,":
                pos += 1
        if found:
            return total if total > 0 else None
        bare = _BARE_NUMBER_RE.match(tail)
        if bare is not None:
            value = float(bare.group(1))
            return value if value > 0 else None

    at = _RESETS_AT_RE.search(text)
    if at is not None:
        dt = _parse_iso_utc(at.group(1))
        if dt is None:
            return None
        delta = (dt - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)
    return None


def is_quota_warning_line(line: str) -> bool:
    """Return True for soft ``[quota] ... low on quota`` log lines (not hard failures)."""
    low = line.lower()
    return "[quota]" in low and "low on quota" in low


def parse_quota_warning(line: str) -> tuple[str | None, float | None, str | None]:
    """Parse (model, remaining_pct, resets_at) out of a ``[quota]`` warning line."""
    model_m = _QUOTA_MODEL_RE.search(line)
    pct_m = _QUOTA_PCT_RE.search(line)
    at_m = _RESETS_AT_RE.search(line)
    pct: float | None = None
    if pct_m is not None:
        try:
            pct = float(pct_m.group(1))
        except ValueError:
            pct = None
        if pct is not None and not (0.0 <= pct <= 100.0):
            pct = None
    return (
        model_m.group(1) if model_m else None,
        pct,
        at_m.group(1) if at_m else None,
    )


class ErrorCategory(str, Enum):
    """Normalized categories of Antigravity errors."""

    RATE_LIMIT = "rate_limit"
    CONTEXT_OVERFLOW = "context_overflow"
    SERVER_ERROR = "server_error"
    BILLING = "billing"
    AUTH_PERMANENT = "auth_permanent"
    MODEL_NOT_FOUND = "model_not_found"
    NATIVE_TOOL_ABORT = "native_tool_abort"
    UNKNOWN = "unknown"


class ErrorVerdict(BaseModel):
    """Structured recovery guidance for classified errors."""

    category: ErrorCategory
    retryable: bool
    should_compress: bool
    should_fallback: bool
    message: str


class AntigravityError(Exception):
    """Base exception for Antigravity provider operations."""

    def __init__(
        self,
        message: str,
        verdict: ErrorVerdict | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.verdict = verdict or classify_error(message)
        self.retry_after = retry_after if retry_after is not None else extract_retry_after(message)


class AntigravityRateLimitError(AntigravityError):
    """429 / RESOURCE_EXHAUSTED capacity error."""


class AntigravityContextOverflowError(AntigravityError):
    """Context window or trajectory token limit exceeded."""


class AntigravityAuthError(AntigravityError):
    """Authentication or OAuth expiration error."""


class AntigravityBillingError(AntigravityError):
    """Insufficient AI credits or billing quota exhaustion."""


class AntigravityNativeToolAbort(AntigravityError):
    """Raised when agy attempts unauthorized host tool execution."""


class AntigravityServerError(AntigravityError):
    """Upstream server or output token truncation error."""


def classify_error(error: Exception | str, status_code: int | None = None) -> ErrorVerdict:
    """Classify Antigravity error string or exception into actionable recovery verdict."""
    if isinstance(error, BaseException):
        parts: list[str] = []
        curr: BaseException | None = error
        seen: set[int] = set()
        while curr is not None and id(curr) not in seen:
            seen.add(id(curr))
            parts.append(str(curr))
            curr = curr.__cause__ or curr.__context__
        full_error_str = " | ".join(p for p in parts if p) or str(error)
    else:
        full_error_str = str(error)
    msg = full_error_str.lower()

    if status_code == 429 or re.search(r"\b429\b", msg):
        return ErrorVerdict(
            category=ErrorCategory.RATE_LIMIT,
            retryable=True,
            should_compress=False,
            should_fallback=True,
            message=str(error),
        )

    # 1. Output token truncation / server error
    if any(
        p in msg
        for p in (
            "exceeded the output token limit",
            "previous response was cut off",
            "output token limit",
        )
    ):
        return ErrorVerdict(
            category=ErrorCategory.SERVER_ERROR,
            retryable=True,
            should_compress=False,
            should_fallback=False,
            message=str(error),
        )

    # 2. Context window or trajectory token overflow
    if any(
        p in msg
        for p in (
            "max_trajectory_tokens",
            "trajectory token",
            "subscriber fell behind",
            "prompt is too long",
            "context length exceeded",
            "input exceeds the maximum",
        )
    ):
        return ErrorVerdict(
            category=ErrorCategory.CONTEXT_OVERFLOW,
            retryable=True,
            should_compress=True,
            should_fallback=False,
            message=str(error),
        )

    # 3. AI credits / billing quota
    if any(
        p in msg
        for p in (
            "ai credits balance is too low",
            "credits balance",
            "insufficient credits",
            "quota has been exhausted",
            "upgrade your plan",
        )
    ):
        return ErrorVerdict(
            category=ErrorCategory.BILLING,
            retryable=False,
            should_compress=False,
            should_fallback=True,
            message=str(error),
        )

    # 4. OAuth and account verification
    if any(
        p in msg
        for p in (
            "verify your account",
            "terms of service",
            "oauth token expired",
            "not authenticated",
            "no google antigravity oauth credentials",
            "unauthenticated",
            "invalid_grant",
        )
    ):
        return ErrorVerdict(
            category=ErrorCategory.AUTH_PERMANENT,
            retryable=False,
            should_compress=False,
            should_fallback=True,
            message=str(error),
        )

    # 5. Model not found
    if status_code == 404 or re.search(r"not_found|\b404\b", msg) or any(
        p in msg
        for p in (
            "model not found",
            "unknown model",
            "unsupported model",
            "model_not_found",
        )
    ):
        return ErrorVerdict(
            category=ErrorCategory.MODEL_NOT_FOUND,
            retryable=False,
            should_compress=False,
            should_fallback=True,
            message=str(error),
        )

    # 6. Rate limits & quota
    if any(
        p in msg
        for p in (
            "resource_exhausted",
            "exhausted your capacity",
            "429 too many requests",
            "too many requests",
            "quota exceeded",
            "quota_exceeded",
            "http 429",
            "ratelimitexceeded",
            "rate limit",
        )
    ):
        return ErrorVerdict(
            category=ErrorCategory.RATE_LIMIT,
            retryable=True,
            should_compress=False,
            should_fallback=True,
            message=str(error),
        )

    # 7. Native tool interception
    if "native tool" in msg or "tool execution aborted" in msg:
        return ErrorVerdict(
            category=ErrorCategory.NATIVE_TOOL_ABORT,
            retryable=False,
            should_compress=False,
            should_fallback=False,
            message=str(error),
        )

    return ErrorVerdict(
        category=ErrorCategory.UNKNOWN,
        retryable=False,
        should_compress=False,
        should_fallback=True,
        message=str(error),
    )


def calculate_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> float:
    """Calculate exponential backoff delay with optional jitter."""
    calculated = min(max_delay, base_delay * (2 ** attempt))
    if jitter:
        jitter_val = random.uniform(0.1, 1.0)
        return min(max_delay, calculated + jitter_val)
    return calculated


async def retry_with_backoff(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    rate_limit_max_attempts: int = 1,
    retryable_categories: tuple[ErrorCategory, ...] = (
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.SERVER_ERROR,
    ),
    on_retry: Callable[[Exception, int, float], Any] | None = None,
) -> T:
    """Execute async operation with exponential backoff on retryable errors."""
    for attempt in range(max_attempts):
        try:
            return await operation()
        except Exception as exc:
            verdict = classify_error(exc)
            effective_max = (
                min(max_attempts, max(1, rate_limit_max_attempts))
                if verdict.category == ErrorCategory.RATE_LIMIT
                else max_attempts
            )
            if (
                attempt >= effective_max - 1
                or not verdict.retryable
                or verdict.category not in retryable_categories
            ):
                raise
            retry_after = extract_retry_after(exc) if verdict.category == ErrorCategory.RATE_LIMIT else None
            delay = (
                min(max_delay, retry_after)
                if retry_after is not None and retry_after > 0
                else calculate_backoff(attempt, base_delay=base_delay, max_delay=max_delay)
            )
            logger.warning(
                "Antigravity retryable error (attempt %d/%d): %s. Backing off for %.2fs",
                attempt + 1,
                effective_max,
                exc,
                delay,
            )
            if on_retry is not None:
                on_retry(exc, attempt + 1, delay)
            await asyncio.sleep(delay)

    raise RuntimeError("Unreachable retry loop exit")
