"""Bounded anonymous, process-local session ownership for the public demo."""

from __future__ import annotations

import re
import secrets
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import Literal

from pydantic import BaseModel, ConfigDict

from backend.orchestration.models import OrchestrationResult
from backend.state import ProcessState


RECENT_MESSAGE_LIMIT = 6
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{43,128}$")


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class RecentConversation:
    def __init__(self, max_messages: int = RECENT_MESSAGE_LIMIT):
        self.max_messages = max_messages
        self._messages: deque[ConversationMessage] = deque(maxlen=max_messages)

    def append_turn(self, user_message: str, assistant_message: str) -> None:
        self._messages.append(ConversationMessage(role="user", content=user_message))
        self._messages.append(
            ConversationMessage(role="assistant", content=assistant_message)
        )

    def snapshot(self) -> list[ConversationMessage]:
        return [message.model_copy(deep=True) for message in self._messages]

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


@dataclass(slots=True)
class SessionContext:
    """All mutable process-discovery data owned by one anonymous visitor."""

    process_state: ProcessState = field(default_factory=ProcessState)
    recent_messages: RecentConversation = field(default_factory=RecentConversation)
    orchestration_result: OrchestrationResult | None = None
    created_at: float = 0.0
    last_accessed_at: float = 0.0
    chat_turns: int = 0
    analysis_runs: int = 0
    chat_requests: deque[float] = field(default_factory=deque)
    analysis_requests: deque[float] = field(default_factory=deque)
    active_requests: int = 0
    state_lock: RLock = field(default_factory=RLock, repr=False)
    usage_lock: Lock = field(default_factory=Lock, repr=False)
    discovery_guard: Lock = field(default_factory=Lock, repr=False)
    analysis_guard: Lock = field(default_factory=Lock, repr=False)

    def state_snapshot(self) -> ProcessState:
        with self.state_lock:
            return self.process_state.model_copy(deep=True)

    def reset(self) -> ProcessState:
        """Clear visitor-derived data while retaining abuse counters for this cookie."""
        with self.state_lock:
            self.process_state = ProcessState()
            self.recent_messages.clear()
            self.orchestration_result = None
            return self.process_state.model_copy(deep=True)

    def begin_request(self) -> None:
        with self.usage_lock:
            self.active_requests += 1

    def end_request(self) -> None:
        with self.usage_lock:
            self.active_requests = max(0, self.active_requests - 1)

    def is_active(self) -> bool:
        with self.usage_lock:
            return self.active_requests > 0


@dataclass(frozen=True, slots=True)
class SessionResolution:
    session_id: str
    context: SessionContext
    created: bool


class SessionStore:
    """Thread-safe bounded LRU store with lazy idle expiration."""

    def __init__(
        self,
        *,
        idle_ttl_seconds: int = 7_200,
        max_sessions: int = 100,
        recent_message_limit: int = RECENT_MESSAGE_LIMIT,
        clock: Callable[[], float] = time.monotonic,
        id_factory: Callable[[], str] | None = None,
    ):
        if idle_ttl_seconds <= 0 or max_sessions <= 0:
            raise ValueError("session TTL and capacity must be positive")
        self.idle_ttl_seconds = idle_ttl_seconds
        self.max_sessions = max_sessions
        self.recent_message_limit = recent_message_limit
        self._clock = clock
        self._id_factory = id_factory or (lambda: secrets.token_urlsafe(32))
        self._sessions: dict[str, SessionContext] = {}
        self._lock = RLock()

    def get_or_create(self, session_id: str | None) -> SessionResolution:
        now = self._clock()
        with self._lock:
            self._cleanup_expired_locked(now)
            context = self._get_valid_locked(session_id)
            if context is not None:
                context.last_accessed_at = now
                return SessionResolution(session_id=session_id or "", context=context, created=False)

            self._make_capacity_locked()
            new_id = self._new_unique_id_locked()
            context = SessionContext(
                recent_messages=RecentConversation(self.recent_message_limit),
                created_at=now,
                last_accessed_at=now,
            )
            self._sessions[new_id] = context
            return SessionResolution(session_id=new_id, context=context, created=True)

    def get(self, session_id: str | None) -> SessionContext | None:
        now = self._clock()
        with self._lock:
            self._cleanup_expired_locked(now)
            context = self._get_valid_locked(session_id)
            if context is not None:
                context.last_accessed_at = now
            return context

    def reset(self, session_id: str) -> ProcessState | None:
        context = self.get(session_id)
        return None if context is None else context.reset()

    def touch(self, session_id: str, expected_context: SessionContext | None = None) -> bool:
        with self._lock:
            context = self._sessions.get(session_id)
            if context is None or (
                expected_context is not None and context is not expected_context
            ):
                return False
            context.last_accessed_at = self._clock()
            return True

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def cleanup_expired(self) -> int:
        with self._lock:
            return self._cleanup_expired_locked(self._clock())

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def session_ids(self) -> tuple[str, ...]:
        """Expose opaque IDs only for deterministic tests and diagnostics."""
        with self._lock:
            return tuple(self._sessions)

    def _get_valid_locked(self, session_id: str | None) -> SessionContext | None:
        if not session_id or not SESSION_ID_PATTERN.fullmatch(session_id):
            return None
        return self._sessions.get(session_id)

    def _cleanup_expired_locked(self, now: float) -> int:
        expired = [
            session_id
            for session_id, context in self._sessions.items()
            if now - context.last_accessed_at >= self.idle_ttl_seconds
            and not context.is_active()
        ]
        for session_id in expired:
            del self._sessions[session_id]
        return len(expired)

    def _make_capacity_locked(self) -> None:
        while len(self._sessions) >= self.max_sessions:
            idle = [
                (session_id, context)
                for session_id, context in self._sessions.items()
                if not context.is_active()
            ]
            candidates = idle or list(self._sessions.items())
            session_id, _ = min(
                candidates,
                key=lambda item: (item[1].last_accessed_at, item[1].created_at, item[0]),
            )
            del self._sessions[session_id]

    def _new_unique_id_locked(self) -> str:
        for _ in range(10):
            candidate = self._id_factory()
            if SESSION_ID_PATTERN.fullmatch(candidate) and candidate not in self._sessions:
                return candidate
        raise RuntimeError("could not generate a valid unique session identifier")


class OperationAdmission:
    """Atomic per-session rate, lifetime-budget, and execution-guard admission."""

    def __init__(
        self,
        *,
        chat_limit: int,
        chat_window_seconds: int,
        analysis_limit: int,
        analysis_window_seconds: int,
        max_chat_turns: int,
        max_analysis_runs: int,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.limits = {"chat": chat_limit, "analysis": analysis_limit}
        self.windows = {
            "chat": chat_window_seconds,
            "analysis": analysis_window_seconds,
        }
        self.lifetime_limits = {
            "chat": max_chat_turns,
            "analysis": max_analysis_runs,
        }
        self._clock = clock

    def acquire(self, context: SessionContext, operation: Literal["chat", "analysis"]) -> str | None:
        guard = context.discovery_guard if operation == "chat" else context.analysis_guard
        if not guard.acquire(blocking=False):
            return "already_running"
        now = self._clock()
        with context.usage_lock:
            requests = context.chat_requests if operation == "chat" else context.analysis_requests
            cutoff = now - self.windows[operation]
            while requests and requests[0] <= cutoff:
                requests.popleft()
            count = context.chat_turns if operation == "chat" else context.analysis_runs
            if len(requests) >= self.limits[operation]:
                guard.release()
                return "rate_limited"
            if count >= self.lifetime_limits[operation]:
                guard.release()
                return "session_budget_exhausted"
            requests.append(now)
            if operation == "chat":
                context.chat_turns += 1
            else:
                context.analysis_runs += 1
            context.active_requests += 1
        return None

    @staticmethod
    def release(context: SessionContext, operation: Literal["chat", "analysis"]) -> None:
        guard = context.discovery_guard if operation == "chat" else context.analysis_guard
        context.end_request()
        if guard.locked():
            guard.release()


class GlobalOperationBudget:
    """Optional process-lifetime emergency fuse for accepted paid requests."""

    def __init__(self, maximum: int | None):
        self.maximum = maximum
        self.used = 0
        self._lock = Lock()

    def consume(self) -> bool:
        if self.maximum is None:
            return True
        with self._lock:
            if self.used >= self.maximum:
                return False
            self.used += 1
            return True


class IpRateLimiter:
    """Coarse, bounded network-origin throttle; never treated as identity."""

    def __init__(
        self,
        *,
        chat_limit: int,
        chat_window_seconds: int,
        analysis_limit: int,
        analysis_window_seconds: int,
        max_keys: int,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.limits = {"chat": chat_limit, "analysis": analysis_limit}
        self.windows = {
            "chat": chat_window_seconds,
            "analysis": analysis_window_seconds,
        }
        self.max_keys = max_keys
        self._clock = clock
        self._requests: dict[tuple[str, str], deque[float]] = {}
        self._last_seen: dict[tuple[str, str], float] = {}
        self._lock = Lock()

    def allow(self, address: str | None, operation: Literal["chat", "analysis"]) -> bool:
        if not address:
            return True
        now = self._clock()
        key = (operation, address)
        with self._lock:
            self._cleanup_locked(now)
            if key not in self._requests and len(self._requests) >= self.max_keys:
                oldest = min(self._last_seen, key=lambda item: self._last_seen[item])
                self._requests.pop(oldest, None)
                self._last_seen.pop(oldest, None)
            requests = self._requests.setdefault(key, deque())
            cutoff = now - self.windows[operation]
            while requests and requests[0] <= cutoff:
                requests.popleft()
            self._last_seen[key] = now
            if len(requests) >= self.limits[operation]:
                return False
            requests.append(now)
            return True

    def _cleanup_locked(self, now: float) -> None:
        expired = [
            key
            for key, seen_at in self._last_seen.items()
            if now - seen_at >= self.windows[key[0]]
        ]
        for key in expired:
            self._requests.pop(key, None)
            self._last_seen.pop(key, None)
