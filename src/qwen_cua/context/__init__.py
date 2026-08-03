"""Task-local context memory for long-running computer-use agents."""

from .memory import (
    META_CONTEXT_SKILL,
    ContextUpdateResult,
    EvolvingTaskContext,
    TaskContextSnapshot,
    context_update_from_json,
    has_context_update,
    strip_context_update,
)

__all__ = [
    "META_CONTEXT_SKILL",
    "ContextUpdateResult",
    "EvolvingTaskContext",
    "TaskContextSnapshot",
    "context_update_from_json",
    "has_context_update",
    "strip_context_update",
]
