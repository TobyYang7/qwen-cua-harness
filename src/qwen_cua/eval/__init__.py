"""Adapters for unattended benchmark and batch-evaluation environments."""

from __future__ import annotations

from typing import Any

__all__ = ["QwenCUAAgent", "compile_action"]


def __getattr__(name: str) -> Any:
    """Keep lightweight evaluation helpers usable without runtime dependencies."""

    if name in __all__:
        from .osworld import QwenCUAAgent, compile_action

        return {"QwenCUAAgent": QwenCUAAgent, "compile_action": compile_action}[name]
    raise AttributeError(name)
