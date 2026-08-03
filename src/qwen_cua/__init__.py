"""Qwen CUA browser agent, batch adapters, and operator-console runner."""

from typing import TYPE_CHECKING, Any

from .config import Settings

if TYPE_CHECKING:
    from .runner import RunnerManager

__all__ = ["RunnerManager", "Settings"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Keep lightweight protocol/eval imports independent of Playwright."""
    if name == "RunnerManager":
        from .runner import RunnerManager

        return RunnerManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
