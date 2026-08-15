"""Hardware configuration package (settings only; legacy re-exports moved to legacy/)."""

from src.core.settings import effective_worker_count

__all__ = ["effective_worker_count"]
