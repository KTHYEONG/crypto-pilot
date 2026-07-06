"""Centralized Smart Cache Manager with Dependency-Aware Hashing and LRU Cleanup."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


class CacheManager:
    """Manages file-based caching with automatic invalidation and storage limits."""

    def __init__(self, cache_dir: Path, max_files: int = 20, max_size_mb: float = 2000.0):
        self.cache_dir = cache_dir
        self.max_files = max_files
        self.max_size_mb = max_size_mb
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Auto-cleanup on init to enforce storage limits immediately
        self.cleanup_lru(pattern="*.parquet")

    @staticmethod
    def generate_hash(dependencies: dict[str, Any], source_files: list[Path] | None = None) -> str:
        """Generate a SHA-256 hash based on configuration and source code state.

        Args:
            dependencies: Dictionary of configuration parameters and metadata.
            source_files: List of source files whose content should be included in the hash.

        Returns:
            A short 8-character hash string.

        """
        dna = dependencies.copy()

        # Add source code hashes if provided
        if source_files:
            dna["_source_hashes"] = {}
            for f_path in source_files:
                if f_path.exists():
                    with open(f_path, "rb") as f:
                        f_hash = hashlib.sha256(f.read()).hexdigest()[:8]
                        dna["_source_hashes"][f_path.name] = f_hash

        dna_json = json.dumps(dna, sort_keys=True, default=str)
        full_hash = hashlib.sha256(dna_json.encode()).hexdigest()
        return full_hash[:8]

    def cleanup_lru(self, pattern: str = "*") -> None:
        """Remove old cache files based on Least Recently Used policy and capacity limits."""
        files = list(self.cache_dir.glob(pattern))
        if not files:
            return

        # Sort files by Access Time (atime) or Modification Time (mtime) - mtime is more reliable for cache
        # We want the oldest (smallest mtime) at the beginning
        files.sort(key=lambda x: x.stat().st_mtime)

        # 1. Limit by file count
        if len(files) > self.max_files:
            to_delete = files[: len(files) - self.max_files]
            for f in to_delete:
                try:
                    f.unlink()
                except Exception as e:
                    _logger.warning("Failed to delete cache file %s: %s", f.name, e)
            files = files[len(files) - self.max_files :]

        # 2. Limit by total size
        total_size = sum(f.stat().st_size for f in files)
        while total_size > self.max_size_mb * 1024 * 1024 and files:
            f = files.pop(0)  # Remove the oldest
            f_size = f.stat().st_size
            try:
                f.unlink()
                total_size -= f_size
            except Exception as e:
                _logger.warning("Failed to delete cache file %s: %s", f.name, e)

    def get_cache_path(self, prefix: str, suffix: str, hash_str: str) -> Path:
        """Construct a full cache path."""
        return self.cache_dir / f"{prefix}_{hash_str}{suffix}"
