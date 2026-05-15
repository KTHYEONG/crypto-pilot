"""Project package bootstrap."""

from __future__ import annotations

import os

# Prevent JAX runtime from probing CUDA plugins in this environment.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
