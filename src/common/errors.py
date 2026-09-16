from __future__ import annotations


class DataIntegrityError(ValueError):
    """Fail-closed integrity error for missing, non-UTC, non-finite, or incomplete inputs.

    Raised whenever a canonical market-data input is incomplete or invalid.
    Missing costs (funding/borrow) are never replaced with zero-cost series.
    """

    pass


class ProvisioningError(ValueError):
    """Fail-closed provisioning error for workstation-to-VPS secret installs.

    Raised when a declared runtime key has no non-empty workstation source
    value, or when an in-scope key is assigned twice. The message carries
    only the missing/duplicated key name, never a secret value.
    """

    pass
