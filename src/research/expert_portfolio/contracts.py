"""Compatibility re-exports for the split expert-portfolio contract modules.

The frozen public surface lives in three canonical modules — ``models.py``,
``admission_types.py``, and ``admission_reports.py`` — and this facade only
re-exports their objects so the ``src.research.expert_portfolio.contracts``
import path and object identity are preserved.
"""

from __future__ import annotations

from src.research.expert_portfolio.admission_reports import (
    LibraryAdmissionBacktestReport,
    LibraryAdmissionPipelineReport,
    LibraryAdmissionReport,
)
from src.research.expert_portfolio.admission_types import (
    LIBRARY_ADMISSION_PROFILES,
    ROLLING_LIBRARY_ADMISSION_PROFILES,
    AdmissionProposal,
    CandidateAdmissionResult,
    LibraryAdmissionConfig,
    TechnicalLibraryAdmissionBacktestRequest,
    TechnicalLibraryAdmissionPipelineRequest,
    TechnicalLibraryAdmissionRequest,
    admission_proposal_id,
    expert_ids_from_admission_proposal_id,
    resolve_library_admission_profile,
    resolve_rolling_library_admission_profile,
    technical_5symbol_2022_v1_profile,
    technical_5symbol_rolling_v1_profile,
)
from src.research.expert_portfolio.models import (
    ContextualRouterSpec,
    ExpertDefinition,
    ExpertPortfolioEvaluationRequest,
    ExpertPortfolioSpec,
    lcb_z_score,
)

__all__ = [
    "LIBRARY_ADMISSION_PROFILES",
    "ROLLING_LIBRARY_ADMISSION_PROFILES",
    "AdmissionProposal",
    "CandidateAdmissionResult",
    "ContextualRouterSpec",
    "ExpertDefinition",
    "ExpertPortfolioEvaluationRequest",
    "ExpertPortfolioSpec",
    "LibraryAdmissionBacktestReport",
    "LibraryAdmissionConfig",
    "LibraryAdmissionPipelineReport",
    "LibraryAdmissionReport",
    "TechnicalLibraryAdmissionBacktestRequest",
    "TechnicalLibraryAdmissionPipelineRequest",
    "TechnicalLibraryAdmissionRequest",
    "admission_proposal_id",
    "expert_ids_from_admission_proposal_id",
    "lcb_z_score",
    "resolve_library_admission_profile",
    "resolve_rolling_library_admission_profile",
    "technical_5symbol_2022_v1_profile",
    "technical_5symbol_rolling_v1_profile",
]


def _check_contract() -> None:
    """Executable import-identity assertions locking the facade to its canon."""
    assert ExpertDefinition.__module__ == "src.research.expert_portfolio.models"
    assert ContextualRouterSpec.__module__ == "src.research.expert_portfolio.models"
    assert ExpertPortfolioSpec.__module__ == "src.research.expert_portfolio.models"
    assert (
        ExpertPortfolioEvaluationRequest.__module__
        == "src.research.expert_portfolio.models"
    )
    assert lcb_z_score.__module__ == "src.research.expert_portfolio.models"
    assert LibraryAdmissionConfig.__module__ == (
        "src.research.expert_portfolio.admission_types"
    )
    assert TechnicalLibraryAdmissionRequest.__module__ == (
        "src.research.expert_portfolio.admission_types"
    )
    assert TechnicalLibraryAdmissionBacktestRequest.__module__ == (
        "src.research.expert_portfolio.admission_types"
    )
    assert TechnicalLibraryAdmissionPipelineRequest.__module__ == (
        "src.research.expert_portfolio.admission_types"
    )
    assert LibraryAdmissionPipelineReport.__module__ == (
        "src.research.expert_portfolio.admission_reports"
    )
    assert technical_5symbol_2022_v1_profile.__module__ == (
        "src.research.expert_portfolio.admission_types"
    )
    assert resolve_library_admission_profile.__module__ == (
        "src.research.expert_portfolio.admission_types"
    )
    assert (
        resolve_rolling_library_admission_profile.__module__
        == "src.research.expert_portfolio.admission_types"
    )
    assert technical_5symbol_rolling_v1_profile.__module__ == (
        "src.research.expert_portfolio.admission_types"
    )
    assert admission_proposal_id.__module__ == (
        "src.research.expert_portfolio.admission_types"
    )
    assert expert_ids_from_admission_proposal_id.__module__ == (
        "src.research.expert_portfolio.admission_types"
    )
    assert CandidateAdmissionResult.__module__ == (
        "src.research.expert_portfolio.admission_types"
    )
    assert AdmissionProposal.__module__ == "src.research.expert_portfolio.admission_types"
    assert LibraryAdmissionReport.__module__ == (
        "src.research.expert_portfolio.admission_reports"
    )
    assert LibraryAdmissionBacktestReport.__module__ == (
        "src.research.expert_portfolio.admission_reports"
    )


_check_contract()
