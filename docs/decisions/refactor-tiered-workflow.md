# ADR: Refactor Tiered Workflow to Subpackage

- **Context**: `tiered_workflow.py`가 3,000라인을 초과하여 높은 복잡도와 낮은 가독성 문제를 야기함.
- **Decision**: 도메인 역할(metrics, diagnostics, awf_sim, signal_selection, pipeline)에 맞춰 7개의 소형 서브모듈 패키지로 분할함.
- **Rationale**: 기존 패키지 진입점(`__init__.py`)에서 함수 및 logger를 Re-export하여 100% 하위 호환성을 확보하고 회귀 테스트가 정상 작동하도록 함.
