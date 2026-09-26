# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_NO_SYNC=1

ARG GIT_SHA=""
ENV GIT_SHA=${GIT_SHA}
# UV_NO_SYNC: 빌드시 --no-dev --frozen 으로 굳힌 venv를 uv run 이 컨테이너 런타임에
# 다시 동기화(=dev 의존성까지 재설치)하지 않도록 막는다. 실측(VPS): 이 변수 없이
# 컨테이너 기동 시마다 mypy/ruff 등 17개 dev 패키지를 네트워크로 재설치했다.

# 의존성 레이어: pyproject/uv.lock 변경 시에만 재설치된다.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked uv sync --frozen --no-dev --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked uv sync --frozen --no-dev

# 배포는 이 값과 실행 중인 캡처 슬롯의 값을 비교해 핸드오버 여부를 판단한다.
RUN /app/.venv/bin/python -m src.application.ops.capture_fingerprint --root /app > /app/.capture_fingerprint

# 24/7 무인 데몬(live daemon)을 PID 1로 구동한다(exec form: 종료 시그널 전달 보장).
CMD ["uv", "run", "python", "-m", "src.cli.main", "live", "daemon"]
