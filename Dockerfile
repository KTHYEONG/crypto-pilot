FROM python:3.11-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC \
    UV_LINK_MODE=copy

# 의존성 레이어: pyproject/uv.lock 변경 시에만 재설치된다.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

# 24/7 무인 데몬(live daemon)을 PID 1로 구동한다(exec form: 종료 시그널 전달 보장).
CMD ["uv", "run", "python", "-m", "src.cli.main", "live", "daemon"]
