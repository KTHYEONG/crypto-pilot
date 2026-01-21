# /Dockerfile
FROM python:3.10-slim

# 보안 및 효율을 위한 환경 설정
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV TZ=Asia/Seoul

WORKDIR /app

# 시스템 의존성 설치 (필요시)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사 (이때 .dockerignore가 적용되어 CSV 등은 복사 안됨)
COPY . .

# 필요한 디렉토리 생성
RUN mkdir -p logs data

# 기본적으로 현물 봇을 실행하도록 설정 (Compose에서 덮어쓰기 가능)
CMD ["python", "src/spot_strategy/real_trader_spot.py"]
