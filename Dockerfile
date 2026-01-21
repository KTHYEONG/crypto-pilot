FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV TZ=Asia/Seoul
ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /app

# 1. 필수 빌드 도구 설치 (numba 설치를 위해 llvm 등 관련 도구 대비)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. TA-Lib C 라이브러리 설치
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# 3. 파이썬 패키지 설치
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel

# 핵심 의존성 패키지들을 명시적으로 선행 설치
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir TA-Lib
# 추가: numba 패키지 명시적 설치
RUN pip install --no-cache-dir numba

# 나머지 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 4. 소스 코드 복사 및 실행 환경 설정
COPY . .
RUN mkdir -p logs data

CMD ["python", "src/spot_strategy/real_trader_spot.py"]