FROM python:3.10-slim

# 1. 환경 변수 설정 (라이브러리 경로 지정 필수)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV TZ=Asia/Seoul
# 중요: 시스템이 C 라이브러리를 찾을 수 있도록 경로 명시
ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /app

# 2. 필수 빌드 도구 설치
# python3-dev: TA-Lib 파이썬 래퍼 컴파일 시 헤더 파일 필요
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. TA-Lib C 라이브러리 다운로드 및 컴파일
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# 4. 파이썬 패키지 설치 (순서 중요)
COPY requirements.txt .

# (1) 빌드 도구 최신화
RUN pip install --upgrade pip setuptools wheel

# (2) Numpy 선행 설치 (TA-Lib 설치 시 필수 의존성)
RUN pip install --no-cache-dir numpy

# (3) TA-Lib 파이썬 패키지 '명시적' 설치
# requirements.txt에 있더라도 여기서 먼저 확실하게 설치합니다.
RUN pip install TA-Lib

# (4) 나머지 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY . .

# 6. 로그 폴더 생성 및 실행
RUN mkdir -p logs data
CMD ["python", "src/spot_strategy/real_trader_spot.py"]