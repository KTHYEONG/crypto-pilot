FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV TZ=Asia/Seoul
ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /app

# 1. Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 2. Build & Install TA-Lib C Library
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# 3. Install Python Dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy && \
    pip install --no-cache-dir TA-Lib && \
    pip install --no-cache-dir numba && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy Project Files
COPY . .
RUN mkdir -p logs data results

# Default CMD runs nothing to allow docker-compose to override with specific bots
CMD ["python", "-c", "print('Please use docker-compose to start specific bots.')"]
