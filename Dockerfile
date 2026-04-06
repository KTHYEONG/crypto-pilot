FROM python:3.10-slim

# System Dependencies & Build Tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib C Library Build
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Seoul \
    LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:$LD_LIBRARY_PATH \
    NUMBA_CACHE_DIR=/tmp/numba_cache

WORKDIR /app

# Install Python Dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir TA-Lib && \
    pip install --no-cache-dir -r requirements.txt

# Copy Project Files
COPY . .

# Seed strategy JSON configs outside the volume mount point.
# docker-compose command copies these into /app/results/ at container startup.
# Using COPY (not cp) so missing files cause a loud build failure, not silent skip.
COPY results/best_params_*.json /app/results_seed/

RUN mkdir -p logs data results /tmp/numba_cache && \
    chmod -R 755 /app

CMD ["python", "-c", "print('Bot is ready. Use docker-compose to start.')"]
