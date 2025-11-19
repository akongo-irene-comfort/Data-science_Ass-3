# ============================
# Stage 1: Builder
# ============================
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ============================
# Stage 2: Runtime
# ============================
FROM python:3.10-slim

LABEL maintainer="RL Traffic Control Team"
LABEL description="FastAPI inference server for DQN traffic signal control"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Make sure pip-installed scripts are visible
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# ====================================
# Copy ONLY the files that actually exist
# ====================================
COPY ml/api/ ./api/
COPY ml/utils.py ./
COPY ml/dqn_agent.py ./
COPY ml/environment.py ./
COPY ml/train.py ./
COPY ml/evaluate.py ./

# Create non-root user
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app

USER apiuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Start API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
