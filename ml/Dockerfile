# ----------------------------
# Builder stage
# ----------------------------
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --user --no-cache-dir --default-timeout=600 -r requirements.txt

# ----------------------------
# Runtime stage
# ----------------------------
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Ensure scripts in .local are in PATH
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create directories and non-root user
RUN mkdir -p /app/models /app/logs && \
    useradd -m -r appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose port and healthcheck
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
