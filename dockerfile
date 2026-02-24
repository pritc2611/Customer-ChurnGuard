# ── STAGE 1: BUILDER ──────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install compilation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies into a specific folder (wheels)
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt


# ── STAGE 2: RUNNER ───────────────────────────────────────────────────
FROM python:3.11-slim AS runner

# Create a non-privileged user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install only runtime essentials (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy wheels from builder and install them
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application source and assets
# Note: Ensure ownership is set to our non-root user
COPY --chown=appuser:appuser app/ .
COPY --chown=appuser:appuser util/ ./utils/
COPY --chown=appuser:appuser template/ ./template/
COPY --chown=appuser:appuser static/ ./static/
COPY --chown=appuser:appuser data/ ./data/


EXPOSE 8000

# Fixed CMD: Using module notation
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1
