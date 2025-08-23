# Multi-stage build for optimized image size and build speed

# ============================================
# Stage 1: Builder - Compile dependencies
# ============================================
FROM python:3.12-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install base dependencies first (better caching)
COPY requirements-base.txt .
RUN pip install --user --no-cache-dir -r requirements-base.txt

# Install Docker-specific requirements
COPY requirements-docker.txt .
RUN pip install --user --no-cache-dir -r requirements-docker.txt

# ============================================
# Stage 2: Runtime - Minimal final image
# ============================================
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10234

# Install only runtime dependencies (no gcc/g++)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security BEFORE copying files
RUN useradd -m -u 1000 appuser

# Copy installed packages from builder to appuser's home
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser setup.py .

# Install the package as appuser
USER appuser
RUN pip install --user --no-deps -e .

# Update PATH for appuser
ENV PATH=/home/appuser/.local/bin:$PATH

# Expose the API port
EXPOSE 10234

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:10234/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10234"]