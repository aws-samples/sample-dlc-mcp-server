FROM python:3.12-slim

LABEL maintainer="AWS Samples"
LABEL description="AWS Deep Learning Containers MCP Server"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY aws_samples/ ./aws_samples/

# Install the package
RUN pip install --no-cache-dir -e .

# Create non-root user for security
RUN useradd -m -u 1000 mcpuser
USER mcpuser

# Default command
ENTRYPOINT ["dlc-mcp-server"]
