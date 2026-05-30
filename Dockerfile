# Use official lightweight Python image
FROM python:3.9-slim

# Set system configurations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Create root directories
WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy production requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create runtime data directories (bypasses git empty folder check)
RUN mkdir -p data/raw data/processed models

# Copy application source directories
COPY src/ ./src/

# Pre-seed the SQLite feature store database inside the image build stage
RUN python -c "from src.core.feature_store import get_feature_store_instance; fs = get_feature_store_instance(); fs.seed_initial_data(3000)"

# Expose server port (standard Hugging Face Spaces port)
EXPOSE 7860

# CMD to launch ASGI application via Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]
