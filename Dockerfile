# Use slim Python base
FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers logs (better for Docker)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV, DocTR, and PDF/image handling

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libjpeg-dev zlib1g libxml2 poppler-utils \
    libpq-dev build-essential gcc pkg-config \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Set working directory
WORKDIR /app

# Copy dependency files first (better Docker caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port (Railway uses PORT environment variable automatically)
ENV PYTHONHASHSEED=random
ENV PORT=8080

# Start FastAPI app with Uvicorn
CMD uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 5
