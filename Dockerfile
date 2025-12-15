# Use slim Python base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies required for OpenCV, DocTR, and PDF/image handling

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libjpeg-dev zlib1g poppler-utils libxml2 \
    gcc libpq-dev pkg-config \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Set working directory
WORKDIR /app

# copy requirements first for layer caching
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --prefer-binary -r /app/requirements.txt

# Copy app sources
COPY . /app

ENV PYTHONHASHSEED=random \
    PORT=8080

# Start FastAPI app with Uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 5 --lifespan off"]
