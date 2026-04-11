# Google Cloud Run Deployment Guide for APLSys Backend

This guide provides step-by-step instructions to deploy the APLSys Backend (FastAPI application) to Google Cloud Run.

## Prerequisites

1. **Google Cloud Account**: Active GCP account with billing enabled
2. **Google Cloud SDK**: Install [gcloud CLI](https://cloud.google.com/sdk/docs/install)
3. **Docker**: Installed locally for building containers
4. **Git**: For version control
5. **Project Access**: Access to the APLSys Backend codebase

## Step 1: Set Up Google Cloud Project

### 1.1 Create or Select a Project
```bash
# Login to Google Cloud
gcloud auth login

# Set your project ID (replace with your actual project ID)
gcloud config set project YOUR_PROJECT_ID

# Verify the project is set
gcloud config get-value project
```

### 1.2 Enable Required APIs
Enable the necessary Google Cloud APIs for your project:

```bash
# Enable Cloud Run API
gcloud services enable run.googleapis.com

# Enable Cloud Vision API (required for OCR functionality)
gcloud services enable vision.googleapis.com

# Enable Container Registry API (for storing Docker images)
gcloud services enable containerregistry.googleapis.com

# Enable Cloud Build API (for building containers)
gcloud services enable cloudbuild.googleapis.com
```

## Step 2: Prepare Service Account and Credentials

### 2.1 Create Service Account for Vision API
```bash
# Create a service account for Vision API access
gcloud iam service-accounts create aplsys-vision-sa \
    --description="Service account for APLSys Vision API access" \
    --display-name="APLSys Vision SA"

# Grant Vision API access to the service account
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:aplsys-vision-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/cloudvision.user"
```

### 2.2 Generate Service Account Key
```bash
# Generate a key for the service account
gcloud iam service-accounts keys create vision-creds.json \
    --iam-account=aplsys-vision-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

**Important**: Download and securely store the `vision-creds.json` file. This will replace your existing `VISION_CREDS.json`.

## Step 3: Prepare the Application

### 3.1 Clone and Prepare Repository
```bash
# Clone your repository
git clone https://github.com/SenArmstrong07/aplservice.git
cd aplservice

# Copy the new service account key
cp ~/vision-creds.json VISION_CREDS.json
```

### 3.2 Update Environment Variables
Create a `.env` file with your production environment variables:

```bash
# Copy the existing .env and modify as needed
cp .env .env.production

# Edit .env.production with production values
# Make sure to use production API keys and endpoints
```

### 3.3 Update Dockerfile for Production (Optional)
Your existing Dockerfile should work, but you may want to optimize it for production:

```dockerfile
# Use slim Python base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libjpeg-dev zlib1g poppler-utils libxml2 \
    gcc libpq-dev pkg-config \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --prefer-binary -r /app/requirements.txt

# Copy app sources
COPY . /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Start FastAPI app with Uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 5"]
```

## Step 4: Build and Push Docker Image

### 4.1 Authenticate Docker with GCR
```bash
# Configure Docker to use gcloud as a credential helper
gcloud auth configure-docker
```

### 4.2 Build and Push the Image
```bash
# Build the Docker image
docker build -t gcr.io/YOUR_PROJECT_ID/aplservice:latest .

# Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT_ID/aplservice:latest
```

## Step 5: Deploy to Cloud Run

### 5.1 Deploy the Service
```bash
# Deploy to Cloud Run
gcloud run deploy aplservice \
    --image gcr.io/YOUR_PROJECT_ID/aplservice:latest \
    --platform managed \
    --region asia-southeast1 \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 1 \
    --max-instances 10 \
    --timeout 300 \
    --concurrency 80
```

### 5.2 Set Environment Variables
```bash
# Set environment variables for the deployed service
gcloud run services update aplservice \
    --set-env-vars "GEMINI_API_KEY=YOUR_GEMINI_KEY" \
    --set-env-vars "OPENROUTER_API_KEY=YOUR_OPENROUTER_KEY" \
    --set-env-vars "OPENROUTER_MODEL=deepseek/deepseek-chat-v3.1:free" \
    --set-env-vars "FREEOCR=YOUR_FREEOCR_KEY" \
    --region asia-southeast1
```

**Note**: Replace the placeholder values with your actual API keys. For security, consider using Google Cloud Secret Manager for sensitive credentials.

## Step 6: Configure Service Account Permissions

### 6.1 Grant Cloud Run Service Access to Vision API
```bash
# Get the Cloud Run service account
CLOUD_RUN_SA=$(gcloud iam service-accounts list --filter="email ~ ^YOUR_PROJECT_NUMBER-compute@" --format="value(email)")

# Grant Vision API access to Cloud Run service account
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:$CLOUD_RUN_SA" \
    --role="roles/cloudvision.user"
```

## Step 7: Test the Deployment

### 7.1 Get the Service URL
```bash
# Get the deployed service URL
SERVICE_URL=$(gcloud run services describe aplservice \
    --region asia-southeast1 \
    --format="value(status.url)")

echo "Service deployed at: $SERVICE_URL"
```

### 7.2 Test Health Endpoints
```bash
# Test the root endpoint
curl -X GET "$SERVICE_URL/"

# Test Vision API health
curl -X GET "$SERVICE_URL/ocr/vision-health"

# Test OCR endpoints (replace with actual file)
curl -X POST "$SERVICE_URL/ocr/extract-full" \
    -F "file=@/path/to/test/image.png"
```

## Step 8: Configure Custom Domain (Optional)

### 8.1 Map Custom Domain
```bash
# Add custom domain
gcloud run domain-mappings create \
    --service aplservice \
    --domain your-domain.com \
    --region asia-southeast1
```

### 8.2 Configure SSL Certificate
SSL certificates are automatically provisioned by Cloud Run for custom domains.

## Step 9: Monitoring and Logging

### 9.1 View Logs
```bash
# View Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=aplservice" \
    --limit 50 \
    --format "table(timestamp,severity,textPayload)"
```

### 9.2 Set Up Monitoring
```bash
# Enable Cloud Monitoring
gcloud services enable monitoring.googleapis.com

# Create uptime check
gcloud monitoring uptime-check-configs create "APLSys Backend Uptime" \
    --resource-type=uptime-url \
    --resource-labels=host="$SERVICE_URL" \
    --check-type=GET \
    --timeout=10s
```

## Step 10: Security Best Practices

### 10.1 Use Secret Manager for API Keys
```bash
# Create secrets for sensitive data
echo -n "your-gemini-api-key" | gcloud secrets create gemini-api-key --data-file=-
echo -n "your-openrouter-key" | gcloud secrets create openrouter-api-key --data-file=-

# Grant access to Cloud Run service account
gcloud secrets add-iam-policy-binding gemini-api-key \
    --member="serviceAccount:$CLOUD_RUN_SA" \
    --role="roles/secretmanager.secretAccessor"

# Update service to use secrets
gcloud run services update aplservice \
    --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" \
    --set-secrets "OPENROUTER_API_KEY=openrouter-api-key:latest" \
    --region asia-southeast1
```

### 10.2 Configure VPC and Firewall Rules
For additional security, consider setting up VPC networks and firewall rules.

## Step 11: Cost Optimization

### 11.1 Set Resource Limits
```bash
# Update resource allocation based on usage
gcloud run services update aplservice \
    --memory 1Gi \
    --cpu 0.5 \
    --max-instances 5 \
    --region asia-southeast1
```

### 11.2 Set Up Budget Alerts
```bash
# Create budget alert
gcloud billing budgets create "APLSys Budget" \
    --billing-account=YOUR_BILLING_ACCOUNT_ID \
    --amount=100 \
    --thresholds=50,90,100
```

## Troubleshooting

### Common Issues

1. **Container Build Failures**
   - Check Docker build logs
   - Ensure all dependencies are in requirements.txt
   - Verify system dependencies in Dockerfile

2. **Vision API Authentication Errors**
   - Verify service account key is correct
   - Check service account has proper permissions

3. **Memory/CPU Issues**
   - Increase memory allocation
   - Reduce concurrency
   - Optimize model loading

4. **Timeout Errors**
   - Increase timeout settings
   - Optimize OCR processing
   - Check for memory leaks

### Useful Commands

```bash
# Check service status
gcloud run services describe aplservice --region asia-southeast1

# View service logs
gcloud logging read "resource.type=cloud_run_revision" --limit 100

# Update service
gcloud run services update aplservice --image gcr.io/project-829fc85b-1852-401e-950/aplservice:v2 --region asia-southeast1

# Delete service
gcloud run services delete aplservice --region asia-southeast1
```

## Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | Yes |
| `OPENROUTER_API_KEY` | OpenRouter API key | Yes |
| `OPENROUTER_MODEL` | OpenRouter model name | Yes |
| `FREEOCR` | Free OCR service key | Optional |
| `PORT` | Port for the application (default: 8080) | No |

## Support

For issues with this deployment:
1. Check Cloud Run logs
2. Verify API keys and permissions
3. Test locally with `docker run`
4. Review Google Cloud documentation

---

**Note**: This guide assumes you have basic familiarity with Google Cloud Platform. Replace `YOUR_PROJECT_ID` with your actual Google Cloud project ID throughout the guide.</content>
<parameter name="filePath">c:\Projects\backend\GOOGLE_CLOUD_RUN_DEPLOYMENT_GUIDE.md