FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD echo "MY_ARTIFACT_ROOT=$MY_ARTIFACT_ROOT" && mlflow server \
    --host 0.0.0.0 \
    --port 7860 \
    --backend-store-uri $BACKEND_STORE_URI \
    --artifacts-destination $MY_ARTIFACT_ROOT \
    --serve-artifacts \
    --allowed-hosts atomik31-mlflow-cdsd.hf.space
