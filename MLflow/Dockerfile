FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/mlruns

EXPOSE 7860

CMD mlflow server \
    --host 0.0.0.0 \
    --port 7860 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ${ARTIFACT_ROOT:-/app/mlruns}
