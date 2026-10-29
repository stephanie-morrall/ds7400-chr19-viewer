FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
COPY training/results/three_class_residual_f1_v1/requirements-model.txt \
    training/results/three_class_residual_f1_v1/
RUN pip install --no-cache-dir torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r backend/requirements.txt \
    -r training/results/three_class_residual_f1_v1/requirements-model.txt

COPY backend/main.py backend/predict.py backend/training_service.py backend/
COPY training/*.py training/
COPY training/results/three_class_residual_f1_v1/weights.pt \
    training/results/three_class_residual_f1_v1/run.json \
    training/results/three_class_residual_f1_v1/validation_metrics.json \
    training/results/three_class_residual_f1_v1/test_metrics.json \
    training/results/three_class_residual_f1_v1/validation_exploration.json \
    training/results/three_class_residual_f1_v1/history.csv \
    training/results/three_class_residual_f1_v1/freeze.json \
    training/results/three_class_residual_f1_v1/baseline.pt training/results/three_class_residual_f1_v1/

ENV SQLITE_PATH=/data/annotations.db
ENV MODEL_DIR=/app/training/results/three_class_residual_f1_v1
ENV TRAINING_DATA_DIR=/app/data
ENV TRAINING_OUTPUT_DIR=/app/training-runs

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
