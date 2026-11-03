FROM python:3.11-slim

WORKDIR /app

# Install pinned dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application + the saved model artifact (no training at build time)
COPY app ./app
COPY tests ./tests
COPY pytest.ini .
COPY models /app/models

ENV MODEL_PATH=/app/models/best_price_model.pkl
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
