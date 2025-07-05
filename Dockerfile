FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir \
      fastapi uvicorn python-multipart \
      transformers accelerate sentencepiece safetensors \
      torch==2.2.2 --extra-index-url https://download.pytorch.org/whl/cpu
ENV PORT=8080
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]
