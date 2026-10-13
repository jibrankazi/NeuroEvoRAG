FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1     HF_HOME=/models/huggingface PYTHONPATH=/app
WORKDIR /app
COPY requirements-service.txt .
RUN python -m pip install --no-cache-dir -r requirements-service.txt
COPY benchmarks/__init__.py benchmarks/real_squad_qa.py benchmarks/
COPY service/ service/
RUN useradd --uid 10001 --create-home research && mkdir -p /models/huggingface && chown -R research:research /models
USER research
EXPOSE 8000
# Port 8000 must be mapped ONLY to loopback for local use; unauthenticated.
CMD ["uvicorn","service.api:app","--host","0.0.0.0","--port","8000","--workers","1"]
