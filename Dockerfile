FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY reciprec ./reciprec
RUN python -m pip install --no-cache-dir '.[serving]'
COPY configs ./configs
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "reciprec.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
