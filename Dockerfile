FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen2.5-0.5B-Instruct-GGUF', filename='qwen2.5-0.5b-instruct-q4_k_m.gguf', local_dir='/app/models')"

COPY *.py ./

CMD ["python", "run.py"]
