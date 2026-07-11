FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir huggingface-hub && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen2.5-0.5B-Instruct-GGUF', filename='qwen2.5-0.5b-instruct-q4_k_m.gguf', local_dir='/app/models')"

COPY *.py ./

CMD ["python", "run.py"]
