FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('all-MiniLM-L6-v2')"

COPY *.py ./

CMD ["python", "run.py"]
