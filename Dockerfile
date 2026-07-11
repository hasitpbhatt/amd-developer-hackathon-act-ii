FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_2.9.0

WORKDIR /app

RUN pip install --no-cache-dir jupyter nbconvert vllm==0.16.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tokensage.ipynb .

RUN jupyter nbconvert --to script tokensage.ipynb --output tokensage_script

CMD ["python", "tokensage_script.py"]
