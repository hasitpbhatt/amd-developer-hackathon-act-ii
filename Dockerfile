FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1

WORKDIR /app

RUN pip install --no-cache-dir vllm==0.16.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tokensage.ipynb .

RUN python -c "import json; nb=json.load(open('tokensage.ipynb')); code='\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'); open('tokensage_script.py','w').write(code)"

CMD ["python", "tokensage_script.py"]
