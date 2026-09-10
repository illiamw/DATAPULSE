FROM python:3.12-slim

# Evita arquivos .pyc
ENV PYTHONDONTWRITEBYTECODE=1

# Logs aparecem imediatamente
ENV PYTHONUNBUFFERED=1

WORKDIR /app_mlops

COPY requirements.txt .

RUN pip install --upgrade pip\
    && pip install --no-cache-dir -r requirements.txt\
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . .


# Porta da API
EXPOSE 8000

# Inicia FastAPI
CMD ["python","-m","uvicorn", "src.api.main:app_mlops", "--host", "0.0.0.0", "--port", "8000"]