FROM python:3.12-slim

WORKDIR /app

# Evita arquivos .pyc
ENV PYTHONDONTWRITEBYTECODE=1

# Logs aparecem imediatamente
ENV PYTHONUNBUFFERED=1

# Instala dependências
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copia o projeto
COPY . .

# Porta da API
EXPOSE 8000

# Inicia FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]