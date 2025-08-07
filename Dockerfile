# Usar imagem base com Python
FROM python:3.10-slim

# Diretório de trabalho
WORKDIR /app

# Copiar arquivos para dentro do container
COPY . .

# Instalar dependências
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Expõe a porta 8000
EXPOSE 8000

# Comando para iniciar a API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

