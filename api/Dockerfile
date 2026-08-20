FROM python:3.12-slim

WORKDIR /app

# Instalar uv para resolución rápida de dependencias en contenedor
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copiar archivos de configuración de dependencias y código
COPY pyproject.toml ./
COPY src/ ./src/
COPY api/ ./api/

# Instalar proyecto
RUN uv pip install --system -e .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
