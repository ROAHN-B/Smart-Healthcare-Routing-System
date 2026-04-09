FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./

RUN echo "VITE_API_URL=" > .env
RUN npm run build


FROM python:3.10-slim
WORKDIR /app


RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*


RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx


COPY . .


RUN uv sync --frozen --no-editable


COPY --from=frontend-builder /app/frontend/dist ./static


ENV PATH="/app/.venv/bin:$PATH"


RUN pip install mysql-connector-python


ENV PYTHONPATH=/app:/app/openenv_env:/app/rl
ENV PORT=7860


EXPOSE 7860

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Start FastAPI and serve everything
CMD ["sh", "-c", "python -u inference.py & uvicorn backend.api.main:app --host 0.0.0.0 --port 7860"]