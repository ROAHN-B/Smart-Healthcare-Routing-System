# --- STAGE 1: Build React Frontend ---
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
# We set VITE_API_URL to empty so it calls the same origin in production
RUN echo "VITE_API_URL=" > .env
RUN npm run build

# --- STAGE 2: Final Backend Image ---
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies (added curl for the healthcheck)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*


COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY openenv_env/ ./openenv_env/
COPY rl/ ./rl/
COPY backend/ ./backend/
COPY server/ ./server/


COPY inference.py pyproject.toml uv.lock ./


RUN pip install -e .


COPY --from=frontend-builder /app/frontend/dist ./static


ENV PYTHONPATH=/app:/app/openenv_env:/app/rl
ENV PORT=7860


EXPOSE 7860

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

CMD ["sh", "-c", "python -u inference.py & uvicorn backend.api.main:app --host 0.0.0.0 --port 7860"]