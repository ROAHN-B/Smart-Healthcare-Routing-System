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

# Install system dependencies (curl is required to download uv)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- PRO FIX: Install 'uv' (The blazing fast package manager) ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Copy all project files into the container
COPY . .

# --- PRO FIX: Use uv to install dependencies instantly based on uv.lock ---
RUN uv sync --frozen --no-editable

# Copy built frontend from Stage 1 to a 'static' folder in backend
COPY --from=frontend-builder /app/frontend/dist ./static

# Set PATH so the container uses the newly created uv virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app:/app/openenv_env:/app/rl
ENV PORT=7860

# Hugging Face runs on port 7860
EXPOSE 7860

# Docker Healthcheck so the OpenEnv Grader doesn't timeout
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Start FastAPI and serve everything
CMD ["sh", "-c", "python -u inference.py & uvicorn backend.api.main:app --host 0.0.0.0 --port 7860"]