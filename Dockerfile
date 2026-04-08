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

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Backend Dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy RL, Environment, and new Server modules
COPY openenv_env/ ./openenv_env/
COPY rl/ ./rl/
COPY backend/ ./backend/
COPY server/ ./server/

# Copy scripts and OpenEnv configuration files
COPY inference.py pyproject.toml uv.lock ./

# Install the project as a package (This automatically installs openenv-core, openai, torch, etc.)
RUN pip install -e .

# Copy built frontend from Stage 1 to a 'static' folder in backend
COPY --from=frontend-builder /app/frontend/dist ./static

# Set Environment Variables for RL pathing
ENV PYTHONPATH=/app:/app/openenv_env:/app/rl
ENV PORT=7860

# Hugging Face runs on port 7860
EXPOSE 7860

# Start FastAPI and serve everything
CMD ["sh", "-c", "python -u inference.py & uvicorn backend.api.main:app --host 0.0.0.0 --port 7860"]