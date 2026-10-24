# ChurnLens runtime image: Streamlit app (default) or the MCP server.
# Data, models and reports are NOT baked in; mount them (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# libgomp is needed by LightGBM.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY configs ./configs
COPY app ./app
COPY docs ./docs

# Run as a non-root user.
RUN useradd --create-home churnlens && chown -R churnlens /app
USER churnlens

EXPOSE 8501 8000
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
