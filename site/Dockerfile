FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from site/requirements.txt
COPY site/requirements.txt site/
RUN pip install --no-cache-dir -r site/requirements.txt

# Copy source code and data preserving structure
COPY src/ src/
COPY data/dataset/ data/dataset/
COPY site/ site/

# Set working directory to site/ for running the app
WORKDIR /app/site

RUN mkdir -p /fastf1_cache && chmod 777 /fastf1_cache

EXPOSE 5050

ENV PYTHONUNBUFFERED=1

# Command to run the application using Gunicorn (from /app/site)
CMD ["gunicorn", "--bind", "0.0.0.0:5050", "--workers", "4", "--threads", "2", "api_server:app"]
