FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .


RUN mkdir -p /fastf1_cache && chmod 777 /fastf1_cache

EXPOSE 5050

ENV PYTHONUNBUFFERED=1

# Command to run the application using Gunicorn
# Bind to 0.0.0.0 to allow external access
# 4 workers and 2 threads per worker for concurrency
CMD ["gunicorn", "--bind", "0.0.0.0:5050", "--workers", "4", "--threads", "2", "api_server:app"]
