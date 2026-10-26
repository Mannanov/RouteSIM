FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SQLITE_PATH=data/routesim.db

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    libcairo2-dev \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    proj-bin \
    proj-data \
    libgeos-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY app/ /app/

RUN mkdir -p /app/log /app/static/userdata /app/data

EXPOSE 5000

CMD ["sh", "-c", "python base_init.py && exec gunicorn --bind 0.0.0.0:5000 --workers 2 wsgi:app"]
