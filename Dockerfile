FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    PORT=8080

# 1. Install g++ and build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# If this step fails because Neuroglancer has no ready-made package for this Python,
# add:  RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt && python -c "import neuroglancer, h5py, scipy, networkx; print('imports ok')"

COPY src/ ./

# The 180 MB CREMI file and the edit log live here, NOT in the image.
VOLUME ["/data"]
EXPOSE 8080
CMD ["python", "app.py"]
