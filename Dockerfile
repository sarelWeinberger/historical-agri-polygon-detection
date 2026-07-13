# Milestone 1 environment. GDAL/GEOS/PROJ come from the base image so that
# rasterio / geopandas / pyproj build cleanly.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # OpenCV: allow the very large historical scans
    OPENCV_IO_MAX_IMAGE_PIXELS=2000000000

RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin libgdal-dev libgeos-dev libproj-dev proj-data proj-bin \
        libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app

# Default: run the Milestone 1 dataset pipeline (mount imagery at /data)
CMD ["python", "scripts/prepare_dataset.py", "--src", "/data", "--out", "outputs/milestone1"]
