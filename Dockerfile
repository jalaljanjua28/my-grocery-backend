# Use the official Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install system dependencies.
RUN apt-get update -qqy \
    && apt-get install -qqy \
    tesseract-ocr \
    libtesseract-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
RUN pip install --no-cache-dir Flask gunicorn \
    && pip install --no-cache-dir -r requirements.txt

# Run the application with Gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app

COPY Data-Folder ./Data-Folder
