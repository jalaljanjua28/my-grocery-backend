# Use the official Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install Flask gunicorn
RUN pip install -r requirements.txt

#Install tesseract
RUN apt-get update -qqy && apt-get install -qqy \
        tesseract-ocr \
        libtesseract-dev

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app