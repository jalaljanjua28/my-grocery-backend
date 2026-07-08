FROM python:3.9-slim
ENV PORT=8080
CMD exec echo "Port is $PORT"
