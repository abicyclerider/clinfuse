FROM python:3.11-slim

WORKDIR /app

COPY requirements-runtime.txt augmentation/
RUN pip install --no-cache-dir -r augmentation/requirements-runtime.txt

COPY . augmentation/

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python"]
