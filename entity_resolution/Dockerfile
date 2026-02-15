FROM python:3.11-slim

WORKDIR /app

COPY entity_resolution/requirements-runtime.txt entity_resolution/
RUN pip install --no-cache-dir -r entity_resolution/requirements-runtime.txt

COPY entity_resolution/ entity_resolution/
COPY shared/ shared/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
