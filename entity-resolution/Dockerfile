FROM python:3.11-slim

WORKDIR /app

COPY entity-resolution/requirements-runtime.txt entity-resolution/
RUN pip install --no-cache-dir -r entity-resolution/requirements-runtime.txt

COPY entity-resolution/ entity-resolution/
COPY shared/ shared/

ENV PYTHONUNBUFFERED=1
