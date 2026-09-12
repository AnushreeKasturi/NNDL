FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /tmp/root-requirements.txt
COPY pyproject.toml /tmp/pyproject.toml
COPY src /tmp/src
COPY apps/api/requirements.txt /tmp/api-requirements.txt

RUN pip install --no-cache-dir -r /tmp/root-requirements.txt && \
    pip install --no-cache-dir -r /tmp/api-requirements.txt && \
    pip install --no-cache-dir -e /tmp

COPY apps/api /app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

