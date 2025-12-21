FROM python:3.11-slim
WORKDIR /workspace

RUN python -m pip install --upgrade pip

COPY requirements/base.txt /req/base.txt
COPY requirements/backend.txt /req/backend.txt
RUN pip install --no-cache-dir -r /req/base.txt -r /req/backend.txt

COPY backend /workspace/backend

ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
EXPOSE $PORT

CMD ["sh", "backend/entrypoint.sh"]
