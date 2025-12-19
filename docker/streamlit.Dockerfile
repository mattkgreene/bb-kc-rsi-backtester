FROM python:3.11-slim
WORKDIR /workspace

RUN python -m pip install --upgrade pip
COPY requirements/base.txt /req/base.txt
COPY requirements/streamlit.txt /req/streamlit.txt
RUN pip install --no-cache-dir -r /req/base.txt -r /req/streamlit.txt

# Copy code to a stable path and expose it on PYTHONPATH
COPY app /workspace/app
ENV PYTHONPATH=/workspace/app

# Use PORT env var for Railway compatibility (defaults to 8501 for local Docker)
ENV PORT=8501
EXPOSE $PORT
CMD streamlit run app/ui/app.py --server.address=0.0.0.0 --server.port=$PORT
