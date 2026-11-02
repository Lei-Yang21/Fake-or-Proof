FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p predicted_images uploads

ENV PORT 8080
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
