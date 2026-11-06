FROM python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy UV_NO_CACHE=1 \
    PORT=8080 OMP_NUM_THREADS=2 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN python -m pip install --no-cache-dir pip==26.3 uv==0.13.7
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --group cloud \
    && python -m pip uninstall -y uv pip

COPY app.py ./
COPY fake_or_proof/ ./fake_or_proof/
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY modeling/fake_or_proof_model.pth modeling/model_manifest.json ./modeling/
RUN useradd --uid 10001 --create-home appuser
USER 10001:10001
EXPOSE 8080
CMD ["/app/.venv/bin/python", "-m", "fake_or_proof.serve"]
