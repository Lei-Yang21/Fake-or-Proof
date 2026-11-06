# Build from the repository root: docker build -f pipeline/Dockerfile .
FROM python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54

WORKDIR /pipeline
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy UV_NO_CACHE=1 \
    MPLBACKEND=Agg OMP_NUM_THREADS=2 \
    PATH=/pipeline/.venv/bin:$PATH PYTHONPATH=/pipeline
RUN python -m pip install --no-cache-dir pip==26.3 uv==0.13.7
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-default-groups --group pipeline
RUN /usr/local/bin/python -m pip uninstall -y uv pip \
    && useradd --uid 10001 --create-home pipelineuser
COPY fake_or_proof/ ./fake_or_proof/
USER 10001:10001
