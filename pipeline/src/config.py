"""Explicit Vertex AI configuration; no cloud calls on import."""

import os
from pathlib import Path

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
SERVICE_ACCOUNT = os.environ.get("GCP_PIPELINE_SERVICE_ACCOUNT", "")
REGION = os.environ.get("GCP_REGION", "")
BUCKET_PATH = os.environ.get("FOP_DATASET_URI", "")
PIPELINE_ROOT = os.environ.get("FOP_PIPELINE_ROOT", "")
PUBLISHED_MODEL_URI = os.environ.get("FOP_PUBLISHED_MODEL_URI", "")
# Permit CI compilation with a sentinel; reject it during cloud
# submission.
BASE_IMAGE = os.environ.get(
    "PIPELINE_BASE_IMAGE", "pipeline-image-not-configured"
)
TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "build"
    / "fake_or_proof_pipeline.json"
)
