"""Validated configuration without import-time cloud clients."""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "modeling" / "fake_or_proof_model.pth"
CLASS_LABELS = ("AI Generated Image", "Real Image")
IMAGE_SIZE = 224
NORMALIZATION_MEAN = (0.5, 0.5, 0.5)
NORMALIZATION_STD = (0.5, 0.5, 0.5)
MAX_MODEL_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class Settings:
    model_uri: str
    model_sha256: str
    max_upload_bytes: int = 5 * 1024 * 1024
    max_image_dimension: int = 4096
    max_image_pixels: int = 12_000_000
    history_limit: int = 100
    prediction_interval_seconds: float = 2.0

    @classmethod
    def from_env(cls):
        model_uri = os.environ.get("FOP_MODEL_URI", "bundled")
        expected_hash = os.environ.get("FOP_MODEL_SHA256", "")
        if model_uri != "bundled" and not model_uri.startswith("gs://"):
            raise ValueError(
                "FOP_MODEL_URI must be bundled or a gs://bucket/object URI"
            )
        if not expected_hash and model_uri == "bundled":
            manifest = json.loads(
                (ROOT / "modeling" / "model_manifest.json").read_text()
            )
            expected_hash = manifest["sha256"]
        if not re.fullmatch(r"[a-fA-F0-9]{64}", expected_hash):
            raise ValueError(
                "FOP_MODEL_SHA256 must be a trusted 64-character SHA-256"
            )
        upload_mb = int(os.environ.get("FOP_MAX_UPLOAD_MB", "5"))
        if not 1 <= upload_mb <= 5:
            raise ValueError("FOP_MAX_UPLOAD_MB must be between 1 and 5")
        return cls(
            model_uri=model_uri,
            model_sha256=expected_hash.lower(),
            max_upload_bytes=upload_mb * 1024 * 1024,
        )
