"""Preflight real inference before opening the HTTP server."""

import os
import sys

from PIL import Image

from .config import ROOT, Settings
from .model import InferenceService


def main():
    settings = Settings.from_env()
    service = InferenceService.load(settings)
    service.predict(Image.new("RGB", (224, 224)))
    port = int(os.environ.get("PORT", "8080"))
    if not 1024 <= port <= 65535:
        raise ValueError("PORT must be between 1024 and 65535")
    os.chdir(ROOT)
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "app.py"),
            f"--server.port={port}",
            "--server.address=0.0.0.0",
        ],
    )


if __name__ == "__main__":
    main()
