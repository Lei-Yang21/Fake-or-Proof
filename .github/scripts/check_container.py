"""CI model/GradCAM checks excluded from the production image."""

import os
import subprocess
import sys
import time
import urllib.request

from PIL import Image

from fake_or_proof.config import Settings
from fake_or_proof.model import InferenceService

service = InferenceService.load(Settings.from_env())
image = Image.new("RGB", (224, 224), color="white")
class_index, probabilities = service.predict(image)
assert class_index in (0, 1) and len(probabilities) == 2
assert service.explain(image, class_index).size == image.size
print("Verified release artifact, inference and GradCAM")

server = subprocess.Popen(
    [sys.executable, "-m", "fake_or_proof.serve"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
try:
    port = int(os.environ.get("PORT", "8080"))
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(server.stdout.read())
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/_stcore/health", timeout=1
            ) as response:
                if response.status == 200:
                    print("Verified Cloud Run entry point and HTTP readiness")
                    break
        except OSError:
            time.sleep(0.2)
    else:
        raise RuntimeError("Cloud Run HTTP readiness timeout")
finally:
    server.terminate()
    server.wait(timeout=15)
