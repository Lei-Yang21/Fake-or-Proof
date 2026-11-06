"""Restricted model artifacts and synchronized CPU inference."""

import hashlib
import hmac
import io
import threading
from contextlib import contextmanager

import torch
from torch import nn
from torchvision import models, transforms

from .config import (
    CLASS_LABELS,
    DEFAULT_MODEL,
    IMAGE_SIZE,
    MAX_MODEL_BYTES,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    Settings,
)

FORMAT_VERSION = 1


class ModelUnavailable(RuntimeError):
    pass


class InferenceBusy(RuntimeError):
    pass


def build_network(pretrained=False):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    network = models.mobilenet_v2(weights=weights)
    network.classifier[1] = nn.Linear(
        network.classifier[1].in_features, len(CLASS_LABELS)
    )
    return network


def image_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD
            ),
        ]
    )


def checkpoint_for(network):
    return {
        "format_version": FORMAT_VERSION,
        "architecture": "mobilenet_v2",
        "class_labels": list(CLASS_LABELS),
        "image_size": IMAGE_SIZE,
        "mean": list(NORMALIZATION_MEAN),
        "std": list(NORMALIZATION_STD),
        "state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in network.state_dict().items()
        },
    }


def load_checkpoint(source):
    # Serving must reject arbitrary Python objects without extra
    # safe_globals.
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format_version") != FORMAT_VERSION
        or checkpoint.get("architecture") != "mobilenet_v2"
        or checkpoint.get("class_labels") != list(CLASS_LABELS)
        or checkpoint.get("image_size") != IMAGE_SIZE
        or checkpoint.get("mean") != list(NORMALIZATION_MEAN)
        or checkpoint.get("std") != list(NORMALIZATION_STD)
    ):
        raise ValueError("Unsupported model contract")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Missing model weights")
    network = build_network()
    expected = network.state_dict()
    if state_dict.keys() != expected.keys():
        raise ValueError("Unexpected model weight names")
    for name, tensor in state_dict.items():
        if (
            type(tensor) is not torch.Tensor
            or tensor.shape != expected[name].shape
            or tensor.dtype != expected[name].dtype
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("Invalid model weights")
    network.load_state_dict(state_dict, strict=True)
    return network.eval()


def read_model_bytes(settings: Settings) -> bytes:
    if settings.model_uri.startswith("gs://"):
        from google.cloud import storage

        bucket, separator, object_name = settings.model_uri[5:].partition("/")
        if not bucket or not separator or not object_name:
            raise ValueError("Expected gs://bucket/object")
        blob = storage.Client().bucket(bucket).blob(object_name)
        blob.reload(timeout=30)
        if blob.size is None or blob.size > MAX_MODEL_BYTES:
            raise ValueError("Model is too large")
        contents = blob.download_as_bytes(
            if_generation_match=blob.generation, timeout=30
        )
    elif settings.model_uri == "bundled":
        with DEFAULT_MODEL.open("rb") as model_file:
            contents = model_file.read(MAX_MODEL_BYTES + 1)
    else:
        raise ValueError(
            "Only the release artifact or a GCS object can be served"
        )
    if len(contents) > MAX_MODEL_BYTES:
        raise ValueError("Model is too large")
    if not hmac.compare_digest(
        hashlib.sha256(contents).hexdigest(), settings.model_sha256
    ):
        raise ValueError(
            "Model checksum does not match the trusted configuration"
        )
    return contents


class InferenceService:
    def __init__(self, network):
        self.network = network
        self.transform = image_transform()
        self._lock = threading.Lock()

    @classmethod
    def load(cls, settings):
        torch.set_num_threads(2)
        return cls(load_checkpoint(io.BytesIO(read_model_bytes(settings))))

    @contextmanager
    def exclusive(self):
        # Hooks and gradients mutate the model, so serialize model
        # access.
        if not self._lock.acquire(blocking=False):
            raise InferenceBusy(
                "The classifier is busy. Please try again shortly."
            )
        try:
            yield
        finally:
            self._lock.release()

    def predict(self, image):
        with self.exclusive(), torch.no_grad():
            probabilities = self.network(
                self.transform(image).unsqueeze(0)
            ).softmax(dim=1)[0]
            if (
                probabilities.shape != (len(CLASS_LABELS),)
                or not torch.isfinite(probabilities).all()
            ):
                raise ModelUnavailable("Invalid model output")
            predicted_class_index = int(probabilities.argmax())
            return predicted_class_index, probabilities.tolist()

    def explain(self, image, predicted_class_index):
        from torchcam.methods import GradCAM
        from torchcam.utils import overlay_mask
        from torchvision.transforms.functional import to_pil_image

        with self.exclusive(), torch.enable_grad():
            input_tensor = (
                self.transform(image).unsqueeze(0).requires_grad_(True)
            )
            try:
                with GradCAM(
                    self.network, target_layer="features.18"
                ) as extractor:
                    scores = self.network(input_tensor)
                    activation = extractor(predicted_class_index, scores)[0]
                    heatmap = to_pil_image(
                        activation.detach().cpu(), mode="F"
                    ).resize(image.size)
                    return overlay_mask(image, heatmap, alpha=0.6)
            finally:
                self.network.zero_grad(set_to_none=True)
