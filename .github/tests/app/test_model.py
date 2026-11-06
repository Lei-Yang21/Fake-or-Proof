import hashlib
import io
import pickle
import threading
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from fake_or_proof.config import Settings
from fake_or_proof.model import (
    InferenceBusy,
    InferenceService,
    checkpoint_for,
    load_checkpoint,
    read_model_bytes,
)


def serialize(value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    buffer.seek(0)
    return buffer


def test_bundled_artifact_integrity_and_real_prediction(service, settings):
    contents = read_model_bytes(settings)
    assert hashlib.sha256(contents).hexdigest() == settings.model_sha256
    class_index, probabilities = service.predict(Image.new("RGB", (224, 224)))
    assert class_index in (0, 1)
    assert len(probabilities) == 2 and sum(probabilities) == pytest.approx(1.0)


def test_modified_artifact_is_rejected_before_deserialization(settings):
    with patch("fake_or_proof.model.torch.load") as load:
        with pytest.raises(ValueError, match="checksum"):
            InferenceService.load(replace(settings, model_sha256="0" * 64))
        load.assert_not_called()


def test_gcs_read_is_pinned_to_the_inspected_generation(settings):
    contents = read_model_bytes(settings)
    with patch("google.cloud.storage.Client") as client:
        blob = client.return_value.bucket.return_value.blob.return_value
        blob.size, blob.generation = len(contents), 77
        blob.download_as_bytes.return_value = contents
        cloud_settings = replace(
            settings, model_uri="gs://test-models/models/checkpoint.pth"
        )
        assert read_model_bytes(cloud_settings) == contents
        blob.download_as_bytes.assert_called_once_with(
            if_generation_match=77, timeout=30
        )


def test_oversized_gcs_artifact_is_refused_before_download(settings):
    with patch("google.cloud.storage.Client") as client:
        blob = client.return_value.bucket.return_value.blob.return_value
        blob.size = 100 * 1024 * 1024
        with pytest.raises(ValueError, match="too large"):
            read_model_bytes(
                replace(
                    settings,
                    model_uri="gs://test-models/models/checkpoint.pth",
                )
            )
        blob.download_as_bytes.assert_not_called()


def test_custom_model_requires_a_trusted_hash(monkeypatch):
    monkeypatch.setenv("FOP_MODEL_URI", "gs://example/models/checkpoint.pth")
    monkeypatch.delenv("FOP_MODEL_SHA256", raising=False)
    with pytest.raises(ValueError, match="SHA-256"):
        Settings.from_env()


def test_cloud_configuration_rejects_arbitrary_filesystem_models(monkeypatch):
    monkeypatch.setenv("FOP_MODEL_URI", "some-local-model.pth")
    monkeypatch.setenv("FOP_MODEL_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="bundled or a gs://"):
        Settings.from_env()


def test_general_python_objects_are_rejected():
    class UnexpectedPython:
        def __reduce__(self):
            return eval, ("1 + 1",)

    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(serialize(UnexpectedPython()))


@pytest.mark.parametrize("corruption", ["labels", "shape", "nan"])
def test_invalid_weight_contracts_are_rejected(service, corruption):
    checkpoint = checkpoint_for(service.network)
    name = next(iter(checkpoint["state_dict"]))
    if corruption == "labels":
        checkpoint["class_labels"].reverse()
    elif corruption == "shape":
        checkpoint["state_dict"][name] = torch.zeros(1)
    else:
        checkpoint["state_dict"][name] = checkpoint["state_dict"][name].clone()
        checkpoint["state_dict"][name].view(-1)[0] = float("nan")
    with pytest.raises(ValueError):
        load_checkpoint(serialize(checkpoint))


def test_concurrent_inference_is_refused_without_waiting(service):
    outcomes = []

    def attempt_prediction():
        try:
            service.predict(Image.new("RGB", (224, 224)))
        except InferenceBusy:
            outcomes.append("busy")

    with service.exclusive():
        thread = threading.Thread(target=attempt_prediction)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert outcomes == ["busy"]


def hook_count(network):
    return sum(
        len(layer._forward_hooks) + len(layer._backward_hooks)
        for layer in network.modules()
    )


def test_gradcam_cleans_up_hooks_on_success_and_failure(service):
    image = Image.new("RGB", (224, 224))
    before = hook_count(service.network)
    assert service.explain(image, 0).size == image.size
    assert hook_count(service.network) == before
    with patch.object(
        service.network, "forward", side_effect=RuntimeError("test failure")
    ):
        with pytest.raises(RuntimeError, match="test failure"):
            service.explain(image, 0)
    assert hook_count(service.network) == before
    assert all(
        parameter.grad is None for parameter in service.network.parameters()
    )
    service.predict(image)
