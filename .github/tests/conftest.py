import pytest
import torch

from fake_or_proof.config import Settings
from fake_or_proof.model import InferenceService


@pytest.fixture(scope="session", autouse=True)
def bounded_test_runtime():
    torch.set_num_threads(2)


@pytest.fixture
def settings():
    return Settings.from_env()


@pytest.fixture(scope="session")
def service():
    return InferenceService.load(Settings.from_env())
