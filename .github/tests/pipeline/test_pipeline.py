"""CI component checks without cloud calls or pretrained downloads."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from PIL import Image
from torch import nn

from fake_or_proof.model import checkpoint_for, load_checkpoint
from pipeline.src.compiler import compile_pipeline
from pipeline.src.config import BASE_IMAGE
from pipeline.src.data_ingestion import data_ingestion
from pipeline.src.evaluation import evaluation
from pipeline.src.preprocessing import preprocessing
from pipeline.src.training import training


class TinyMobileNet(nn.Module):
    """Use a small network with MobileNet's classifier interface."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Sequential(nn.Identity(), nn.Linear(3, 2))
        with torch.no_grad():
            self.classifier[1].weight.zero_()
            self.classifier[1].weight[1].fill_(1 / 3)
            self.classifier[1].bias.zero_()

    def forward(self, images):
        return self.classifier(self.features(images))


class RankingModel(TinyMobileNet):
    """Rank both classes correctly while predicting class zero."""

    def __init__(self, direction=1):
        super().__init__()
        with torch.no_grad():
            self.classifier[1].weight[1].fill_(direction / 3)
            self.classifier[1].bias[1] = -2


class FakeBlob:
    def __init__(self, cloud, bucket, name):
        self.cloud = cloud
        self.bucket = bucket
        self.name = name

    @property
    def key(self):
        return self.bucket, self.name

    def reload(self, **kwargs):
        return None

    @property
    def generation(self):
        return self.cloud.generations.get(self.key, 1)

    def download_as_bytes(self, **kwargs):
        self.cloud.downloads.append(self.key)
        return self.cloud.objects[self.key]

    def exists(self):
        return self.key in self.cloud.objects

    def upload_from_string(
        self, contents, content_type=None, if_generation_match=None
    ):
        expected = self.generation if self.key in self.cloud.objects else 0
        if if_generation_match is not None and if_generation_match != expected:
            raise RuntimeError("Generation precondition failed")
        self.cloud.generations[self.key] = expected + 1
        self.cloud.uploads.append(self.key)
        self.cloud.objects[self.key] = contents


class FakeBucket:
    def __init__(self, cloud, name):
        self.cloud = cloud
        self.name = name

    def blob(self, name):
        return FakeBlob(self.cloud, self.name, name)

    def list_blobs(self, prefix):
        return [
            self.blob(name)
            for bucket, name in sorted(self.cloud.objects)
            if bucket == self.name and name.startswith(prefix)
        ]


class FakeCloud:
    def __init__(self):
        self.objects = {}
        self.generations = {}
        self.downloads = []
        self.uploads = []

    def bucket(self, name):
        return FakeBucket(self, name)


def artifact(path):
    return SimpleNamespace(path=str(path), metadata={})


def metrics_artifact():
    values = {}
    return SimpleNamespace(
        log_metric=lambda name, value: values.update({name: value}),
        metadata=values,
    )


def image_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, format="PNG")
    return buffer.getvalue()


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cloud = FakeCloud()
        self.addCleanup(patch.stopall)
        patch("google.cloud.storage.Client", return_value=self.cloud).start()
        patch("torch.cuda.is_available", return_value=False).start()
        patch(
            "torchvision.models.mobilenet_v2",
            side_effect=lambda **_: TinyMobileNet(),
        ).start()
        self.csv = artifact(self.root / "ingested.csv")
        self.cleaned = artifact(self.root / "cleaned.csv")
        self.model = artifact(self.root / "model.pth")
        self.html = artifact(self.root / "report.html")
        self.manifest = artifact(self.root / "published_manifest.json")
        self.published = ("test-dataset", "models/model.pth")
        for label, folder in ((0, "fake"), (1, "real")):
            contents = image_bytes("white" if label else "black")
            for index in range(20):
                self.cloud.objects[
                    ("test-dataset", f"data/dataset/{folder}/{index:02}.PNG")
                ] = contents

    def prepare(self):
        data_ingestion.python_func("gs://test-dataset/data/dataset", self.csv)
        preprocessing.python_func(self.csv, self.cleaned, split_seed=42)
        return pd.read_csv(self.cleaned.path)

    def evaluate(self, network):
        torch.save(checkpoint_for(network), self.model.path)
        metrics = metrics_artifact()
        evaluation.python_func(
            self.model,
            self.cleaned,
            metrics,
            self.html,
            self.manifest,
            published_model_uri="gs://test-dataset/models/model.pth",
        )
        return metrics.metadata

    def store_published(self, checkpoint):
        import hashlib

        buffer = io.BytesIO()
        torch.save(checkpoint_for(checkpoint), buffer)
        self.cloud.objects[self.published] = buffer.getvalue()
        digest = hashlib.sha256(buffer.getvalue()).hexdigest()
        self.cloud.objects[(self.published[0], f"models/{digest}.pth")] = (
            buffer.getvalue()
        )

    def test_compile_has_four_tasks_and_disables_external_state_caches(self):
        output = compile_pipeline(self.root / "compiled.json")
        specification = json.loads(output.read_text(encoding="utf-8"))
        tasks = specification["root"]["dag"]["tasks"]
        self.assertEqual(
            set(tasks),
            {"data-ingestion", "preprocessing", "training", "evaluation"},
        )
        self.assertFalse(
            tasks["data-ingestion"]["cachingOptions"].get("enableCache", False)
        )
        self.assertFalse(
            tasks["evaluation"]["cachingOptions"].get("enableCache", False)
        )
        self.assertEqual(
            tasks["training"]["dependentTasks"], ["preprocessing"]
        )
        self.assertIn("training", tasks["evaluation"]["dependentTasks"])
        for executor in specification["deploymentSpec"]["executors"].values():
            self.assertEqual(executor["container"]["image"], BASE_IMAGE)

    def test_split_is_shared_reproducible_and_stratified(self):
        self.cloud.objects[
            ("test-dataset", "data/dataset/fake/broken.png")
        ] = b"broken"
        self.cloud.objects[
            ("test-dataset", "data/dataset-extra/real/outside.png")
        ] = image_bytes("white")
        self.cloud.objects[
            ("test-dataset", "data/dataset/other/real/nested.png")
        ] = image_bytes("white")
        first = self.prepare()
        self.assertEqual(len(first), 40)
        self.assertEqual(
            first["split"].value_counts().to_dict(),
            {"train": 28, "validation": 6, "test": 6},
        )
        self.assertEqual(first["image_path"].nunique(), 40)
        for name in ("train", "validation", "test"):
            self.assertEqual(
                set(first.loc[first["split"] == name, "label"]), {0, 1}
            )
        # Listing order must not change the images assigned to each
        # split.
        pd.read_csv(self.csv.path).iloc[::-1].to_csv(
            self.csv.path, index=False
        )
        preprocessing.python_func(self.csv, self.cleaned, split_seed=42)
        pd.testing.assert_frame_equal(first, pd.read_csv(self.cleaned.path))

    def test_training_saves_restricted_checkpoint_and_never_reads_test_images(
        self,
    ):
        dataframe = self.prepare()
        self.cloud.downloads.clear()
        with patch(
            "torchvision.models.mobilenet_v2",
            side_effect=lambda **_: TinyMobileNet(),
        ):
            metrics = metrics_artifact()
            training.python_func(
                self.cleaned,
                self.model,
                metrics,
                epochs=1,
                learning_rate=0.001,
                seed=42,
            )
        loaded = load_checkpoint(self.model.path)
        self.assertIsInstance(loaded, nn.Module)
        loaded.eval()
        self.assertEqual(
            tuple(loaded(torch.zeros(1, 3, 224, 224)).shape), (1, 2)
        )
        self.assertIn("validation_accuracy", metrics.metadata)
        read_uris = {
            f"gs://{bucket}/{name}" for bucket, name in self.cloud.downloads
        }
        forbidden = set(
            dataframe.loc[dataframe["split"] == "test", "image_path"]
        )
        self.assertFalse(read_uris & forbidden)

    def test_first_model_promotion_uses_probabilities_and_embeds_charts(self):
        self.prepare()
        values = self.evaluate(RankingModel())
        # Classes are all 0; probabilities rank both classes perfectly.
        self.assertEqual(values["AUC"], 1.0)
        self.assertEqual(values["F1_score"], 0.0)
        self.assertEqual(values["model_promoted"], 1.0)
        published = load_checkpoint(
            io.BytesIO(self.cloud.objects[self.published])
        )
        self.assertIsInstance(published, nn.Module)
        report = Path(self.html.path).read_text(encoding="utf-8")
        self.assertEqual(report.count("data:image/png;base64,"), 2)
        self.assertNotIn('src="confusion_matrix.png"', report)

    def test_better_incumbent_is_kept_without_rewriting(self):
        self.prepare()
        self.store_published(RankingModel())
        original = self.cloud.objects[self.published]
        values = self.evaluate(RankingModel(direction=-1))
        self.assertEqual(values["model_promoted"], 0.0)
        self.assertEqual(values["incumbent_validation_AUC"], 1.0)
        self.assertEqual(self.cloud.objects[self.published], original)
        self.assertFalse(self.cloud.uploads)

    def test_promotion_uses_validation_instead_of_test(self):
        dataframe = self.prepare()
        # Reverse test appearance only: the candidate is worse on test,
        # but better on validation.
        for row in dataframe[dataframe["split"] == "test"].itertuples(
            index=False
        ):
            bucket, name = row.image_path[5:].split("/", 1)
            self.cloud.objects[(bucket, name)] = image_bytes(
                "black" if row.label else "white"
            )
        self.store_published(RankingModel(direction=-1))
        values = self.evaluate(RankingModel())
        self.assertEqual(values["AUC"], 0.0)
        self.assertEqual(values["validation_AUC"], 1.0)
        self.assertEqual(values["model_promoted"], 1.0)

    def test_unsafe_legacy_models_are_rejected_without_publication(self):
        self.prepare()
        buffer = io.BytesIO()
        torch.save(RankingModel(), buffer)
        self.cloud.objects[self.published] = buffer.getvalue()
        import pickle

        with self.assertRaises(pickle.UnpicklingError):
            self.evaluate(RankingModel())
        self.assertFalse(self.cloud.uploads)

    def test_promoted_artifact_is_content_addressed(self):
        import hashlib

        self.prepare()
        self.evaluate(RankingModel())
        contents = self.cloud.objects[self.published]
        digest = hashlib.sha256(contents).hexdigest()
        self.assertEqual(
            self.cloud.objects[(self.published[0], f"models/{digest}.pth")],
            contents,
        )
        manifest = json.loads(Path(self.manifest.path).read_text())
        self.assertEqual(manifest["sha256"], digest)
        self.assertEqual(
            manifest["uri"], f"gs://test-dataset/models/{digest}.pth"
        )

    def test_one_class_dataset_fails_before_training(self):
        self.cloud.objects = {
            key: value
            for key, value in self.cloud.objects.items()
            if "/fake/" in key[1]
        }
        data_ingestion.python_func("gs://test-dataset/data/dataset", self.csv)
        with self.assertRaisesRegex(ValueError, "Both fake and real"):
            preprocessing.python_func(self.csv, self.cleaned)

    def test_empty_ingestion_fails_with_clear_error(self):
        with self.assertRaisesRegex(ValueError, "No images found"):
            data_ingestion.python_func("gs://test-dataset/empty", self.csv)


if __name__ == "__main__":
    unittest.main()
