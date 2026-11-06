from kfp.dsl import (
    HTML,
    Artifact,
    Dataset,
    Input,
    Metrics,
    Model,
    Output,
    component,
)

from .config import BASE_IMAGE


@component(base_image=BASE_IMAGE, install_kfp_package=False)
def evaluation(
    model: Input[Model],
    preprocessed_dataset: Input[Dataset],
    metrics: Output[Metrics],
    html: Output[HTML],
    published_manifest: Output[Artifact],
    published_model_uri: str,
):
    """Report test metrics; promote models using validation AUC."""
    import base64
    import hashlib
    import io
    import json
    import logging
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from google.cloud import storage
    from PIL import Image
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from torch.utils.data import DataLoader, Dataset

    from fake_or_proof.model import (
        checkpoint_for,
        image_transform,
        load_checkpoint,
    )

    if (
        not published_model_uri.startswith("gs://")
        or "/" not in published_model_uri[5:]
    ):
        raise ValueError(
            "published_model_uri must be a gs://bucket/object URI"
        )
    transform = image_transform()
    client = storage.Client()

    class GCSImageDataset(Dataset):
        def __init__(self, dataframe):
            self.data = dataframe.reset_index(drop=True)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            row = self.data.iloc[index]
            bucket_name, blob_name = row["image_path"][5:].split("/", 1)
            contents = (
                client.bucket(bucket_name).blob(blob_name).download_as_bytes()
            )
            with Image.open(io.BytesIO(contents)) as image:
                tensor = transform(image.convert("RGB"))
            return tensor, int(row["label"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def score(network, loader):
        labels, predictions, probabilities = [], [], []
        with torch.no_grad():
            for images, batch_labels in loader:
                outputs = network(images.to(device))
                predictions.extend(outputs.argmax(dim=1).cpu().tolist())
                probabilities.extend(
                    outputs.softmax(dim=1)[:, 1].cpu().tolist()
                )
                labels.extend(batch_labels.tolist())
        return (
            np.asarray(labels),
            np.asarray(predictions),
            np.asarray(probabilities),
        )

    dataframe = pd.read_csv(preprocessed_dataset.path)
    if not {"image_path", "label", "split"}.issubset(dataframe.columns):
        raise ValueError(
            "Expected the preprocessed CSV with a persisted split column"
        )
    validation = dataframe[dataframe["split"] == "validation"]
    test = dataframe[dataframe["split"] == "test"]
    if set(validation["label"]) != {0, 1} or set(test["label"]) != {0, 1}:
        raise ValueError(
            "Validation and test partitions must each contain both classes"
        )
    validation_loader = DataLoader(GCSImageDataset(validation), batch_size=32)
    test_loader = DataLoader(GCSImageDataset(test), batch_size=32)
    candidate = load_checkpoint(model.path).to(device)
    val_labels, _, val_probabilities = score(candidate, validation_loader)
    validation_auc = float(roc_auc_score(val_labels, val_probabilities))
    labels, predictions, probabilities = score(candidate, test_loader)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, _, _ = cm.ravel()
    values = {
        "F1_score": float(f1_score(labels, predictions, zero_division=0)),
        "AUC": float(roc_auc_score(labels, probabilities)),
        "Sensitivity": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "Specificity": float(tn / (tn + fp)),
        "Precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "validation_AUC": validation_auc,
    }
    for name, value in values.items():
        metrics.log_metric(name, value)

    def embed_figure(figure):
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(figure)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.imshow(cm, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(
                column, row, str(cm[row, column]), ha="center", va="center"
            )
    class_names = ["AI-Generated", "Real"]
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="Actual",
        title="Test confusion matrix",
    )
    cm_image = embed_figure(figure)
    fpr, tpr, _ = roc_curve(labels, probabilities)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, label=f"AUC = {values['AUC']:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--")
    axis.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Test ROC",
    )
    axis.legend()
    roc_image = embed_figure(figure)

    bucket_name, blob_name = published_model_uri[5:].split("/", 1)
    blob = client.bucket(bucket_name).blob(blob_name)
    old_auc = None
    incumbent_generation = 0
    incumbent_contents = None
    if blob.exists():
        blob.reload()
        incumbent_generation = blob.generation
        incumbent_contents = blob.download_as_bytes(
            if_generation_match=incumbent_generation
        )
        incumbent = load_checkpoint(io.BytesIO(incumbent_contents)).to(device)
        old_labels, _, old_probabilities = score(incumbent, validation_loader)
        old_auc = float(roc_auc_score(old_labels, old_probabilities))
        metrics.log_metric("incumbent_validation_AUC", old_auc)

    promoted = old_auc is None or validation_auc > old_auc
    if promoted:
        buffer = io.BytesIO()
        torch.save(checkpoint_for(candidate.cpu().eval()), buffer)
        contents = buffer.getvalue()
    else:
        contents = incumbent_contents
    digest = hashlib.sha256(contents).hexdigest()
    immutable_name = str(Path(blob_name).parent / f"{digest}.pth").replace(
        "\\", "/"
    )
    immutable_blob = client.bucket(bucket_name).blob(immutable_name)
    if not immutable_blob.exists():
        immutable_blob.upload_from_string(
            contents,
            content_type="application/octet-stream",
            if_generation_match=0,
        )
    elif (
        hashlib.sha256(immutable_blob.download_as_bytes()).hexdigest()
        != digest
    ):
        raise ValueError("Immutable model checksum mismatch")
    if promoted:
        # Do not overwrite the compared model after a competing
        # promotion.
        blob.upload_from_string(
            contents,
            content_type="application/octet-stream",
            if_generation_match=incumbent_generation,
        )
    manifest = {
        "format_version": 1,
        "sha256": digest,
        "uri": f"gs://{bucket_name}/{immutable_name}",
        "candidate_promoted": promoted,
    }
    Path(published_manifest.path).parent.mkdir(parents=True, exist_ok=True)
    Path(published_manifest.path).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    published_manifest.metadata.update(manifest)
    metrics.log_metric("model_promoted", float(promoted))
    logging.info(
        "Candidate validation AUC %.4f; incumbent %s; promoted %s",
        validation_auc,
        old_auc,
        promoted,
    )

    summary = "".join(
        f"<li>{name}: {value:.4f}</li>" for name, value in values.items()
    )
    Path(html.path).parent.mkdir(parents=True, exist_ok=True)
    Path(html.path).write_text(
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        "<title>Model evaluation</title></head><body><h1>Model evaluation</h1>"
        "<p>Metrics below use the held-out test split. "
        "Promotion uses validation AUC.</p>"
        f"<p>Candidate promoted: {promoted}</p><ul>{summary}</ul>"
        '<img alt="Test confusion matrix" '
        f'src="data:image/png;base64,{cm_image}">'
        f'<img alt="Test ROC curve" src="data:image/png;base64,{roc_image}">'
        "</body></html>",
        encoding="utf-8",
    )
