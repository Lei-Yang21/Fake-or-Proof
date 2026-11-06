from kfp.dsl import Dataset, Input, Metrics, Model, Output, component

from .config import BASE_IMAGE


@component(base_image=BASE_IMAGE, install_kfp_package=False)
def training(
    preprocessed_dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    epochs: int = 10,
    learning_rate: float = 0.001,
    seed: int = 42,
):
    """Train MobileNetV2 on persisted training and validation splits."""
    import io
    import logging
    from pathlib import Path

    import pandas as pd
    import torch
    from google.cloud import storage
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    from fake_or_proof.model import (
        build_network,
        checkpoint_for,
        image_transform,
    )

    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs and lr must be positive")
    torch.manual_seed(seed)
    transform = image_transform()

    class GCSImageDataset(Dataset):
        def __init__(self, dataframe):
            self.data = dataframe.reset_index(drop=True)
            self.client = storage.Client()

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            row = self.data.iloc[index]
            bucket_name, blob_name = row["image_path"][5:].split("/", 1)
            contents = (
                self.client.bucket(bucket_name)
                .blob(blob_name)
                .download_as_bytes()
            )
            with Image.open(io.BytesIO(contents)) as image:
                tensor = transform(image.convert("RGB"))
            return tensor, int(row["label"])

    dataframe = pd.read_csv(preprocessed_dataset.path)
    if not {"image_path", "label", "split"}.issubset(dataframe.columns):
        raise ValueError(
            "Expected the preprocessed CSV with a persisted split column"
        )
    train = dataframe[dataframe["split"] == "train"]
    validation = dataframe[dataframe["split"] == "validation"]
    if train.empty or validation.empty:
        raise ValueError("Train and validation partitions must not be empty")
    train_loader = DataLoader(
        GCSImageDataset(train), batch_size=32, shuffle=True
    )
    validation_loader = DataLoader(GCSImageDataset(validation), batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ft = build_network(pretrained=True)
    model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model_ft.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model_ft(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        logging.info(
            "Epoch %d/%d, loss %.4f",
            epoch + 1,
            epochs,
            total_loss / len(train),
        )

    model_ft.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in validation_loader:
            predicted = model_ft(images.to(device)).argmax(dim=1).cpu()
            correct += int((predicted == labels).sum())
    metrics.log_metric(
        "validation_accuracy", 100.0 * correct / len(validation)
    )
    Path(model.path).parent.mkdir(parents=True, exist_ok=True)
    # Only tensors and validated primitive metadata cross the model
    # boundary.
    torch.save(checkpoint_for(model_ft.cpu().eval()), model.path)
    model.metadata["architecture"] = "MobileNetV2"
    model.metadata["format"] = "restricted_checkpoint_v1"
    model.metadata["class_labels"] = {
        "0": "AI Generated Image",
        "1": "Real Image",
    }
