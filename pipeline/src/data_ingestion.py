from kfp.dsl import Dataset, Output, component

from .config import BASE_IMAGE


@component(base_image=BASE_IMAGE, install_kfp_package=False)
def data_ingestion(bucket_path: str, dataset: Output[Dataset]):
    """List images under <bucket_path>/fake/ and <bucket_path>/real/."""
    import logging
    from pathlib import Path

    import pandas as pd
    from google.cloud import storage

    if not bucket_path.startswith("gs://"):
        raise ValueError("bucket_path must start with gs://")
    bucket_name, _, prefix = bucket_path[5:].partition("/")
    if not bucket_name:
        raise ValueError("bucket_path must contain a bucket name")
    prefix = prefix.strip("/")
    if prefix:
        prefix += "/"
    prefix_length = len(prefix)

    rows = []
    bucket = storage.Client().bucket(bucket_name)
    for blob in bucket.list_blobs(prefix=prefix):
        folder = blob.name[prefix_length:].split("/", 1)[0]
        if folder not in {"fake", "real"}:
            continue
        if not blob.name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        rows.append(
            {
                "image_path": f"gs://{bucket_name}/{blob.name}",
                "label": int(folder == "real"),
            }
        )

    if not rows:
        raise ValueError(f"No images found under {bucket_path}/fake or /real")
    dataframe = (
        pd.DataFrame(rows).sort_values("image_path").reset_index(drop=True)
    )
    Path(dataset.path).parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(dataset.path, index=False)
    logging.info("Found %d images", len(dataframe))
