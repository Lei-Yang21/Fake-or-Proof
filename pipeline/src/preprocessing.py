from kfp.dsl import Dataset, Input, Output, component

from .config import BASE_IMAGE


@component(base_image=BASE_IMAGE, install_kfp_package=False)
def preprocessing(
    input_dataset: Input[Dataset],
    preprocessed_dataset: Output[Dataset],
    split_seed: int = 42,
):
    """Validate images and persist one stratified 70/15/15 split."""
    import io
    import logging
    from pathlib import Path

    import pandas as pd
    from google.cloud import storage
    from PIL import Image
    from sklearn.model_selection import train_test_split

    dataframe = pd.read_csv(input_dataset.path)
    if not {"image_path", "label"}.issubset(dataframe.columns):
        raise ValueError(
            "The dataset must contain image_path and label columns"
        )
    dataframe = dataframe.drop_duplicates("image_path").sort_values(
        "image_path"
    )
    client = storage.Client()
    valid_rows = []
    for row in dataframe.itertuples(index=False):
        if row.label not in (0, 1) or not str(row.image_path).startswith(
            "gs://"
        ):
            raise ValueError(
                "Expected GCS image paths and labels 0 (fake) or 1 (real)"
            )
        bucket_name, blob_name = row.image_path[5:].split("/", 1)
        # Fail on storage errors rather than silently changing the
        # split.
        contents = (
            client.bucket(bucket_name).blob(blob_name).download_as_bytes()
        )
        try:
            with Image.open(io.BytesIO(contents)) as image:
                image.convert("RGB").load()
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            logging.warning(
                "Skipping invalid image %s: %s", row.image_path, error
            )
            continue
        valid_rows.append(
            {"image_path": row.image_path, "label": int(row.label)}
        )

    cleaned = pd.DataFrame(valid_rows, columns=["image_path", "label"])
    if set(cleaned["label"]) != {0, 1}:
        raise ValueError(
            "Both fake and real images are required after validation"
        )
    try:
        train, held_out = train_test_split(
            cleaned,
            test_size=0.30,
            stratify=cleaned["label"],
            random_state=split_seed,
        )
        validation, test = train_test_split(
            held_out,
            test_size=0.50,
            stratify=held_out["label"],
            random_state=split_seed,
        )
    except ValueError as error:
        raise ValueError(
            "Not enough images per class for a stratified 70/15/15 split"
        ) from error

    partitions = []
    for name, partition in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        if set(partition["label"]) != {0, 1}:
            raise ValueError(
                f"The {name} split must contain both classes; add more images"
            )
        partitions.append(partition.assign(split=name))
    result = pd.concat(partitions, ignore_index=True)
    Path(preprocessed_dataset.path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(preprocessed_dataset.path, index=False)
    preprocessed_dataset.metadata["split_seed"] = split_seed
    logging.info(
        "Saved split sizes: %s", result["split"].value_counts().to_dict()
    )
