from kfp import dsl

from .config import BUCKET_PATH, PIPELINE_ROOT, PUBLISHED_MODEL_URI
from .data_ingestion import data_ingestion
from .evaluation import evaluation
from .preprocessing import preprocessing
from .training import training


@dsl.pipeline(
    name="fake-or-proof-pipeline", pipeline_root=PIPELINE_ROOT or None
)
def fake_or_proof_pipeline(
    bucket_path: str = BUCKET_PATH,
    published_model_uri: str = PUBLISHED_MODEL_URI,
    epochs: int = 10,
    learning_rate: float = 0.001,
    split_seed: int = 42,
):
    # GCS contents can change while the path stays the same.
    ingestion_task = data_ingestion(
        bucket_path=bucket_path
    ).set_caching_options(False)
    preprocessing_task = preprocessing(
        input_dataset=ingestion_task.outputs["dataset"],
        split_seed=split_seed,
    )
    training_task = training(
        preprocessed_dataset=preprocessing_task.outputs[
            "preprocessed_dataset"
        ],
        epochs=epochs,
        learning_rate=learning_rate,
        seed=split_seed,
    )
    # Promotion always compares with the current published model.
    evaluation(
        model=training_task.outputs["model"],
        preprocessed_dataset=preprocessing_task.outputs[
            "preprocessed_dataset"
        ],
        published_model_uri=published_model_uri,
    ).set_caching_options(False)
