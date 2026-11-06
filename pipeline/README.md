# Vertex AI pipeline

This pipeline targets Google Cloud Vertex AI. It is separate from Cloud Run application deployment, and no application release starts training automatically.

## Cloud prerequisites

Prepare billing and Vertex AI APIs, a dataset bucket with `fake/` and `real/` image folders, a pipeline artifact bucket, an Artifact Registry repository, a submitting identity and a dedicated pipeline runtime service account. The runtime needs dataset read, artifact/model write, image read and the necessary Vertex permissions on your resources. The Cloud Run serving identity is separate and only needs model-read access when GCS serving is selected.

Build the training image through your cloud build/release tooling using `pipeline/Dockerfile` and the **repository root** as build context. The image installs CPU dependencies from `uv.lock`, includes the shared model contract and preinstalls KFP; task startup does not run pip. Publish the image and use its immutable `@sha256:...` reference.

## Configuration and submission

Configure these variables in the CI environment or Google Cloud Shell used to submit the job:

| Variable | Required value |
| --- | --- |
| `GCP_PROJECT_ID` | Target project |
| `GCP_REGION` | Vertex region |
| `GCP_PIPELINE_SERVICE_ACCOUNT` | Pipeline runtime service-account email |
| `FOP_DATASET_URI` | `gs://YOUR_BUCKET/data/dataset` |
| `FOP_PIPELINE_ROOT` | `gs://YOUR_BUCKET/pipeline-artifacts` |
| `FOP_PUBLISHED_MODEL_URI` | Comparison object, e.g. `gs://YOUR_BUCKET/models/current.pth` |
| `PIPELINE_BASE_IMAGE` | Your training image pinned by SHA-256 digest |

With repository sources available in that cloud environment and the submitting identity authenticated, install the locked pipeline dependencies, compile and submit:

```text
uv sync --locked --no-default-groups --group pipeline
uv run --no-sync python -m pipeline.src.compiler
uv run --no-sync python -m pipeline.src.pipeline_job --epochs 1
```

The last command **starts a billable Vertex AI job**. Import and compilation alone do not submit anything. Cloud submission requires explicit resource settings and rejects unpinned images. The compiler's unconfigured-image sentinel exists only to let CI validate the graph without selecting a real project.

CLI options can override the corresponding settings: `--project`, `--region`, `--service-account`, `--bucket-path`, `--pipeline-root`, `--published-model-uri`, `--epochs`, `--learning-rate`, `--split-seed`. Recompile when changing the image reference.

## Components and model publication

1. List JPEG/PNG image paths under the two class folders.
2. Validate/decode images and persist one stratified 70/15/15 partition.
3. Train MobileNetV2 and compute validation metrics.
4. Report held-out test metrics and promote using **validation AUC**, never test performance.

The model contract matches serving: 224×224 RGB, mean/std 0.5, class 0 AI-generated and class 1 real. Training downloads the official initial ImageNet weights; the trained serving image does not need this training download.

Only restricted tensor checkpoints are supported. A selected model is published as `models/<SHA256>.pth`; the evaluation task's explicit `published_manifest` output records its URI/hash. The mutable comparison object is updated only on promotion with a generation precondition, preventing a concurrent run from silently replacing the model evaluated.

To use a newly selected model in Cloud Run, take its URI/hash from the trusted pipeline output, configure `FOP_MODEL_URI`/`FOP_MODEL_SHA256` in deployment settings and redeploy. A sidecar downloaded next to an unknown model is not an independent source of trust. Serving does not automatically follow a mutable model pointer.

Existing legacy full-model pickle objects are rejected. The bundled release checkpoint has already been migrated; use a new comparison object or publish a known safe checkpoint before starting a pipeline that previously used the old format. Never disable restricted loading to accept a legacy object.

Pipeline tests are maintained with the other CI tests under `.github/tests/pipeline/`, with simulated cloud storage. They do not access production data or start cloud jobs.
