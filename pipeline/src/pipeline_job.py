"""Submit an explicitly configured Vertex AI pipeline."""

import argparse
import json
import re
from pathlib import Path

from .config import (
    BUCKET_PATH,
    PIPELINE_ROOT,
    PROJECT_ID,
    PUBLISHED_MODEL_URI,
    REGION,
    SERVICE_ACCOUNT,
    TEMPLATE_PATH,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=TEMPLATE_PATH)
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--pipeline-root", default=PIPELINE_ROOT)
    parser.add_argument("--bucket-path", default=BUCKET_PATH)
    parser.add_argument("--published-model-uri", default=PUBLISHED_MODEL_URI)
    parser.add_argument(
        "--service-account",
        default=SERVICE_ACCOUNT,
        help="Runtime service account email",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    if not args.template.is_file():
        parser.error(
            "Template not found. Run python -m pipeline.src.compiler first."
        )
    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error("epochs and learning-rate must be positive")
    for name in (
        "project",
        "region",
        "pipeline_root",
        "bucket_path",
        "published_model_uri",
        "service_account",
    ):
        if not getattr(args, name):
            parser.error(
                f"--{name.replace('_', '-')} or its environment variable "
                "is required"
            )
    for uri in (
        args.pipeline_root,
        args.bucket_path,
        args.published_model_uri,
    ):
        if not re.fullmatch(r"gs://[^/]+/.+", uri):
            parser.error(
                "Dataset, pipeline root and published model must be "
                "gs://bucket/path URIs"
            )
    specification = json.loads(args.template.read_text(encoding="utf-8"))
    for executor in specification["deploymentSpec"]["executors"].values():
        if not re.fullmatch(
            r".+@sha256:[a-f0-9]{64}", executor["container"]["image"]
        ):
            parser.error(
                "Compile with PIPELINE_BASE_IMAGE set to an immutable "
                "image@sha256:digest"
            )

    from google.cloud import aiplatform

    aiplatform.init(project=args.project, location=args.region)
    job = aiplatform.PipelineJob(
        display_name="fake-or-proof-pipeline",
        template_path=str(args.template.resolve()),
        pipeline_root=args.pipeline_root,
        parameter_values={
            "bucket_path": args.bucket_path,
            "published_model_uri": args.published_model_uri,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "split_seed": args.split_seed,
        },
    )
    job.run(service_account=args.service_account)


if __name__ == "__main__":
    main()
