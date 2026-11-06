"""Compile a Vertex AI pipeline template."""

import argparse
from pathlib import Path

from kfp import compiler

from .config import TEMPLATE_PATH
from .pipeline import fake_or_proof_pipeline


def compile_pipeline(output_path: Path = TEMPLATE_PATH) -> Path:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(
        pipeline_func=fake_or_proof_pipeline,
        package_path=str(output_path),
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=TEMPLATE_PATH)
    args = parser.parse_args()
    print(compile_pipeline(args.output))


if __name__ == "__main__":
    main()
