"""Audit locked registry packages without reinstalling them."""

import argparse
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    # PyPI advisories use the public version, including for PyTorch CPU
    # wheels.
    requirements = sorted(
        {
            f"{package['name']}=={package['version'].split('+')[0]}"
            for package in lock["package"]
            if "registry" in package.get("source", {})
        }
    )
    directory = ROOT / ".pytest_cache"
    directory.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        requirements_path = Path(temporary) / "requirements.txt"
        requirements_path.write_text(
            "\n".join(requirements) + "\n", encoding="utf-8"
        )
        command = [
            sys.executable,
            "-m",
            "pip_audit",
            "--disable-pip",
            "--no-deps",
            "--progress-spinner",
            "off",
            "-r",
            str(requirements_path),
        ]
        if args.output:
            command.extend(["--format", "json", "--output", str(args.output)])
        result = subprocess.run(command, cwd=ROOT, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
