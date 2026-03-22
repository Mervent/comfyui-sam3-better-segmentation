"""
ComfyUI install script — runs automatically when the node is first detected.

Installs all dependencies declared in pyproject.toml.
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("=" * 70)
    print("[SAM3BS] Installing dependencies from pyproject.toml ...")
    print("=" * 70)

    here = Path(__file__).parent
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", str(here)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[SAM3BS] pip install failed:\n{result.stderr}")
        sys.exit(1)

    print("[SAM3BS] Dependencies installed successfully.")


if __name__ == "__main__":
    main()
