#!/usr/bin/env python3
"""Download the two face models sort_photos.py needs (one-time, ~280 MB download).

Fetches InsightFace's buffalo_l pack and keeps only det_10g.onnx and w600k_r50.onnx
in models/buffalo_l/. The models are too big for GitHub, so they aren't in the repo.
"""

import hashlib
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
DEST = Path(__file__).resolve().parent / "models" / "buffalo_l"
FILES = {
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if all((DEST / n).exists() and sha256(DEST / n) == s for n, s in FILES.items()):
        print(f"Models already present in {DEST}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "buffalo_l.zip"
        print(f"Downloading {URL} ...")

        def progress(blocks, block_size, total):
            if total > 0:
                pct = min(100, blocks * block_size * 100 // total)
                print(f"\r  {pct}%", end="", flush=True)

        urllib.request.urlretrieve(URL, zpath, progress)
        print()
        with zipfile.ZipFile(zpath) as z:
            for name in FILES:
                member = next(m for m in z.namelist() if m.endswith(name))
                (DEST / name).write_bytes(z.read(member))
    for name, expected in FILES.items():
        if sha256(DEST / name) != expected:
            sys.exit(f"Checksum mismatch for {name}; delete {DEST} and try again.")
    print(f"Done. Models saved to {DEST}")


if __name__ == "__main__":
    main()
