#!/usr/bin/env python3
"""Sort photos into per-person folders using face recognition.

Reads photos from --source (e.g. an external HDD), compares every face to the
reference photos in --refs/<PersonName>/, and COPIES each photo into
--out/<PersonName>/ for every person found in it. Originals are never touched.

Face detection: SCRFD (det_10g.onnx). Face recognition: ArcFace (w600k_r50.onnx).
Both come from InsightFace's "buffalo_l" model pack and run via onnxruntime.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from tqdm import tqdm

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # HEIC photos will be reported as unreadable
    pass

HERE = Path(__file__).resolve().parent
DEFAULT_MODELS = HERE / "models" / "buffalo_l"
DEFAULT_CACHE = HERE / "faces.sqlite"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff", ".bmp"}
SKIP_DIRS = {"$RECYCLE.BIN", "System Volume Information", "@eaDir", "#recycle"}
MAX_SIDE = 1920  # photos are downscaled to this before face detection
CACHE_VERSION = "buffalo_l-v1"

# Where the 5 face landmarks should sit in a 112x112 ArcFace input.
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


# --------------------------------------------------------------------------- models

def make_session(path: Path, use_coreml: bool) -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"]
    if use_coreml and "CoreMLExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CoreMLExecutionProvider")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


class FaceDetector:
    """SCRFD face detector: returns boxes (x1,y1,x2,y2,score) and 5 landmarks per face."""

    def __init__(self, model: Path, det_size: int, use_coreml: bool):
        self.sess = make_session(model, use_coreml)
        self.input_name = self.sess.get_inputs()[0].name
        self.det_size = det_size
        self.strides = (8, 16, 32)
        self.num_anchors = 2
        self._centers: dict[tuple, np.ndarray] = {}

    def _anchor_centers(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        if key not in self._centers:
            c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            c = (c * stride).reshape(-1, 2)
            c = np.stack([c] * self.num_anchors, axis=1).reshape(-1, 2)
            self._centers[key] = c
        return self._centers[key]

    def detect(self, img_bgr: np.ndarray, thresh: float = 0.5, nms_thresh: float = 0.4):
        size = self.det_size
        h, w = img_bgr.shape[:2]
        scale = size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(img_bgr, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (size, size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.sess.run(None, {self.input_name: blob})

        boxes, scores, kpss = [], [], []
        fmc = len(self.strides)
        for i, stride in enumerate(self.strides):
            sc = outs[i]
            bb = outs[i + fmc]
            kp = outs[i + fmc * 2]
            if sc.ndim == 3:  # batched output
                sc, bb, kp = sc[0], bb[0], kp[0]
            sc = sc.reshape(-1)
            bb = bb * stride
            kp = kp * stride
            centers = self._anchor_centers(size // stride, size // stride, stride)
            keep = np.where(sc >= thresh)[0]
            if keep.size == 0:
                continue
            c = centers[keep]
            d = bb[keep]
            boxes.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1],
                                   c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], axis=-1))
            k = kp[keep].reshape(-1, 5, 2) + c[:, None, :]
            kpss.append(k)
            scores.append(sc[keep])

        if not boxes:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        boxes = np.concatenate(boxes) / scale
        kpss = np.concatenate(kpss) / scale
        scores = np.concatenate(scores)
        order = scores.argsort()[::-1]
        boxes, kpss, scores = boxes[order], kpss[order], scores[order]
        keep = _nms(boxes, scores, nms_thresh)
        dets = np.hstack([boxes[keep], scores[keep, None]]).astype(np.float32)
        return dets, kpss[keep].astype(np.float32)


def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.arange(len(scores))  # already sorted by score
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= thresh]
    return keep


class FaceEmbedder:
    """ArcFace: turns an aligned face into a 512-number fingerprint (unit length)."""

    def __init__(self, model: Path, use_coreml: bool):
        self.sess = make_session(model, use_coreml)
        self.input_name = self.sess.get_inputs()[0].name

    def embed(self, img_bgr: np.ndarray, kpss: np.ndarray) -> np.ndarray:
        crops = []
        for kps in kpss:
            M, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST, method=cv2.LMEDS)
            if M is None:  # degenerate landmarks; fall back to a plain fit
                M, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST)
            crops.append(cv2.warpAffine(img_bgr, M, (112, 112), borderValue=0.0))
        blob = cv2.dnn.blobFromImages(crops, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        # Run one face at a time: the model's batch size isn't always dynamic.
        embs = np.vstack([self.sess.run(None, {self.input_name: blob[i:i + 1]})[0] for i in range(len(crops))])
        return embs / np.linalg.norm(embs, axis=1, keepdims=True)


class FaceEngine:
    def __init__(self, models_dir: Path, det_size: int, use_coreml: bool):
        det = models_dir / "det_10g.onnx"
        rec = models_dir / "w600k_r50.onnx"
        missing = [p.name for p in (det, rec) if not p.exists()]
        if missing:
            sys.exit(f"Model files missing in {models_dir}: {', '.join(missing)}\n"
                     "See README.md, step 'Download the face models'.")
        # Measured on Apple Silicon: the detector is fastest on CPU, the
        # recognizer about 10x faster on CoreML.
        self.detector = FaceDetector(det, det_size, use_coreml=False)
        self.embedder = FaceEmbedder(rec, use_coreml)

    def faces(self, img_bgr: np.ndarray, min_face: int):
        """Return list of (det_score, face_width_px, embedding) for faces >= min_face px."""
        dets, kpss = self.detector.detect(img_bgr)
        if len(dets) == 0:
            return []
        widths = dets[:, 2] - dets[:, 0]
        heights = dets[:, 3] - dets[:, 1]
        keep = np.where(np.minimum(widths, heights) >= min_face)[0]
        if keep.size == 0:
            return []
        embs = self.embedder.embed(img_bgr, kpss[keep])
        return [(float(dets[i, 4]), int(widths[i]), embs[j]) for j, i in enumerate(keep)]


# --------------------------------------------------------------------------- images

def load_image(path: str) -> np.ndarray:
    """Open any supported photo, fix its rotation, shrink it, and return BGR pixels."""
    with Image.open(path) as im:
        if im.format == "JPEG":
            im.draft("RGB", (MAX_SIDE, MAX_SIDE))  # fast JPEG decode at reduced size
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        if max(im.size) > MAX_SIDE:
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.BILINEAR)
        arr = np.asarray(im)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def list_images(source: Path, exclude: list[Path], skip_names: tuple[str, ...] = ()) -> list[Path]:
    """Walk folders in sorted order (kinder to spinning disks), skipping junk and output folders.

    `exclude` are exact folders to skip; `skip_names` are folder names skipped wherever they appear.
    """
    excl = {os.path.realpath(p) for p in exclude}
    skip = {n.lower() for n in skip_names}
    found = []
    for root, dirs, files in os.walk(source):
        dirs[:] = sorted(
            d for d in dirs
            if not d.startswith(".") and d not in SKIP_DIRS and d.lower() not in skip
            and os.path.realpath(os.path.join(root, d)) not in excl
        )
        for f in sorted(files):
            if f.startswith("."):  # ._AppleDouble files, .DS_Store, etc.
                continue
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                found.append(Path(root) / f)
    return found


# --------------------------------------------------------------------------- cache

class Cache:
    """SQLite store of face fingerprints per photo, so photos are only scanned once."""

    def __init__(self, path: Path):
        self.db = sqlite3.connect(str(path))
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY, size INTEGER, mtime REAL,
                nfaces INTEGER, error TEXT);
            CREATE TABLE IF NOT EXISTS faces (
                path TEXT, idx INTEGER, score REAL, width INTEGER, emb BLOB,
                PRIMARY KEY (path, idx));
            CREATE TABLE IF NOT EXISTS copies (
                src TEXT, person TEXT, dest TEXT, PRIMARY KEY (src, person));
        """)
        row = self.db.execute("SELECT v FROM meta WHERE k='version'").fetchone()
        if row and row[0] != CACHE_VERSION:
            sys.exit(f"Cache {path} was made with a different model ({row[0]}). Delete it or use --cache.")
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('version', ?)", (CACHE_VERSION,))
        self.db.commit()

    def is_fresh(self, path: str, size: int, mtime: float) -> bool:
        row = self.db.execute("SELECT size, mtime FROM files WHERE path=?", (path,)).fetchone()
        return bool(row) and row[0] == size and abs(row[1] - mtime) < 1.0

    def put(self, path: str, size: int, mtime: float, faces, error: str | None):
        self.db.execute("DELETE FROM faces WHERE path=?", (path,))
        self.db.execute("INSERT OR REPLACE INTO files VALUES (?,?,?,?,?)",
                        (path, size, mtime, len(faces), error))
        self.db.executemany(
            "INSERT INTO faces VALUES (?,?,?,?,?)",
            [(path, i, s, w, e.astype(np.float32).tobytes()) for i, (s, w, e) in enumerate(faces)],
        )

    def faces_for(self, paths: list[str]) -> dict[str, list[np.ndarray]]:
        wanted = set(paths)
        out: dict[str, list[np.ndarray]] = {p: [] for p in paths}
        for path, emb in self.db.execute("SELECT path, emb FROM faces ORDER BY path, idx"):
            if path in wanted:
                out[path].append(np.frombuffer(emb, dtype=np.float32))
        return out

    def errors_for(self, paths: list[str]) -> dict[str, str]:
        wanted = set(paths)
        return {p: e for p, e in self.db.execute("SELECT path, error FROM files WHERE error IS NOT NULL")
                if p in wanted}

    def copied_dest(self, src: str, person: str) -> str | None:
        row = self.db.execute("SELECT dest FROM copies WHERE src=? AND person=?", (src, person)).fetchone()
        return row[0] if row else None

    def record_copy(self, src: str, person: str, dest: str):
        self.db.execute("INSERT OR REPLACE INTO copies VALUES (?,?,?)", (src, person, dest))

    def commit(self):
        self.db.commit()


# --------------------------------------------------------------------------- stages

def enroll(engine: FaceEngine, refs: Path, min_face: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Build fingerprints from references/<Name>/*. Returns names, per-ref owner index, embeddings."""
    people = sorted(d for d in refs.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not people:
        sys.exit(f"No person folders found in {refs}. Create e.g. {refs}/Mom/ with a few photos of Mom.")
    names, owners, embs = [], [], []
    print(f"Enrolling {len(people)} people from {refs}")
    for pi, person in enumerate(people):
        names.append(person.name)
        files = list_images(person, [])
        used = 0
        for f in files:
            try:
                img = load_image(str(f))
            except Exception as e:
                print(f"  ! {person.name}/{f.name}: can't open ({e})")
                continue
            faces = engine.faces(img, min_face=20)
            if not faces:
                print(f"  ! {person.name}/{f.name}: no face found, skipped")
                continue
            if len(faces) > 1:
                print(f"  ! {person.name}/{f.name}: {len(faces)} faces, using the largest one")
            best = max(faces, key=lambda t: t[1])
            owners.append(pi)
            embs.append(best[2])
            used += 1
        flag = "" if used >= 3 else "   <- add more photos (3-10 recommended)"
        print(f"  {person.name}: {used} reference face(s){flag}")
        if used == 0:
            sys.exit(f"No usable reference faces for {person.name}.")
    return names, np.array(owners), np.vstack(embs)


def scan(engine: FaceEngine, cache: Cache, images: list[Path], min_face: int, workers: int, rescan: bool):
    """Detect and fingerprint faces in every photo not already in the cache."""
    todo = []
    for p in images:
        try:
            st = p.stat()
        except OSError:
            continue
        if rescan or not cache.is_fresh(str(p), st.st_size, st.st_mtime):
            todo.append((p, st.st_size, st.st_mtime))
    print(f"Scanning: {len(images)} photos found, {len(images) - len(todo)} already in cache, {len(todo)} to scan")
    if not todo:
        return

    def load(item):
        p = item[0]
        try:
            return item, load_image(str(p)), None
        except Exception as e:
            return item, None, f"{type(e).__name__}: {e}"

    last_commit = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool, tqdm(total=len(todo), unit="photo") as bar:
        pending: deque = deque()
        it = iter(todo)
        for item in it:  # prime a small read-ahead window
            pending.append(pool.submit(load, item))
            if len(pending) >= workers * 3:
                break
        while pending:
            (p, size, mtime), img, err = pending.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(pool.submit(load, nxt))
            if err and not p.parent.exists():
                cache.commit()
                sys.exit(f"\nThe source drive seems to have disconnected ({p.parent}). "
                         "Progress is saved; reconnect and run the same command again.")
            faces = []
            if img is not None:
                try:
                    faces = engine.faces(img, min_face)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
            cache.put(str(p), size, mtime, faces, err)
            bar.update(1)
            if time.time() - last_commit > 5:
                cache.commit()
                last_commit = time.time()
    cache.commit()


def assign(cache: Cache, images: list[Path], names, owners, ref_embs, threshold: float):
    """Match each face to the closest person. Returns rows for the report."""
    paths = [str(p) for p in images]
    faces = cache.faces_for(paths)
    errors = cache.errors_for(paths)
    rows = []
    for path in paths:
        people, details = [], []
        for emb in faces[path]:
            sims = ref_embs @ emb  # cosine similarity to every reference face
            per_person = np.full(len(names), -1.0)
            np.maximum.at(per_person, owners, sims)
            best = int(per_person.argmax())
            score = float(per_person[best])
            if score >= threshold:
                if names[best] not in people:
                    people.append(names[best])
                details.append(f"{names[best]} {score:.2f}")
            else:
                details.append(f"(closest {names[best]} {score:.2f})")
        rows.append({"file": path, "faces": len(faces[path]), "people": people,
                     "details": details, "error": errors.get(path, "")})
    return rows


def write_report(rows, report_path: Path, dest_folder):
    # utf-8-sig so Excel on Windows shows non-English names correctly
    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "faces_found", "matched_people", "copy_to", "face_scores", "error"])
        for r in rows:
            dests = " ; ".join(str(dest_folder(Path(r["file"]), p)) for p in r["people"])
            w.writerow([r["file"], r["faces"], "; ".join(r["people"]), dests, " | ".join(r["details"]), r["error"]])


def unique_dest(folder: Path, src: Path) -> Path:
    dest = folder / src.name
    if not dest.exists():
        return dest
    stem, ext = src.stem, src.suffix
    dest = folder / f"{stem}__{src.parent.name}{ext}"
    n = 2
    while dest.exists():
        dest = folder / f"{stem}__{src.parent.name}_{n}{ext}"
        n += 1
    return dest


def copy_files(cache: Cache, rows, dest_folder, space_check: Path):
    """Copy each matched photo into dest_folder(src, person). space_check: a path on the output drive."""
    jobs = []
    for r in rows:
        for person in r["people"]:
            done = cache.copied_dest(r["file"], person)
            if done and os.path.exists(done):
                continue
            jobs.append((Path(r["file"]), person))
    if not jobs:
        print("Copying: nothing new to copy.")
        return
    need = sum(src.stat().st_size for src, _ in jobs)
    free = shutil.disk_usage(space_check).free
    print(f"Copying {len(jobs)} files ({need / 1e9:.2f} GB); {free / 1e9:.2f} GB free on the output drive")
    if need + 200e6 > free:
        sys.exit("Not enough free space on the output drive. Nothing was copied.")
    for src, person in tqdm(jobs, unit="file"):
        folder = dest_folder(src, person)
        folder.mkdir(parents=True, exist_ok=True)
        dest = unique_dest(folder, src)
        tmp = dest.with_name(dest.name + ".partial")
        shutil.copy2(src, tmp)  # copy2 keeps the original dates
        os.replace(tmp, dest)
        cache.record_copy(str(src), person, str(dest))
        cache.commit()


# --------------------------------------------------------------------------- checks

def fs_type(path: Path) -> str:
    """Drive format (apfs, exfat, ntfs, ...) for the volume holding `path`, read from `mount`."""
    p = path.expanduser().absolute()
    while not p.exists() and p != p.parent:
        p = p.parent
    real = os.path.realpath(p)
    try:
        lines = subprocess.run(["mount"], capture_output=True, text=True).stdout.splitlines()
    except Exception:
        return "unknown"
    best, kind = "", "unknown"
    for line in lines:  # e.g. "/dev/disk4s1 on /Volumes/HDD (exfat, local, nodev)"
        if " on " not in line or " (" not in line:
            continue
        mnt, rest = line.split(" on ", 1)[1].rsplit(" (", 1)
        if (real == mnt or real.startswith(mnt.rstrip("/") + "/")) and len(mnt) > len(best):
            best, kind = mnt, rest.split(",")[0]
    return kind


def check_writable(out: Path) -> bool:
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Copy photos into per-person folders using face recognition.")
    ap.add_argument("--source", required=True, type=Path, help="folder with your photos (scanned recursively)")
    ap.add_argument("--refs", required=True, type=Path, help="folder with one sub-folder of reference photos per person")
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--out", type=Path, help="one folder where all per-person folders are created")
    where.add_argument("--per-event", metavar="NAME",
                       help="instead of --out: create a folder called NAME inside each top-level "
                            "sub-folder of --source (e.g. Event 1\\NAME\\Person 1)")
    ap.add_argument("--threshold", type=float, default=0.40,
                    help="match strictness 0-1 (default 0.40). Lower = more matches, higher = fewer mistakes")
    ap.add_argument("--dry-run", action="store_true", help="only scan and write report.csv; copy nothing")
    ap.add_argument("--min-face", type=int, default=40, help="ignore faces smaller than this many pixels (default 40)")
    ap.add_argument("--det-size", type=int, default=640, help="face detector resolution (640 default; 1024 finds tinier faces, slower)")
    ap.add_argument("--workers", type=int, default=3, help="photo-loading threads (default 3)")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="face cache file (default: next to this script)")
    ap.add_argument("--models", type=Path, default=DEFAULT_MODELS, help="folder with det_10g.onnx and w600k_r50.onnx")
    ap.add_argument("--rescan", action="store_true", help="ignore the cache and scan every photo again")
    ap.add_argument("--no-coreml", action="store_true", help="use CPU only (try this if you see CoreML errors)")
    args = ap.parse_args()

    def clean(p: Path) -> Path:
        # On Windows, a quoted path ending in a backslash ("E:\Main Folder\") arrives with a stray quote.
        return Path(str(p).strip().strip('"')).expanduser().absolute()

    source, refs = clean(args.source), clean(args.refs)
    if args.out:
        args.out = clean(args.out)
    for label, p in (("--source", source), ("--refs", refs)):
        if not p.is_dir():
            sys.exit(f"{label} folder not found: {p}  (is the drive plugged in?)")
    if is_inside(refs, source):
        print("Note: the references folder is inside the source; it will be skipped while scanning.")

    if args.per_event:
        name = args.per_event.strip().strip("\\/")
        if not name or any(c in name for c in '\\/:*?"<>|'):
            sys.exit("--per-event needs a plain folder name, e.g. --per-event Sorted_People")

        def dest_folder(src: Path, person: str) -> Path:
            rel = src.relative_to(source)
            base = source / rel.parts[0] if len(rel.parts) > 1 else source  # the event folder
            return base / name / person

        out = source  # used for the writable and free-space checks
        skip_names: tuple[str, ...] = (name,)
        where_text = f"a '{name}' folder inside each event folder of {source}"
        # The report goes next to this script, so a dry run writes nothing to the photo drive.
        report_path = HERE / "report.csv"
        writable = True if args.dry_run else check_writable(source)
    else:
        out = args.out.expanduser().absolute()

        def dest_folder(src: Path, person: str) -> Path:
            return out / person

        skip_names = ()
        where_text = str(out)
        writable = check_writable(out)
        report_path = out / "report.csv"
    if not writable:
        kind = fs_type(out.parent if not out.exists() else out)
        msg = (f"Can't write to {out} (drive format: {kind}).\n"
               + ("This drive is NTFS (Windows format); macOS can only read it.\n" if "ntfs" in kind.lower() else "")
               + "Check the drive isn't read-only or write-protected, or choose an --out folder on another drive.")
        if not args.dry_run:
            sys.exit(msg)
        report_path = HERE / "report.csv"
        print(msg + f"\nDry run: the report will be saved to {report_path} instead.")

    engine = FaceEngine(args.models, args.det_size, not args.no_coreml)
    names, owners, ref_embs = enroll(engine, refs, args.min_face)

    images = list_images(source, [refs] if args.per_event else [out, refs], skip_names)
    if not images:
        sys.exit(f"No photos found in {source}")
    cache = Cache(args.cache)
    scan(engine, cache, images, args.min_face, args.workers, args.rescan)

    rows = assign(cache, images, names, owners, ref_embs, args.threshold)
    write_report(rows, report_path, dest_folder)

    counts = Counter(p for r in rows for p in r["people"])
    no_faces = sum(1 for r in rows if r["faces"] == 0 and not r["error"])
    unmatched = sum(1 for r in rows if r["faces"] > 0 and not r["people"])
    errors = sum(1 for r in rows if r["error"])
    print(f"\nResults (threshold {args.threshold:.2f}):")
    for n in names:
        print(f"  {n:<20} {counts.get(n, 0)} photos")
    print(f"  {'(unknown faces only)':<20} {unmatched} photos - left alone")
    print(f"  {'(no faces)':<20} {no_faces} photos - left alone")
    if errors:
        print(f"  {'(could not read)':<20} {errors} photos - see 'error' column in the report")
    print(f"Report: {report_path}")

    if args.dry_run:
        print("Dry run: nothing copied. Check the report, then run again without --dry-run.")
    else:
        copy_files(cache, rows, dest_folder, out)
        print(f"Done. Sorted folders are in {where_text}")


if __name__ == "__main__":
    main()
