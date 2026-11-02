# Photo Organizer

Copies your photos into one folder per person, using face recognition.
Everything runs offline on your computer, and your original photos are never moved or changed.

- A photo with several known people is copied into **each** of their folders.
- Photos with only strangers, or no faces at all, are left alone.
- Supported formats: JPG, PNG, HEIC/HEIF (iPhone), WEBP, TIFF, BMP. Other files are ignored.
- Works on Windows, macOS and Linux (Python 3.10 or newer).

## 1. Install (once per computer)

You need [Python 3.10+](https://www.python.org/downloads/) and Git. On Windows, tick "Add Python to PATH" in the installer.

**Windows (PowerShell):**
```powershell
git clone https://github.com/Srikesh2197/photo_organizer.git
cd photo_organizer
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python download_models.py
```

**macOS / Linux:**
```bash
git clone https://github.com/Srikesh2197/photo_organizer.git
cd photo_organizer
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python download_models.py
```

`download_models.py` fetches the face models once (about 280 MB download, about 190 MB kept). They're too big for GitHub, so they aren't in the repo.

## 2. Add reference photos

Create one folder per person inside `references/` and put **3–10 photos** of them in it:

```
references/
  Mom/   mom1.jpg  mom2.heic  mom3.jpg ...
  Dad/   ...
```

Tips for good results:
- One face per photo is best. If there are more, the largest face is used.
- Vary them: different years, angles, lighting, with and without glasses.
- The folder name becomes the output folder name.

## 3. Find your hard drive's path

- **Windows:** check "This PC" for the drive letter, e.g. `E:\Photos`.
- **macOS:** run `ls /Volumes`, e.g. `/Volumes/MyHDD/Photos`. If the drive is formatted NTFS (Windows format), a Mac can read it but not write to it; the script detects this and tells you.
- **Linux:** usually `/media/<you>/<drive>`.

## 4. Dry run first (copies nothing)

Windows example:
```powershell
.venv\Scripts\python sort_photos.py --source E:\Photos --refs references --out E:\Sorted_People --dry-run
```
Mac example:
```bash
.venv/bin/python sort_photos.py --source /Volumes/MyHDD/Photos --refs references --out /Volumes/MyHDD/Sorted_People --dry-run
```

This scans every photo and writes `report.csv` into the `--out` folder. Open it in Excel or Numbers:

| column | meaning |
|---|---|
| `matched_people` | who it would be copied to |
| `face_scores` | each face's best score, e.g. `Mom 0.71`, or `(closest Dad 0.31)` when below the threshold |
| `error` | files that couldn't be opened (corrupt etc.) |

Try one small sub-folder first to check the results.

## 5. Tune the threshold if needed

The default `--threshold 0.40` is a good start. Scores for the same person are usually 0.5–0.9. Different people are usually below 0.3.

- **Known people missed?** Lower it (e.g. `--threshold 0.35`) or add more varied reference photos.
- **Wrong people matched?** Raise it (e.g. `--threshold 0.45`).

Re-running is fast: face data is cached in `faces.sqlite` next to the script, so photos aren't re-read from the drive. Adding a new person folder later also re-runs quickly.

## 6. Real run

Run the same command (from inside the `photo_organizer` folder) without `--dry-run`. Files are copied with their original dates.
If two different photos share a name, the second becomes e.g. `IMG_0001__2020.jpg` (the parent folder's name is added).

It is safe to stop it (Ctrl-C) or unplug the drive. Run the same command again and it continues where it left off. Photos already copied are never copied twice.

## Options

| flag | default | what it does |
|---|---|---|
| `--threshold` | 0.40 | match strictness |
| `--min-face` | 40 | ignore faces smaller than this many pixels (tiny background faces) |
| `--det-size` | 640 | detector resolution. 1024 finds smaller faces in big group shots, but is slower |
| `--workers` | 3 | photo-loading threads |
| `--rescan` | off | ignore the cache and re-scan everything |
| `--no-coreml` | off | Mac only: CPU only, in case you see CoreML errors |
| `--cache` | `faces.sqlite` here | where face data is cached |

On Apple Silicon Macs, face matching uses the GPU automatically (CoreML). Elsewhere it runs on the CPU.

## Credits

Face models: [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` (SCRFD detector + ArcFace recognizer), released for non-commercial research use.
