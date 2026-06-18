# Changelog

All notable changes to this project are documented here.

## [0.1.0] - 2026-06-16

Initial version: a local CLI that turns a NAS photo library into iPhone-style memory videos.

### Catalog & indexing
- Incremental, parallel scanner over SMB; one SQLite row per media file.
- Date resolution from folder names (`YYYYMMDD` / `YYYY_MM_DD Title`) and file names,
  with EXIF `DateTimeOriginal` as fallback.
- EXIF **GPS** extraction + offline reverse geocoding to place labels.
- `index`, `stats` commands. (Catalog built for ~237K photos, 1997–2026.)

### Grouping
- Monthly and yearly buckets with per-period "main activity" labels.
- **GPS-aware trip detection**: home location auto-estimated; consecutive away-from-home
  days form trips, named by place. `list-months`, `list-events`, `list-trips`.

### Selection
- Near-duplicate removal via perceptual hash (prefers edited `-LR` variants).
- Quality scoring: sharpness, exposure, GPU **NIMA aesthetic** (pyiqa), face bonus.
- Scores and downscaled render frames cached for fast rebuilds.
- Per-period candidate cap + even cross-group sampling so year-scale builds stay fast.
- Selection distributed across days/events for full-period coverage.

### Face recognition (enroll family, boost their photos)
- GPU detection + embeddings via facenet-pytorch (MTCNN + VGGFace2).
- Auto-cluster faces (`faces cluster`), refine over-merged clusters (`faces split`),
  label clusters (`faces label`), or enroll from reference photos (`faces enroll`,
  picks the dominant recurring face and saves a verification montage).
- **Age-aware matching**: k-NN over per-person prototypes, using birthdates
  (`faces birthday`) and each prototype's photo date to disambiguate look-alike siblings.
- `faces lifespan --until` bounds the dates a person can appear (e.g. someone who has passed).
- Family presence strongly boosts selection (`scoring.w_family`).

### Rendering
- moviepy + bundled ffmpeg (`imageio-ffmpeg`) — no system ffmpeg needed.
- Fit-whole-photo with a blurred fill background (no hard cropping), gentle Ken Burns,
  crossfades, title + lower-third caption cards, and synthesized royalty-free music.
- `build month|year|trip`, 1080p H.264 output.

### Notes
- Optional GPU stack (torch **cu124** — cu121 has no Python 3.13 wheels), pyiqa,
  scikit-learn, and `--no-deps` facenet-pytorch. The pipeline degrades to heuristics
  if the ML extras aren't installed.
