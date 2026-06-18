# Photo Memory Generator

iPhone-Memories-style recap videos from a large photo library on a NAS. It catalogs
your photos, detects what belongs together (months, years, multi-day trips), picks the
best shots, and renders a video with Ken Burns motion, crossfades, title/caption cards,
and music — all locally on Windows.

Built for a library laid out as:
- `Z:\Triees\<year>\<YYYYMMDD Description>` (or `YYYY_MM_DD Description`) — the folder
  name supplies both the date and a human event title.
- `Z:\Amazon Photos Downloads\...\<device>\YYYY-MM-DD_HH-MM-SS_*.jpeg` — phone photos
  (date from filename, GPS from EXIF).

## How it works

1. **Index** — a fast, incremental, parallel walk of the NAS. Dates come from folder /
   file names (cheap); EXIF is read per image for **GPS** (and as a date fallback). GPS
   is reverse-geocoded offline to a place label. One row per file in a local SQLite
   catalog (`data/catalog.db`).
2. **Group** — events (folders), monthly/yearly buckets, and **GPS-aware trip
   detection** (consecutive days; a location away from home promotes & names a trip).
3. **Score & select** — per candidate: near-duplicate removal (perceptual hash, prefers
   `-LR` edits), sharpness, exposure, face detection, and optional GPU **NIMA aesthetic**
   scoring. Scores are cached. The selector spreads the picks across days/events for
   coverage and orders them chronologically.
4. **Render** — moviepy with a bundled ffmpeg (`imageio-ffmpeg`, no system install) →
   1080p H.264 mp4 in `output/`.

## Setup

```powershell
cd D:\dev\Claude\photo-memory
python -m pip install -r requirements.txt
python tools\gen_music.py          # generate the bundled royalty-free tracks

# Optional GPU aesthetic scoring (large download, big quality gain):
# Use cu124 — cu121 has no Python 3.13 wheels.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install pyiqa
```

Edit `config.toml` to set your NAS paths, performance (`workers`), scoring weights, and
optional home GPS coordinates.

## Usage

```powershell
# 1. Build / update the catalog (incremental — safe to re-run).
python -m photomemory.cli index                      # everything
python -m photomemory.cli index --root triees --subpath 2024   # just one year (good first test)
python -m photomemory.cli stats

# 2. Explore what memories are available.
python -m photomemory.cli list-months 2024
python -m photomemory.cli list-events 2024-10
python -m photomemory.cli list-trips --year 2024

# 2b. (Optional) Enroll family members so their photos get strongly favored.
python -m photomemory.cli faces extract --sample 6000   # build a face sample for clustering
python -m photomemory.cli faces cluster                 # writes review montages to output/people_review/
#    Look at output/people_review/cluster_XXX_nNN.jpg, then name the ones you recognize:
python -m photomemory.cli faces label 9 "Erwan"
python -m photomemory.cli faces label 3 "Wei"
python -m photomemory.cli faces people                  # list enrolled people
#    From now on every build auto-detects faces on candidates and boosts family shots.

# 3. Build videos (output/*.mp4).
python -m photomemory.cli build month 2024-10 --length 60
python -m photomemory.cli build year 2024
python -m photomemory.cli build trip 0 --year 2024
python -m photomemory.cli build trip --around 2024-06-15
```

Add `--no-music` to any `build` command for a silent video.

## Face recognition (enrolling family)

Teaches the app who your family is so their photos are strongly favored in every build.
Faces are detected/embedded on the GPU (facenet-pytorch) and matched per-memory against
enrolled people; matching runs on the cached frames, so no extra NAS reads.

There are two ways to enroll, and they can be combined:

**A. Auto-cluster, then label** — good for people who appear a lot and look distinct:
```powershell
python -m photomemory.cli faces extract --sample 6000   # detect+embed a library sample
python -m photomemory.cli faces cluster                 # -> montages in output/people_review/
python -m photomemory.cli faces split 9                 # refine an over-merged cluster
python -m photomemory.cli faces label 9 "Erwan"         # name a cluster after viewing its montage
```

**B. Enroll from reference photos** — reliable for anyone clustering can't separate
(e.g. young siblings, or to avoid a look-alike friend). Point it at folders or specific
files; it picks the *dominant recurring face* and saves `output/people_review/enrolled_<name>.jpg`
so you can verify:
```powershell
python -m photomemory.cli faces enroll "Julien" "Z:\...\Portrait Julien" --add
python -m photomemory.cli faces enroll "Julien" "Z:\...\IMG_4197.JPG" "Z:\...\IMG_4316.JPG" --add
```

**Disambiguating look-alikes (e.g. brothers):** give each person a birthdate, and the
matcher compares a candidate against each person's *age-appropriate* prototypes (using each
photo's date), so siblings stop collapsing together:
```powershell
python -m photomemory.cli faces birthday "Erwan"  2008-10-12
python -m photomemory.cli faces birthday "Julien" 2012-01-16
```

**Date bounds:** restrict when a person can appear (prevents impossible matches):
```powershell
python -m photomemory.cli faces lifespan "Benedicte" --until 2016-04-30
```

`faces people` lists everyone enrolled. Strength of the boost is `scoring.w_family` in
`config.toml`. Once anyone is enrolled, every `build` favors them automatically.

## Performance notes

- The first `index` of 350K files reads EXIF per image over SMB; expect it to take a
  while (raise `scan.workers` in `config.toml` for a faster link). Re-runs skip unchanged
  files instantly.
- Scoring reads each candidate full-res once, then caches a downscaled render frame
  (`data/cache/frames`) and its scores — so rebuilding/re-rendering a memory is fast and
  never touches the NAS again.

## Project layout

```
photomemory/   config, db, dates, exif, geocode, scanner, grouping,
               dedupe, selector, memory, render, cli,
               facerec (GPU embeddings), faceindex (cluster/enroll/match)
               quality/  heuristics, faces, aesthetic, score
assets/music/  bundled synthesized tracks (gen_music.py)
data/          catalog.db + frame cache (gitignored)
output/        rendered mp4s
```

## Roadmap (not yet implemented)

- Web UI to browse the library, review/label face clusters, and approve a selection.
- Including short video clips in the montage.
- Beat-synced cuts.
