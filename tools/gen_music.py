"""Generate simple, fully royalty-free background tracks (synthesized from scratch).

Produces a few mood-tagged WAV files under assets/music/. These are intentionally
gentle ambient beds (pad chords + soft arpeggio) meant to sit under a photo montage.
Run:  python tools/gen_music.py
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 44100
OUT = Path(__file__).resolve().parent.parent / "assets" / "music"

A4 = 440.0
NOTES = {  # semitone offsets from A4
    "C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4, "F#": -3,
    "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2,
}


def freq(name: str, octave: int) -> float:
    semis = NOTES[name] + (octave - 4) * 12
    return A4 * (2 ** (semis / 12))


def adsr(n: int, a=0.02, d=0.1, s=0.7, r=0.2) -> np.ndarray:
    env = np.ones(n)
    ai, di, ri = int(a * SR), int(d * SR), int(r * SR)
    ai, di, ri = min(ai, n), min(di, n), min(ri, n)
    if ai:
        env[:ai] = np.linspace(0, 1, ai)
    if di:
        env[ai:ai + di] = np.linspace(1, s, di)
    env[ai + di:n - ri] = s
    if ri:
        env[n - ri:] = np.linspace(s, 0, ri)
    return env


def pad(f: float, dur: float, gain=0.2) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    # detuned partials for warmth
    sig = np.zeros(n)
    for mult, g, det in [(1, 1.0, 0.0), (1, 0.6, 0.3), (2, 0.25, 0.0), (3, 0.12, 0.0)]:
        sig += g * np.sin(2 * np.pi * (f * mult + det) * t)
    sig *= adsr(n, a=0.4, d=0.5, s=0.8, r=0.6)
    return sig * gain


def pluck(f: float, dur: float, gain=0.18) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 2 * f * t)
    sig *= np.exp(-4.0 * t)  # fast decay -> plucky
    return sig * gain


def chord(names_oct: list[tuple[str, int]], dur: float, gain=0.2) -> np.ndarray:
    n = int(dur * SR)
    out = np.zeros(n)
    for name, octv in names_oct:
        out += pad(freq(name, octv), dur, gain / len(names_oct))
    return out


def soft_reverb(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    for delay_ms, decay in [(60, 0.3), (130, 0.2), (200, 0.12)]:
        d = int(delay_ms / 1000 * SR)
        echo = np.zeros_like(x)
        echo[d:] = x[:-d] * decay
        out += echo
    return out


def normalize(x: np.ndarray, peak=0.85) -> np.ndarray:
    m = np.max(np.abs(x)) or 1.0
    return x / m * peak


def write_wav(path: Path, mono: np.ndarray):
    mono = normalize(mono)
    stereo = np.stack([mono, np.roll(mono, 200)], axis=1)  # tiny width
    data = (stereo * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())
    print(f"wrote {path.name}  ({len(mono)/SR:.0f}s)")


def build_track(progression, arp_notes, beat=2.0, repeats=8, octave=3) -> np.ndarray:
    """Lay chords end-to-end, sprinkle an arpeggio over the top."""
    chord_dur = beat * 2
    parts = []
    for _ in range(repeats):
        for ch in progression:
            parts.append(chord([(nm, octave) for nm in ch], chord_dur))
    bed = np.concatenate(parts)

    # Arpeggio layer aligned to a faster grid.
    arp = np.zeros_like(bed)
    step = int(beat / 2 * SR)
    notes = arp_notes
    pos = 0
    i = 0
    while pos + step < len(arp):
        nm, octv = notes[i % len(notes)]
        p = pluck(freq(nm, octv), beat / 2 * 1.1)
        end = min(pos + len(p), len(arp))
        arp[pos:end] += p[:end - pos]
        pos += step
        i += 1
    mix = bed + 0.7 * arp
    return soft_reverb(mix)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Uplifting: I–V–vi–IV in C major.
    uplifting = build_track(
        progression=[["C", "E", "G"], ["G", "B", "D"], ["A", "C", "E"], ["F", "A", "C"]],
        arp_notes=[("C", 5), ("E", 5), ("G", 5), ("E", 5), ("A", 4), ("C", 5)],
        beat=1.6, repeats=8,
    )
    write_wav(OUT / "uplifting.wav", uplifting)

    # Calm/nostalgic: vi–IV–I–V in A minor feel, slower.
    calm = build_track(
        progression=[["A", "C", "E"], ["F", "A", "C"], ["C", "E", "G"], ["G", "B", "D"]],
        arp_notes=[("A", 4), ("E", 5), ("C", 5), ("E", 5)],
        beat=2.2, repeats=6,
    )
    write_wav(OUT / "calm.wav", calm)

    # Reflective: longer pads, sparse arp.
    reflective = build_track(
        progression=[["D", "F", "A"], ["A", "C", "E"], ["B", "D", "F"], ["G", "B", "D"]],
        arp_notes=[("D", 5), ("A", 4), ("F", 5), ("A", 4)],
        beat=2.6, repeats=6,
    )
    write_wav(OUT / "reflective.wav", reflective)


if __name__ == "__main__":
    main()
