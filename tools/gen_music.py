"""Generate royalty-free background tracks (synthesized from scratch).

Produces upbeat, energetic beds (pad chords + arpeggio + a kick/hat groove) under
assets/music/. The earlier slow "calm/reflective" tracks are replaced with faster,
more energetic ones. Run:  python tools/gen_music.py
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 44100
OUT = Path(__file__).resolve().parent.parent / "assets" / "music"

A4 = 440.0
NOTES = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4, "F#": -3,
         "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}


def freq(name: str, octave: int) -> float:
    return A4 * (2 ** ((NOTES[name] + (octave - 4) * 12) / 12))


def adsr(n, a=0.02, d=0.1, s=0.7, r=0.2):
    env = np.ones(n)
    ai, di, ri = min(int(a * SR), n), min(int(d * SR), n), min(int(r * SR), n)
    if ai:
        env[:ai] = np.linspace(0, 1, ai)
    if di:
        env[ai:ai + di] = np.linspace(1, s, di)
    env[ai + di:n - ri] = s
    if ri:
        env[n - ri:] = np.linspace(s, 0, ri)
    return env


def pad(f, dur, gain=0.2):
    n = int(dur * SR); t = np.arange(n) / SR
    sig = np.zeros(n)
    for mult, g, det in [(1, 1.0, 0.0), (1, 0.6, 0.3), (2, 0.3, 0.0), (3, 0.14, 0.0)]:
        sig += g * np.sin(2 * np.pi * (f * mult + det) * t)
    return sig * adsr(n, a=0.05, d=0.3, s=0.85, r=0.25) * gain


def chord(names_oct, dur, gain=0.18):
    n = int(dur * SR); out = np.zeros(n)
    for name, octv in names_oct:
        out += pad(freq(name, octv), dur, gain / len(names_oct))
    return out


def pluck(f, dur, gain=0.16):
    n = int(dur * SR); t = np.arange(n) / SR
    sig = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 2 * f * t) + 0.25 * np.sin(2 * np.pi * 3 * f * t)
    return sig * np.exp(-5.0 * t) * gain


def kick(beat, gain=0.95):
    dur = min(beat * 0.9, 0.35); n = int(dur * SR); t = np.arange(n) / SR
    f = 110 * np.exp(-t * 32) + 47           # punchy pitch drop
    sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)
    return sig * gain


def hat(gain=0.22):
    dur = 0.045; n = int(dur * SR); t = np.arange(n) / SR
    return np.random.randn(n) * np.exp(-t * 90) * gain


def soft_reverb(x):
    out = x.copy()
    for delay_ms, decay in [(70, 0.25), (140, 0.16), (210, 0.1)]:
        d = int(delay_ms / 1000 * SR)
        echo = np.zeros_like(x); echo[d:] = x[:-d] * decay; out += echo
    return out


def normalize(x, peak=0.9):
    return x / (np.max(np.abs(x)) or 1.0) * peak


def write_wav(path, mono):
    mono = normalize(mono)
    stereo = np.stack([mono, np.roll(mono, 150)], axis=1)
    data = (stereo * 32767).astype(np.int16)
    with wave.open(str(path), "w") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(data.tobytes())
    print(f"wrote {path.name}  ({len(mono) / SR:.0f}s)")


def build_energetic(progression, arp_notes, bpm=124, bars=32, octave=4, swing_hat=True):
    """Driving track: sustained pad chords + eighth-note arpeggio + kick/hat groove."""
    beat = 60 / bpm
    total_beats = bars * 4
    length = total_beats * beat
    n = int(length * SR)
    bed = np.zeros(n); arp = np.zeros(n); perc = np.zeros(n)

    # One chord per bar, in a lower octave.
    for bar in range(bars):
        ch = progression[bar % len(progression)]
        s = int(bar * 4 * beat * SR)
        c = chord([(nm, octave - 1) for nm in ch], 4 * beat, gain=0.16)
        e = min(s + len(c), n); bed[s:e] += c[:e - s]

    # Arpeggio on eighth notes.
    step = beat / 2; pos = 0.0; i = 0
    while pos < length - step:
        nm, o = arp_notes[i % len(arp_notes)]
        p = pluck(freq(nm, o), step * 1.3)
        s = int(pos * SR); e = min(s + len(p), n); arp[s:e] += p[:e - s]
        pos += step; i += 1

    # Kick on every beat; hat on the off-beats.
    k = kick(beat)
    for b in range(total_beats):
        s = int(b * beat * SR); e = min(s + len(k), n); perc[s:e] += k[:e - s]
    h = hat(); pos = beat / 2
    while pos < length:
        s = int(pos * SR); e = min(s + len(h), n); perc[s:e] += h[:e - s]
        pos += beat

    return soft_reverb(bed + 0.85 * arp) + 0.8 * perc   # keep drums dry & punchy


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Remove the old slow tracks if present.
    for old in ("calm.wav", "reflective.wav"):
        p = OUT / old
        if p.exists():
            p.unlink(); print(f"removed {old} (too slow)")

    # Energetic, bright — I–V–vi–IV in C, fast arpeggio.
    write_wav(OUT / "energetic.wav", build_energetic(
        progression=[["C", "E", "G"], ["G", "B", "D"], ["A", "C", "E"], ["F", "A", "C"]],
        arp_notes=[("C", 5), ("E", 5), ("G", 5), ("C", 6), ("G", 5), ("E", 5)],
        bpm=126, bars=32))

    # Upbeat, driving — vi–IV–I–V, slightly faster.
    write_wav(OUT / "upbeat.wav", build_energetic(
        progression=[["A", "C", "E"], ["F", "A", "C"], ["C", "E", "G"], ["G", "B", "D"]],
        arp_notes=[("A", 5), ("C", 6), ("E", 5), ("G", 5)],
        bpm=132, bars=34))

    # Uplifting — mid-up tempo with a lighter groove.
    write_wav(OUT / "uplifting.wav", build_energetic(
        progression=[["D", "F#", "A"], ["A", "C#", "E"], ["B", "D", "F#"], ["G", "B", "D"]],
        arp_notes=[("D", 5), ("F#", 5), ("A", 5), ("F#", 5)],
        bpm=118, bars=30))


if __name__ == "__main__":
    main()
