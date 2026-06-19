"""Generate royalty-free background tracks (synthesized from scratch).

Energetic beds with actual song structure (intro -> verse -> chorus w/ lead melody ->
variation -> outro), drum fills and a snare backbeat, so they evolve instead of looping
the same bars. Output under assets/music/. Run:  python tools/gen_music.py
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


def freq(name, octave):
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


# --- instruments ---------------------------------------------------------------

def pad(f, dur, gain=0.2):
    n = int(dur * SR); t = np.arange(n) / SR
    sig = sum(g * np.sin(2 * np.pi * (f * m + det) * t)
              for m, g, det in [(1, 1.0, 0), (1, 0.6, 0.3), (2, 0.3, 0), (3, 0.14, 0)])
    return sig * adsr(n, a=0.05, d=0.3, s=0.85, r=0.25) * gain


def chord(names_oct, dur, gain=0.15):
    n = int(dur * SR); out = np.zeros(n)
    for name, octv in names_oct:
        out += pad(freq(name, octv), dur, gain / len(names_oct))
    return out


def pluck(f, dur, gain=0.16):
    n = int(dur * SR); t = np.arange(n) / SR
    sig = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 2 * f * t) + 0.25 * np.sin(2 * np.pi * 3 * f * t)
    return sig * np.exp(-5.0 * t) * gain


def lead(f, dur, gain=0.24):
    n = int(dur * SR); t = np.arange(n) / SR
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.5 * t)
    sig = np.sin(2 * np.pi * f * t * vib) + 0.3 * np.sin(2 * np.pi * 2 * f * t * vib)
    return sig * adsr(n, a=0.02, d=0.12, s=0.8, r=0.12) * gain


def kick(beat, gain=0.95):
    dur = min(beat * 0.9, 0.35); n = int(dur * SR); t = np.arange(n) / SR
    f = 110 * np.exp(-t * 32) + 47
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9) * gain


def snare(gain=0.5):
    dur = 0.13; n = int(dur * SR); t = np.arange(n) / SR
    return (np.random.randn(n) + 0.5 * np.sin(2 * np.pi * 190 * t)) * np.exp(-t * 28) * gain


def hat(gain=0.2):
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
    with wave.open(str(path), "w") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((stereo * 32767).astype(np.int16).tobytes())
    print(f"wrote {path.name}  ({len(mono) / SR:.0f}s)")


# --- arrangement ---------------------------------------------------------------

def _section(prog, bars, beat, arp=None, octave=4, drums=True, fill=False, lead_phrase=None):
    """Render one section's layers (bed, arp, perc, lead) as same-length arrays."""
    length = bars * 4 * beat; n = int(length * SR)
    bed = np.zeros(n); arpb = np.zeros(n); perc = np.zeros(n); leadb = np.zeros(n)

    for bar in range(bars):
        s = int(bar * 4 * beat * SR)
        c = chord([(nm, octave - 1) for nm in prog[bar % len(prog)]], 4 * beat)
        e = min(s + len(c), n); bed[s:e] += c[:e - s]

    if arp:
        step = beat / 2; pos = 0.0; i = 0
        while pos < length - 1e-6:
            nm, o = arp[i % len(arp)]
            p = pluck(freq(nm, o), step * 1.3)
            s = int(pos * SR); e = min(s + len(p), n); arpb[s:e] += p[:e - s]
            pos += step; i += 1

    if drums:
        total = bars * 4; k = kick(beat); sn = snare(); h = hat()
        for b in range(total):
            if fill and b >= total - 2:        # last half-bar -> snare fill
                continue
            s = int(b * beat * SR)
            e = min(s + len(k), n); perc[s:e] += k[:e - s]
            if b % 4 in (1, 3):                 # backbeat on 2 & 4
                e = min(s + len(sn), n); perc[s:e] += sn[:e - s]
        pos = beat / 2                          # offbeat hats
        while pos < length:
            s = int(pos * SR); e = min(s + len(h), n); perc[s:e] += h[:e - s]; pos += beat
        if fill:                                # 16th snare roll into next section
            pos = (total - 2) * beat
            while pos < length:
                sr = snare(0.4); s = int(pos * SR); e = min(s + len(sr), n)
                perc[s:e] += sr[:e - s]; pos += beat / 4

    if lead_phrase:
        pos = 0.0; i = 0
        while pos < length - 1e-6:
            nm, o, bts = lead_phrase[i % len(lead_phrase)]
            dur = bts * beat
            l = lead(freq(nm, o), dur * 0.92)
            s = int(pos * SR); e = min(s + len(l), n); leadb[s:e] += l[:e - s]
            pos += dur; i += 1

    return bed, arpb, perc, leadb


def build_song(prog_a, prog_b, arp_a, arp_b, lead_b, bpm=126, key_oct=4):
    """intro -> verse -> chorus(lead) -> verse(var) -> chorus(lead, var) -> outro."""
    beat = 60 / bpm
    arrangement = [
        (prog_a[:1], 2, dict(arp=None, drums=False)),                                  # intro pads
        (prog_a, 8, dict(arp=arp_a, drums=True, fill=True)),                           # verse 1
        (prog_b, 8, dict(arp=arp_b, drums=True, lead_phrase=lead_b, fill=True)),       # chorus 1
        (prog_a, 8, dict(arp=[(n, o + 1) for n, o in arp_a], drums=True, fill=True)),  # verse 2 (8va arp)
        (prog_b, 8, dict(arp=arp_b, drums=True,
                         lead_phrase=[(n, o, b) for n, o, b in lead_b])),              # chorus 2
        (prog_a[:1], 2, dict(arp=None, drums=False)),                                  # outro pads
    ]
    layers = [[], [], [], []]
    for prog, bars, opts in arrangement:
        for i, arr in enumerate(_section(prog, bars, beat, octave=key_oct, **opts)):
            layers[i].append(arr)
    bed, arp, perc, ld = (np.concatenate(L) for L in layers)
    return soft_reverb(bed + 0.8 * arp + 0.85 * ld) + 0.85 * perc


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in ("calm.wav", "reflective.wav"):
        p = OUT / old
        if p.exists():
            p.unlink(); print(f"removed {old}")

    C = {"C": ["C", "E", "G"], "G": ["G", "B", "D"], "Am": ["A", "C", "E"], "F": ["F", "A", "C"]}
    write_wav(OUT / "energetic.wav", build_song(
        prog_a=[C["C"], C["G"], C["Am"], C["F"]],
        prog_b=[C["F"], C["C"], C["G"], C["Am"]],
        arp_a=[("C", 5), ("E", 5), ("G", 5), ("E", 5)],
        arp_b=[("C", 5), ("G", 5), ("C", 6), ("G", 5)],
        lead_b=[("E", 5, 1), ("G", 5, 1), ("A", 5, 1), ("G", 5, 1),
                ("C", 6, 2), ("B", 5, 1), ("G", 5, 1)],
        bpm=126))

    Am = {"Am": ["A", "C", "E"], "F": ["F", "A", "C"], "C": ["C", "E", "G"], "G": ["G", "B", "D"]}
    write_wav(OUT / "upbeat.wav", build_song(
        prog_a=[Am["Am"], Am["F"], Am["C"], Am["G"]],
        prog_b=[Am["C"], Am["G"], Am["Am"], Am["F"]],
        arp_a=[("A", 5), ("C", 6), ("E", 6), ("C", 6)],
        arp_b=[("A", 5), ("E", 6), ("A", 6), ("E", 6)],
        lead_b=[("A", 5, 1), ("C", 6, 1), ("E", 6, 2), ("D", 6, 1), ("C", 6, 1), ("A", 5, 2)],
        bpm=132))

    D = {"D": ["D", "F#", "A"], "A": ["A", "C#", "E"], "Bm": ["B", "D", "F#"], "G": ["G", "B", "D"]}
    write_wav(OUT / "uplifting.wav", build_song(
        prog_a=[D["D"], D["A"], D["Bm"], D["G"]],
        prog_b=[D["G"], D["D"], D["A"], D["Bm"]],
        arp_a=[("D", 5), ("F#", 5), ("A", 5), ("F#", 5)],
        arp_b=[("D", 5), ("A", 5), ("D", 6), ("A", 5)],
        lead_b=[("F#", 5, 1), ("A", 5, 1), ("B", 5, 2), ("A", 5, 1), ("F#", 5, 1), ("D", 5, 2)],
        bpm=118))


if __name__ == "__main__":
    main()
