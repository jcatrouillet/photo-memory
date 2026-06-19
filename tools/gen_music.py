"""Generate three royalty-free background tracks (synthesized from scratch), each a
distinct style/mood (not just different tempos):

  trip_hop.wav  - trip-hop, inspired by Massive Attack "Teardrop": harpsichord arpeggio,
                  sub-bass, halftime beat with a big reverb snare. Moody.
  downtempo.wav - downtempo soul, inspired by Morcheeba "The Sea": warm Rhodes 7th chords,
                  laid-back swung groove, mellow sax lead. Smooth.
  stomp.wav     - stadium stomp, inspired by Queen "We Will Rock You": stomp-stomp-clap
                  body percussion, power-chord stabs, anthem lead. Big.

Run:  python tools/gen_music.py
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


def _t(dur):
    n = int(dur * SR)
    return n, np.arange(n) / SR


def place(buf, arr, t):
    s = int(t * SR); e = min(s + len(arr), len(buf))
    if e > s:
        buf[s:e] += arr[:e - s]


def soft_reverb(x, taps=((70, .25), (140, .16), (210, .1), (310, .06))):
    out = x.copy()
    for ms, decay in taps:
        d = int(ms / 1000 * SR); echo = np.zeros_like(x); echo[d:] = x[:-d] * decay; out += echo
    return out


def normalize(x, peak=0.9):
    return x / (np.max(np.abs(x)) or 1.0) * peak


def write_wav(path, mono):
    mono = normalize(mono)
    stereo = np.stack([mono, np.roll(mono, 220)], axis=1)
    with wave.open(str(path), "w") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((stereo * 32767).astype(np.int16).tobytes())
    print(f"wrote {path.name}  ({len(mono) / SR:.0f}s)")


# --- instruments ---------------------------------------------------------------

def harpsichord(f, dur, gain=0.22):
    n, t = _t(dur)
    sig = sum((1 / h) * np.sin(2 * np.pi * f * h * t) for h in (1, 2, 3, 4, 5, 6))
    env = np.exp(-3.2 * t) * (1 - np.exp(-300 * t))
    return sig * env * gain


def rhodes(f, dur, gain=0.2):
    n, t = _t(dur)
    car = np.sin(2 * np.pi * f * t + 3 * np.exp(-4 * t) * np.sin(2 * np.pi * 2 * f * t))
    trem = 1 + 0.12 * np.sin(2 * np.pi * 5 * t)
    return car * adsr(n, a=0.01, d=0.5, s=0.6, r=0.3) * trem * gain


def subbass(f, dur, gain=0.55):
    n, t = _t(dur)
    sig = np.sin(2 * np.pi * f * t) + 0.2 * np.sin(2 * np.pi * 2 * f * t)
    return sig * adsr(n, a=0.01, d=0.1, s=0.9, r=0.12) * gain


def powerchord(root, octave, dur, gain=0.16):
    n, t = _t(dur)
    sig = np.zeros(n)
    for f in (freq(root, octave), freq(root, octave) * 1.5, freq(root, octave) * 2):
        sig += sum((1 / h) * np.sin(2 * np.pi * f * h * t) for h in (1, 2, 3))
    return sig * adsr(n, a=0.005, d=0.25, s=0.5, r=0.2) * gain / 3


def lead(f, dur, gain=0.24, bright=False):
    n, t = _t(dur)
    vib = 1 + 0.007 * np.sin(2 * np.pi * 5.5 * t)
    sig = np.sin(2 * np.pi * f * t * vib)
    sig += (0.5 if bright else 0.25) * np.sin(2 * np.pi * 2 * f * t * vib)
    if bright:
        sig += 0.2 * np.sin(2 * np.pi * 3 * f * t * vib)
    return sig * adsr(n, a=0.03, d=0.12, s=0.8, r=0.15) * gain


def kick(beat, gain=0.95):
    dur = min(beat * 0.9, 0.35); n, t = _t(dur)
    f = 110 * np.exp(-t * 32) + 47
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9) * gain


def snare(gain=0.5, decay=28):
    n, t = _t(0.16)
    return (np.random.randn(n) + 0.5 * np.sin(2 * np.pi * 190 * t)) * np.exp(-t * decay) * gain


def brush(gain=0.3):
    n, t = _t(0.18)
    return np.random.randn(n) * np.exp(-t * 12) * gain


def hat(gain=0.2):
    n, t = _t(0.045)
    return np.random.randn(n) * np.exp(-t * 90) * gain


def stomp(gain=0.95):
    n, t = _t(0.22)
    f = 90 * np.exp(-t * 18) + 50
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 13)
    thud = np.random.randn(n) * np.exp(-t * 45) * 0.4
    return (body + thud) * gain


def clap(gain=0.6):
    n, t = _t(0.2); out = np.zeros(n)
    for off in (0.0, 0.009, 0.018, 0.028):
        out += np.random.randn(n) * np.exp(-np.maximum(t - off, 0) * 55)
    return out * gain * 0.4


# --- tracks --------------------------------------------------------------------

def build_trip_hop(bpm=78):
    """Massive Attack 'Teardrop' vibe: harpsichord arpeggio + sub + halftime reverb-snare."""
    beat = 60 / bpm; bar = 4 * beat
    chords = {"Am": ("A", "C", "E", "A"), "F": ("F", "A", "C", "F"),
              "C": ("C", "E", "G", "C"), "G": ("G", "B", "D", "G")}
    seq = ["Am", "Am", "F", "F", "C", "C", "G", "G"]
    bars = len(seq) * 3
    N = int(bars * bar * SR)
    harp = np.zeros(N); bass = np.zeros(N); kicks = np.zeros(N); sn = np.zeros(N); hats = np.zeros(N)
    pat = [0, 2, 1, 2, 3, 2, 1, 2]
    for b in range(bars):
        tones = chords[seq[b % len(seq)]]; t0 = b * bar
        for e in range(8):
            place(harp, harpsichord(freq(tones[pat[e]], 4 if pat[e] < 3 else 5), beat * 0.55),
                  t0 + e * (beat / 2))
        place(bass, subbass(freq(tones[0], 2), bar * 0.95), t0)
        place(kicks, kick(beat, 1.0), t0)
        place(kicks, kick(beat, 0.55), t0 + 2.5 * beat)   # syncopated ghost kick
        place(sn, snare(0.6), t0 + 2 * beat)              # halftime backbeat (beat 3)
        for e in range(8):                                # swung closed hats
            place(hats, hat(0.13), t0 + e * (beat / 2) + (0.06 * beat if e % 2 else 0))
    return soft_reverb(harp * 0.9) + bass + kicks + soft_reverb(sn) * 0.85 + hats * 0.6


def build_downtempo(bpm=92):
    """Morcheeba 'The Sea' vibe: warm Rhodes 7th chords, swung soft groove, sax lead."""
    beat = 60 / bpm; bar = 4 * beat
    ch = {"Fmaj7": ["F", "A", "C", "E"], "Dm7": ["D", "F", "A", "C"],
          "Gm7": ["G", "A#", "D", "F"], "C7": ["C", "E", "G", "A#"]}
    seq = ["Fmaj7", "Dm7", "Gm7", "C7"]
    reps = 4; bars = len(seq) * 2 * reps
    N = int(bars * bar * SR)
    keys = np.zeros(N); bass = np.zeros(N); perc = np.zeros(N); ld = np.zeros(N)
    sax = [("F", 5, 2), ("A", 5, 1), ("G", 5, 1), ("F", 5, 2), ("E", 5, 1), ("D", 5, 1),
           ("C", 5, 4)]
    for b in range(bars):
        name = seq[(b // 2) % len(seq)]; tones = ch[name]; t0 = b * bar
        if b % 2 == 0:                                    # sustain chord across 2 bars
            for nm in tones:
                place(keys, rhodes(freq(nm, 4), 2 * bar * 0.98, gain=0.16), t0)
        place(bass, subbass(freq(tones[0], 2), beat * 1.4, 0.5), t0)               # root
        place(bass, subbass(freq(tones[2], 2), beat * 1.4, 0.45), t0 + 2 * beat)   # fifth
        place(perc, kick(beat, 0.7), t0); place(perc, kick(beat, 0.5), t0 + 2 * beat)
        place(perc, brush(0.32), t0 + beat); place(perc, brush(0.32), t0 + 3 * beat)
        for e in range(4):
            place(perc, hat(0.1), t0 + e * beat + beat / 2 + 0.05 * beat)
    pos = 8 * bar; i = 0                                  # sax lead enters second half
    while pos < bars * bar - beat:
        nm, o, bts = sax[i % len(sax)]
        place(ld, lead(freq(nm, o), bts * beat * 0.95, gain=0.2), pos); pos += bts * beat; i += 1
    return soft_reverb(keys + 0.9 * ld) + bass + 0.7 * perc


def build_stomp(bpm=81):
    """Queen 'We Will Rock You' vibe: stomp-stomp-clap, power-chord stabs, anthem lead."""
    beat = 60 / bpm; bar = 4 * beat
    # intro(4) verse(8) chorus(8) verse(8) chorus(8) outro(4)
    plan = [("intro", 4), ("verse", 8), ("chorus", 8), ("verse", 8), ("chorus", 8), ("outro", 4)]
    bars = sum(n for _, n in plan)
    N = int(bars * bar * SR)
    perc = np.zeros(N); chords_buf = np.zeros(N); bass = np.zeros(N); ld = np.zeros(N)
    melody = [("E", 4, 1), ("E", 4, 1), ("G", 4, 1), ("E", 4, 1),
              ("A", 4, 2), ("G", 4, 1), ("E", 4, 1)]
    bar_i = 0
    for sec, count in plan:
        for _ in range(count):
            t0 = bar_i * bar
            for half in (0, 2):                          # two "stomp stomp clap" per bar
                place(perc, stomp(0.95), t0 + half * beat)
                place(perc, stomp(0.95), t0 + (half + 0.5) * beat)
                place(perc, clap(0.7), t0 + (half + 1) * beat)
            if sec in ("verse", "chorus", "outro"):
                place(bass, subbass(freq("E", 2), bar * 0.6, 0.5), t0)
            if sec in ("chorus", "outro"):               # power-chord stabs on the claps
                for half in (0, 2):
                    place(chords_buf, powerchord("E", 3, beat * 0.8, 0.18), t0 + (half + 1) * beat)
            bar_i += 1
    pos = (4 + 8) * bar; i = 0                            # anthem lead over first chorus on
    end = (4 + 8 + 8 + 8 + 8) * bar
    while pos < end - beat:
        nm, o, bts = melody[i % len(melody)]
        place(ld, lead(freq(nm, o), bts * beat * 0.9, gain=0.26, bright=True), pos)
        pos += bts * beat; i += 1
    return soft_reverb(chords_buf + 0.9 * ld) + bass + perc


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in ("calm.wav", "reflective.wav", "energetic.wav", "upbeat.wav", "uplifting.wav"):
        p = OUT / old
        if p.exists():
            p.unlink(); print(f"removed {old}")
    write_wav(OUT / "trip_hop.wav", build_trip_hop())
    write_wav(OUT / "downtempo.wav", build_downtempo())
    write_wav(OUT / "stomp.wav", build_stomp())


if __name__ == "__main__":
    main()
