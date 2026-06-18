"""Video rendering with moviepy: Ken Burns slides, crossfades, title/caption cards, music.

Reads only the cached render frames (created during scoring), so rendering is fast and
never has to touch the NAS or decode HEIC at render time. Title and caption cards are
drawn with Pillow (no ImageMagick dependency).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from .config import get_config
from .memory import Memory
from .quality.score import ensure_frame

# moviepy 1.x API
from moviepy.editor import (  # noqa: E402
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_videoclips,
)
from moviepy.audio.fx import all as afx  # noqa: E402


# --- Fonts ---------------------------------------------------------------------

def _font(size: int) -> ImageFont.FreeTypeFont:
    for cand in (
        get_config().raw["paths"].get("font"),
        r"C:\Windows\Fonts\seguisb.ttf",   # Segoe UI Semibold
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ):
        if cand and Path(cand).exists():
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                continue
    return ImageFont.load_default()


# --- Text cards ----------------------------------------------------------------

def _text_card(size: tuple[int, int], title: str, subtitle: str | None,
               duration: float) -> ImageClip:
    w, h = size
    img = Image.new("RGB", size, (12, 12, 14))
    draw = ImageDraw.Draw(img)
    tf = _font(int(h * 0.085))
    cx, cy = w // 2, h // 2

    def centered(text, font, y, fill):
        bb = draw.textbbox((0, 0), text, font=font)
        draw.text((cx - (bb[2] - bb[0]) // 2, y), text, font=font, fill=fill)

    centered(title, tf, cy - int(h * 0.07), (245, 245, 245))
    if subtitle:
        sf = _font(int(h * 0.04))
        centered(subtitle, sf, cy + int(h * 0.03), (180, 180, 185))
    return ImageClip(_to_array(img)).set_duration(duration)


def _to_array(img: Image.Image):
    import numpy as np

    return np.asarray(img)


def _caption_overlay(size: tuple[int, int], text: str, duration: float) -> ImageClip:
    """A lower-third caption with a soft gradient backing, transparent elsewhere."""
    import numpy as np

    w, h = size
    rgba = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(rgba)
    band_h = int(h * 0.18)
    for i in range(band_h):  # bottom-up gradient
        a = int(150 * (i / band_h))
        draw.line([(0, h - i), (w, h - i)], fill=(0, 0, 0, a))
    f = _font(int(h * 0.045))
    bb = draw.textbbox((0, 0), text, font=f)
    draw.text((int(w * 0.05), h - int(h * 0.11)), text, font=f, fill=(255, 255, 255, 255))
    arr = np.asarray(rgba)
    clip = ImageClip(arr[:, :, :3]).set_duration(duration)
    mask = ImageClip(arr[:, :, 3] / 255.0, ismask=True).set_duration(duration)
    return clip.set_mask(mask)


# --- Ken Burns slides ----------------------------------------------------------

def _compose_frame(frame_path: Path, size: tuple[int, int]) -> np.ndarray:
    """Fit the WHOLE photo inside the frame (no cropping), filling the leftover space
    with a blurred, darkened copy of itself instead of black bars."""
    cw, ch = size
    im = Image.open(frame_path).convert("RGB")

    # Background: cover-crop to canvas, then blur + darken.
    bg = ImageOps.fit(im, size, method=Image.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(40))
    bg = ImageEnhance.Brightness(bg).enhance(0.45)

    # Foreground: contain the entire image within the canvas.
    fg = im.copy()
    fg.thumbnail(size, Image.LANCZOS)
    canvas = bg
    canvas.paste(fg, ((cw - fg.width) // 2, (ch - fg.height) // 2))
    return np.asarray(canvas)


def _slide(frame_path: Path, duration: float, size: tuple[int, int],
           caption: str | None) -> CompositeVideoClip:
    zoom = float(get_config().render.get("ken_burns_zoom", 0.04))
    base = ImageClip(_compose_frame(frame_path, size))  # already exactly canvas-sized

    # Gentle zoom only (whole photo is visible at t=0, barely cropped by the end).
    if random.choice([True, False]):
        moving = base.resize(lambda t: 1 + zoom * t / duration)
    else:
        moving = base.resize(lambda t: 1 + zoom * (1 - t / duration))
    moving = moving.set_duration(duration).set_position(("center", "center"))

    layers = [moving]
    if caption:
        layers.append(_caption_overlay(size, caption, duration))
    return CompositeVideoClip(layers, size=size).set_duration(duration)


# --- Music ---------------------------------------------------------------------

def _pick_music() -> Path | None:
    music_dir = get_config().raw["paths"].get("music_dir")
    base = Path(music_dir) if music_dir else (get_config().db_path.parent.parent / "assets" / "music")
    if not base.exists():
        return None
    tracks = [p for p in base.iterdir() if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".ogg")]
    return random.choice(tracks) if tracks else None


# --- Public entrypoint ---------------------------------------------------------

def render_memory(memory: Memory, out_path: Path, *, music: bool = True,
                  length_seconds: float | None = None) -> Path:
    cfg = get_config().render
    size = (int(cfg["width"]), int(cfg["height"]))
    fps = int(cfg["fps"])
    transition = float(cfg["transition_seconds"])

    n = len(memory.selection)
    if n == 0:
        raise ValueError("Memory has no photos to render.")
    spp = float(cfg["seconds_per_photo"])
    if length_seconds:
        spp = max(1.2, length_seconds / n)

    # Build slides, inserting a caption only when the group (event) changes.
    slides = []
    last_group = None
    for s in memory.selection:
        frame = ensure_frame(s.media_id, Path(s.path))
        if frame is None:
            continue
        caption = None
        if s.group != last_group:
            caption = memory.captions.get(s.group)
            last_group = s.group
        slides.append(_slide(frame, spp, size, caption))

    title_card = _text_card(size, memory.title, memory.subtitle, max(2.5, spp))
    clips = [title_card] + slides

    faded = [clips[0]]
    for c in clips[1:]:
        faded.append(c.crossfadein(transition))
    video = concatenate_videoclips(faded, method="compose", padding=-transition)

    if music:
        track = _pick_music()
        if track:
            audio = AudioFileClip(str(track))
            audio = afx.audio_loop(audio, duration=video.duration)
            audio = audio.fx(afx.audio_fadein, 1.0).fx(afx.audio_fadeout, 2.0)
            video = video.set_audio(audio)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    video.write_videofile(
        str(out_path), fps=fps, codec="libx264", audio_codec="aac",
        threads=4, preset="medium", logger=None,
    )
    return out_path
