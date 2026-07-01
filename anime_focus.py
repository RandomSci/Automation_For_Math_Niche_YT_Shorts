# ============================================================
# VAULTS OF HISTORY - AI Video Generator v3
# Two-call GPT: Story Beats → Render Decisions
# OpenCV + Pillow renderer with fade in/out animation
# Proper timing via timestamp_hint + fuzzy anchor matching
# ============================================================

import subprocess
import os
import json
import random
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import traceback
import glob
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import cv2
from PIL import Image

load_dotenv()

from openai import OpenAI
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Vaults of History v3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

current_job = {"status": "idle", "progress": 0, "output": None, "error": None, "started_at": None}

OUTPUT_WIDTH  = 1920
OUTPUT_HEIGHT = 1080
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    print("⚠  WARNING: OPENAI_API_KEY not set.")

# Toggle: True = generate animated background procedurally (no broll, no
# clip failures, zero cost). False = use the old broll-clip pipeline.
USE_PROCEDURAL_BACKGROUND = True

MUSIC_MAP = {
    "space":    "bg_musics/space_ambient.mp3",
    "death":    "bg_musics/dark_ambient.mp3",
    "ancient":  "bg_musics/ancient_ambient.mp3",
    "religion": "bg_musics/sacred_ambient.mp3",
    "human":    "bg_musics/human_ambient.mp3",
    "default":  "bg_musics/vaults_ambient.mp3",
}

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

FONT_BLACK_CANDIDATES = [
    os.path.join(FONTS_DIR, "Anton-Regular.ttf"),
    os.path.join(FONTS_DIR, "Montserrat-Black.ttf"),
    os.path.join(FONTS_DIR, "Montserrat-ExtraBold.ttf"),
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
FONT_BOLD_CANDIDATES = [
    os.path.join(FONTS_DIR, "Montserrat-ExtraBold.ttf"),
    os.path.join(FONTS_DIR, "Montserrat-Bold.ttf"),
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
FONT_REGULAR_CANDIDATES = [
    os.path.join(FONTS_DIR, "Montserrat-Bold.ttf"),
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

def find_font(candidates):
    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            print(f"  ✓ Font: {path}")
            return path
    return None

FONT_BLACK   = find_font(FONT_BLACK_CANDIDATES)
FONT_BOLD    = find_font(FONT_BOLD_CANDIDATES)
FONT_REGULAR = find_font(FONT_REGULAR_CANDIDATES)

def get_primary_font_path(bold: bool = True) -> str:
    """Return best available font: Black > ExtraBold > Bold > anything."""
    if FONT_BLACK:   return FONT_BLACK
    if FONT_BOLD:    return FONT_BOLD
    if FONT_REGULAR: return FONT_REGULAR
    return None

def _probe_clip_health(filepath: str) -> tuple[bool, str]:
    """Quick ffprobe check: can this file actually be decoded?
    Returns (is_healthy, reason_if_not)."""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height,duration,codec_name',
           '-of', 'json', filepath]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "ffprobe failed"
        return False, err[:120]
    try:
        data = json.loads(result.stdout)
        streams = data.get('streams', [])
        if not streams:
            return False, "no video stream found"
        s = streams[0]
        if not s.get('width') or not s.get('height'):
            return False, "missing width/height"
        return True, ""
    except Exception as e:
        return False, f"parse error: {e}"


@app.on_event("startup")
async def startup_event():
    print("🚀 Vaults of History v3 starting...")
    # Audit broll folders so missing clips are immediately visible
    broll_dirs = ['space_vids','ancient_ruins_vids','cosmic_vids',
                  'dark_sky_vids','temple_vids']
    print("📁 Broll folder audit:")
    bad_clips = []
    for d in broll_dirs:
        if os.path.exists(d):
            files = [f for f in os.listdir(d) if f.lower().endswith(('.mp4','.mov','.avi'))]
            status = f"✅ {len(files)} clips" if files else "❌ EMPTY -- add Seedance clips here"
            print(f"  {d}: {status}")
            # Health-check each clip so broken files are caught before a run
            for f in files:
                fpath = os.path.join(d, f)
                healthy, reason = _probe_clip_health(fpath)
                if not healthy:
                    bad_clips.append((fpath, reason))
        else:
            print(f"  {d}: ❌ MISSING -- folder doesn't exist")

    if bad_clips:
        print("⚠️  BROKEN CLIPS DETECTED (these will render as black filler):")
        for fpath, reason in bad_clips:
            print(f"    ✗ {fpath} -- {reason}")
        print(f"  → Replace or remove these {len(bad_clips)} file(s) to eliminate black segments.")
    else:
        print("  ✅ All clips passed health check")



# ============================================================
# PROCEDURAL BACKGROUND GENERATOR
# Replaces broll entirely. No clip failures, no black fillers,
# zero cost, and a visual identity that actually matches a
# "mind-bending facts" channel instead of random stock footage.
#
# One primary animated style per topic, with intensity (from
# GPT Call 1's per-beat "intensity" field) smoothly driving
# animation speed/density/glow over the course of the video.
# ============================================================

def _bgr(r, g, b):
    """Convenience: define colors in RGB, return BGR for OpenCV."""
    return (b, g, r)


TOPIC_STYLES = {
    'space': {
        'bg':      _bgr(6, 8, 18),
        'accent':  _bgr(255, 225, 170),   # warm starlight
        'accent2': _bgr(150, 110, 255),   # violet nebula
        'styles':  ['starfield', 'nebula'],
    },
    'cosmic': {
        'bg':      _bgr(10, 4, 22),
        'accent':  _bgr(190, 110, 255),
        'accent2': _bgr(255, 200, 90),
        'styles':  ['nebula', 'particles'],
    },
    'ancient': {
        'bg':      _bgr(12, 14, 20),
        'accent':  _bgr(255, 210, 120),   # gold
        'accent2': _bgr(170, 175, 190),   # stone grey
        'styles':  ['geometric', 'particles'],
    },
    'religion': {
        'bg':      _bgr(14, 10, 22),
        'accent':  _bgr(255, 215, 130),
        'accent2': _bgr(180, 140, 255),
        'styles':  ['geometric', 'aurora'],
    },
    'human': {
        'bg':      _bgr(8, 10, 16),
        'accent':  _bgr(160, 210, 255),   # cool blue
        'accent2': _bgr(255, 200, 110),
        'styles':  ['particles', 'aurora'],
    },
    'death': {
        'bg':      _bgr(5, 5, 9),
        'accent':  _bgr(200, 60, 60),     # deep red
        'accent2': _bgr(150, 150, 160),
        'styles':  ['starfield', 'geometric'],
    },
    'default': {
        'bg':      _bgr(8, 9, 16),
        'accent':  _bgr(255, 255, 255),
        'accent2': _bgr(255, 200, 90),
        'styles':  ['starfield', 'particles'],
    },
}


class _Starfield:
    """Deterministic starfield: positions fixed, twinkle + slow horizontal drift."""
    def __init__(self, width, height, n_stars=220, seed=42):
        rng = random.Random(seed)
        self.stars = []
        for _ in range(n_stars):
            self.stars.append({
                'x': rng.uniform(0, width),
                'y': rng.uniform(0, height),
                'r': rng.uniform(0.6, 2.4),
                'speed': rng.uniform(2, 10),
                'phase': rng.uniform(0, 6.283),
                'tw_speed': rng.uniform(0.8, 2.5),
            })
        self.width, self.height = width, height

    def draw(self, frame, t, intensity, color):
        w = self.width
        bright_base = 0.35 + 0.04 * intensity
        for s in self.stars:
            x = (s['x'] + t * s['speed'] * (0.5 + 0.08 * intensity)) % w
            tw = 0.5 + 0.5 * math.sin(t * s['tw_speed'] + s['phase'])
            brightness = bright_base + 0.5 * tw
            r = max(1, int(round(s['r'] * (0.8 + 0.5 * tw))))
            col = tuple(int(c * min(brightness, 1.0)) for c in color)
            cv2.circle(frame, (int(x), int(s['y'])), r, col, -1, lineType=cv2.LINE_AA)
        return frame


def _draw_nebula(frame, t, intensity, color):
    """Soft slow-moving glow blobs, rendered at low-res and upscaled for a
    cheap painterly blur (full-res GaussianBlur on 1920x1080 every frame is
    too slow for 2500+ frames)."""
    h, w = frame.shape[:2]
    sw, sh = max(w // 3, 8), max(h // 3, 8)
    small = np.zeros((sh, sw, 3), dtype=np.float32)
    n_blobs = 3 + int(min(intensity, 10) // 3)
    for i in range(n_blobs):
        bx = sw * (0.2 + 0.6 * ((i * 0.37 + 0.06 * t + 0.5 * math.sin(t * 0.04 + i)) % 1))
        by = sh * (0.2 + 0.6 * ((i * 0.61 + 0.04 * t + 0.5 * math.cos(t * 0.035 + i * 1.3)) % 1))
        radius = int(min(sw, sh) * (0.35 + 0.08 * math.sin(t * 0.08 + i)))
        cv2.circle(small, (int(bx), int(by)), max(radius, 4), color, -1, lineType=cv2.LINE_AA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sw * 0.18)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    strength = 0.10 + 0.012 * intensity
    out = np.clip(frame.astype(np.float32) + big * strength, 0, 255).astype(np.uint8)
    return out


def _draw_geometric(frame, t, intensity, color):
    """Slowly rotating concentric hexagons -- 'sacred geometry' motif."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    base_r = int(min(w, h) * 0.32)
    n_shapes = 3
    speed = 4 + intensity * 1.2  # degrees/sec
    for i in range(n_shapes):
        angle0 = math.radians(t * speed * (1 if i % 2 == 0 else -1) + i * 40)
        r = base_r - i * int(base_r * 0.22)
        sides = 6
        pts = []
        for k in range(sides):
            a = angle0 + 2 * math.pi * k / sides
            pts.append((int(cx + r * math.cos(a)), int(cy + r * math.sin(a))))
        pts = np.array(pts, dtype=np.int32)
        alpha = 0.10 + 0.01 * intensity
        overlay = frame.copy()
        cv2.polylines(overlay, [pts], True, color, 2, lineType=cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return frame


def _draw_aurora(frame, t, intensity, color):
    """Flowing horizontal energy bands, low-res + blur for performance."""
    h, w = frame.shape[:2]
    sw, sh = max(w // 4, 8), max(h // 4, 8)
    small = np.zeros((sh, sw, 3), dtype=np.float32)
    n_bands = 3
    for b in range(n_bands):
        y_center = sh * (0.25 + 0.22 * b) + 4 * math.sin(t * 0.3 + b)
        for x in range(sw):
            y_off = 3 * math.sin(x * 0.25 + t * (0.4 + 0.05 * intensity) + b * 2)
            y = int(y_center + y_off)
            if 0 <= y < sh:
                cv2.line(small, (x, max(0, y - 1)), (x, min(sh, y + 1)), color, 1)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sw * 0.06)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    strength = 0.18 + 0.015 * intensity
    out = np.clip(frame.astype(np.float32) + big * strength, 0, 255).astype(np.uint8)
    return out


def _draw_particles(frame, t, intensity, color, seed=21, n=70):
    h, w = frame.shape[:2]
    rng = random.Random(seed)
    bright_base = 0.25 + 0.04 * intensity
    for i in range(n):
        sx = rng.uniform(0, w)
        sy = rng.uniform(0, h)
        speed = rng.uniform(8, 25) * (0.5 + 0.08 * intensity)
        phase = rng.uniform(0, 6.283)
        x = (sx + t * speed) % w
        y = (sy + 25 * math.sin(t * 0.6 + phase)) % h
        tw = 0.5 + 0.5 * math.sin(t * 1.5 + phase)
        r = 1 + int(2 * tw)
        col = tuple(int(c * min(bright_base + 0.5 * tw, 1.0)) for c in color)
        cv2.circle(frame, (int(x), int(y)), r, col, -1, lineType=cv2.LINE_AA)
    return frame


_BG_DRAW_FNS = {
    'starfield': lambda frame, t, intensity, color, sf: sf.draw(frame, t, intensity, color),
    'nebula':    lambda frame, t, intensity, color, sf: _draw_nebula(frame, t, intensity, color),
    'geometric': lambda frame, t, intensity, color, sf: _draw_geometric(frame, t, intensity, color),
    'aurora':    lambda frame, t, intensity, color, sf: _draw_aurora(frame, t, intensity, color),
    'particles': lambda frame, t, intensity, color, sf: _draw_particles(frame, t, intensity, color),
}


# ============================================================
# PROCEDURAL ILLUSTRATIONS -- "drawing" system
#
# A small library of line-art subjects, each defined as a list of
# STROKES (point paths in 0-1 normalized space). When a beat's
# content clearly evokes one of these (GPT Call 1 sets
# beat["visual_subject"]), the renderer progressively "draws" the
# shape -- a pen-reveal animation -- then holds it with a gentle
# pulse for the rest of the beat. Sits centered, low-opacity,
# beneath the text layer.
# ============================================================

def _circle_pts(cx, cy, r, n=36, a0=0.0, a1=2*math.pi):
    return [(cx + r*math.cos(a0 + (a1-a0)*i/(n-1)),
             cy + r*math.sin(a0 + (a1-a0)*i/(n-1))) for i in range(n)]


def _ellipse_pts(cx, cy, rx, ry, n=36, a0=0.0, a1=2*math.pi, rot=0.0):
    pts = []
    for i in range(n):
        a = a0 + (a1-a0)*i/(n-1)
        x, y = rx*math.cos(a), ry*math.sin(a)
        xr = x*math.cos(rot) - y*math.sin(rot)
        yr = x*math.sin(rot) + y*math.cos(rot)
        pts.append((cx+xr, cy+yr))
    return pts


def _lumpy_circle_pts(cx, cy, r, bumps=5, bump_amt=0.15, n=48):
    pts = []
    for i in range(n):
        a = 2*math.pi*i/(n-1)
        rr = r * (1 + bump_amt*math.sin(bumps*a))
        pts.append((cx + rr*math.cos(a), cy + rr*math.sin(a)))
    return pts


def _build_illustration_shapes():
    shapes = {}

    # PLANET: sphere + tilted ring + small moon
    shapes['planet'] = [
        _circle_pts(0.5, 0.5, 0.26, n=36),
        _ellipse_pts(0.5, 0.5, 0.42, 0.10, n=30, rot=math.radians(-15)),
        _circle_pts(0.85, 0.18, 0.05, n=14),
    ]

    # PYRAMID: triangle body + horizontal band + entrance
    shapes['pyramid'] = [
        [(0.22, 0.82), (0.50, 0.16), (0.78, 0.82), (0.22, 0.82)],
        [(0.34, 0.55), (0.66, 0.55)],
        [(0.465, 0.82), (0.465, 0.68), (0.535, 0.68), (0.535, 0.82)],
    ]

    # BRAIN: two lumpy lobes + wavy center divide
    divide = []
    for i in range(12):
        yy = 0.28 + (0.72-0.28) * i/11
        xx = 0.5 + 0.04*math.sin(yy*20)
        divide.append((xx, yy))
    shapes['brain'] = [
        _lumpy_circle_pts(0.40, 0.50, 0.24, bumps=5, bump_amt=0.16, n=40),
        _lumpy_circle_pts(0.60, 0.50, 0.24, bumps=5, bump_amt=0.16, n=40),
        divide,
    ]

    # EYE: upper lid arc, lower lid arc, pupil, highlight
    upper, lower = [], []
    for i in range(20):
        xx = 0.15 + (0.85-0.15) * i/19
        upper.append((xx, 0.50 - 0.28*math.sin(math.pi*(xx-0.15)/0.70)))
        lower.append((xx, 0.50 + 0.16*math.sin(math.pi*(xx-0.15)/0.70)))
    shapes['eye'] = [
        upper, lower,
        _circle_pts(0.5, 0.5, 0.10, n=22),
        _circle_pts(0.55, 0.45, 0.03, n=10),
    ]

    # DNA: two sine strands + connecting rungs
    strandA, strandB = [], []
    for i in range(30):
        yy = i/29
        strandA.append((0.5 + 0.22*math.sin(2*math.pi*yy*2), yy))
        strandB.append((0.5 + 0.22*math.sin(2*math.pi*yy*2 + math.pi), yy))
    rungs = []
    for k in range(6):
        yy = k/5
        xa = 0.5 + 0.22*math.sin(2*math.pi*yy*2)
        xb = 0.5 + 0.22*math.sin(2*math.pi*yy*2 + math.pi)
        rungs.append([(xa, yy), (xb, yy)])
    shapes['dna'] = [strandA, strandB] + rungs

    # ATOM: nucleus + three orbit ellipses at different rotations
    shapes['atom'] = [
        _circle_pts(0.5, 0.5, 0.05, n=14),
        _ellipse_pts(0.5, 0.5, 0.40, 0.16, n=36, rot=0.0),
        _ellipse_pts(0.5, 0.5, 0.40, 0.16, n=36, rot=math.radians(60)),
        _ellipse_pts(0.5, 0.5, 0.40, 0.16, n=36, rot=math.radians(-60)),
    ]

    # HOURGLASS: bowtie frame + top/bottom bars
    shapes['hourglass'] = [
        [(0.22, 0.12), (0.78, 0.12), (0.50, 0.50), (0.78, 0.88),
         (0.22, 0.88), (0.50, 0.50), (0.22, 0.12)],
        [(0.16, 0.10), (0.84, 0.10)],
        [(0.16, 0.90), (0.84, 0.90)],
    ]

    return shapes


ILLUSTRATION_SHAPES = _build_illustration_shapes()

_SHAPE_LEN_CACHE: dict = {}


def _stroke_length(pts):
    return sum(math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
               for i in range(len(pts)-1))


def _draw_illustration(frame, t, beat_start, beat_dur, subject, color):
    """Progressively 'draw' the named shape (pen-reveal), then hold with a
    gentle pulse for the remainder of the beat. No-op if subject unknown."""
    strokes = ILLUSTRATION_SHAPES.get(subject)
    if not strokes:
        return frame

    h, w = frame.shape[:2]
    size = min(w, h) * 0.55
    cx, cy = w * 0.5, h * 0.5

    def to_px(p):
        return (int(cx + (p[0]-0.5)*size), int(cy + (p[1]-0.5)*size))

    if subject not in _SHAPE_LEN_CACHE:
        lens = [_stroke_length(s) for s in strokes]
        _SHAPE_LEN_CACHE[subject] = (lens, sum(lens) or 1.0)
    stroke_lens, total_len = _SHAPE_LEN_CACHE[subject]

    el_t = t - beat_start
    reveal_dur = min(1.2, max(0.4, beat_dur * 0.6))
    progress = max(0.0, min(1.0, el_t / reveal_dur))
    target = progress * total_len

    overlay = frame.copy()
    remaining = target
    for stroke, slen in zip(strokes, stroke_lens):
        if slen <= 1e-6:
            continue
        if remaining >= slen:
            pts = np.array([to_px(p) for p in stroke], dtype=np.int32)
            cv2.polylines(overlay, [pts], False, color, 2, lineType=cv2.LINE_AA)
            remaining -= slen
        elif remaining > 0:
            frac_len = remaining
            acc = 0.0
            pts_px = []
            for i in range(len(stroke)-1):
                seg_len = math.hypot(stroke[i+1][0]-stroke[i][0], stroke[i+1][1]-stroke[i][1])
                pts_px.append(to_px(stroke[i]))
                if acc + seg_len >= frac_len:
                    seg_frac = (frac_len - acc) / seg_len if seg_len > 0 else 0
                    ix = stroke[i][0] + (stroke[i+1][0]-stroke[i][0]) * seg_frac
                    iy = stroke[i][1] + (stroke[i+1][1]-stroke[i][1]) * seg_frac
                    pts_px.append(to_px((ix, iy)))
                    break
                acc += seg_len
            if len(pts_px) >= 2:
                pts = np.array(pts_px, dtype=np.int32)
                cv2.polylines(overlay, [pts], False, color, 2, lineType=cv2.LINE_AA)
            remaining = 0
        else:
            break

    if progress < 1.0:
        alpha = 0.35
    else:
        pulse = 0.5 + 0.5 * math.sin((el_t - reveal_dur) * 2.2)
        alpha = 0.28 + 0.12 * pulse

    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def _build_subject_timeline(beats, total_duration, fps):
    """Per-frame (visual_subject, beat_start, beat_dur), discrete (not
    interpolated) so the illustration matches whichever beat is speaking."""
    n_frames = max(1, int(total_duration * fps))
    out = []
    bi = 0
    nb = len(beats)
    for f in range(n_frames):
        t = f / fps
        while bi + 1 < nb and float(beats[bi+1].get('start_time', 0.0)) <= t:
            bi += 1
        b = beats[bi] if nb else {}
        subj = (b.get('visual_subject') or 'none').strip().lower()
        bs = float(b.get('start_time', 0.0))
        be = float(b.get('end_time', bs + 1.0))
        out.append((subj, bs, max(be - bs, 0.1)))
    return out



def _build_intensity_curve(beats, total_duration, fps):
    """Per-frame intensity (1-10), linearly interpolated between beat
    midpoints and smoothed slightly so it drifts rather than jumps."""
    n_frames = max(1, int(total_duration * fps))
    control_t = []
    control_v = []
    for b in beats:
        s = float(b.get('start_time', 0.0))
        e = float(b.get('end_time', s + 1.0))
        mid = (s + e) / 2.0
        val = float(b.get('intensity', 5))
        control_t.append(mid)
        control_v.append(val)
    if not control_t:
        return [5.0] * n_frames

    curve = np.interp(
        [f / fps for f in range(n_frames)],
        control_t, control_v,
        left=control_v[0], right=control_v[-1]
    )
    # Light smoothing (moving average) so intensity drifts, not snaps
    if len(curve) > 5:
        kernel = np.ones(5) / 5
        curve = np.convolve(curve, kernel, mode='same')
    return curve.tolist()


def generate_procedural_background(beats: list, topic: str, total_duration: float,
                                     output_path: str, width: int = 1920,
                                     height: int = 1080, fps: int = 30) -> str:
    """Generate a fully procedural animated background video. No broll, no
    clip failures, no black fillers. One visual identity per topic, with
    intensity smoothly tracking the narration's emotional arc."""
    import cv2

    style_cfg = TOPIC_STYLES.get(topic, TOPIC_STYLES['default'])
    bg_color      = style_cfg['bg']
    accent        = style_cfg['accent']
    accent2       = style_cfg['accent2']
    style_names   = style_cfg['styles']

    n_frames = max(1, int(total_duration * fps))
    print(f"  🎨 Procedural background: topic={topic}, styles={style_names}, {n_frames} frames")

    intensity_curve = _build_intensity_curve(beats, total_duration, fps)
    subject_timeline = _build_subject_timeline(beats, total_duration, fps)
    n_with_subject = sum(1 for s, _, _ in subject_timeline if s != 'none' and s in ILLUSTRATION_SHAPES)
    if n_with_subject:
        print(f"  ✏️  Illustrations active on {n_with_subject}/{n_frames} frames")

    # Render at half-resolution then upscale -- the background is intentionally
    # soft/abstract (it sits behind text), so this is ~4x faster with no
    # visible quality loss after upscale + video compression.
    rw, rh = width // 2, height // 2
    starfield = _Starfield(rw, rh, n_stars=220, seed=hash(topic) & 0xffff)

    raw_path = output_path.replace('.mp4', '_bg_raw.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))

    # Precompute a subtle vignette mask once (multiplicative, darkens edges)
    yv, xv = np.mgrid[0:rh, 0:rw].astype(np.float32)
    cx, cy = rw / 2, rh / 2
    dist = np.sqrt(((xv - cx) / (rw / 2)) ** 2 + ((yv - cy) / (rh / 2)) ** 2)
    vignette = np.clip(1.0 - 0.35 * np.clip(dist - 0.5, 0, 1), 0.55, 1.0)
    vignette3 = vignette[:, :, None]

    for f in range(n_frames):
        t = f / fps
        intensity = intensity_curve[f]

        frame = np.full((rh, rw, 3), bg_color, dtype=np.uint8)

        # Primary style at full strength
        frame = _BG_DRAW_FNS[style_names[0]](frame, t, intensity, accent, starfield)
        # Secondary style as subtle accent layer
        if len(style_names) > 1:
            frame = _BG_DRAW_FNS[style_names[1]](frame, t, intensity * 0.7, accent2, starfield)

        # Content-aware illustration (drawing system) -- if this beat
        # evokes a known subject (planet, brain, dna, etc.), draw it with
        # a pen-reveal animation, centered, beneath the text layer
        subj, b_start, b_dur = subject_timeline[f]
        if subj in ILLUSTRATION_SHAPES:
            frame = _draw_illustration(frame, t, b_start, b_dur, subj, accent)

        # Vignette
        frame = (frame.astype(np.float32) * vignette3).astype(np.uint8)

        # Upscale to final output resolution
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        writer.write(frame)

        if f % (fps * 5) == 0:
            print(f"    {f}/{n_frames} frames...", end='\r')

    writer.release()
    print(f"    {n_frames}/{n_frames} frames... done")

    # Re-encode to H.264 yuv420p for downstream ffmpeg compatibility
    r = subprocess.run([
        'ffmpeg', '-y', '-i', raw_path,
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20',
        '-pix_fmt', 'yuv420p', '-r', str(fps), '-an', output_path
    ], capture_output=True)
    os.remove(raw_path)
    if r.returncode != 0:
        raise Exception(f"Background re-encode failed: {r.stderr.decode()[-200:]}")

    print(f"  ✅ Procedural background: {output_path}")
    return output_path


# ============================================================
# RECURRING CHARACTER COMPOSITOR - OPTION B
# Base CLOSED character + open-mouth transparent overlay layers.
#
# This fixes the two big problems:
# 1) The closed mouth is NOT an overlay anymore. Idle/closed state uses the
#    real full character image with the correct closed mouth.
# 2) The character sprite is resized/rotated with PREMULTIPLIED ALPHA so
#    hidden black RGB pixels in transparent PNGs do not leak into edges.
#
# Recommended folder:
#   character_test/
#     01_base_closed.png      full character, transparent PNG, good closed mouth
#     01_base_no_mouth.png    optional but recommended for talking frames
#     mouth_small.png         full-canvas transparent mouth layer only
#     mouth_medium.png        full-canvas transparent mouth layer only
#     mouth_wide.png          full-canvas transparent mouth layer only
#
# If 01_base_no_mouth.png exists, the renderer uses it while talking so the
# closed mouth line never ghosts under open mouths. If it does not exist, it
# falls back to 01_base_closed.png and overlays the open mouths directly.
# ============================================================
CHARACTER_ENABLED = True
CHARACTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "character_test")

# OPTION B: closed mouth is the base image, NOT a mouth layer.
CHARACTER_BASE_CLOSED_IMAGE = "01_base_closed.png"

# Optional but strongly recommended. Used only while mouth_openness > 0.
# This prevents the closed mouth line from showing under open mouth overlays.
CHARACTER_BASE_NO_MOUTH_IMAGE = "01_base_no_mouth.png"

# These must be FULL-CANVAS transparent PNG layers with the exact same canvas
# size and alignment as the base image. They should contain ONLY the open mouth
# pixels, transparent everywhere else.
CHARACTER_MOUTH_SOURCES = {
    "small":  "mouth_small.png",
    "medium": "mouth_medium.png",
    "wide":   "mouth_wide.png",
}

# Placement on the OUTPUT_WIDTH x OUTPUT_HEIGHT canvas -- centered,
# bust framing with a slight overflow off the bottom edge.
CHARACTER_HEIGHT_RATIO = 0.95
CHARACTER_Y_OFFSET_RATIO = 0.06

# Mouth animation tuning. Lower MOUTH_MAX_OPENNESS if the mouth still opens too
# wide. Raise ENVELOPE_RELEASE slightly if the mouth closes too slowly.
MOUTH_PRE_ROLL_SECONDS = 0.035
MOUTH_POST_ROLL_SECONDS = 0.110
MOUTH_MIN_WORD_DURATION = 0.080
MOUTH_SHORT_WORD_THRESHOLD = 0.180
MOUTH_SHORT_WORD_OPENNESS = 0.42
MOUTH_NORMAL_WORD_OPENNESS = 0.64
MOUTH_LONG_WORD_OPENNESS = 0.78
MOUTH_MAX_OPENNESS = 0.82
ENVELOPE_ATTACK = 0.36
ENVELOPE_RELEASE = 0.15
ENVELOPE_BLUR_SECONDS = 0.055
MOUTH_CACHE_STEPS = 72
CLOSED_RETURN_THRESHOLD = 0.025


def _load_char_rgba(path: str) -> np.ndarray:
    """Load an image as RGBA.

    If the file has no alpha and a pure white/checker-ish exported background,
    it will try to remove near-white pixels. Real production assets should be
    proper transparent PNGs, though.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Character asset not found: {path}")

    im = Image.open(path).convert("RGBA")
    arr = np.array(im).astype(np.uint8)

    # Fallback for accidental white-background exports. This is not meant to
    # replace proper transparency, but it prevents total failure.
    if arr[:, :, 3].min() == 255:
        rgb = arr[:, :, :3].astype(np.int16)
        near_white = np.all(rgb > 245, axis=2)
        if near_white.mean() > 0.15:
            arr[near_white, 3] = 0

    return arr


def _clean_transparent_rgb(rgba: np.ndarray, alpha_cutoff: int = 2) -> np.ndarray:
    """Zero RGB where alpha is basically invisible.

    This removes hidden black/white RGB garbage from fully transparent pixels.
    Premultiplied transforms below handle semi-transparent edges.
    """
    out = rgba.copy()
    a = out[:, :, 3]
    out[a <= alpha_cutoff, :3] = 0
    out[a <= alpha_cutoff, 3] = 0
    return out


def _assert_same_canvas(name: str, layer: np.ndarray, base: np.ndarray):
    if layer.shape != base.shape:
        raise ValueError(
            f"Character layer '{name}' is {layer.shape[1]}x{layer.shape[0]}, "
            f"but base is {base.shape[1]}x{base.shape[0]}. Every character "
            f"asset must use the exact same canvas size and alignment."
        )


def load_character_assets():
    """Load Option B character rig.

    Returns:
        base_closed_rgba: full character with good closed mouth.
        patches: dict containing:
            _base_closed: full character, closed mouth
            _base_talking: no-mouth base if available, otherwise base_closed
            small/medium/wide: full-canvas transparent mouth layers
            _cache: rendered mouth composite cache
    """
    closed_path = os.path.join(CHARACTER_DIR, CHARACTER_BASE_CLOSED_IMAGE)

    # Friendly fallback so existing folders still run, but best practice is to
    # create 01_base_closed.png explicitly.
    if not os.path.exists(closed_path):
        fallback_names = [
            "source_transparent_reference.png",
            "01_neutral.png",
            "01_neutral(1).png",
        ]
        for fname in fallback_names:
            fallback = os.path.join(CHARACTER_DIR, fname)
            if os.path.exists(fallback):
                print(f"  ⚠ {CHARACTER_BASE_CLOSED_IMAGE} not found. Using fallback: {fname}")
                closed_path = fallback
                break

    base_closed = _clean_transparent_rgb(_load_char_rgba(closed_path))

    no_mouth_path = os.path.join(CHARACTER_DIR, CHARACTER_BASE_NO_MOUTH_IMAGE)
    if os.path.exists(no_mouth_path):
        base_talking = _clean_transparent_rgb(_load_char_rgba(no_mouth_path))
        _assert_same_canvas(CHARACTER_BASE_NO_MOUTH_IMAGE, base_talking, base_closed)
        print(f"  ✓ Talking base: {CHARACTER_BASE_NO_MOUTH_IMAGE}")
    else:
        base_talking = base_closed.copy()
        print(f"  ⚠ {CHARACTER_BASE_NO_MOUTH_IMAGE} not found. Open mouths will overlay directly on the closed-mouth base.")

    patches = {
        "_base_closed": base_closed.astype(np.float32),
        "_base_talking": base_talking.astype(np.float32),
        "_cache": {},
    }

    for name, fname in CHARACTER_MOUTH_SOURCES.items():
        path = os.path.join(CHARACTER_DIR, fname)
        layer = _clean_transparent_rgb(_load_char_rgba(path))
        _assert_same_canvas(fname, layer, base_closed)
        patches[name] = layer.astype(np.float32)

    print("  ✓ Character rig loaded: base_closed + open-mouth layers")
    return base_closed, patches


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _word_open_strength(word: dict) -> float:
    """Estimate how open the mouth should be for one Whisper word."""
    start = float(word.get("start", 0.0))
    end = float(word.get("end", start))
    dur = max(0.0, end - start)
    text = str(word.get("word", "")).strip()

    if dur < MOUTH_SHORT_WORD_THRESHOLD:
        openness = MOUTH_SHORT_WORD_OPENNESS
    elif dur > 0.34:
        openness = MOUTH_LONG_WORD_OPENNESS
    else:
        openness = MOUTH_NORMAL_WORD_OPENNESS

    letters = [c for c in text.upper() if c.isalpha()]
    if letters:
        vowel_ratio = sum(1 for c in letters if c in "AEIOU") / max(1, len(letters))
        openness += 0.07 * vowel_ratio

    return max(0.0, min(MOUTH_MAX_OPENNESS, openness))


def build_mouth_envelope(whisper_word_list: list, total_frames: int, fps: float) -> np.ndarray:
    """Build smooth per-frame mouth openness from Whisper word timestamps.

    Returns float32 values from 0.0 to MOUTH_MAX_OPENNESS.
    """
    total_frames = max(0, int(total_frames))
    if total_frames <= 0:
        return np.zeros(0, dtype=np.float32)

    fps = float(fps) if fps else 30.0
    target = np.zeros(total_frames, dtype=np.float32)

    for word in whisper_word_list or []:
        try:
            raw_start = float(word.get("start", 0.0))
            raw_end = float(word.get("end", raw_start))
        except Exception:
            continue

        if raw_end <= raw_start:
            raw_end = raw_start + MOUTH_MIN_WORD_DURATION

        start = max(0.0, raw_start - MOUTH_PRE_ROLL_SECONDS)
        end = max(start + MOUTH_MIN_WORD_DURATION, raw_end + MOUTH_POST_ROLL_SECONDS)

        f0 = max(0, int(math.floor(start * fps)))
        f1 = min(total_frames, int(math.ceil(end * fps)))
        if f1 <= f0:
            continue

        peak = _word_open_strength(word)
        duration = max(end - start, 1.0 / fps)

        for f in range(f0, f1):
            ft = f / fps
            u = max(0.0, min(1.0, (ft - start) / duration))

            # Fast-ish opening, soft closing, stable middle hold.
            if u < 0.26:
                local = _smoothstep(u / 0.26)
            elif u > 0.74:
                local = _smoothstep((1.0 - u) / 0.26)
            else:
                local = 1.0

            value = peak * max(0.32, local)
            target[f] = max(target[f], value)

    # Attack/release filter prevents open/close snapping.
    envelope = np.zeros_like(target)
    value = 0.0
    for i, desired in enumerate(target):
        if desired > value:
            value += (desired - value) * ENVELOPE_ATTACK
        else:
            value += (desired - value) * ENVELOPE_RELEASE
        envelope[i] = value

    # Final tiny temporal blur removes frame jitter.
    blur_frames = int(round(ENVELOPE_BLUR_SECONDS * fps))
    if blur_frames >= 2 and len(envelope) > blur_frames:
        if blur_frames % 2 == 0:
            blur_frames += 1
        kernel = np.hanning(blur_frames).astype(np.float32)
        if kernel.sum() > 0:
            kernel /= kernel.sum()
            pad = blur_frames // 2
            padded = np.pad(envelope, (pad, pad), mode="edge")
            envelope = np.convolve(padded, kernel, mode="valid").astype(np.float32)

    return np.clip(envelope, 0.0, MOUTH_MAX_OPENNESS).astype(np.float32)


def _premultiply_rgba(rgba: np.ndarray) -> np.ndarray:
    """Convert straight-alpha RGBA to premultiplied-alpha float32 RGBA."""
    arr = rgba.astype(np.float32)
    a = arr[:, :, 3:4] / 255.0
    arr[:, :, :3] *= a
    return arr


def _unpremultiply_rgba(premul: np.ndarray) -> np.ndarray:
    """Convert premultiplied-alpha float32 RGBA back to straight-alpha uint8 RGBA."""
    arr = premul.astype(np.float32).copy()
    a = arr[:, :, 3:4] / 255.0
    valid = a[:, :, 0] > 1e-6
    arr[:, :, :3][valid] /= a[:, :, :][valid]
    arr[:, :, :3] = np.clip(arr[:, :, :3], 0, 255)
    arr[:, :, 3] = np.clip(arr[:, :, 3], 0, 255)
    out = arr.astype(np.uint8)
    out[out[:, :, 3] <= 2, :3] = 0
    out[out[:, :, 3] <= 2, 3] = 0
    return out


def _resize_rgba_premultiplied(rgba: np.ndarray, size_xy: tuple[int, int]) -> np.ndarray:
    """Resize RGBA without black/white transparent-edge halos."""
    premul = _premultiply_rgba(rgba)
    resized = cv2.resize(premul, size_xy, interpolation=cv2.INTER_LINEAR)
    return _unpremultiply_rgba(resized)


def _warp_rgba_premultiplied(rgba: np.ndarray, M: np.ndarray, size_xy: tuple[int, int]) -> np.ndarray:
    """Affine-warp RGBA without black/white transparent-edge halos."""
    premul = _premultiply_rgba(rgba)
    warped = cv2.warpAffine(
        premul, M, size_xy,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return _unpremultiply_rgba(warped)


def _composite_rgba_over(base_rgba: np.ndarray, layer_rgba: np.ndarray) -> np.ndarray:
    """Composite layer over base using proper alpha-over math."""
    base = base_rgba.astype(np.float32)
    layer = layer_rgba.astype(np.float32)

    ba = base[:, :, 3:4] / 255.0
    la = layer[:, :, 3:4] / 255.0

    out_a = la + ba * (1.0 - la)
    out_rgb_premul = layer[:, :, :3] * la + base[:, :, :3] * ba * (1.0 - la)

    out_rgb = np.zeros_like(out_rgb_premul)
    valid = out_a[:, :, 0] > 1e-6
    out_rgb[valid] = out_rgb_premul[valid] / out_a[valid]

    out = np.zeros_like(base)
    out[:, :, :3] = out_rgb
    out[:, :, 3:4] = out_a * 255.0
    return _clean_transparent_rgb(np.clip(out, 0, 255).astype(np.uint8))


def _crossfade_rgba(a_rgba: np.ndarray, b_rgba: np.ndarray, amount: float) -> np.ndarray:
    """Crossfade two RGBA images in premultiplied-alpha space."""
    t = _smoothstep(max(0.0, min(1.0, float(amount))))
    a = _premultiply_rgba(a_rgba)
    b = _premultiply_rgba(b_rgba)
    mixed = a * (1.0 - t) + b * t
    return _unpremultiply_rgba(mixed)


def _scale_layer_alpha(layer_rgba: np.ndarray, amount: float) -> np.ndarray:
    """Fade a transparent mouth layer in/out by scaling its alpha channel."""
    t = _smoothstep(max(0.0, min(1.0, float(amount))))
    out = layer_rgba.astype(np.float32).copy()
    out[:, :, 3] *= t
    return _clean_transparent_rgb(np.clip(out, 0, 255).astype(np.uint8))


def _blend_mouth(base_closed: np.ndarray, patches: dict, openness: float) -> np.ndarray:
    """Build the character sprite for the current mouth openness.

    Idle state returns the closed-mouth base exactly.
    Talking state uses the no-mouth base if available, then overlays small /
    medium / wide mouth layers with smooth transitions.
    """
    openness = max(0.0, min(MOUTH_MAX_OPENNESS, float(openness)))

    if openness <= CLOSED_RETURN_THRESHOLD:
        return base_closed.astype(np.uint8)

    cache_key = int(round((openness / max(MOUTH_MAX_OPENNESS, 1e-6)) * MOUTH_CACHE_STEPS))
    cache = patches.get("_cache")
    if isinstance(cache, dict) and cache_key in cache:
        return cache[cache_key]

    base_closed_u8 = patches.get("_base_closed", base_closed).astype(np.uint8)
    base_talking_u8 = patches.get("_base_talking", base_closed).astype(np.uint8)

    small = patches["small"].astype(np.uint8)
    medium = patches["medium"].astype(np.uint8)
    wide = patches["wide"].astype(np.uint8)

    o = openness / max(MOUTH_MAX_OPENNESS, 1e-6)

    # 0.00-0.25: closed base -> no-mouth base + small mouth faded in
    # 0.25-0.62: small -> medium
    # 0.62-1.00: medium -> wide
    if o < 0.25:
        t = o / 0.25
        mouth = _scale_layer_alpha(small, t)
        open_sprite = _composite_rgba_over(base_talking_u8, mouth)
        out = _crossfade_rgba(base_closed_u8, open_sprite, t)
    elif o < 0.62:
        t = (o - 0.25) / 0.37
        mouth = _crossfade_rgba(small, medium, t)
        out = _composite_rgba_over(base_talking_u8, mouth)
    else:
        t = (o - 0.62) / 0.38
        mouth = _crossfade_rgba(medium, wide, t)
        out = _composite_rgba_over(base_talking_u8, mouth)

    if isinstance(cache, dict):
        cache[cache_key] = out
    return out


def _character_idle_transform(t: float):
    """Continuous idle motion: breathing, slow sway, subtle rotation."""
    breathe = math.sin(2 * math.pi * t / 2.8)
    scale = 1.0 + 0.010 * breathe
    dy = -4 * breathe
    dx = 3 * math.sin(2 * math.pi * t / 4.6 + 1.0)
    angle = 0.9 * math.sin(2 * math.pi * t / 5.2)
    return scale, dx, dy, angle


def render_character_frame(frame: np.ndarray, base: np.ndarray, patches: dict,
                            t: float, mouth_openness: float,
                            width: int, height: int) -> np.ndarray:
    """Composite the character onto a BGR frame with smooth mouth animation."""
    if mouth_openness > 0.04:
        natural_variation = 0.018 * math.sin(2 * math.pi * t * 2.4)
    else:
        natural_variation = 0.0

    mouth_value = max(0.0, min(MOUTH_MAX_OPENNESS, mouth_openness + natural_variation))
    char = _blend_mouth(base, patches, mouth_value)

    scale, dx, dy, angle = _character_idle_transform(t)
    char_h = int(height * CHARACTER_HEIGHT_RATIO)
    char_w = char_h  # source frames are square
    size = max(1, int(char_w * scale))

    # IMPORTANT: premultiplied-alpha resize/rotate prevents the black fringe
    # that appears when transparent PNGs have hidden black RGB values.
    resized = _resize_rgba_premultiplied(char, (size, size))
    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    rotated = _warp_rgba_premultiplied(resized, M, (size, size))

    base_x = (width - char_w) // 2
    base_y = height - char_h + int(height * CHARACTER_Y_OFFSET_RATIO)
    x0 = base_x + (char_w - size) // 2 + int(dx)
    y0 = base_y + (char_h - size) // 2 + int(dy)

    src_x0, src_y0 = max(0, -x0), max(0, -y0)
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1 = min(width, x0 + size)
    dst_y1 = min(height, y0 + size)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return frame

    sprite_rgb = rotated[src_y0:src_y1, src_x0:src_x1, :3][:, :, ::-1].astype(np.float32)  # RGB->BGR
    sprite_a = rotated[src_y0:src_y1, src_x0:src_x1, 3:4].astype(np.float32) / 255.0

    roi = frame[dst_y0:dst_y1, dst_x0:dst_x1, :].astype(np.float32)
    blended = sprite_rgb * sprite_a + roi * (1.0 - sprite_a)
    frame[dst_y0:dst_y1, dst_x0:dst_x1, :] = np.clip(blended, 0, 255).astype(np.uint8)
    return frame


# ============================================================
# GPT CALL 1 -- STORY BEAT ANALYZER
# Sends Whisper segments (with timestamps) to GPT so it can
# produce accurate timing without word-level lookup
# ============================================================
def build_whisper_word_list(whisper_segments: list) -> list:
    """Flatten Whisper segments into an ordered word list with timestamps."""
    words = []
    for seg in whisper_segments:
        for we in seg.get('words', []):
            wc = we.get('word', '').upper().strip('.,!?;:\'"()[]- ')
            if not wc:
                continue
            words.append({
                'word':  wc,
                'start': float(we.get('start', 0.0)),
                'end':   float(we.get('end',   0.0)),
            })
    return words


def realign_beat_times(beats: list, whisper_word_list: list) -> list:
    """Recompute start_time/end_time for every beat by sequentially matching
    each beat's verbatim text against Whisper's word-level timestamps.

    GPT Call 1 is only given segment-level [start-end] brackets. When it splits
    one segment into multiple beats, it INVENTS the split-point timestamps --
    it has no word-level data. Those guessed boundaries cause every downstream
    word-matching step to look in the wrong time window, producing words that
    appear far too early or too late.

    Walk through the Whisper word list with a single forward-only pointer.
    Bounded lookahead handles normal drift; if that fails, fall back to an
    UNBOUNDED search from the global pointer so one bad match can't strand
    every subsequent beat. If a beat truly can't be matched, estimate its
    timing sequentially rather than keeping GPT's possibly-wild guess.
    """
    ptr = 0
    n = len(whisper_word_list)
    LOOKAHEAD = 20

    def norm(w):
        return w.upper().strip('.,!?;:\'"()[]- ')

    for beat in beats:
        text = (beat.get("text") or "").strip()
        words = [norm(w) for w in text.split() if norm(w)]

        if not words:
            continue

        start_idx = None
        end_idx = None
        local_ptr = ptr

        for w in words:
            found = None
            for look in range(local_ptr, min(local_ptr + LOOKAHEAD, n)):
                ww = whisper_word_list[look]['word']
                if ww == w or w in ww or ww in w:
                    found = look
                    break
            if found is None:
                for look in range(ptr, n):
                    ww = whisper_word_list[look]['word']
                    if ww == w or w in ww or ww in w:
                        found = look
                        break
            if found is None:
                continue
            if start_idx is None:
                start_idx = found
            end_idx = found
            local_ptr = found + 1

        if start_idx is not None and end_idx is not None:
            beat["start_time"] = whisper_word_list[start_idx]['start']
            beat["end_time"]   = whisper_word_list[end_idx]['end']
            ptr = end_idx + 1
        else:
            if ptr < n:
                est_start = whisper_word_list[ptr]['start']
            elif n > 0:
                est_start = whisper_word_list[-1]['end']
            else:
                est_start = float(beat.get("start_time", 0.0))
            est_dur = max(0.3, 0.35 * len(words))
            beat["start_time"] = est_start
            beat["end_time"]   = est_start + est_dur
            print(f"    ⚠ Could not align beat text '{text[:40]}' -- estimated timing")
            ptr = min(ptr + max(1, len(words)), n)

    for i in range(1, len(beats)):
        prev_end = float(beats[i-1].get("end_time", 0.0))
        cur_start = float(beats[i].get("start_time", 0.0))
        if cur_start < prev_end:
            beats[i]["start_time"] = prev_end
            if float(beats[i].get("end_time", 0.0)) <= prev_end:
                beats[i]["end_time"] = prev_end + 0.3

    return beats


def analyze_story_beats(transcript_text: str, whisper_segments: list,
                        topic_hint: str, total_duration: float) -> dict:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not set.")

    print(f"  🎭 Call 1: Story beats ({len(transcript_text)} chars, {total_duration:.1f}s)...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Build timed transcript — each Whisper segment on its own line with [start - end]
    timed_lines = []
    for seg in whisper_segments:
        s = float(seg.get('start', 0))
        e = float(seg.get('end', 0))
        t = seg.get('text', '').strip()
        if t:
            timed_lines.append(f"[{s:.2f}s - {e:.2f}s] {t}")
    timed_transcript = "\n".join(timed_lines)

    system_prompt = f"""You are the story producer for VAULTS OF HISTORY -- a viral mind-bending facts channel.
Style: @sackfeels on TikTok. Dramatic. Eerie. Cinematic.
Total audio duration: {total_duration:.1f} seconds.

You will receive a transcript with EXACT timestamps from Whisper speech recognition.
Each line is formatted as: [start - end] spoken words

YOUR JOB: Segment the transcript into story beats for cinematic video editing.

RULES:
- Use the Whisper timestamps directly -- they are accurate. Copy start_time and end_time from the brackets.
- Beat text MUST be copied VERBATIM from the transcript. Exact words, exact spelling. No paraphrasing.
- Keep beats 2-10 words -- natural spoken phrases or short clauses.
- A single Whisper segment can become 1-3 beats if it contains multiple natural phrases.
- Cover the ENTIRE transcript -- every word must appear in some beat.
- "pause" beats only for clear silence gaps (>0.5s) between segments.

beat_type: "hook"|"buildup"|"reveal"|"shock"|"pause"|"resolution"|"outro"

VISUAL_SUBJECT (drawing system): if this beat's text CLEARLY and SPECIFICALLY
evokes one of these objects, set visual_subject to it -- the renderer will
draw it as a line-art animation. Options: "none"|"planet"|"pyramid"|"brain"|
"eye"|"dna"|"atom"|"hourglass". Be CONSERVATIVE -- most beats should be "none".
Only use a specific subject when the beat is clearly ABOUT that thing (e.g.
"planet" only for beats actually discussing a planet/moon/sphere in space,
"brain" only for beats about the brain/mind/neurons, "hourglass" only for
beats about time running out / ancient time / countdown). Never force a match.

Return ONLY valid JSON:
{{
  "topic": "space|ancient|religion|human|death|default",
  "music_mood": "eerie|dark|mysterious|sacred|cosmic|haunting",
  "beats": [
    {{
      "beat_type": "hook|buildup|reveal|shock|pause|resolution|outro",
      "text": "verbatim words from transcript",
      "start_time": 0.0,
      "end_time": 2.5,
      "intensity": 8,
      "broll_category": "space|ancient|cosmic|sky|temple|any",
      "clip_duration": 4.0,
      "visual_subject": "none|planet|pyramid|brain|eye|dna|atom|hourglass"
    }}
  ]
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Topic hint: {topic_hint}\n\nTimed transcript:\n{timed_transcript}\n\nSegment every line into beats. Use the timestamps shown. Copy text verbatim."}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            timeout=90,
        )
        result = json.loads(response.choices[0].message.content)
        beats  = result.get('beats', [])
        print(f"  ✅ {len(beats)} beats, topic={result.get('topic')}")
        return result
    except Exception as e:
        print(f"  ❌ Story beats failed: {e}")
        raise


# ============================================================
# GPT CALL 2 -- RENDER DECISION GENERATOR
# ============================================================

def _build_batch_prompt(topic: str, batch: list) -> str:
    """Build the GPT Call 2 user prompt, annotating each beat with its real duration
    so GPT can set start_offset values that actually fit within the beat window."""
    annotated = []
    for b in batch:
        dur = round(float(b.get("end_time", 0)) - float(b.get("start_time", 0)), 2)
        entry = dict(b)
        entry["_duration_seconds"] = dur  # injected so GPT knows the real budget
        annotated.append(entry)
    return (
        f"Topic: {topic}\n\n"
        f"Beats ({len(batch)} total -- output exactly {len(batch)} scenes):\n"
        f"{json.dumps(annotated, indent=2)}\n\n"
        f"IMPORTANT: Each beat has a _duration_seconds field. "
        f"All start_offset values for elements in that beat MUST be less than _duration_seconds. "
        f"If _duration_seconds is 0.8s, valid start_offsets are 0.0, 0.2, 0.4 -- NOT 0.6 or higher (element would never show). "
        f"For beats shorter than 0.5s: use only 1 element with start_offset 0.0. "
        f"For beats 0.5-1.0s: max 2 elements, stagger by 0.2s. "
        f"For beats >1.0s: up to 3 elements, stagger by 0.3s. "
        f"Compose each scene cinematically. Vary layouts. White dominant, yellow for one key word per scene."
    )

def generate_render_decisions(beats: list, topic: str) -> list:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not set.")

    print(f"  🎨 Call 2: Scene compositions for {len(beats)} beats...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = f"""You are an elite short-form video editor. You compose every frame like a motion designer -- choosing position, size, color, animation, and timing for each visual element. You are not picking from preset templates. You are designing each scene.

Channel: VAULTS OF HISTORY -- mind-bending facts about space, ancient civilizations, and human consciousness. Audience is impossible to impress. The aesthetic is CINEMATIC and DARK -- like a documentary trailer, not a social media post.

=== FONT BEHAVIOR ===
The renderer uses Anton (ultra-condensed, cinematic) as the primary font. This font is TALL and NARROW. All text is automatically rendered in ALL CAPS -- so write content in ALL CAPS.
SIZE RULES (strictly enforced by renderer):
- Single impact word: 120-160px max. Centered or slightly off-center.
- Sentence words (2+ words in a beat): 70-110px each. Cascade across canvas.
- Use SIZE VARIATION within a scene: vary between 70px and 110px across words for rhythm.
- DO NOT go above 160px for any element -- it will be clamped.
- Fewer elements per scene is better. 3-5 elements max. Dense scenes are unreadable.

=== YOUR RENDERING ENGINE ===
Python OpenCV + Pillow on a {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} canvas.

For each beat, you output a SCENE -- a list of ELEMENTS placed and animated however you want.

=== ELEMENT TYPES ===

TEXT element:
{{
  "type": "text",
  "content": "WORD",              // the text string
  "x": 0.5,                       // 0.0-1.0 horizontal position (anchor)
  "y": 0.4,                       // 0.0-1.0 vertical position (anchor)
  "anchor": "center",             // "center" | "left" | "right" -- how x,y aligns the text
  "size": 120,                    // pixels
  "color": "#FFFFFF",             // hex
  "weight": "black",              // "regular" | "bold" | "black" (use black for impact)
  "outline": 4,                   // pixels of black outline (3-6 typical)
  "anim": "fade_in",              // see ANIMATIONS below
  "start_offset": 0.0,            // seconds after beat starts when element appears
  "duration": null,               // seconds visible (null = until next beat)
  "anim_duration": 0.15,          // how long entrance animation takes
  "effect": "none"                // see EFFECTS below
}}

LINE element (for fractions, dividers, underlines):
{{
  "type": "line",
  "x1": 0.3, "y1": 0.5, "x2": 0.7, "y2": 0.5,
  "thickness": 8,
  "color": "#FFFFFF",
  "anim": "draw_horizontal",      // "draw_horizontal" draws left-to-right, "fade_in" fades, "none" appears instantly
  "start_offset": 0.2,
  "duration": null,
  "anim_duration": 0.3
}}

RECT element (for boxes, backgrounds, highlight bars):
{{
  "type": "rect",
  "x": 0.4, "y": 0.5, "w": 0.2, "h": 0.1,   // x,y is top-left corner. w,h are width/height
  "color": "#FBC02D",
  "filled": true,                   // true=filled, false=outline only
  "thickness": 4,                   // only used if filled=false
  "anim": "fade_in",
  "start_offset": 0.0,
  "duration": null
}}

CIRCLE element:
{{
  "type": "circle",
  "x": 0.5, "y": 0.5, "radius": 0.05,
  "color": "#FFFFFF",
  "filled": false,
  "thickness": 4,
  "anim": "fade_in",
  "start_offset": 0.0,
  "duration": null
}}

=== ANIMATIONS ===
- "none": appears instantly
- "fade_in": opacity 0→100% over anim_duration
- "slide_in_left": slides in from off-screen left
- "slide_in_right": slides in from off-screen right
- "slide_in_top": slides in from off-screen top
- "slide_in_bottom": slides in from off-screen bottom
- "scale_in": starts at 1.3x scale and snaps to 1.0x (punch effect)
- "snap": appears instantly with a 1-frame white flash
- "draw_horizontal": (lines only) draws progressively left-to-right

=== EFFECTS (applied during display, not just entrance) ===
- "none": static
- "flicker": rapid on/off blinking for first 0.3s (for shock words)
- "shake": position jitters slightly (for impact)
- "glow": adds soft colored glow halo around element

=== HOW TO COMPOSE SCENES ===

ELEMENT LIMIT: Maximum 4 elements per scene. Less is more. 2-3 elements is ideal.

STAGGER ALL ELEMENTS: Words must appear one at a time. Use start_offset to sequence them. CRITICAL: start_offset must be less than the beat's _duration_seconds or the element will NEVER appear.
- Beat <0.5s: 1 element only, start_offset 0.0
- Beat 0.5-1.0s: max 2 elements, offsets 0.0 and 0.3
- Beat >1.0s: up to 3 elements, offsets 0.0 / 0.35 / 0.7
NEVER set start_offset >= _duration_seconds. The renderer clamps it and the word disappears.

=== POSITIONING GRID (1920x1080 canvas) ===
Safe zone: x: 0.08-0.92, y: 0.12-0.88. Never place text center-point outside this box -- it gets clipped or looks cramped against edges.

Three vertical bands -- rotate between them across consecutive beats so the video doesn't feel static:
- UPPER band:  y: 0.20-0.35
- CENTER band: y: 0.42-0.58
- LOWER band:  y: 0.65-0.80

Size-to-position pairing:
- Large text (140-160px): keep near CENTER band, x: 0.25-0.75 (needs room to breathe)
- Medium text (90-130px): any band, x: 0.15-0.85
- Small text (70-90px): can sit closer to edges, x: 0.10-0.90

For a SPOKEN SENTENCE (3-5 words): Pick the 2-3 most impactful words. One element per word, each ~90-120px. Place them across DIFFERENT bands (e.g. word 1 in UPPER, word 2 in CENTER) so they don't visually stack on top of each other. Vary x positions too -- don't put every word at x:0.5.

For a SHORT BEAT (1-2 words): 1 or 2 elements max, 100-140px. Pick ONE band (not split across two) and place words side-by-side or stacked within that band, x: 0.2-0.8.

For a SINGLE IMPACT WORD: One TEXT element, 120-160px, CENTER band, x: 0.3-0.7. scale_in animation.

For a NUMBER or STATISTIC: TEXT for the number (large, CENTER band), LINE divider just below it, TEXT for the unit in LOWER band. 3 elements.

For EMPHASIS: RECT highlight bar behind the key word, then the TEXT word on top, same position. 2 elements.

VARY bands across consecutive beats -- if the previous beat used CENTER, this beat should favor UPPER or LOWER. This creates visual rhythm instead of every beat looking the same.

=== HARD RULES ===
1. Output exactly {len(beats)} scenes, one per beat, in order.
2. Every "content" in TEXT elements must be ALL CAPS and use words VERBATIM from the beat text.
3. For pause beats: output {{"elements": []}} (empty scene).
4. start_offset values must fit within the beat duration. STAGGER them -- never all 0.0.
5. x, y values are 0.0-1.0. NEVER use percentages or pixels.
6. MAX 4 elements per scene. Violating this makes the video unreadable.
7. Never repeat the same word twice in one scene.
8. Content must be a SINGLE WORD or SHORT PHRASE -- never a full sentence in one element.

=== COLOR DISCIPLINE ===
White (#FFFFFF) is your dominant color. Yellow (#FBC02D) for ONE key word per scene maximum. Other colors (red, blue, purple) only for very specific moments.

Return ONLY valid JSON:
{{
  "scenes": [
    {{
      "beat_index": <int>,
      "beat_type": "<hook|shock|reveal|buildup|pause|resolution|outro>",
      "elements": [
        // list of element objects as specified above
      ]
    }}
    // ... exactly {len(beats)} scenes
  ]
}}"""

    # Batch to stay safely under GPT-4o's 16384 output token limit.
    # 6 beats per batch ~ 5-7k tokens output, well within limit even with long scenes.
    BATCH_SIZE = 3
    all_scenes = []
    batches = [beats[i:i+BATCH_SIZE] for i in range(0, len(beats), BATCH_SIZE)]

    def _run_batch(batch_idx, batch):
        start_beat = batch_idx * BATCH_SIZE
        print(f"  🎨 Batch {batch_idx+1}/{len(batches)}: beats {start_beat}-{start_beat+len(batch)-1}...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _build_batch_prompt(topic, batch)}
            ],
            response_format={"type": "json_object"},
            temperature=0.85,
            max_tokens=8000,
            timeout=120,
        )
        result = json.loads(response.choices[0].message.content)
        batch_scenes = result.get('scenes', [])
        if len(batch_scenes) > len(batch):
            # GPT split a beat into 2 scenes -- if we keep the extra,
            # EVERY subsequent scene shifts by +1 relative to its beat,
            # causing cascading "hallucination" drops in the validator.
            # Trim to guarantee scenes[i] always corresponds to beats[i].
            print(f"  ⚠️  Batch {batch_idx+1}: expected {len(batch)} scenes, got {len(batch_scenes)} -- trimming extras")
            batch_scenes = batch_scenes[:len(batch)]
        elif len(batch_scenes) < len(batch):
            print(f"  ⚠️  Batch {batch_idx+1}: expected {len(batch)} scenes, got {len(batch_scenes)} -- padding with empty scenes")
            while len(batch_scenes) < len(batch):
                batch_scenes.append({"beat_index": start_beat + len(batch_scenes), "elements": []})
        print(f"  ✅ Batch {batch_idx+1} done: {len(batch_scenes)} scenes")
        return batch_idx, batch_scenes

    # Run batches with 4 concurrent workers. Each batch is an independent
    # GPT call, so this is purely an I/O-bound speedup -- order is preserved
    # by collecting results into a pre-sized list before flattening.
    results = [None] * len(batches)
    MAX_WORKERS = 4
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_run_batch, i, b): i for i, b in enumerate(batches)}
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                idx, batch_scenes = future.result()
                results[idx] = batch_scenes
            except Exception as e:
                print(f"  ❌ Batch {batch_idx+1} failed: {e}")
                raise

    for batch_scenes in results:
        all_scenes.extend(batch_scenes)

    print(f"  ✅ {len(all_scenes)} total scenes composed")
    return all_scenes


# ============================================================
# VALIDATION PASS - validates scene compositions
# ============================================================
def _ensure_bright_color(hex_color: str, min_luminance: float = 130.0) -> str:
    """If a color is too dark to read against the near-black procedural
    background, brighten it. White and the brand yellow (#FBC02D) pass
    through unchanged -- they're already bright. Dark/muted colors get
    scaled up toward white while preserving hue, so 'dark grey' becomes
    'light grey' rather than just snapping to pure white for everything."""
    try:
        h = hex_color.strip().lstrip('#')
        if len(h) != 6:
            return "#FFFFFF"
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except Exception:
        return "#FFFFFF"

    # Perceived luminance (standard weights)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    if luminance >= min_luminance:
        return hex_color

    if luminance < 1.0:
        # Pure/near black -- no hue to preserve, just go white
        return "#FFFFFF"

    # Scale up toward white, preserving hue/ratio
    scale = min_luminance / luminance
    r = min(255, int(r * scale))
    g = min(255, int(g * scale))
    b = min(255, int(b * scale))
    return f"#{r:02X}{g:02X}{b:02X}"


def validate_decisions(scenes: list, beats: list) -> list:
    print(f"  🔍 Validating {len(scenes)} scenes...")
    fixed = 0

    for scene_pos, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            scenes[scene_pos] = {"beat_index": scene_pos, "elements": []}
            continue

        # Use enumeration position -- ignore GPT's beat_index
        scene["beat_index"] = scene_pos
        beat = beats[scene_pos] if scene_pos < len(beats) else {}
        beat_text = beat.get("text", "").strip().lower()
        beat_words = set()
        for w in beat_text.split():
            beat_words.add(w.strip('.,!?;:\'"()[]- '))

        elements = scene.get("elements", [])
        if not isinstance(elements, list):
            scene["elements"] = []
            continue

        cleaned = []
        for el in elements:
            if not isinstance(el, dict):
                continue
            etype = el.get("type", "text")

            # Validate TEXT content against beat text
            if etype == "text":
                content = (el.get("content") or "").strip()
                if not content:
                    continue
                # Strip punctuation, lowercase for comparison
                check_words = [w.strip('.,!?;:\'"()[]- ').lower()
                               for w in content.split()
                               if len(w.strip('.,!?;:\'"()[]- ')) > 2]
                if check_words and beat_words:
                    matches = sum(1 for w in check_words if w in beat_words)
                    if matches == 0 and len(check_words) > 0:
                        # Pure hallucination -- skip element
                        print(f"  ⚠ Scene {scene_pos}: dropped hallucinated text '{content[:30]}'")
                        fixed += 1
                        continue

                # Default text properties
                el.setdefault("x", 0.5)
                el.setdefault("y", 0.5)
                el.setdefault("anchor", "center")
                el.setdefault("size", 90)
                el.setdefault("color", "#FFFFFF")
                el.setdefault("weight", "black")
                el.setdefault("outline", 4)
                el.setdefault("anim", "fade_in")
                el.setdefault("start_offset", 0.0)
                el.setdefault("duration", None)
                el.setdefault("anim_duration", 0.15)
                el.setdefault("effect", "none")

                # Enforce minimum brightness -- against the dark procedural
                # background, a muted/dark fill color (e.g. dark grey, olive)
                # combined with a black outline becomes nearly invisible.
                el["color"] = _ensure_bright_color(el["color"])

            elif etype == "line":
                el.setdefault("x1", 0.3)
                el.setdefault("y1", 0.5)
                el.setdefault("x2", 0.7)
                el.setdefault("y2", 0.5)
                el.setdefault("thickness", 6)
                el.setdefault("color", "#FFFFFF")
                el.setdefault("anim", "draw_horizontal")
                el.setdefault("start_offset", 0.0)
                el.setdefault("duration", None)
                el.setdefault("anim_duration", 0.3)

            elif etype == "rect":
                el.setdefault("x", 0.4)
                el.setdefault("y", 0.4)
                el.setdefault("w", 0.2)
                el.setdefault("h", 0.1)
                el.setdefault("color", "#FFFFFF")
                el.setdefault("filled", True)
                el.setdefault("thickness", 3)
                el.setdefault("anim", "fade_in")
                el.setdefault("start_offset", 0.0)
                el.setdefault("duration", None)
                el.setdefault("anim_duration", 0.2)

            elif etype == "circle":
                el.setdefault("x", 0.5)
                el.setdefault("y", 0.5)
                el.setdefault("radius", 0.05)
                el.setdefault("color", "#FFFFFF")
                el.setdefault("filled", False)
                el.setdefault("thickness", 4)
                el.setdefault("anim", "fade_in")
                el.setdefault("start_offset", 0.0)
                el.setdefault("duration", None)
                el.setdefault("anim_duration", 0.2)

            else:
                # Unknown type, skip
                continue

            # Clamp coordinates to safe range
            for k in ("x", "y", "x1", "y1", "x2", "y2", "w", "h", "radius"):
                if k in el and isinstance(el[k], (int, float)):
                    el[k] = max(0.0, min(1.0, float(el[k])))

            cleaned.append(el)

        # HARD CAP: max 4 elements per scene
        if len(cleaned) > 4:
            print(f"  ⚠ Scene {scene_pos}: trimmed {len(cleaned)} elements to 4")
            cleaned = cleaned[:4]
            fixed += 1

        # AUTO-STAGGER: if all text elements have start_offset 0.0, spread them out
        text_els = [e for e in cleaned if e.get("type", "text") == "text" and isinstance(e.get("content", ""), str)]
        all_zero = all(float(e.get("start_offset", 0.0)) < 0.05 for e in text_els)
        if all_zero and len(text_els) > 1:
            beat_dur = max(0.5, float(beats[scene_pos].get("end_time", 2.0)) - float(beats[scene_pos].get("start_time", 0.0))) if scene_pos < len(beats) else 1.0
            step = min(0.25, beat_dur / (len(text_els) + 1))
            for i, e in enumerate(text_els):
                e["start_offset"] = round(i * step, 2)
            fixed += 1

        scene["elements"] = cleaned

    print(f"  ✅ Validated {len(scenes)} scenes, fixed {fixed} issues")
    return scenes



# ============================================================
# SCENE-BASED RENDERER v5
# GPT outputs scene compositions (lists of elements).
# This renderer executes any combination of text/line/rect/circle
# with per-element animation and timing.
# ============================================================
def render_text_overlay_opencv(video_path: str, scenes: list, beats: list,
                               whisper_segments: list, output_path: str):
    print(f"🎨 Scene renderer v5: {len(scenes)} scenes...")

    try:
        import cv2
        from PIL import Image, ImageDraw, ImageFont
        print(f"  ✓ OpenCV {cv2.__version__} + Pillow ready")
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        subprocess.run(['ffmpeg', '-y', '-i', video_path, '-c', 'copy', output_path],
                       check=True, capture_output=True)
        return

    if not os.path.exists(video_path):
        raise Exception(f"Video not found: {video_path}")

    # ── basic helpers ──────────────────────────────────────────────
    def load_pil_font(path, size, weight="black"):
        # weight selects between Black / ExtraBold / Bold
        try:
            if weight == "regular":
                p = FONT_BOLD or path
            elif weight == "black":
                p = FONT_BLACK or FONT_BOLD or path
            else:
                p = FONT_BOLD or FONT_BLACK or path
            if p and os.path.exists(p):
                return ImageFont.truetype(p, size)
        except: pass
        return ImageFont.load_default()

    def hex_to_rgb(hex_str):
        h = (hex_str or "#FFFFFF").lstrip('#')
        try: return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        except: return (255, 255, 255)

    def apply_vignette(frame):
        import numpy as np
        rows, cols = frame.shape[:2]
        X = cv2.getGaussianKernel(cols, cols * 0.6)
        Y = cv2.getGaussianKernel(rows, rows * 0.6)
        mask = (Y * X.T) / (Y * X.T).max()
        out = frame.copy().astype(np.float32)
        for i in range(3): out[:,:,i] *= mask
        return np.clip(out, 0, 255).astype(np.uint8)

    def apply_warm_grade(frame):
        import numpy as np
        out = frame.copy().astype(np.float32)
        out[:,:,2] = np.clip(out[:,:,2] * 1.04, 0, 255)
        return out.astype(np.uint8)

    def to_pil(frame):
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def to_frame(pil_img):
        import numpy as np
        return cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)

    def composite_layer(frame, layer):
        pil = to_pil(frame).convert('RGBA')
        merged = Image.alpha_composite(pil, layer)
        return to_frame(merged)

    # ── video metadata ─────────────────────────────────────────────
    def _ffprobe_dur(path):
        try:
            r = subprocess.run(['ffprobe','-v','error','-show_entries','format=duration',
                                '-of','default=noprint_wrappers=1:nokey=1', path],
                               capture_output=True, text=True)
            return float(r.stdout.strip())
        except: return 0.0

    TARGET_FPS = 30.0
    vid_dur = _ffprobe_dur(video_path)

    # Transcode to CFR for predictable frame timing
    cfr_video = output_path.replace(".mp4", "_cfr_tmp.mp4")
    subprocess.run(['ffmpeg','-y','-i',video_path,
                    '-vf',f'fps={TARGET_FPS:.0f}',
                    '-c:v','libx264','-preset','ultrafast','-crf','18',
                    '-an', cfr_video],
                   capture_output=True, check=True)

    # ── Build ordered Whisper word list — exact timestamps in speech order ──
    whisper_word_list = []
    for seg in whisper_segments:
        for we in seg.get('words', []):
            raw = we.get('word', '')
            wc = raw.upper().strip('.,!?;:\'"()[]- ')
            if not wc:
                continue
            whisper_word_list.append({
                'word':  wc,
                'start': float(we.get('start', 0.0)),
                'end':   float(we.get('end',   0.0)),
            })

    def get_beat_whisper_words(beat_start, beat_end):
        """All Whisper words whose start falls within this beat window."""
        return [w for w in whisper_word_list
                if beat_start - 0.15 <= w['start'] <= beat_end + 0.15]

    def match_word_in_list(word, candidates):
        """Find the best matching Whisper word entry for `word` within candidates."""
        wc = word.upper().strip('.,!?;:\'"()[]- ')
        for w in candidates:
            if w['word'] == wc:
                return w
        for w in candidates:
            if wc in w['word'] or w['word'] in wc:
                return w
        return None

    def clamp(v, lo, hi): return max(lo, min(v, hi))

    # ── Build flat timeline — pure Whisper, scoped per beat ─────────────────
    #
    # SINGLE-WORD BEAT: 1 text element → IMPACT mode
    #   - Appears at exact Whisper word start
    #   - Lasts up to 1.0s with bzzt flicker, but never overlaps the next beat
    #
    # MULTI-WORD BEAT: multiple text elements → CAPTION mode
    #   - Each word appears at its Whisper timestamp
    #   - Disappears when next word is spoken
    #   - For multi-word content (e.g. "THE WORLD"), anim_end accounts for
    #     the LAST word's end time, not just the first word's start
    #
    timeline = []
    for scene_pos, scene in enumerate(scenes):
        beat = beats[scene_pos] if scene_pos < len(beats) else {}
        beat_start = clamp(float(beat.get("start_time", 0.0)), 0, vid_dur - 0.1)
        beat_end   = clamp(float(beat.get("end_time", beat_start + 2.0)),
                           beat_start + 0.05, vid_dur)
        next_beat_start = None
        if scene_pos + 1 < len(beats):
            next_beat_start = float(beats[scene_pos + 1].get("start_time", beat_end))
            beat_end = min(beat_end, next_beat_start)

        elements = scene.get("elements", [])
        if not elements:
            continue

        text_els  = [e for e in elements if e.get("type", "text") == "text"]
        other_els = [e for e in elements if e.get("type", "text") != "text"]

        # Whisper words available in this beat's time window (beat-scoped to
        # avoid matching the wrong occurrence of common words like THE/IS/A)
        beat_words = get_beat_whisper_words(beat_start, beat_end)

        # Resolve each text element to Whisper timestamps within this beat
        resolved = []  # [(whisper_start, whisper_end, el)]
        used_indices = set()
        for el in text_els:
            raw = (el.get("content") or "").strip()
            if not raw:
                continue
            words_in_content = raw.split()
            available = [w for i, w in enumerate(beat_words) if i not in used_indices]

            first_match = match_word_in_list(words_in_content[0], available)
            if first_match:
                idx = beat_words.index(first_match)
                used_indices.add(idx)
                el_start = first_match['start']
                el_end   = first_match['end']

                # If content has multiple words, extend el_end to cover the
                # LAST word's Whisper end time too (so "THE WORLD" stays
                # visible until "WORLD" is actually spoken, not just "THE")
                if len(words_in_content) > 1:
                    available2 = [w for i, w in enumerate(beat_words) if i not in used_indices]
                    last_match = match_word_in_list(words_in_content[-1], available2)
                    if last_match:
                        idx2 = beat_words.index(last_match)
                        used_indices.add(idx2)
                        el_end = max(el_end, last_match['end'])

                resolved.append((el_start, el_end, el))
            else:
                # No Whisper match — fallback to beat_start
                resolved.append((beat_start, beat_end, el))

        resolved.sort(key=lambda x: x[0])
        is_single_word = len(resolved) == 1

        for i, (ws, we_t, el) in enumerate(resolved):
            anim_start = clamp(ws, 0.0, vid_dur - 0.1)

            if is_single_word:
                # IMPACT: up to 1.0s with bzzt flicker, but never bleed into
                # the next beat's start time
                impact_end = anim_start + 1.0
                if next_beat_start is not None:
                    impact_end = min(impact_end, next_beat_start)
                anim_end = clamp(impact_end, anim_start + 0.1, vid_dur)
                impact = True
            else:
                # CAPTION: hold until next word's Whisper start, but at least
                # until this element's own Whisper end time (covers multi-word content)
                min_end = max(we_t, anim_start + 0.08)
                if i + 1 < len(resolved):
                    anim_end = clamp(max(resolved[i + 1][0], min_end), anim_start + 0.08, vid_dur)
                else:
                    anim_end = clamp(max(beat_end, min_end), anim_start + 0.08, vid_dur)
                impact = False

            timeline.append({
                "el":            el,
                "start":         anim_start,
                "end":           anim_end,
                "anim_duration": 0.06 if impact else float(el.get("anim_duration", 0.10)),
                "impact":        impact,
            })

        for el in other_els:
            timeline.append({
                "el":            el,
                "start":         beat_start,
                "end":           beat_end,
                "anim_duration": float(el.get("anim_duration", 0.2)),
                "impact":        False,
            })

    timeline.sort(key=lambda x: x["start"])
    print(f"  📊 Timeline: {len(timeline)} elements")

    # ── Frame-by-frame render ─────────────────────────────────────
    cap = cv2.VideoCapture(cfr_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_vid = TARGET_FPS

    temp_video = output_path.replace(".mp4", "_noaudio_tmp.mp4")
    out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps_vid, (fw, fh))

    print(f"  🎬 {total_frames} frames @ {fps_vid:.0f}fps...")
    frame_idx = 0
    prev_pct = -1

    # ── Character compositor setup ──────────────────────────────────
    character_ok = False
    char_base, char_patches, char_mouth_envelope = None, None, None
    if CHARACTER_ENABLED:
        try:
            char_base, char_patches = load_character_assets()
            char_mouth_envelope = build_mouth_envelope(whisper_word_list, total_frames, fps_vid)
            character_ok = True
            active_frames = int(np.count_nonzero(char_mouth_envelope > 0.03))
            avg_open = float(char_mouth_envelope.mean()) if len(char_mouth_envelope) else 0.0
            peak_open = float(char_mouth_envelope.max()) if len(char_mouth_envelope) else 0.0
            print(f"  🎭 Character loaded -- smooth mouth active on "
                  f"{active_frames}/{total_frames} frames "
                  f"(avg={avg_open:.3f}, peak={peak_open:.3f})")
        except Exception as e:
            print(f"  ⚠ Character disabled (load failed): {e}")

    # ── Element drawing functions ──────────────────────────────────
    def get_anim_progress(el_t, start, end, anim_dur):
        """Return (entrance_progress, exit_progress) both 0..1.
        entrance_progress: 0=not started, 1=fully appeared
        exit_progress: 1=visible, 0=fully gone (only at end)
        """
        if anim_dur <= 0:
            anim_dur = 0.001
        entrance = clamp((el_t - start) / anim_dur, 0.0, 1.0)
        # No exit fade by default - just snap out at end
        return entrance

    def draw_text_element(layer, el, el_t, anim_t):
        """Draw a TEXT element with animation."""
        draw = ImageDraw.Draw(layer)
        content = el.get("content", "").upper().strip()  # Always uppercase for Anton
        if not content:
            return
        x_pct = float(el.get("x", 0.5))
        y_pct = float(el.get("y", 0.5))
        # Hard cap: 130px max for sentence words, 160px for single-word impact
        raw_size = int(el.get("size", 90))
        word_count = len(content.split())
        size_cap = 160 if word_count == 1 else 110
        size = max(20, min(raw_size, size_cap))
        color = hex_to_rgb(el.get("color", "#FFFFFF"))
        weight = el.get("weight", "black")
        outline = max(0, min(int(el.get("outline", 4)), 12))
        anim = el.get("anim", "fade_in")
        anchor = el.get("anchor", "center")
        effect = el.get("effect", "none")

        font = load_pil_font(get_primary_font_path(), size, weight)
        try:
            bbox = font.getbbox(content)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except:
            tw = size * len(content) * 0.55
            th = size

        # Compute base position
        target_x = int(OUTPUT_WIDTH * x_pct)
        target_y = int(OUTPUT_HEIGHT * y_pct)
        if anchor == "center":
            base_x = target_x - tw // 2
            base_y = target_y - th // 2
        elif anchor == "left":
            base_x = target_x
            base_y = target_y - th // 2
        elif anchor == "right":
            base_x = target_x - tw
            base_y = target_y - th // 2
        else:
            base_x = target_x - tw // 2
            base_y = target_y - th // 2

        # Clamp to screen with padding so text never goes off-edge
        pad = 30
        max_x = OUTPUT_WIDTH - tw - pad
        max_y = OUTPUT_HEIGHT - th - pad
        base_x = max(pad, min(base_x, max_x))
        base_y = max(pad, min(base_y, max_y))

        # Apply entrance animation
        draw_x, draw_y = base_x, base_y
        alpha = 1.0
        scale = 1.0

        if anim == "fade_in":
            alpha = anim_t
        elif anim == "slide_in_left":
            slide_dist = int(OUTPUT_WIDTH * 0.3)
            draw_x = base_x - int(slide_dist * (1.0 - anim_t))
            alpha = anim_t
        elif anim == "slide_in_right":
            slide_dist = int(OUTPUT_WIDTH * 0.3)
            draw_x = base_x + int(slide_dist * (1.0 - anim_t))
            alpha = anim_t
        elif anim == "slide_in_top":
            slide_dist = int(OUTPUT_HEIGHT * 0.2)
            draw_y = base_y - int(slide_dist * (1.0 - anim_t))
            alpha = anim_t
        elif anim == "slide_in_bottom":
            slide_dist = int(OUTPUT_HEIGHT * 0.2)
            draw_y = base_y + int(slide_dist * (1.0 - anim_t))
            alpha = anim_t
        elif anim == "scale_in":
            scale = 1.3 - 0.3 * anim_t
            alpha = anim_t
        elif anim == "snap":
            alpha = 1.0
        elif anim == "none":
            alpha = 1.0

        # Effects (applied during display)
        if effect == "flicker":
            # Blink during first 0.3s of element life
            if el_t - 0 < 0.3:
                frame_no = int(el_t * 30)
                if frame_no % 2 == 1:
                    return
        elif effect == "shake":
            import random as _r
            draw_x += _r.randint(-3, 3)
            draw_y += _r.randint(-3, 3)

        # Re-render text at scaled size if scale != 1.0
        render_font = font
        if abs(scale - 1.0) > 0.02:
            new_size = max(20, int(size * scale))
            render_font = load_pil_font(get_primary_font_path(), new_size, weight)
            try:
                bbox = render_font.getbbox(content)
                tw2 = bbox[2] - bbox[0]
                th2 = bbox[3] - bbox[1]
                draw_x = base_x + (tw - tw2) // 2
                draw_y = base_y + (th - th2) // 2
            except: pass

        # Clamp draw position AFTER animation offsets (slide/shake can push text off-screen)
        draw_x = max(pad, min(draw_x, OUTPUT_WIDTH - tw - pad))
        draw_y = max(pad, min(draw_y, OUTPUT_HEIGHT - th - pad))

        a_int = max(0, min(int(255 * alpha), 255))
        if a_int < 5:
            return

        # Draw outline (multiple offsets for thick outline)
        if outline > 0:
            for ox in range(-outline, outline + 1):
                for oy in range(-outline, outline + 1):
                    if ox * ox + oy * oy <= outline * outline:
                        if ox == 0 and oy == 0:
                            continue
                        draw.text((draw_x + ox, draw_y + oy), content,
                                  font=render_font, fill=(0, 0, 0, a_int))
        # Draw fill
        draw.text((draw_x, draw_y), content, font=render_font,
                  fill=(color[0], color[1], color[2], a_int))

    def draw_line_element(layer, el, el_t, anim_t):
        """Draw a LINE element with animation."""
        draw = ImageDraw.Draw(layer)
        x1 = int(OUTPUT_WIDTH * float(el.get("x1", 0.3)))
        y1 = int(OUTPUT_HEIGHT * float(el.get("y1", 0.5)))
        x2 = int(OUTPUT_WIDTH * float(el.get("x2", 0.7)))
        y2 = int(OUTPUT_HEIGHT * float(el.get("y2", 0.5)))
        thickness = max(1, int(el.get("thickness", 6)))
        color = hex_to_rgb(el.get("color", "#FFFFFF"))
        anim = el.get("anim", "draw_horizontal")

        alpha = 1.0
        end_x, end_y = x2, y2

        if anim == "fade_in":
            alpha = anim_t
        elif anim == "draw_horizontal":
            end_x = x1 + int((x2 - x1) * anim_t)
            end_y = y1 + int((y2 - y1) * anim_t)
            alpha = 1.0
        elif anim == "none":
            alpha = 1.0

        a_int = max(0, min(int(255 * alpha), 255))
        if a_int < 5:
            return

        draw.line([(x1, y1), (end_x, end_y)],
                  fill=(color[0], color[1], color[2], a_int),
                  width=thickness)

    def draw_rect_element(layer, el, el_t, anim_t):
        """Draw a RECT element."""
        draw = ImageDraw.Draw(layer)
        x = int(OUTPUT_WIDTH * float(el.get("x", 0.4)))
        y = int(OUTPUT_HEIGHT * float(el.get("y", 0.4)))
        w = int(OUTPUT_WIDTH * float(el.get("w", 0.2)))
        h = int(OUTPUT_HEIGHT * float(el.get("h", 0.1)))
        color = hex_to_rgb(el.get("color", "#FFFFFF"))
        filled = bool(el.get("filled", True))
        thickness = max(1, int(el.get("thickness", 3)))
        anim = el.get("anim", "fade_in")

        alpha = 1.0
        if anim == "fade_in":
            alpha = anim_t
        elif anim == "scale_in":
            scale = anim_t
            cx, cy = x + w // 2, y + h // 2
            w = int(w * scale); h = int(h * scale)
            x = cx - w // 2; y = cy - h // 2
            alpha = anim_t

        a_int = max(0, min(int(255 * alpha), 255))
        if a_int < 5:
            return

        rgba = (color[0], color[1], color[2], a_int)
        if filled:
            draw.rectangle([x, y, x + w, y + h], fill=rgba)
        else:
            draw.rectangle([x, y, x + w, y + h], outline=rgba, width=thickness)

    def draw_circle_element(layer, el, el_t, anim_t):
        """Draw a CIRCLE element."""
        draw = ImageDraw.Draw(layer)
        cx = int(OUTPUT_WIDTH * float(el.get("x", 0.5)))
        cy = int(OUTPUT_HEIGHT * float(el.get("y", 0.5)))
        r = int(min(OUTPUT_WIDTH, OUTPUT_HEIGHT) * float(el.get("radius", 0.05)))
        color = hex_to_rgb(el.get("color", "#FFFFFF"))
        filled = bool(el.get("filled", False))
        thickness = max(1, int(el.get("thickness", 4)))
        anim = el.get("anim", "fade_in")

        alpha = 1.0
        if anim == "fade_in":
            alpha = anim_t
        elif anim == "scale_in":
            r = int(r * anim_t)
            alpha = anim_t

        a_int = max(0, min(int(255 * alpha), 255))
        if a_int < 5 or r <= 0:
            return

        rgba = (color[0], color[1], color[2], a_int)
        bbox = [cx - r, cy - r, cx + r, cy + r]
        if filled:
            draw.ellipse(bbox, fill=rgba)
        else:
            draw.ellipse(bbox, outline=rgba, width=thickness)

    # ── Main frame loop ──────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret: break

        t = frame_idx / fps_vid
        frame = apply_vignette(frame)
        frame = apply_warm_grade(frame)

        # Character: centered, idle motion + smooth mouth envelope driven by
        # Whisper word timing. Drawn before the text layer so on-screen
        # text remains on top.
        if character_ok:
            mouth_openness = (
                float(char_mouth_envelope[frame_idx])
                if frame_idx < len(char_mouth_envelope)
                else 0.0
            )
            frame = render_character_frame(frame, char_base, char_patches, t, mouth_openness, fw, fh)

        # Find all elements active at time t
        raw_active = [item for item in timeline
                      if item["start"] <= t < item["end"]]

        # Deduplicate: if two elements share the same content + near-same x/y position,
        # keep only the one whose start time is closest to t (most recently activated).
        # This prevents "WE WE" doubles when GPT repeats a word across adjacent beats.
        seen_keys = {}
        for item in sorted(raw_active, key=lambda x: x["start"], reverse=True):
            el = item["el"]
            if el.get("type") == "text":
                content_key = (el.get("content", "").upper().strip(),
                               round(float(el.get("x", 0.5)), 1),
                               round(float(el.get("y", 0.5)), 1))
                if content_key not in seen_keys:
                    seen_keys[content_key] = item
            else:
                seen_keys[id(item["el"])] = item
        active = list(seen_keys.values())

        if active:
            # Slight darkening overlay when text is on screen
            import numpy as np
            frame = cv2.addWeighted(frame, 0.82, np.zeros_like(frame), 0.18, 0)

            # Build composite layer
            layer = Image.new('RGBA', (OUTPUT_WIDTH, OUTPUT_HEIGHT), (0, 0, 0, 0))

            for item in active:
                el    = item["el"]
                el_t  = t - item["start"]   # seconds since element appeared
                el_dur = max(item["end"] - item["start"], 0.01)
                impact = item.get("impact", False)
                etype  = el.get("type", "text")

                if impact and etype == "text":
                    # BZZT FLICKER: flash-flash-flash-hold pattern over 1.0s
                    # 0.00-0.08s: ON  (flash 1)
                    # 0.08-0.16s: OFF
                    # 0.16-0.24s: ON  (flash 2)
                    # 0.24-0.32s: OFF
                    # 0.32-0.40s: ON  (flash 3)
                    # 0.40s+    : HOLD fully visible with slow fade-out at end
                    in_flash = (
                        (0.00 <= el_t < 0.08) or
                        (0.16 <= el_t < 0.24) or
                        (0.32 <= el_t < 0.40)
                    )
                    in_off = (0.08 <= el_t < 0.16) or (0.24 <= el_t < 0.32)
                    if in_off:
                        continue  # skip drawing — element is "off"
                    # During flashes use full alpha; after 0.4s hold then fade out near end
                    if el_t >= 0.40:
                        fade_window = 0.15
                        time_left = el_dur - el_t
                        if time_left < fade_window:
                            anim_t = max(0.0, time_left / fade_window)
                        else:
                            anim_t = 1.0
                    else:
                        anim_t = 1.0  # instant full alpha during flashes
                    try:
                        draw_text_element(layer, el, el_t, anim_t)
                    except Exception as e:
                        print(f"  ⚠ impact render error: {e}")
                else:
                    anim_t = get_anim_progress(el_t, 0, el_dur, item["anim_duration"])
                    try:
                        if etype == "text":
                            draw_text_element(layer, el, el_t, anim_t)
                        elif etype == "line":
                            draw_line_element(layer, el, el_t, anim_t)
                        elif etype == "rect":
                            draw_rect_element(layer, el, el_t, anim_t)
                        elif etype == "circle":
                            draw_circle_element(layer, el, el_t, anim_t)
                    except Exception as e:
                        print(f"  ⚠ element render error: {e}")

            frame = composite_layer(frame, layer)

        out.write(frame)
        frame_idx += 1
        pct = int(frame_idx / max(total_frames, 1) * 20)
        if pct != prev_pct:
            print(f"  [{'█' * pct}{'░' * (20 - pct)}] {frame_idx}/{total_frames}",
                  end='\r')
            prev_pct = pct

    cap.release(); out.release()
    if os.path.exists(cfr_video): os.remove(cfr_video)
    print(f"\n  ✓ Frames done")

    result = subprocess.run([
        'ffmpeg', '-y', '-i', temp_video, '-i', video_path,
        '-map', '0:v', '-map', '1:a',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'copy', output_path
    ], capture_output=True)

    if os.path.exists(temp_video): os.remove(temp_video)
    if result.returncode != 0:
        raise Exception(f"Audio merge failed: {result.stderr.decode()[-200:]}")

    print(f"  ✅ Render complete: {output_path}")



# ============================================================
# MAIN GENERATOR
# ============================================================
class VaultsGenerator:
    def __init__(self, audio_path: str, output_path: str = "output.mp4", niche_config: dict = None):
        self.audio_path  = audio_path
        self.output_path = output_path

        if niche_config:
            self.broll_dirs  = niche_config.get('broll_dirs', {})
            self.keyword_map = niche_config.get('keyword_map', {})
        else:
            self.broll_dirs = {
                'space':   'space_vids',
                'ancient': 'ancient_ruins_vids',
                'cosmic':  'cosmic_vids',
                'sky':     'dark_sky_vids',
                'temple':  'temple_vids',
            }
            self.keyword_map = {
                'space':   ['universe', 'galaxy', 'black hole', 'star', 'planet', 'cosmos'],
                'ancient': ['ancient', 'civilization', 'pyramid', 'ruins', 'lost', 'forgotten'],
                'cosmic':  ['time', 'reality', 'dimension', 'quantum', 'existence', 'consciousness'],
                'sky':     ['sky', 'atmosphere', 'above', 'beyond', 'vast', 'endless'],
                'temple':  ['religion', 'god', 'sacred', 'ritual', 'belief', 'worship'],
            }

    def get_audio_duration(self) -> float:
        cmd    = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                  '-of', 'default=noprint_wrappers=1:nokey=1', self.audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"ffprobe failed: {result.stderr}")
        return float(result.stdout.strip())

    def get_video_info(self, filepath: str):
        cmd    = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                  '-show_entries', 'stream=width,height', '-of', 'json', filepath]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            data = json.loads(result.stdout)
            w    = data['streams'][0]['width']
            h    = data['streams'][0]['height']
            return w, h, w / h
        except:
            return None, None, None

    def get_all_files_from_dir(self, directory: str) -> list:
        if not os.path.exists(directory):
            return []
        files = [os.path.join(directory, f) for f in os.listdir(directory)
                 if f.lower().endswith(('.mp4', '.mov', '.avi'))]
        if not files:
            print(f"  ⚠ Folder exists but is EMPTY: {directory}")
        return files

    def transcribe_with_whisper(self, model: str = "base") -> dict | None:
        cache_file = f"{os.path.splitext(self.audio_path)[0]}_transcription.json"
        if os.path.exists(cache_file):
            print(f"  ✅ Cached transcription")
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        try:
            import whisper
            if not hasattr(whisper, 'load_model'):
                raise ImportError("Wrong whisper. Run: pip install openai-whisper")
            print(f"  🎤 Transcribing ({model})...")
            wm     = whisper.load_model(model)
            result = wm.transcribe(self.audio_path, word_timestamps=True, language="en")
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
            return result
        except Exception as e:
            print(f"  ❌ Whisper error: {e}")
            return None

    def match_broll_categories(self, full_text: str) -> list:
        text   = full_text.lower()
        scores = {cat: sum(text.count(k) for k in kws)
                  for cat, kws in self.keyword_map.items()}
        sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = [self.broll_dirs[c] for c, s in sorted_cats if s > 0 and c in self.broll_dirs]

        # FIX 2: Filter to only folders that actually exist and have clips
        valid_top = []
        for folder in top:
            files = self.get_all_files_from_dir(folder)
            if files:
                valid_top.append(folder)
            else:
                print(f"  ⚠ Skipping empty/missing broll folder: {folder}")

        if not valid_top:
            # Fallback: scan ALL configured broll dirs and use any that have clips
            print(f"  ⚠ No keyword-matched folders had clips -- scanning all broll dirs...")
            for folder in self.broll_dirs.values():
                files = self.get_all_files_from_dir(folder)
                if files:
                    valid_top.append(folder)
                    print(f"  ✓ Found clips in: {folder} ({len(files)} files)")

        if not valid_top:
            raise Exception(
                "No broll clips found in ANY configured folder.\n"
                f"Configured dirs: {list(self.broll_dirs.values())}\n"
                "Add your Seedance space/ancient/cosmic clips to these folders."
            )

        return valid_top

    def create_segment_plan(self, duration: float, beats: list, top_categories: list) -> list:
        segments = []
        # Build complete pool of all folders that have clips
        all_folders = []
        for folder in self.broll_dirs.values():
            if self.get_all_files_from_dir(folder):
                all_folders.append(folder)
        if not all_folders:
            raise Exception("No broll clips found in any folder.")

        # Per-folder clip pools with used tracking
        folder_pools = {}
        for folder in all_folders:
            folder_pools[folder] = list(self.get_all_files_from_dir(folder))

        # Beat category → preferred folder (best-effort, falls back to rotation)
        broll_cat_to_folder = {
            'space':   self.broll_dirs.get('space',   'space_vids'),
            'ancient': self.broll_dirs.get('ancient', 'ancient_ruins_vids'),
            'cosmic':  self.broll_dirs.get('cosmic',  'cosmic_vids'),
            'sky':     self.broll_dirs.get('sky',     'dark_sky_vids'),
            'temple':  self.broll_dirs.get('temple',  'temple_vids'),
        }

        # Per-folder shuffle indices so we cycle without repeating
        folder_idx = {f: 0 for f in all_folders}
        for f in all_folders:
            random.shuffle(folder_pools[f])

        base_dur   = 4.0
        n_segs     = max(int(duration / base_dur), 1)
        folder_rot = 0

        for i in range(n_segs):
            seg_dur = float(beats[i].get('clip_duration', base_dur)) if i < len(beats) else base_dur
            # Pure round-robin across all folders -- guaranteed variety
            target_folder = all_folders[folder_rot % len(all_folders)]
            folder_rot += 1

            # Pick next clip from folder, cycling
            pool = folder_pools[target_folder]
            idx  = folder_idx[target_folder]
            if idx >= len(pool):
                random.shuffle(pool)
                idx = 0
            chosen = pool[idx]
            folder_idx[target_folder] = idx + 1

            segments.append({
                'type':     'broll',
                'category': target_folder,
                'file':     chosen,
                'duration': seg_dur,
            })
            print(f"    seg {i+1}: {os.path.basename(chosen)} [{os.path.basename(target_folder)}]")

        if not segments:
            raise Exception("No segments created.")

        total = sum(s['duration'] for s in segments)
        if total < duration:
            segments[-1]['duration'] += (duration - total)

        return segments

    def _make_black_filler(self, output_file: str, dur: float, fps: int = 30) -> str:
        """Generate a black video segment of exact duration — used when broll clip fails."""
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', f'color=c=black:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:r={fps}:d={dur}',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-r', str(fps), '-an', '-t', str(dur),
            output_file
        ]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            raise Exception(f"Black filler failed: {r.stderr.decode()[-200:]}")
        return output_file

    def process_segment_to_file(self, segment: dict, output_file: str,
                                fps: int = 30, progress_callback=None) -> str:
        """Process one broll segment. ALWAYS returns a valid file — never skips.
        If the clip fails, falls back to a black filler of the correct duration
        so total video length is preserved and text timestamps stay in sync."""
        dur = segment['duration']
        source_file = segment['file']
        w, h, aspect = self.get_video_info(source_file)

        cmd = ['ffmpeg', '-y', '-progress', 'pipe:1', '-nostats',
               '-i', source_file, '-t', str(dur)]

        vf = []
        if aspect and aspect < (OUTPUT_WIDTH / OUTPUT_HEIGHT):
            vf += [f"scale={OUTPUT_WIDTH}:-2:force_original_aspect_ratio=decrease",
                   f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black"]
        else:
            vf += [f"scale=-2:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase",
                   f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}"]

        # fps=30 MUST come first to convert VFR → CFR before any other filter
        vf = [f"fps={fps}"] + vf
        vf += ["eq=brightness=0.02:contrast=1.05:saturation=1.1", "format=yuv420p"]
        cmd += ['-vf', ','.join(vf), '-c:v', 'libx264', '-preset', 'ultrafast',
                '-crf', '23', '-pix_fmt', 'yuv420p', '-r', str(fps), '-an', output_file]

        success  = False
        err_text = ""
        if progress_callback:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True, bufsize=1)
            total_f = int(dur * fps)
            last_f  = 0
            for line in proc.stdout:
                if line.startswith('frame='):
                    try:
                        cf = int(line.split('=')[1].strip())
                        if cf > last_f:
                            last_f = cf
                            progress_callback(cf, total_f)
                    except:
                        pass
            stderr_out = proc.stderr.read() if proc.stderr else ""
            proc.wait()
            success = proc.returncode == 0
            err_text = stderr_out
        else:
            r = subprocess.run(cmd, capture_output=True)
            success = r.returncode == 0
            err_text = r.stderr.decode(errors='replace')

        if not success:
            err_line = err_text.strip().splitlines()[-1] if err_text.strip() else "unknown error"
            print(f"\n  ⚠ Clip failed ({os.path.basename(source_file)}): {err_line[:150]}")
            print(f"  ⚠ Using black filler ({dur:.2f}s)")
            return self._make_black_filler(output_file, dur, fps)

        return output_file

    def _add_cta_overlay(self, video_input: str, output_path: str, duration: float):
        end_time = round(max(duration - 4, 1), 3)
        vf = (
            f"drawtext=text='Vaults of History'"
            f":fontcolor=yellow:fontsize=42:font=Arial"
            f":borderw=2:bordercolor=black:shadowx=2:shadowy=2"
            f":x=(w-text_w)/2:y=h*0.91:enable='gt(t\\,{end_time})'"
        )
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', video_input, '-vf', vf, '-c:a', 'copy', output_path],
            capture_output=True
        )
        if result.returncode != 0:
            subprocess.run(['ffmpeg', '-y', '-i', video_input, '-c', 'copy', output_path],
                           check=True, capture_output=True)
        else:
            print(f"  ✨ CTA added")

    def create_vaults_video(self, bg_volume: float = 0.12, fps: int = 30) -> bool:
        import time
        t0 = time.time()

        print(f"\n{'='*70}")
        print(f"🏛  VAULTS OF HISTORY v3")
        print(f"{'='*70}")

        try:
            duration = self.get_audio_duration()
            print(f"⏱  {duration:.2f}s")
        except Exception as e:
            raise Exception(f"STEP 1 FAILED: {e}")

        print(f"\n[STEP 2] Transcribing...")
        transcription = self.transcribe_with_whisper()
        if not transcription:
            raise Exception("Transcription failed")

        full_text        = transcription.get('text', '').strip()
        whisper_segments = transcription.get('segments', [])
        print(f"  ✅ {len(full_text)} chars, {len(whisper_segments)} segments")

        print(f"\n[STEP 3] GPT Call 1: Story Beats...")
        try:
            topic_hint   = list(self.broll_dirs.keys())[0] if self.broll_dirs else "space"
            beats_result = analyze_story_beats(full_text, whisper_segments, topic_hint, duration)
            topic        = beats_result.get('topic', 'default')
            beats        = beats_result.get('beats', [])

            # CRITICAL: GPT Call 1 only sees segment-level [start-end] brackets.
            # When it splits one segment into multiple beats, it INVENTS the
            # split-point timestamps. Recompute every beat's start_time/end_time
            # from actual Whisper word timestamps for frame-accurate rendering.
            _whisper_words = build_whisper_word_list(whisper_segments)
            beats = realign_beat_times(beats, _whisper_words)
            print(f"  🎯 Realigned {len(beats)} beat timestamps to Whisper word boundaries")
        except Exception as e:
            raise Exception(f"STEP 3 FAILED: {e}")

        print(f"\n[STEP 4] GPT Call 2: Render Decisions...")
        try:
            decisions = generate_render_decisions(beats, topic)
        except Exception as e:
            raise Exception(f"STEP 4 FAILED: {e}")

        print(f"\n[STEP 5] Validating...")
        decisions = validate_decisions(decisions, beats)

        print(f"\n[STEP 6] Music...")
        bg_music = MUSIC_MAP.get(topic, MUSIC_MAP['default'])
        if not os.path.exists(bg_music):
            bg_music = MUSIC_MAP['default']
            if not os.path.exists(bg_music):
                for fname in (os.listdir('bg_musics') if os.path.exists('bg_musics') else []):
                    if fname.endswith('.mp3'):
                        bg_music = os.path.join('bg_musics', fname)
                        break
                else:
                    bg_music = None
        print(f"  🎵 {bg_music}")

        temp_files    = []
        concat_list   = "concat_list.txt"
        concat_output = "concatenated_video.mp4"
        audio_output  = "audio_mixed.mp4"

        try:
            if USE_PROCEDURAL_BACKGROUND:
                print(f"\n[STEP 7-10] Procedural background...")
                generate_procedural_background(beats, topic, duration, concat_output,
                                                 width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT, fps=fps)
            else:
                print(f"\n[STEP 7] B-roll matching...")
                top_categories = self.match_broll_categories(full_text)
                print(f"  📊 {top_categories}")

                print(f"\n[STEP 8] Segment plan...")
                try:
                    video_segments = self.create_segment_plan(duration, beats, top_categories)
                    print(f"  ✅ {len(video_segments)} segments")
                except Exception as e:
                    raise Exception(f"STEP 8 FAILED: {e}")

                print(f"\n[STEP 9] Processing segments...")
                try:
                    from tqdm import tqdm
                    use_tqdm = True
                except:
                    use_tqdm = False

                for i, seg in enumerate(video_segments):
                    temp_file = f"temp_segment_{i:02d}.mp4"
                    t_seg     = time.time()

                    if use_tqdm:
                        total_f = int(seg['duration'] * fps)
                        pbar    = tqdm(total=total_f,
                                       desc=f"  Seg {i+1}/{len(video_segments)}: {os.path.basename(seg['file'])[:28]}",
                                       unit='frame')
                        def upd(c, t, pb=pbar):
                            pb.n = min(c, t); pb.refresh()
                        result = None
                        try:
                            result = self.process_segment_to_file(seg, temp_file, fps, upd)
                        finally:
                            pbar.n = pbar.total; pbar.refresh(); pbar.close()
                            print(f"    ✓ {time.time()-t_seg:.1f}s")
                    else:
                        print(f"  {i+1}/{len(video_segments)}: {os.path.basename(seg['file'])}", end='', flush=True)
                        result = self.process_segment_to_file(seg, temp_file, fps)
                        print(f" ✓ ({time.time()-t_seg:.1f}s)")

                    if os.path.exists(temp_file):
                        temp_files.append(temp_file)

                if not temp_files:
                    raise Exception("No segments processed")

                print(f"\n[STEP 10] Concatenating {len(temp_files)} segments...")
                with open(concat_list, 'w') as f:
                    for tf in temp_files:
                        f.write(f"file '{tf}'\n")
                r = subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                                     '-i', concat_list, '-c', 'copy', concat_output],
                                    capture_output=True)
                if r.returncode != 0:
                    raise Exception(f"Concat failed: {r.stderr.decode()[-200:]}")
                print(f"  ✅ Done")

            print(f"\n[STEP 11] Audio mix...")
            cmd = ['ffmpeg', '-y', '-i', concat_output, '-i', self.audio_path]
            if bg_music and os.path.exists(bg_music):
                cmd += ['-i', bg_music]
                fc = (
                    f'[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume=1.0[voice];'
                    f'[2:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,'
                    f'volume={bg_volume},aloop=loop=-1:size=2e+09[bg];'
                    f'[voice][bg]amix=inputs=2:duration=first:dropout_transition=2,aresample=48000[aout]'
                )
                cmd += ['-filter_complex', fc, '-map', '0:v', '-map', '[aout]']
            else:
                cmd += ['-map', '0:v', '-map', '1:a']
            cmd += ['-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                    '-ar', '48000', '-ac', '2', '-t', str(duration), audio_output]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                raise Exception(f"Audio failed: {r.stderr.decode()[-200:]}")
            print(f"  ✅ Mixed")

            print(f"\n[STEP 12] OpenCV text render...")
            try:
                render_text_overlay_opencv(audio_output, decisions, beats, whisper_segments, self.output_path)
            except Exception as e:
                print(f"  ❌ Render failed: {e}")
                traceback.print_exc()
                subprocess.run(['ffmpeg', '-y', '-i', audio_output, '-c', 'copy', self.output_path],
                               check=True, capture_output=True)

            if os.path.exists(audio_output):
                os.remove(audio_output)

            print(f"\n[STEP 13] CTA...")
            cta_output = self.output_path.replace(".mp4", "_cta.mp4")
            self._add_cta_overlay(self.output_path, cta_output, duration)
            self.output_path = cta_output

            if not os.path.exists(self.output_path):
                raise Exception(f"Output missing: {self.output_path}")

            file_size  = os.path.getsize(self.output_path) / (1024 * 1024)
            total_time = time.time() - t0

            print(f"\n{'='*70}")
            print(f"✅ COMPLETE!")
            print(f"📁 {self.output_path}")
            print(f"💾 {file_size:.2f} MB | ⏱ {duration:.1f}s | ⚡ {total_time:.0f}s")
            print(f"🎭 {len(beats)} beats | 📝 {len(decisions)} decisions")
            print(f"{'='*70}\n")
            return True

        except Exception as e:
            print(f"\n❌ Pipeline error: {e}")
            traceback.print_exc()
            return False

        finally:
            print(f"\n🧹 Cleanup...")
            for tf in temp_files:
                if os.path.exists(tf):
                    try: os.remove(tf)
                    except: pass
            for f in [concat_list, concat_output, audio_output]:
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass
            for tmp in glob.glob("*TEMP_MPY*.mp4") + glob.glob("*_noaudio_tmp.mp4") + glob.glob("*_cfr_tmp.mp4") + glob.glob("temp_segment_*.mp4"):
                try: os.remove(tmp)
                except: pass


# ============================================================
# NICHE TEMPLATES
# ============================================================
NICHE_TEMPLATES = {
    'vaults': {
        'broll_dirs': {
            'space':   'space_vids',
            'ancient': 'ancient_ruins_vids',
            'cosmic':  'cosmic_vids',
            'sky':     'dark_sky_vids',
            'temple':  'temple_vids',
        },
        'keyword_map': {
            'space':   ['universe', 'galaxy', 'black hole', 'star', 'planet', 'cosmos'],
            'ancient': ['ancient', 'civilization', 'pyramid', 'ruins', 'lost', 'forgotten'],
            'cosmic':  ['time', 'reality', 'dimension', 'quantum', 'existence', 'consciousness'],
            'sky':     ['sky', 'atmosphere', 'above', 'beyond', 'vast', 'endless'],
            'temple':  ['religion', 'god', 'sacred', 'ritual', 'belief', 'worship'],
        }
    },
}


# ============================================================
# FASTAPI
# ============================================================
@app.get("/")
def root():
    return {"service": "Vaults of History v3", "status": "running",
            "openai_key": bool(OPENAI_API_KEY)}

@app.post("/generate")
async def generate_video_api(background_tasks: BackgroundTasks, niche: str = "vaults"):
    global current_job
    if current_job["status"] == "processing":
        return {"message": "Already processing", "status": "processing"}
    current_job = {"status": "processing", "progress": 0, "output": None,
                   "error": None, "started_at": datetime.now().isoformat(), "niche": niche}
    background_tasks.add_task(process_video, niche)
    return {"message": f"Started niche={niche}", "status": "processing"}

def process_video(niche: str = "vaults"):
    global current_job
    try:
        current_job["progress"] = 5
        audio_url   = "https://raw.githubusercontent.com/RandomSci/Automation_For_Love_Niche/main/Audio_Voice/vaults_narration.mp3"
        audio_file  = "Audio_Voice/vaults_narration.mp3"
        output_file = "vaults_output.mp4"
        trans_file  = f"{os.path.splitext(audio_file)[0]}_transcription.json"

        print(f"\n📥 Downloading audio...")
        os.makedirs("Audio_Voice", exist_ok=True)
        resp = requests.get(audio_url, timeout=30)
        if resp.status_code != 200:
            raise Exception(f"HTTP {resp.status_code}")
        with open(audio_file, "wb") as f:
            f.write(resp.content)
        print(f"  ✅ {len(resp.content)//1024}KB")

        current_job["progress"] = 10

        for old in [output_file, output_file.replace(".mp4", "_cta.mp4"),
                    "audio_mixed.mp4", trans_file]:
            if os.path.exists(old):
                os.remove(old)

        current_job["progress"] = 15
        niche_config = NICHE_TEMPLATES.get(niche, NICHE_TEMPLATES['vaults'])
        gen = VaultsGenerator(audio_path=audio_file, output_path=output_file,
                              niche_config=niche_config)

        current_job["progress"] = 20
        success = gen.create_vaults_video(bg_volume=0.12, fps=30)
        current_job["progress"] = 95

        final = output_file.replace(".mp4", "_cta.mp4")
        if success and os.path.exists(final):
            current_job.update({"status": "completed", "progress": 100, "output": final})
            print(f"\n🎉 DONE: {final}")
        else:
            raise Exception("Pipeline failed or output missing")

    except Exception as e:
        current_job.update({"status": "error", "error": str(e), "progress": 0})
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()

@app.get("/status")
def check_status():
    return {**current_job, "ready": current_job["status"] == "completed",
            "niche": current_job.get("niche", "vaults")}

@app.get("/download")
def download_video():
    if current_job["status"] != "completed":
        raise HTTPException(400, f"Not ready: {current_job['status']}")
    if not current_job["output"] or not os.path.exists(current_job["output"]):
        raise HTTPException(404, "File not found")
    return FileResponse(current_job["output"], media_type="video/mp4",
                        filename=f"vaults_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Vaults v3 on :{port} | Key: {'set' if OPENAI_API_KEY else 'MISSING'}")
    uvicorn.run(app, host="0.0.0.0", port=port)