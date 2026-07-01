import subprocess
import os
import json
import random
import math
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import requests
import traceback
import glob
import tempfile
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import cv2
from PIL import Image
import shutil

load_dotenv()

from openai import OpenAI
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Math Unlocked")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

current_job = {"status": "idle", "progress": 0, "output": None, "error": None, "started_at": None}

OUTPUT_WIDTH  = 1920
OUTPUT_HEIGHT = 1080

ENCODE_PRESET = os.environ.get("FINANCE_ENCODE_PRESET", "medium")
ENCODE_CRF    = os.environ.get("FINANCE_ENCODE_CRF", "15")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def safe_int(d, key, default):
    """Like dict.get(key, default), but ALSO falls back to default
    when the key is present with an explicit None/null value, not just
    when the key is absent. Plain .get(key, default) only applies its
    default for a MISSING key -- if a GPT call's JSON ever returns
    {"decimals": null} (a completely normal way to express "not
    applicable"), .get() happily returns None, and int(None) crashes.
    This was a real, repeated production bug across multiple element
    types in both validate_decisions and the renderer."""
    v = d.get(key, default)
    return int(v) if v is not None else int(default)


def safe_float(d, key, default):
    """Same fix as safe_int, for float fields."""
    v = d.get(key, default)
    return float(v) if v is not None else float(default)


def safe_set_default(d, key, default):
    """Like dict.setdefault(key, default), but ALSO overwrites an
    existing explicit None value with default -- plain setdefault only
    fills in a MISSING key, so {"size": None} survives setdefault
    untouched and still crashes downstream at the first int()/float()
    call on it."""
    if d.get(key) is None:
        d[key] = default
    return d[key]

import threading
import time as _time

_GPT4O_CONCURRENCY = threading.Semaphore(3)
_GPT4O_LOCK = threading.Lock()
_GPT4O_LAST_CALL_TS = [0.0]
_GPT4O_TPM_LIMIT = 30000
_GPT4O_REMAINING_TOKENS = [_GPT4O_TPM_LIMIT]
_GPT4O_FALLBACK_GAP_SECONDS = 1.5

def _adaptive_gap_seconds():
    """Scale the minimum gap between call starts based on the most
    recently observed remaining-token headroom. Plenty of headroom ->
    near-zero extra gap. Low headroom -> wider gap, up to a 4s ceiling
    so a single bad reading can't stall the pipeline indefinitely."""
    remaining = _GPT4O_REMAINING_TOKENS[0]
    if remaining is None:
        return _GPT4O_FALLBACK_GAP_SECONDS
    headroom_frac = max(0.0, min(1.0, remaining / _GPT4O_TPM_LIMIT))
    return 0.1 + (4.0 - 0.1) * (1.0 - headroom_frac)

def gpt4o_call(client, **kwargs):
    """Wrapper around client.chat.completions.create for gpt-4o that
    throttles across ALL callers (Call 1/2/3 batches alike) so they
    can't collectively exceed the shared TPM budget. Reads the real
    remaining-tokens header off each response to adapt the pacing."""
    with _GPT4O_CONCURRENCY:
        with _GPT4O_LOCK:
            gap = _adaptive_gap_seconds()
            wait = gap - (_time.time() - _GPT4O_LAST_CALL_TS[0])
            if wait > 0:
                _time.sleep(wait)
            _GPT4O_LAST_CALL_TS[0] = _time.time()

        raw = client.chat.completions.with_raw_response.create(**kwargs)
        try:
            remaining_hdr = raw.headers.get("x-ratelimit-remaining-tokens")
            if remaining_hdr is not None:
                with _GPT4O_LOCK:
                    _GPT4O_REMAINING_TOKENS[0] = int(remaining_hdr)
        except (TypeError, ValueError):
            pass
        return raw.parse()


def _call_with_retry(fn, label="gpt-4o call", max_retries=3):
    """Retry on 429 (rate limit) with exponential backoff. OpenAI's 429
    error includes a suggested wait time in its message; this is a
    simple fixed-backoff fallback since parsing that out reliably isn't
    worth the fragility. Re-raises on the final attempt so a genuinely
    persistent failure still surfaces instead of silently vanishing."""
    delay = 2.0
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if is_rate_limit and attempt < max_retries - 1:
                print(f"  ⏳ {label}: rate limited, retrying in {delay:.0f}s (attempt {attempt+1}/{max_retries})...")
                _time.sleep(delay)
                delay *= 2
                continue
            raise


def dynamic_batch_size(n_beats: int, min_size: int = 3, max_size: int = 10) -> int:
    """Scale batch size with script length so long scripts don't end up
    with dozens of tiny batches all queueing behind the same shared
    throttle. Short scripts keep the original small batch size (more
    batches, but there aren't many beats anyway, so total wait time is
    low regardless). Long scripts get larger batches -- fewer total
    requests, each one a bit bigger, which is a better trade once a
    script has enough beats that request COUNT (not request size) is
    what's actually slowing the pipeline down.
        <20 beats  -> 3   (matches original behavior)
        20-60 beats -> scales 3 to 6
        60+ beats   -> scales 6 to 10 (capped at max_size)
    """
    if n_beats <= 20:
        size = min_size
    elif n_beats <= 60:
        size = round(min_size + (6 - min_size) * (n_beats - 20) / 40)
    else:
        size = round(6 + (max_size - 6) * min(1.0, (n_beats - 60) / 90))
    return max(min_size, min(max_size, size))

if not OPENAI_API_KEY:
    print("⚠  WARNING: OPENAI_API_KEY not set.")

USE_PROCEDURAL_BACKGROUND = True

MUSIC_MAP = {
    "markets":   "bg_musics/finance_ambient.mp3",
    "growth":    "bg_musics/finance_ambient.mp3",
    "warning":   "bg_musics/dark_ambient.mp3",
    "history":   "bg_musics/finance_ambient.mp3",
    "default":   "bg_musics/finance_ambient.mp3",
}


SFX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sfx")

SFX_MAP = {
    "graph_whoosh":    ("alexzavesa-swoosh-1-463607.mp3",          -6),
    "card_pop":        ("universfield-new-notification-010-352755.mp3", -6),
    "solution_chime":  ("universfield-new-notification-013-363676.mp3", -8),
    "warning_ping":    ("universfield-new-notification-014-363678.mp3", -6),
    "problem_hit":     ("freesound_community-punch-boxing-02wav-14897.mp3", -8),
    "tiny_click":      ("dragon-studio-mouse-click-4-393911.mp3",   -8),
}

def _sfx_path(key):
    entry = SFX_MAP.get(key)
    if not entry:
        return None
    fname, _ = entry
    p = os.path.join(SFX_DIR, fname)
    return p if os.path.exists(p) else None

def _detect_sfx_for_chunk(code: str) -> str:
    if not code:
        return None
    c = code.lower()
    if "fm_animate_line_chart" in c or "fm_animate_line_chart_multi" in c:
        return "graph_whoosh"
    if "fm_animate_waterfall" in c or "fm_animate_comparison_bars" in c:
        if "brand_red" in c or "warning" in c or "danger" in c or "debt" in c:
            return "problem_hit"
        return "graph_whoosh"
    if "fm_animate_gauge" in c or "fm_animate_donut" in c:
        if "brand_red" in c or "warning" in c:
            return "warning_ping"
        return "solution_chime"
    if "fm_animate_bar_chart" in c:
        if "brand_red" in c:
            return "warning_ping"
        return "tiny_click"
    if "fm_animate_counter" in c or "fm_animate_single_value" in c:
        return "tiny_click"
    if "fm_card" in c or "fm_two_cards" in c or "fm_stacked_cards" in c or "fm_card_row" in c:
        if "brand_red" in c:
            return "problem_hit"
        return "card_pop"
    if "fm_animate_glow_reveal" in c or "fm_animate_text_reveal" in c:
        return "solution_chime"
    if "fm_concept_pills" in c or "fm_animate_timeline" in c:
        return "card_pop"
    return None

def _build_sfx_audio_inputs(clip_paths, chunk_code_list):
    return []
    sfx_events = []
    t = 0.0
    MIN_GAP = 6.0
    last_sfx_t = -MIN_GAP
    for i, clip in enumerate(clip_paths):
        if not clip or not os.path.exists(clip):
            t += 4.5
            continue
        try:
            r = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', clip],
                capture_output=True, text=True
            )
            dur = float(r.stdout.strip())
        except Exception:
            dur = 4.5
        item = chunk_code_list[i] if i < len(chunk_code_list) else None
        code = item.get("code", "") if isinstance(item, dict) else (item or "")
        sfx_key = _detect_sfx_for_chunk(code)
        sfx_file = _sfx_path(sfx_key) if sfx_key else None
        if sfx_file and (t - last_sfx_t) >= MIN_GAP:
            _, vol_db = SFX_MAP[sfx_key]
            sfx_events.append((t, sfx_file, vol_db))
            last_sfx_t = t
        t += dur
    return sfx_events

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

FONT_BLACK_CANDIDATES = [
    os.path.join(FONTS_DIR, "Anton-Regular.ttf"),
    os.path.join(FONTS_DIR, "Montserrat-Black.ttf"),
    os.path.join(FONTS_DIR, "Montserrat-ExtraBold.ttf"),
    os.path.join(FONTS_DIR, "Poppins-Bold.ttf"),
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
FONT_BOLD_CANDIDATES = [
    os.path.join(FONTS_DIR, "Montserrat-ExtraBold.ttf"),
    os.path.join(FONTS_DIR, "Montserrat-Bold.ttf"),
    os.path.join(FONTS_DIR, "Poppins-Bold.ttf"),
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
FONT_REGULAR_CANDIDATES = [
    os.path.join(FONTS_DIR, "Montserrat-Bold.ttf"),
    os.path.join(FONTS_DIR, "Poppins-Bold.ttf"),
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

_FUNCTIONS_MANIM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "functions_manim.py")
try:
    with open(_FUNCTIONS_MANIM_PATH) as _fmf:
        _FUNCTIONS_MANIM_CODE = _fmf.read()
    print(f"  ✅ functions_manim.py loaded ({len(_FUNCTIONS_MANIM_CODE)} chars)")
except FileNotFoundError:
    _FUNCTIONS_MANIM_CODE = ""
    print(f"  ⚠ functions_manim.py not found at {_FUNCTIONS_MANIM_PATH} -- fm_* library unavailable")

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
    print("🚀 Math Unlocked YouTube Content Generator Starting...")
    broll_dirs = ['space_vids','ancient_ruins_vids','cosmic_vids',
                  'dark_sky_vids','temple_vids']
    print("📁 Broll folder audit:")
    bad_clips = []
    for d in broll_dirs:
        if os.path.exists(d):
            files = [f for f in os.listdir(d) if f.lower().endswith(('.mp4','.mov','.avi'))]
            status = f"✅ {len(files)} clips" if files else "❌ EMPTY -- add Seedance clips here"
            print(f"  {d}: {status}")
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


def _bgr(r, g, b):
    """Convenience: define colors in RGB, return BGR for OpenCV."""
    return (b, g, r)


TOPIC_STYLES = {
    'finance': {
        'bg':      _bgr(7, 16, 24),
        'accent':  _bgr(138, 148, 166),
        'accent2': _bgr(56, 217, 150),
        'styles':  ['dashboard_grid', 'dashboard_ticker'],
    },
    'markets': {
        'bg':      _bgr(8, 10, 14),
        'accent':  _bgr(120, 230, 170),
        'accent2': _bgr(255, 215, 130),
        'styles':  ['particles', 'geometric'],
    },
    'growth': {
        'bg':      _bgr(6, 12, 14),
        'accent':  _bgr(120, 230, 170),
        'accent2': _bgr(160, 210, 255),
        'styles':  ['geometric', 'particles'],
    },
    'warning': {
        'bg':      _bgr(10, 6, 6),
        'accent':  _bgr(230, 90, 80),
        'accent2': _bgr(255, 200, 90),
        'styles':  ['particles', 'geometric'],
    },
    'history': {
        'bg':      _bgr(10, 10, 14),
        'accent':  _bgr(255, 215, 130),
        'accent2': _bgr(170, 175, 190),
        'styles':  ['geometric', 'particles'],
    },
    'default': {
        'bg':      _bgr(8, 9, 12),
        'accent':  _bgr(255, 255, 255),
        'accent2': _bgr(120, 230, 170),
        'styles':  ['particles', 'geometric'],
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
    speed = 4 + intensity * 1.2
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


def _draw_dashboard_grid(frame, t, intensity, color):
    """Subtle low-opacity grid lines, a couple of faint chart-axis
    lines, plus a slowly drifting line-chart silhouette -- evoking a
    financial dashboard background -- never bright or busy, just
    enough structure and life to feel 'finance', not decorative."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    spacing = max(40, int(min(w, h) * 0.07))
    grid_alpha = 0.05 + 0.005 * min(intensity, 6)
    for x in range(0, w, spacing):
        cv2.line(overlay, (x, 0), (x, h), color, 1, lineType=cv2.LINE_AA)
    for y in range(0, h, spacing):
        cv2.line(overlay, (0, y), (w, y), color, 1, lineType=cv2.LINE_AA)
    frame = cv2.addWeighted(overlay, grid_alpha, frame, 1 - grid_alpha, 0)

    axis_overlay = frame.copy()
    axis_y = int(h * 0.82)
    axis_x = int(w * 0.08)
    cv2.line(axis_overlay, (axis_x, int(h * 0.15)), (axis_x, axis_y), color, 2, lineType=cv2.LINE_AA)
    cv2.line(axis_overlay, (axis_x, axis_y), (int(w * 0.92), axis_y), color, 2, lineType=cv2.LINE_AA)
    axis_alpha = 0.07 + 0.004 * min(intensity, 6)
    frame = cv2.addWeighted(axis_overlay, axis_alpha, frame, 1 - axis_alpha, 0)

    chart_overlay = frame.copy()
    n_pts = 40
    pts = []
    scroll = t * 12
    for i in range(n_pts):
        px = w * (i / (n_pts - 1))
        wobble = (
            18 * math.sin((i * 0.4) + scroll * 0.05) +
            10 * math.sin((i * 0.9) + scroll * 0.09 + 1.7) +
            6 * math.sin((i * 1.7) + scroll * 0.14 + 3.1)
        )
        py = int(h * 0.62 + wobble)
        pts.append((int(px), py))
    pts_arr = np.array(pts, dtype=np.int32)
    cv2.polylines(chart_overlay, [pts_arr], False, color, 2, lineType=cv2.LINE_AA)
    fill_pts = pts + [(w, h), (0, h)]
    fill_arr = np.array(fill_pts, dtype=np.int32)
    cv2.fillPoly(chart_overlay, [fill_arr], color, lineType=cv2.LINE_AA)
    chart_alpha = 0.035 + 0.003 * min(intensity, 6)
    frame = cv2.addWeighted(chart_overlay, chart_alpha, frame, 1 - chart_alpha, 0)
    return frame


def _draw_dashboard_ticker(frame, t, intensity, color):
    """Small moving ticker-like numbers drifting slowly across the
    background, very low alpha -- pure texture, never legible content,
    never distracting from the foreground visual."""
    h, w = frame.shape[:2]
    rng_seed_rows = 5
    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for row in range(rng_seed_rows):
        y = int(h * (0.08 + row * 0.21))
        speed = 14 + row * 6
        direction = 1 if row % 2 == 0 else -1
        offset = (t * speed * direction) % (w + 400) - 200
        rng = random.Random(row * 97)
        x = offset
        while x < w + 100:
            val = rng.choice(["+0.4%", "-0.2%", "+1.1%", "-0.6%", "+2.3%", "-1.4%"])
            cv2.putText(overlay, val, (int(x), y), font, 0.55, color, 1, cv2.LINE_AA)
            x += rng.randint(220, 340)
    alpha = 0.045 + 0.003 * min(intensity, 6)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return frame


_BG_DRAW_FNS = {
    'starfield': lambda frame, t, intensity, color, sf: sf.draw(frame, t, intensity, color),
    'nebula':    lambda frame, t, intensity, color, sf: _draw_nebula(frame, t, intensity, color),
    'geometric': lambda frame, t, intensity, color, sf: _draw_geometric(frame, t, intensity, color),
    'aurora':    lambda frame, t, intensity, color, sf: _draw_aurora(frame, t, intensity, color),
    'particles': lambda frame, t, intensity, color, sf: _draw_particles(frame, t, intensity, color),
    'dashboard_grid':   lambda frame, t, intensity, color, sf: _draw_dashboard_grid(frame, t, intensity, color),
    'dashboard_ticker': lambda frame, t, intensity, color, sf: _draw_dashboard_ticker(frame, t, intensity, color),
}


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

    line_pts = [(0.18, 0.78), (0.36, 0.58), (0.50, 0.66), (0.82, 0.24)]
    arrow_tip = line_pts[-1]
    ang = math.atan2(line_pts[-1][1] - line_pts[-2][1], line_pts[-1][0] - line_pts[-2][0])
    head_len, head_w = 0.09, 0.05
    back_x = arrow_tip[0] - head_len * math.cos(ang)
    back_y = arrow_tip[1] - head_len * math.sin(ang)
    perp = ang + math.pi / 2
    left  = (back_x + head_w * math.cos(perp), back_y + head_w * math.sin(perp))
    right = (back_x - head_w * math.cos(perp), back_y - head_w * math.sin(perp))
    shapes['uptrend'] = [
        line_pts,
        [left, arrow_tip, right],
    ]

    dline_pts = [(0.18, 0.24), (0.36, 0.46), (0.50, 0.38), (0.82, 0.78)]
    dang = math.atan2(dline_pts[-1][1] - dline_pts[-2][1], dline_pts[-1][0] - dline_pts[-2][0])
    dback_x = dline_pts[-1][0] - head_len * math.cos(dang)
    dback_y = dline_pts[-1][1] - head_len * math.sin(dang)
    dperp = dang + math.pi / 2
    dleft  = (dback_x + head_w * math.cos(dperp), dback_y + head_w * math.sin(dperp))
    dright = (dback_x - head_w * math.cos(dperp), dback_y - head_w * math.sin(dperp))
    shapes['downtrend'] = [
        dline_pts,
        [dleft, dline_pts[-1], dright],
    ]

    shapes['coin_stack'] = [
        _ellipse_pts(0.5, 0.70, 0.22, 0.07, n=28),
        _ellipse_pts(0.5, 0.58, 0.22, 0.07, n=28),
        _ellipse_pts(0.5, 0.46, 0.22, 0.07, n=28),
        [(0.28, 0.46), (0.28, 0.70)],
        [(0.72, 0.46), (0.72, 0.70)],
    ]

    hour_ang = math.radians(-60)
    min_ang  = math.radians(60)
    shapes['clock'] = [
        _circle_pts(0.5, 0.5, 0.30, n=40),
        [(0.5, 0.5), (0.5 + 0.14 * math.cos(hour_ang), 0.5 + 0.14 * math.sin(hour_ang))],
        [(0.5, 0.5), (0.5 + 0.22 * math.cos(min_ang),  0.5 + 0.22 * math.sin(min_ang))],
    ]

    diag = [(0.24, 0.76), (0.76, 0.24)]
    shapes['percent'] = [
        _circle_pts(0.30, 0.30, 0.10, n=22),
        _circle_pts(0.70, 0.70, 0.10, n=22),
        diag,
    ]

    tilt = math.radians(-8)

    def _rot(px, py, cx, cy, a):
        dx, dy = px - cx, py - cy
        return (cx + dx * math.cos(a) - dy * math.sin(a),
                cy + dx * math.sin(a) + dy * math.cos(a))

    beam_l = _rot(0.22, 0.32, 0.5, 0.32, tilt)
    beam_r = _rot(0.78, 0.32, 0.5, 0.32, tilt)
    shapes['scale'] = [
        [(0.5, 0.20), (0.5, 0.82)],
        [(0.34, 0.82), (0.66, 0.82)],
        [beam_l, beam_r],
        _ellipse_pts(beam_l[0], beam_l[1] + 0.10, 0.09, 0.035, n=18),
        _ellipse_pts(beam_r[0], beam_r[1] + 0.10, 0.09, 0.035, n=18),
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

    rw, rh = width // 2, height // 2
    starfield = _Starfield(rw, rh, n_stars=220, seed=hash(topic) & 0xffff)

    raw_path = output_path.replace('.mp4', '_bg_raw.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))

    yv, xv = np.mgrid[0:rh, 0:rw].astype(np.float32)
    cx, cy = rw / 2, rh / 2
    dist = np.sqrt(((xv - cx) / (rw / 2)) ** 2 + ((yv - cy) / (rh / 2)) ** 2)
    vignette = np.clip(1.0 - 0.35 * np.clip(dist - 0.5, 0, 1), 0.55, 1.0)
    vignette3 = vignette[:, :, None]

    for f in range(n_frames):
        t = f / fps
        intensity = intensity_curve[f]

        frame = np.full((rh, rw, 3), bg_color, dtype=np.uint8)

        frame = _BG_DRAW_FNS[style_names[0]](frame, t, intensity, accent, starfield)
        if len(style_names) > 1:
            frame = _BG_DRAW_FNS[style_names[1]](frame, t, intensity * 0.7, accent2, starfield)

        subj, b_start, b_dur = subject_timeline[f]
        if subj in ILLUSTRATION_SHAPES:
            frame = _draw_illustration(frame, t, b_start, b_dur, subj, accent)

        frame = (frame.astype(np.float32) * vignette3).astype(np.uint8)

        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        writer.write(frame)

        if f % (fps * 5) == 0:
            print(f"    {f}/{n_frames} frames...", end='\r')

    writer.release()
    print(f"    {n_frames}/{n_frames} frames... done")

    r = subprocess.run([
        'ffmpeg', '-y', '-i', raw_path,
        '-c:v', 'libx264', '-preset', ENCODE_PRESET, '-crf', ENCODE_CRF,
        '-tune', 'animation', '-pix_fmt', 'yuv420p',
        '-r', str(fps), '-an', '-movflags', '+faststart', output_path
    ], capture_output=True)
    os.remove(raw_path)
    if r.returncode != 0:
        raise Exception(f"Background re-encode failed: {r.stderr.decode()[-200:]}")

    print(f"  ✅ Procedural background: {output_path}")
    return output_path


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

    The unbounded fallback search is a known hazard for short connector
    words ("a", "to", "of", "is") -- a loose substring check matches those
    against almost any nearby word, and an unbounded scan will happily grab
    a spurious match hundreds of words away, corrupting that beat's
    end_time by hundreds of seconds. Three guards against this: word
    matching requires an exact match unless both words are long enough
    (4+ chars) that a substring match is actually meaningful; a matched
    span is discarded if it is wildly larger than the beat's word count
    could plausibly justify; and a matched span is discarded if the last
    matched word lands at an earlier word-list position than the first
    matched word. That last case happens because the unbounded fallback
    deliberately restarts its search from the global pointer (not the
    beat-local pointer) to recover when an earlier word in the SAME beat
    over-advanced -- which can make a later word in the beat match a word
    that occurs earlier in the transcript than an already-matched word,
    producing a negative time span. A magnitude check alone does not catch
    this, since any negative number trivially satisfies span <= ceiling.
    """
    ptr = 0
    n = len(whisper_word_list)
    LOOKAHEAD = 20

    def norm(w):
        return w.upper().strip('.,!?;:\'"()[]- ')

    def matches(w, ww):
        if ww == w:
            return True
        if len(w) < 4 or len(ww) < 4:
            return False
        return w in ww or ww in w

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
                if matches(w, whisper_word_list[look]['word']):
                    found = look
                    break
            if found is None:
                for look in range(ptr, n):
                    if matches(w, whisper_word_list[look]['word']):
                        found = look
                        break
            if found is None:
                continue
            if start_idx is None:
                start_idx = found
            end_idx = found
            local_ptr = found + 1

        matched_ok = False
        if start_idx is not None and end_idx is not None and end_idx >= start_idx:
            span = whisper_word_list[end_idx]['end'] - whisper_word_list[start_idx]['start']
            plausible_ceiling = max(8.0, 4.0 * max(0.3, 0.35 * len(words)))
            if span <= plausible_ceiling:
                beat["start_time"] = whisper_word_list[start_idx]['start']
                beat["end_time"]   = whisper_word_list[end_idx]['end']
                ptr = end_idx + 1
                matched_ok = True
            else:
                print(f"    ⚠ Discarding implausible match for '{text[:40]}' "
                      f"({span:.1f}s span for {len(words)} words) -- estimating instead")
        elif start_idx is not None and end_idx is not None:
            print(f"    ⚠ Discarding non-monotonic match for '{text[:40]}' "
                  f"(later word matched an earlier position) -- estimating instead")

        if not matched_ok:
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


def _build_beats_batch_prompt(topic_hint: str, batch_lines: list, is_first_batch: bool) -> str:
    timed_transcript = "\n".join(batch_lines)
    topic_note = (
        "Also include \"topic\" and \"music_mood\" fields at the top level for this chunk -- "
        "they'll be taken from your response if this is the first chunk."
        if is_first_batch else
        "This is a LATER chunk of the same video -- you do not need to include \"topic\" or "
        "\"music_mood\" (only the first chunk's matter), just segment the beats."
    )
    return (
        f"Topic hint: {topic_hint}\n\n"
        f"Timed transcript chunk:\n{timed_transcript}\n\n"
        f"Segment every line in THIS CHUNK into beats. Use the timestamps shown. Copy text verbatim. "
        f"Extract data fields wherever a beat states a real number. {topic_note}"
    )


def analyze_story_beats(transcript_text: str, whisper_segments: list,
                        topic_hint: str, total_duration: float) -> dict:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not set.")

    print(f"  🎭 Call 1: Story beats ({len(transcript_text)} chars, {total_duration:.1f}s)...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    timed_lines = []
    for seg in whisper_segments:
        s = float(seg.get('start', 0))
        e = float(seg.get('end', 0))
        t = seg.get('text', '').strip()
        if t:
            timed_lines.append(f"[{s:.2f}s - {e:.2f}s] {t}")

    system_prompt = f"""You are the producer for a finance/numbers explainer channel. Style: clear, dynamic, data-forward -- think "explain this number visually" rather than dramatic horror-story captions. Audience wants to actually understand the number, not just feel a jump-scare.
Total audio duration: {total_duration:.1f} seconds.

You will receive a CHUNK of a transcript with EXACT timestamps from Whisper speech recognition.
Each line is formatted as: [start - end] spoken words

YOUR JOB: Segment this chunk's transcript into beats for visual data-explainer editing.

RULES:
- Use the Whisper timestamps directly -- they are accurate. Copy start_time and end_time from the brackets.
- Beat text MUST be copied VERBATIM from the transcript. Exact words, exact spelling. No paraphrasing.
- Keep beats 2-12 words -- natural spoken phrases or short clauses. Numbers/stats often need a slightly longer beat to land (e.g. "that's a four hundred percent increase").
- A single Whisper segment can become 1-3 beats if it contains multiple natural phrases.
- Cover the ENTIRE chunk -- every word must appear in some beat.
- "pause" beats only for clear silence gaps (>0.5s) between segments.

beat_type: "hook"|"setup"|"data_point"|"comparison"|"insight"|"warning"|"resolution"|"outro"
- "data_point": beat states a specific number/stat/dollar amount/percentage
- "comparison": beat contrasts two numbers or two things (X vs Y, before vs after)
- "warning": beat flags risk, loss, a downside, a mistake to avoid
- "insight": beat draws a conclusion or "here's what that means" takeaway

DATA EXTRACTION (critical -- this is what makes the visuals possible):
If a beat states an actual quantity, extract it into structured fields so the renderer can animate it precisely instead of guessing from prose:
- "has_data": true/false -- true only if this beat states a concrete number/stat/amount
- "data_value": the numeric value as a plain number (e.g. 400000, 4.5, 23). No currency symbols, no commas, no words.
- "data_unit": "percent"|"dollars"|"years"|"times"|"count"|"none" -- what the number represents
- "data_label": a SHORT (1-4 word) label for what the number IS, verbatim-ish from the beat (e.g. "AVERAGE RETURN", "COMPOUND INTEREST", "MARKET CAP")
- "data_direction": "up"|"down"|"neutral" -- only relevant for comparison/trend beats (does the number represent growth, loss, or neither)
- "compare_value": for "comparison" beats only -- the second number being compared against (numeric, same rules as data_value), else null

VISUAL_SUBJECT (icon drawing system): if this beat CLEARLY evokes one of these
concepts, set visual_subject to it -- the renderer draws it as line-art.
Options: "none"|"uptrend"|"downtrend"|"coin_stack"|"clock"|"percent"|"scale".
Be CONSERVATIVE -- most beats should be "none". Only set when the beat is
genuinely about that concept (e.g. "uptrend" for beats about growth/gains,
"downtrend" for losses/declines, "coin_stack" for savings/wealth/money itself,
"clock" for time-value-of-money/waiting/compounding over time, "percent" for
beats centrally about a rate/percentage, "scale" for risk/reward tradeoffs or
weighing two options). Never force a match.

Return ONLY valid JSON:
{{
  "topic": "markets|growth|warning|history|default",
  "music_mood": "driving|tense|optimistic|neutral|serious",
  "beats": [
    {{
      "beat_type": "hook|setup|data_point|comparison|insight|warning|resolution|outro",
      "text": "verbatim words from transcript",
      "start_time": 0.0,
      "end_time": 2.5,
      "intensity": 8,
      "has_data": true,
      "data_value": 400000,
      "data_unit": "dollars",
      "data_label": "RETIREMENT SAVINGS",
      "data_direction": "up",
      "compare_value": null,
      "visual_subject": "none|uptrend|downtrend|coin_stack|clock|percent|scale"
    }}
  ]
}}"""

    SEGMENTS_PER_BATCH = dynamic_batch_size(len(timed_lines), min_size=15, max_size=40)
    batches = [timed_lines[i:i+SEGMENTS_PER_BATCH] for i in range(0, len(timed_lines), SEGMENTS_PER_BATCH)]
    print(f"  🎭 Call 1: {len(batches)} chunk(s) of ~{SEGMENTS_PER_BATCH} segments each...")

    def _run_batch(batch_idx, batch_lines):
        print(f"  🎭 Call 1 chunk {batch_idx+1}/{len(batches)}: {len(batch_lines)} segments...")
        response = _call_with_retry(lambda: gpt4o_call(client,
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _build_beats_batch_prompt(topic_hint, batch_lines, batch_idx == 0)}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=8000,
            timeout=90,
        ), label=f"Call 1 chunk {batch_idx+1}")
        result = json.loads(response.choices[0].message.content)
        print(f"  ✅ Call 1 chunk {batch_idx+1} done: {len(result.get('beats', []))} beats")
        return batch_idx, result

    results = [None] * len(batches)
    MAX_WORKERS = 3
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_run_batch, i, b): i for i, b in enumerate(batches)}
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                idx, result = future.result()
                results[idx] = result
            except Exception as e:
                print(f"  ❌ Call 1 chunk {batch_idx+1} failed: {e}")
                raise

    all_beats = []
    for r in results:
        all_beats.extend(r.get('beats', []))

    first_result = results[0] if results else {}
    detected_tone = first_result.get("topic", "default")
    final_result = {
        "topic": "finance",
        "detected_tone": detected_tone,
        "music_mood": first_result.get("music_mood", "neutral"),
        "beats": all_beats,
    }
    print(f"  ✅ {len(all_beats)} beats total, topic=finance (detected tone: {detected_tone})")
    return final_result


def extract_section_outline(transcript_text: str, whisper_segments: list,
                             total_duration: float) -> list:
    if not OPENAI_API_KEY:
        return []

    print(f"  🗂  Section outline scan...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    timed_lines = []
    for seg in whisper_segments:
        s = float(seg.get('start', 0))
        e = float(seg.get('end', 0))
        t = seg.get('text', '').strip()
        if t:
            timed_lines.append(f"[{s:.2f}s - {e:.2f}s] {t}")
    timed_transcript = "\n".join(timed_lines)

    system_prompt = f"""You are analyzing a finance/numbers explainer video's transcript to find its NAVIGATIONAL STRUCTURE, not its content.
Total audio duration: {total_duration:.1f} seconds.

YOUR JOB: Decide whether this script walks through a clear ENUMERABLE LIST of distinct items -- warning signs, reasons, steps, mistakes, rules, red flags, ways something happens -- where a viewer would benefit from an on-screen "item 3 of 9" indicator because the list is long enough to lose track of.

Only identify a list if there are genuinely 3 or more distinct enumerable items with clear boundaries in the transcript. A script that is one continuous explanation, a single concept walkthrough, a story, or a before/after comparison with no list does NOT qualify -- return an empty sections array for those. That is the common case, not a rare one.

If a list DOES exist, for each item give:
- "number": 1-indexed position in the list
- "title": a SHORT (2-5 word) label naming that specific item, written for an on-screen tag, not a sentence (e.g. "Income Under $200/mo", "18-Month Lifespan", "Taxes Eat the Income")
- "start_time": the timestamp (from the bracketed transcript) where THIS item's discussion begins
- "end_time": the timestamp where THIS item's discussion ends (the next item's start_time, or the total duration for the last item)

Use the exact timestamps shown in the transcript brackets, do not estimate. Cover the list items in order, do not skip numbers, do not invent items that are not actually distinct beats in the transcript.

Return ONLY valid JSON:
{{
  "has_list_structure": true,
  "sections": [
    {{"number": 1, "title": "Income Under $200/mo", "start_time": 12.40, "end_time": 38.10}}
  ]
}}
If there is no list structure, return {{"has_list_structure": false, "sections": []}}."""

    try:
        response = _call_with_retry(lambda: gpt4o_call(client,
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Timed transcript:\n{timed_transcript}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2000,
            timeout=60,
        ), label="Section outline")
        result = json.loads(response.choices[0].message.content)
        sections = result.get("sections", []) if result.get("has_list_structure") else []
    except Exception as e:
        print(f"  ⚠ Section outline scan failed, skipping navigation overlay: {e}")
        return []

    cleaned = []
    for sec in sections:
        try:
            num   = int(sec["number"])
            title = str(sec["title"]).strip()
            start = max(float(sec["start_time"]), 0.0)
            end   = min(float(sec["end_time"]), total_duration)
            if title and end > start:
                cleaned.append({"number": num, "title": title, "start_time": start, "end_time": end})
        except (KeyError, ValueError, TypeError):
            continue
    cleaned.sort(key=lambda s: s["start_time"])

    if len(cleaned) < 2:
        print(f"  🗂  No list structure detected, skipping navigation overlay")
        return []

    print(f"  ✅ {len(cleaned)}-item list structure detected")
    return cleaned


def _ffmpeg_escape_text(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "'\\''")
    text = text.replace(":", "\\:")
    text = text.replace("%", "\\%")
    text = text.replace(",", "")
    return text


def _build_batch_prompt(topic: str, batch: list) -> str:
    """Build the GPT Call 2 user prompt, annotating each beat with its real duration
    so GPT can set start_offset values that actually fit within the beat window."""
    annotated = []
    for b in batch:
        dur = round(float(b.get("end_time", 0)) - float(b.get("start_time", 0)), 2)
        entry = dict(b)
        entry["_duration_seconds"] = dur
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
        f"Compose each scene to make the number/concept understandable. Vary layouts. White dominant; use number_counter for every real value."
    )

def generate_render_decisions(beats: list, topic: str) -> list:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not set.")

    print(f"  🎨 Call 2: Scene compositions for {len(beats)} beats...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = f"""You are an elite short-form video editor for a finance/numbers explainer channel. You compose every frame like a motion designer -- choosing position, size, color, animation, and timing for each visual element. You are not picking from preset templates. You are designing each scene to make a NUMBER or CONCEPT visually understandable, not just dramatic.

Channel: a finance/numbers explainer. Audience wants to actually grasp the number -- a stat, a comparison, a growth curve, a cost. The aesthetic is CLEAN and DATA-FORWARD -- like a sharp explainer video, not a horror-trailer caption stack. Still punchy and fast-paced, just clearer.

=== FONT BEHAVIOR ===
The renderer uses Anton (ultra-condensed) as the primary font. This font is TALL and NARROW. All text is automatically rendered in ALL CAPS -- so write content in ALL CAPS.
SIZE RULES (strictly enforced by renderer):
- Single impact word or number: 120-160px max. Centered or slightly off-center.
- Sentence words (2+ words in a beat): 70-110px each. Cascade across canvas.
- number_counter elements: 140-220px (numbers need to be the visual anchor of a data beat).
- DO NOT go above 220px for any element -- it will be clamped.
- Fewer elements per scene is better. 2-4 elements max. Dense scenes are unreadable.

=== YOUR RENDERING ENGINE ===
Python OpenCV + Pillow on a {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} canvas.

For each beat, you output a SCENE -- a list of ELEMENTS placed and animated however you want. Each beat in the input includes data fields (has_data, data_value, data_unit, data_label, data_direction, compare_value) extracted from the transcript. USE THESE FIELDS when has_data is true -- they're the actual numbers you should visualize, not something to re-derive from the text.

=== ELEMENT TYPES ===

TEXT element:
{{
  "type": "text",
  "content": "WORD",
  "x": 0.5, "y": 0.4,
  "anchor": "center",              // "center" | "left" | "right"
  "size": 120,
  "color": "#FFFFFF",
  "weight": "black",               // "regular" | "bold" | "black"
  "outline": 4,
  "anim": "fade_in",
  "start_offset": 0.0,
  "duration": null,
  "anim_duration": 0.15,
  "effect": "none"
}}

NUMBER_COUNTER element (NEW -- use this for any beat with has_data=true):
{{
  "type": "number_counter",
  "target_value": 400000,          // copy from the beat's data_value
  "prefix": "$",                   // "$" for dollars, "" otherwise
  "suffix": "%",                   // "%" for percent, "" otherwise -- never put both prefix and suffix unless the unit genuinely needs it
  "decimals": 0,                   // 0 for whole numbers, 1-2 for precise stats
  "x": 0.5, "y": 0.42,
  "anchor": "center",
  "size": 180,
  "color": "#FFFFFF",
  "weight": "black",
  "outline": 5,
  "count_from": 0,                 // where the count-up animation starts (usually 0, or a lower number for a "before" value)
  "count_duration": 0.8,           // seconds for the number to animate from count_from to target_value
  "start_offset": 0.0,
  "duration": null
}}
The renderer animates this counting UP (or down) from count_from to target_value over count_duration seconds, formatted with prefix/suffix/decimals/comma separators automatically. This is your primary tool for making a statistic feel alive instead of just appearing.

GRID element (NEW -- use for beats about scale/quantity/repetition, e.g. "thousands of dollars" or visualizing a large count):
{{
  "type": "grid",
  "glyph": "0",                    // the single character or short string repeated in the grid
  "rows": 4,
  "cols": 14,
  "cell_size": 60,                 // pixel size of each glyph
  "color": "#FBC02D",
  "x": 0.5, "y": 0.55,             // CENTER of the whole grid
  "anim": "fill_sequential",        // "fill_sequential" reveals cell by cell, "fade_in" reveals all at once
  "fill_duration": 1.2,             // total seconds for fill_sequential to complete
  "start_offset": 0.0,
  "duration": null
}}
Use this sparingly -- it's for the rare beat where "a LOT of something" is the point (e.g. visualizing thousands as a wall of repeated digits/symbols). Keep rows*cols under 80 total cells or it gets visually noisy.

LINE element (for dividers, underlines, comparison axes):
{{
  "type": "line",
  "x1": 0.3, "y1": 0.5, "x2": 0.7, "y2": 0.5,
  "thickness": 8,
  "color": "#FFFFFF",
  "anim": "draw_horizontal",
  "start_offset": 0.2,
  "duration": null,
  "anim_duration": 0.3
}}

RECT element (for boxes, comparison bars, highlight bars):
{{
  "type": "rect",
  "x": 0.4, "y": 0.5, "w": 0.2, "h": 0.1,
  "color": "#FBC02D",
  "filled": true,
  "thickness": 4,
  "anim": "fade_in",
  "start_offset": 0.0,
  "duration": null
}}
For a COMPARISON beat (data_direction or compare_value set): two RECT bars side by side, heights proportional to the two values (taller bar = bigger number), is a strong visual. Pair with a TEXT label under each.

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
- "slide_in_left" / "slide_in_right" / "slide_in_top" / "slide_in_bottom"
- "scale_in": starts at 1.3x scale and snaps to 1.0x (punch effect)
- "snap": appears instantly with a 1-frame white flash
- "draw_horizontal": (lines only) draws progressively left-to-right
- "fill_sequential": (grid only) reveals cells one at a time

=== EFFECTS (applied during display, not just entrance) ===
- "none": static
- "flicker": rapid on/off blinking for first 0.3s (for warning/shock numbers)
- "shake": position jitters slightly (for impact)
- "glow": adds soft colored glow halo around element

=== HOW TO COMPOSE SCENES ===

ELEMENT LIMIT: Maximum 4 elements per scene. Less is more. 2-3 elements is ideal.

STAGGER ALL ELEMENTS: start_offset must be less than the beat's _duration_seconds or the element will NEVER appear.
- Beat <0.5s: 1 element only, start_offset 0.0
- Beat 0.5-1.0s: max 2 elements, offsets 0.0 and 0.3
- Beat >1.0s: up to 3 elements, offsets 0.0 / 0.35 / 0.7
NEVER set start_offset >= _duration_seconds.

=== POSITIONING GRID (1920x1080 canvas) ===
Safe zone: x: 0.08-0.92, y: 0.12-0.88.

Three vertical bands:
- UPPER band:  y: 0.20-0.35
- CENTER band: y: 0.42-0.58
- LOWER band:  y: 0.65-0.80

=== BEAT-TYPE -> COMPOSITION MAPPING ===

For a "data_point" beat (has_data=true, no compare_value): ONE number_counter element in CENTER band showing the actual value (use prefix/suffix from data_unit), plus ONE text element in LOWER band with the data_label. 2 elements. This is your bread-and-butter scene type.

For a "comparison" beat (has_data=true AND compare_value set): two RECT bars side by side (proportional heights, taller = larger value) OR two number_counter elements side by side (x: 0.28 and x: 0.72), each with a short text label beneath. 3-4 elements.

For a "warning" beat: text in CENTER band, color #E85D4A or similar warning-red (still pass through _ensure_bright_color), "shake" or "flicker" effect. 1-2 elements.

For an "insight"/"resolution" beat (the takeaway, usually no raw number): SPOKEN SENTENCE treatment -- pick the 1-2 most important words, place across UPPER/CENTER bands, 90-130px, fade_in or slide_in.

For a "hook" or "setup" beat: 1-2 elements, CENTER or UPPER band, sets up the number that's about to land -- don't put the actual data_value here unless the beat itself states it.

For a beat with visual_subject set (uptrend/downtrend/coin_stack/clock/percent/scale): the renderer draws that icon automatically in the background -- you do NOT need to add an element for it. Just compose the text/number elements as normal; they'll appear on top of the icon.

VARY bands across consecutive beats so the video doesn't feel static.

=== HARD RULES ===
1. Output exactly {len(beats)} scenes, one per beat, in order.
2. Every "content" in TEXT elements must be ALL CAPS and use words VERBATIM from the beat text.
3. For pause beats: output {{"elements": []}} (empty scene).
4. start_offset values must fit within the beat duration. STAGGER them -- never all 0.0.
5. x, y values are 0.0-1.0. NEVER use percentages or pixels.
6. MAX 4 elements per scene.
7. Never repeat the same word twice in one scene.
8. Content must be a SINGLE WORD or SHORT PHRASE -- never a full sentence in one element.
9. When has_data is true, ALWAYS use a number_counter element for the actual value -- never spell the number out as a TEXT word (e.g. use number_counter with target_value=400000, not a text element saying "FOUR HUNDRED THOUSAND").

=== COLOR DISCIPLINE ===
White (#FFFFFF) is your dominant color. Market green (#78E6AA) for positive/growth numbers. Warning red (#E85D4A) for losses/risk. Gold (#FFD782) for ONE key highlight per scene maximum.

Return ONLY valid JSON:
{{
  "scenes": [
    {{
      "beat_index": <int>,
      "beat_type": "<hook|setup|data_point|comparison|insight|warning|resolution|outro>",
      "elements": [
        // list of element objects as specified above
      ]
    }}
    // ... exactly {len(beats)} scenes
  ]
}}"""

    BATCH_SIZE = dynamic_batch_size(len(beats))
    all_scenes = []
    batches = [beats[i:i+BATCH_SIZE] for i in range(0, len(beats), BATCH_SIZE)]

    def _run_batch(batch_idx, batch):
        start_beat = batch_idx * BATCH_SIZE
        print(f"  🎨 Batch {batch_idx+1}/{len(batches)}: beats {start_beat}-{start_beat+len(batch)-1}...")
        response = _call_with_retry(lambda: gpt4o_call(client,
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _build_batch_prompt(topic, batch)}
            ],
            response_format={"type": "json_object"},
            temperature=0.85,
            max_tokens=8000,
            timeout=120,
        ), label=f"Call 2 batch {batch_idx+1}")
        result = json.loads(response.choices[0].message.content)
        batch_scenes = result.get('scenes', [])
        if len(batch_scenes) > len(batch):
            print(f"  ⚠️  Batch {batch_idx+1}: expected {len(batch)} scenes, got {len(batch_scenes)} -- trimming extras")
            batch_scenes = batch_scenes[:len(batch)]
        elif len(batch_scenes) < len(batch):
            print(f"  ⚠️  Batch {batch_idx+1}: expected {len(batch)} scenes, got {len(batch_scenes)} -- padding with empty scenes")
            while len(batch_scenes) < len(batch):
                batch_scenes.append({"beat_index": start_beat + len(batch_scenes), "elements": []})
        print(f"  ✅ Batch {batch_idx+1} done: {len(batch_scenes)} scenes")
        return batch_idx, batch_scenes

    results = [None] * len(batches)
    MAX_WORKERS = 3
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


import ast
import multiprocessing
import traceback as _traceback

VISUAL_CODE_TIMEOUT_SECONDS = 8

_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "round": round, "len": len,
    "range": range, "enumerate": enumerate, "zip": zip, "sum": sum,
    "int": int, "float": float, "str": str, "bool": bool, "list": list,
    "tuple": tuple, "dict": dict, "sorted": sorted, "reversed": reversed,
    "map": map, "filter": filter, "all": all, "any": any,
}

_FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "import",
    "os", "sys", "subprocess", "socket", "requests", "shutil",
    "globals", "locals", "vars", "input", "breakpoint", "exit", "quit",
}


def _static_safety_check(code: str) -> tuple[bool, str]:
    """Parse the generated code and reject anything that references a
    forbidden name, imports a module, or otherwise tries to step
    outside pure drawing logic -- BEFORE it ever executes.

    Also verifies draw_beat structurally via the AST, not a substring
    match on the raw text. A substring check like `"def draw_beat(" in
    code` passes even when the string ALSO contains top-level
    executable statements outside any function (e.g. stray code before
    or after the real def, or a malformed response with two defs) --
    those run immediately on exec() and crash with errors like "name
    'draw' is not defined" since they're not inside the function scope
    that actually receives that argument. Real failure seen in
    production: GPT's JSON response contained extra top-level
    statements alongside `def draw_beat`, which `"def draw_beat(" in
    code` happily let through."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "contains an import statement"
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            return False, f"references forbidden name '{node.id}'"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"references dunder attribute '{node.attr}'"

    top_level_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "draw_beat"]
    if not top_level_defs:
        return False, "missing required top-level draw_beat(...) function definition"
    if len(tree.body) > 1:
        other_kinds = sorted({type(n).__name__ for n in tree.body
                              if not (isinstance(n, ast.FunctionDef) and n.name == "draw_beat")})
        return False, f"unexpected top-level statement(s) outside draw_beat: {', '.join(other_kinds)}"

    return True, ""


import hashlib

_PRERENDER_CACHE_DIR = os.path.join(tempfile.gettempdir(), "math_unlocked_beat_cache")
INTERNAL_VISUAL_FPS = 15


def _beat_cache_key(code: str, duration: float, fps: int, w: int, h: int) -> str:
    raw = f"{code}|{duration:.3f}|{fps}|{w}|{h}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


_EMOJI_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto-color-emoji/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",
]
_EMOJI_NATIVE_STRIKE_SIZE = 109

ICONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icons")
KNOWN_ICONS = {
    "trending-up", "trending-down", "chart-line", "chart-bar", "chart-pie",
    "cash-banknote", "coins", "credit-card", "receipt", "invoice",
    "building-bank", "percentage", "gauge", "calendar", "clock", "scale",
    "alert-triangle", "moneybag", "home", "car", "stethoscope", "stairs", "wallet",
}


def _load_emoji_font():
    """Returns a loaded ImageFont, or None if no usable emoji font is
    found on this machine -- callers must treat None as 'just skip
    drawing the emoji', never as a hard failure."""
    from PIL import ImageFont as _ImageFont
    for path in _EMOJI_FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            return _ImageFont.truetype(path, _EMOJI_NATIVE_STRIKE_SIZE)
        except Exception:
            continue
    return None


def _render_beat_frames_worker(code: str, duration: float, fps: int, w: int, h: int, cache_path: str):
    """Runs inside ONE pool worker process for ONE beat. Execs the
    generated code once, then renders every frame this beat needs at
    the reduced internal fps in a tight loop, writes the whole stack
    to a single .npz file, and returns the cache path (or None on
    total failure) plus an error string (empty on success).

    Per-frame failures inside the loop don't abort the whole beat --
    a single bad frame is written as fully transparent and the loop
    continues, since draw_beat might behave correctly at most t values
    and only misbehave at one edge case (e.g. t=0 specifically)."""
    try:
        import numpy as _np
        import math as _math
        from PIL import Image as _Image, ImageDraw as _ImageDraw, ImageFont as _ImageFont

        _emoji_font = _load_emoji_font()
        _emoji_cache_local = {}

        def _make_draw_emoji(get_layer):
            def draw_emoji(emoji_char, cx, cy, size):
                if _emoji_font is None:
                    return
                try:
                    cache_key = (emoji_char, int(size))
                    glyph = _emoji_cache_local.get(cache_key)
                    if glyph is None:
                        tmp = _Image.new("RGBA", (160, 160), (0, 0, 0, 0))
                        _ImageDraw.Draw(tmp).text((20, 20), emoji_char, font=_emoji_font, embedded_color=True)
                        bbox = tmp.getbbox()
                        cropped = tmp.crop(bbox) if bbox else tmp
                        glyph = cropped.resize((max(1, int(size)), max(1, int(size))), _Image.LANCZOS)
                        _emoji_cache_local[cache_key] = glyph
                    layer = get_layer()
                    gw, gh = glyph.size
                    layer.alpha_composite(glyph, (int(cx - gw / 2), int(cy - gh / 2)))
                except Exception:
                    pass
            return draw_emoji

        _current_layer_box = [None]
        draw_emoji = _make_draw_emoji(lambda: _current_layer_box[0])

        _font_cache_local = {}

        def get_font(size, style="bold"):
            size = max(1, int(size))
            style = style if style in ("bold", "black") else "bold"
            cache_key = (size, style)
            font = _font_cache_local.get(cache_key)
            if font is None:
                try:
                    if style == "black":
                        font_path = FONT_BLACK or FONT_BOLD
                    else:
                        font_path = FONT_BOLD or FONT_BLACK
                    font = _ImageFont.truetype(font_path, size) if font_path else _ImageFont.load_default()
                except Exception:
                    font = _ImageFont.load_default()
                _font_cache_local[cache_key] = font
            return font

        def text_size(text, font):
            try:
                tmp = _Image.new("RGBA", (10, 10))
                bbox = _ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
                return (bbox[2] - bbox[0], bbox[3] - bbox[1])
            except Exception:
                return (len(text) * 10, 14)

        def safe_box(cx, cy, half_w, half_h):
            """Returns a guaranteed-valid [x0, y0, x1, y1] box for
            draw.ellipse/draw.rectangle, given a center point and
            half-width/half-height. ALWAYS produces x1 >= x0 and
            y1 >= y0, even if half_w/half_h come in as zero, negative,
            or a float that rounds awkwardly -- this makes the
            'y1 must be greater than or equal to y0' crash structurally
            impossible to produce via this helper, since the absolute
            value is taken and a minimum of 1 pixel is enforced before
            the corners are ever computed. Use this INSTEAD OF manually
            computing [cx-r, cy-r, cx+r, cy+r] yourself, especially for
            any shape whose size shrinks toward zero as t changes."""
            cx = float(cx)
            cy = float(cy)
            half_w = max(1.0, abs(float(half_w)))
            half_h = max(1.0, abs(float(half_h)))
            x0 = int(cx - half_w)
            y0 = int(cy - half_h)
            x1 = int(cx + half_w)
            y1 = int(cy + half_h)
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            if x1 == x0:
                x1 = x0 + 1
            if y1 == y0:
                y1 = y0 + 1
            return [x0, y0, x1, y1]

        _icon_cache_local = {}

        def _load_and_tint_icon(name, size, color):
            cache_key = (name, int(size), tuple(color[:3]))
            cached = _icon_cache_local.get(cache_key)
            if cached is not None:
                return cached
            path = os.path.join(ICONS_DIR, f"{name}.png")
            if not os.path.exists(path):
                return None
            try:
                src = _Image.open(path).convert("RGBA")
                bbox = src.getbbox()
                if bbox:
                    src = src.crop(bbox)
                r, g, b = color[0], color[1], color[2]
                alpha = src.split()[3]
                tinted = _Image.new("RGBA", src.size, (r, g, b, 0))
                tinted.putalpha(alpha)
                side = max(1, int(size))
                sw, sh = src.size
                if sw >= sh:
                    new_w, new_h = side, max(1, int(side * sh / sw))
                else:
                    new_h, new_w = side, max(1, int(side * sw / sh))
                resized = tinted.resize((new_w, new_h), _Image.LANCZOS)
                _icon_cache_local[cache_key] = resized
                return resized
            except Exception:
                return None

        def _make_draw_icon(get_layer):
            def draw_icon(name, cx, cy, size, color=(245, 247, 250, 255)):
                """Draws one of the bundled finance icons (Tabler Icons,
                MIT licensed), tinted to the given RGB(A) color, centered
                at (cx, cy), sized to fit within a `size`x`size` box
                (aspect ratio preserved). Available names: trending-up,
                trending-down, chart-line, chart-bar, chart-pie,
                cash-banknote, coins, credit-card, receipt, invoice,
                building-bank, percentage, gauge, calendar, clock, scale,
                alert-triangle, moneybag, home, car, stethoscope, stairs,
                wallet. If the name isn't recognized or the file is
                missing, this silently does nothing -- never crashes the
                beat. Use this instead of trying to hand-draw a finance
                icon glyph yourself from primitive shapes."""
                glyph = _load_and_tint_icon(name, size, color)
                if glyph is None:
                    return
                layer = get_layer()
                gw, gh = glyph.size
                layer.alpha_composite(glyph, (int(cx - gw / 2), int(cy - gh / 2)))
            return draw_icon

        draw_icon = _make_draw_icon(lambda: _current_layer_box[0])

        namespace = {"__builtins__": _SAFE_BUILTINS, "draw_emoji": draw_emoji,
                     "get_font": get_font, "text_size": text_size, "safe_box": safe_box,
                     "draw_icon": draw_icon}
        exec(code, namespace)
        draw_beat_fn = namespace.get("draw_beat")
        if draw_beat_fn is None:
            return None, "draw_beat not found after exec"

        n_frames = max(1, int(math.ceil(duration * fps)))
        frames = _np.zeros((n_frames, h, w, 4), dtype="uint8")
        any_ok = False
        first_error = None

        for i in range(n_frames):
            t = i / fps
            try:
                layer = _Image.new("RGBA", (w, h), (0, 0, 0, 0))
                _current_layer_box[0] = layer
                draw = _ImageDraw.Draw(layer)
                draw_beat_fn(draw, t, w, h, _np, _math)
                arr = _np.array(layer)
                if arr.shape != (h, w, 4) or not _np.isfinite(arr.astype(_np.float32)).all():
                    if first_error is None:
                        first_error = f"frame {i} (t={t:.2f}) produced invalid pixel data (wrong shape or non-finite values)"
                    continue
                frames[i] = arr.astype("uint8")
                any_ok = True
            except Exception as e:
                if first_error is None:
                    first_error = f"frame {i} (t={t:.2f}): {e}"
                continue

        if not any_ok:
            return None, f"every frame in this beat failed to render -- first failure: {first_error}"

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        _np.savez_compressed(cache_path, frames=frames)
        return cache_path, ""
    except Exception as e:
        return None, f"{e}\n{_traceback.format_exc()[-300:]}"


def prerender_all_beat_visuals(visual_code_timeline: list, w: int, h: int,
                                 fps: int = INTERNAL_VISUAL_FPS) -> dict:
    """Renders every beat's full generated-code frame sequence UP FRONT,
    in parallel across a process pool, before the main video loop
    starts. Returns {beat_index: numpy array of shape (n_frames, h, w, 4)}
    -- beats that fail entirely (timeout, crash, or pre-existing static
    safety rejection) are simply absent from the dict, and the caller
    renders those beats blank.

    Disk-cached by content hash -- re-running the same beat code at the
    same duration/resolution skips straight to loading the cached
    frames instead of re-rendering."""
    if not visual_code_timeline:
        return {}

    print(f"  🎬 Pre-rendering {len(visual_code_timeline)} beats' visual code "
          f"(parallel pool, {INTERNAL_VISUAL_FPS}fps internal)...")
    os.makedirs(_PRERENDER_CACHE_DIR, exist_ok=True)

    tasks = []
    for item in visual_code_timeline:
        duration = max(0.05, item["end"] - item["start"])
        key = _beat_cache_key(item["code"], duration, fps, w, h)
        cache_path = os.path.join(_PRERENDER_CACHE_DIR, f"{key}.npz")
        tasks.append((item["beat_index"], cache_path, not os.path.exists(cache_path),
                      item["code"], duration))

    results = {}
    to_render = [t for t in tasks if t[2]]
    cached = [t for t in tasks if not t[2]]
    loaded_from_cache = 0

    for beat_index, cache_path, _, _, _ in cached:
        try:
            with np.load(cache_path) as data:
                results[beat_index] = data["frames"]
            loaded_from_cache += 1
        except Exception as e:
            print(f"  ⚠ Beat {beat_index}: cached frames unreadable ({e}), will re-render")
            to_render.append(next(t for t in tasks if t[0] == beat_index))

    if loaded_from_cache:
        print(f"  💾 {loaded_from_cache} beat(s) loaded from cache")

    if to_render:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
        print(f"  ⚙️  Rendering {len(to_render)} beat(s) across {n_workers} parallel workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_render_beat_frames_worker, code, duration, fps, w, h, cache_path):
                    (beat_index, cache_path)
                for beat_index, cache_path, _, code, duration in to_render
            }
            done_count = 0
            for future in as_completed(futures):
                beat_index, cache_path = futures[future]
                done_count += 1
                try:
                    result_path, err = future.result(timeout=VISUAL_CODE_TIMEOUT_SECONDS * 30)
                except Exception as e:
                    result_path, err = None, f"worker exception: {e}"

                if result_path is None:
                    print(f"  ⚠ Beat {beat_index}: pre-render failed ({err}) -- will render blank "
                          f"[{done_count}/{len(to_render)}]")
                    continue
                try:
                    with np.load(result_path) as data:
                        results[beat_index] = data["frames"]
                except Exception as e:
                    print(f"  ⚠ Beat {beat_index}: failed to load rendered frames ({e})")
                if done_count % 10 == 0 or done_count == len(to_render):
                    print(f"  ⚙️  Pre-render progress: {done_count}/{len(to_render)}")

    print(f"  ✅ Pre-render complete: {len(results)}/{len(visual_code_timeline)} beats have usable frames")
    return results


def _build_visual_code_batch_prompt(batch: list) -> str:
    annotated = []
    for b in batch:
        dur = round(float(b.get("end_time", 0)) - float(b.get("start_time", 0)), 2)
        entry = dict(b)
        entry["_duration_seconds"] = dur
        annotated.append(entry)
    return (
        f"Beats ({len(batch)} total -- output exactly {len(batch)} code blocks, one per beat, in order):\n"
        f"{json.dumps(annotated, indent=2)}\n\n"
        f"For EACH beat: first identify, in your own reasoning, what this beat is fundamentally "
        f"ABOUT (not the words -- the underlying idea: a quantity growing, a risk, a comparison, "
        f"a moment of surprise, a process taking time, a tradeoff). THEN write code whose visual "
        f"behavior expresses that specific idea. Two beats about different ideas must look "
        f"visually different from each other -- do not reuse the same composition shape across "
        f"beats with different meanings. Each beat's `_duration_seconds` field above tells you "
        f"how long THAT beat lasts -- use it only to PLAN your animation's pacing (e.g. a 0.6s "
        f"beat needs faster motion than a 3s beat). It is NOT a variable available inside your "
        f"draw_beat code -- your function only receives draw, t, w, h, np, math as arguments. "
        f"`t` will range from 0 up to that beat's duration when your code actually runs; write "
        f"code that looks correct across that whole range without ever referencing "
        f"_duration_seconds (or any other field name from the JSON above) directly in the code."
    )


def generate_visual_code(beats: list, topic: str) -> list:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not set.")

    print(f"  🎬 Call 3: Per-beat visual code generation for {len(beats)} beats...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = f"""You are a generative motion graphics engineer for a finance/numbers explainer channel. For each beat of narration you write a Python function that draws that beat's visual directly. You decide what should appear on screen by reasoning from what the beat MEANS, then you write the actual drawing code for it.

Captions are handled by YouTube itself. NEVER draw the beat's sentence as text. The biggest quality failure in this system has been beats rendering as a wall of words instead of an actual visual -- you must actively fight that tendency.

=== TEXT SHOULD BE ALMOST NEVER USED ===
The narration audio and YouTube's own captions already deliver every word of this beat to the viewer -- on-screen text was never necessary for understanding what's being said, and was never your job. Drawing text that restates or parallels the narration is redundant by definition, not just "too much": the viewer already has the words. Your job is the visual the words don't already provide -- the thing you SEE that makes the number/idea land, not a second copy of the dialogue.

Default to NO text at all. The only narrow exception: when a specific numeric value genuinely needs to be visually legible as part of a chart/counter/comparison (e.g. the "$34,000" inside a growing bar, the "60%" inside a donut fill) -- because seeing the exact digits adds something the visual shape alone can't (precision), not because the sentence needs repeating. Even then, it's the number only, never a label restating what was said, never a phrase, never more than a couple of words. If you're not rendering a chart/counter that needs its value displayed, there should usually be ZERO draw.text() calls in your code for that beat.

If you draw text in more than maybe one out of every four or five beats across a whole script, you are almost certainly drawing it too often -- the large majority of beats should have no text at all, just the visual itself.

=== YOUR PRIMARY TOOL: A RICH FINANCE VISUAL VOCABULARY, NOT JUST BARS ===
This is a premium finance dashboard explainer, not a generic chart-of-the-day video. Most beats have a real number (has_data, data_value, data_unit, data_label, compare_value) or a real financial concept -- reach for the SPECIFIC visual metaphor that matches the idea, not the easiest shape to draw. A flat bar or donut works, but using the same one or two chart types for every beat across a whole video is exactly the "simple shapes" failure to avoid. Below is a menu of finance-specific visual primitives, each with enough concrete construction guidance to actually build it -- pick whichever matches the beat's real meaning:

- Bar comparison: two or more rectangles, heights proportional to their values, short label under each. Use for any direct "X vs Y" beat.
- Progress / fill meter: a rounded rectangle or circular ring that fills from 0 toward a target proportion, with the number growing alongside it. Use for "how much of X has happened" beats.
- Gauge / dial: an arc (draw.arc or a manually computed arc path) from one extreme to another (e.g. DANGER to SAFE), with a needle line rotating to the current value, the number below it. Use for risk/security/runway beats.
- Donut / pie percentage: draw.pieslice or two arcs, percent filled, number centered inside. Use for "X% of Y" beats.
- Trend / line chart: 4-8 computed points sloping to match data_direction, end value labeled, optionally with a subtle filled area under the line. Use for any "growing/shrinking over time" beat.
- Compound growth curve: like a line chart but with a deliberately accelerating curve (use an exponential or power easing on the y-values, not a straight slope) -- small deposit marks entering periodically along the bottom, the curve bending upward late. Use specifically for compound interest / investing-over-time beats.
- Cashflow waterfall: a starting bar at the top (e.g. gross income), then a sequence of smaller bars stepping down (each subtracting a cost -- taxes, expenses, fees), ending in a final highlighted bar for the net/take-home amount. Use for "here's what's actually left after costs" beats -- this is one of the most useful primitives for this channel, reach for it often when a beat mentions multiple deductions/costs.
- Funnel: a sequence of horizontal bars or trapezoids, each narrower than the last, top to bottom, with a count or percentage at each stage. Use for "out of N people, only a few succeed" beats.
- Bill / expense card: a simple rounded rectangle styled like a card (a colored top strip or icon, a label like "RENT" or "UTILITIES", an amount). Stack 2-4 of these vertically or slide them in sequentially for "recurring bills" or "costs add up" beats.
- Bank balance / income card: similar card style, but showing a running number that can visibly drop or rise (animate the displayed number, not just a static label) -- use for "paycheck arrives/disappears" or "balance changes" beats.
- Icon-grid / crowd grid: a grid of small circles or squares, a portion filled/colored to represent a percentage of people/cases, the rest left as outlines. Use for "X% of people do this" beats -- this is the right tool for population-style statistics, not a donut.
- Calendar / timeline: a horizontal line or row of tick marks representing time units (months/years), with a marker or flag at a specific point, and optionally a region shaded differently before/after that point. Use for any beat referencing a specific duration or "by month X".
- Treadmill / moving-backward metaphor: a flat horizontal line representing a baseline (e.g. expenses), with a second line that's supposed to be progress but is animated moving backward or staying flat while the baseline rises -- communicates "running in place" or "falling behind" without needing a human figure.
- Leaky bucket / faucet: a container shape (rounded rectangle or simple bucket outline) with a fill level, plus small animated drips leaving through a gap or bottom opening faster than the fill rises (or a thin stream entering from a faucet shape above) -- use for "money in doesn't keep up with money out" beats.
- Portfolio stack: a stack of rectangles or coins of varying size representing a total, with a small percentage/yield indicator beside it. Use for "portfolio size produces X income" beats.

Pick whichever of these (or a reasonable variation) actually matches what the beat is about -- a percentage is not a bar, a trend is not a pie, a "costs eat into income" beat is a waterfall, not a generic comparison bar. Across a whole video, vary which primitive you reach for; repeatedly defaulting to the same one or two regardless of content is a real, previously-observed failure mode.

=== THE FIGURE IS DISABLED FOR NOW -- DO NOT USE IT AS A MAIN VISUAL ===
Do not draw a person/character as the hero or focal point of any beat. This explainer should look like a premium finance dashboard, not an animated character video -- a stick figure or silhouette filling a large part of the frame reads as unprofessional and amateurish for this content, and is explicitly banned as a primary visual.

If, and only if, a beat is fundamentally about human behavior in a way that's hard to express any other way (panic-selling, burnout, a crowd of people), you may include a SMALL, simple silhouette -- no more than about 12% of frame height -- positioned in a corner or edge of the frame (e.g. bottom-right, well within the safe margin), acting as a small supporting accent to a chart or number that remains the actual hero of the beat. Never let it be larger than, or compete with, the main data visual. In the large majority of beats, there should be no figure at all -- reach for a chart, card, gauge, or icon metaphor instead (see the visual primitives list below), since those communicate the finance concept far more clearly than a human figure does.

=== EMOJI ===
A helper function `draw_emoji(emoji_char, cx, cy, size)` is available (already defined, just call it) -- use it instead of trying to draw emoji via draw.text() yourself, since raw emoji font rendering is unreliable at arbitrary sizes. Good for adding tone/color to a beat (💰 📉 😰 ⚠️ 🏠 💸 📊) alongside a chart or number, sparingly -- one emoji accent per beat at most, never the whole visual on its own.

=== REAL FINANCE ICONS (use these instead of hand-drawing icon shapes) ===
A helper function `draw_icon(name, cx, cy, size, color=(r,g,b,a))` is available -- it draws a real, professionally-designed icon (from a licensed open icon set), tinted to whatever color you pass, centered at (cx, cy), sized to fit within a `size`x`size` box with aspect ratio preserved. This is a MUCH better choice than hand-drawing an icon shape from primitive lines/circles, since these are clean, consistent, recognizable glyphs. Exact available names (any other name silently draws nothing, so only use these): trending-up, trending-down, chart-line, chart-bar, chart-pie, cash-banknote, coins, credit-card, receipt, invoice, building-bank, percentage, gauge, calendar, clock, scale, alert-triangle, moneybag, home, car, stethoscope, stairs, wallet. Use the brand palette colors from the QUALITY BAR section for the color argument. Good uses: a small `wallet` or `cash-banknote` icon accenting a paycheck/income card, `building-bank` for a bank/institution reference, `alert-triangle` for a warning card's corner accent, `gauge` alongside a meter you're also drawing numerically, `trending-up`/`trending-down` as a small directional accent next to a chart. Like emoji, use sparingly as an accent to a chart/card/number, not as the entire visual on its own -- one icon per beat at most unless building an icon-grid (e.g. several small `home` or `coins` icons to represent a population/count).

=== NOTICING LIST STRUCTURE ===
Some scripts count through a numbered list ("the first warning sign...", "warning sign number two...", "here's the fourth..."). If THIS beat is the one introducing a new numbered item, it's worth reflecting that (e.g. a small "03" badge in a corner, an outlined number), but only on the beat that actually introduces the item -- not on every beat that elaborates on it afterward. Most beats in a list-style script are elaboration, not new-item beats; don't force a counter onto beats that don't need one.

=== THE OPENING BEAT IS CRITICAL -- THIS HAS FAILED BEFORE, TREAT IT SERIOUSLY ===
If this is beat_index 0, a blank or text-only first 1-2 seconds has been a real, observed problem in actual output -- viewers see nothing but background for the opening moment, which reads as broken, not premium. This is not optional polish, it's a hard requirement: beat_index 0 MUST have a real, fully-formed visual active from t=0, not fading in from nothing, not a placeholder, not "just text." If the beat's content doesn't obviously suggest a number or chart, default to a simple, immediate dashboard-style element -- e.g. a glowing card outline, a single bold icon, a meter at a starting position -- anything that is unmistakably a real graphic on screen at t=0, not an empty frame waiting for content. Treat "the opening beat renders nothing" as a failed beat, the same severity as a crash.

=== 16:9 COMPOSITION ===
Canvas is {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} (always compute from w/h, never hardcode). Keep all content within a safe margin of roughly 8% of w/h from every edge -- nothing should touch or crowd the frame edge. This means BOTH position AND size: before drawing any rectangle/card/bar, check that its full extent (its position PLUS its width/height) still fits inside that safe margin -- a shape positioned correctly but sized too large will still bleed off-frame, which has been an observed real bug (a card or bar extending past the bottom/side of the visible frame). When in doubt, compute the maximum allowed size first (e.g. max_height = h * 0.84 for a vertically-stacked element), then size your shape as a fraction of that maximum, never as a fixed pixel count that could exceed it.

Favor ONE clear underlying idea per beat -- but a single idea rendered as a bare, undecorated shape (one flat rectangle, one plain circle, nothing else) reads as empty and unfinished, not clean. A good beat usually has 3-5 small details supporting its one idea, not 3-5 unrelated competing ideas. For example: a bar chart's "one idea" still includes the bar itself, a baseline/axis line, the value label, and a subtle highlight or motion on the bar -- that's one coherent composition with real detail, not clutter. A gauge's one idea includes the arc, the needle or fill, the number, and a small tick mark or two. Think of it as "one hero element, fully realized with supporting detail" rather than "one hero element floating alone." If your code only calls 1-2 drawing operations total, it is almost certainly too sparse -- add the supporting detail that makes the one idea feel complete and considered, not more competing ideas.

Avoid reusing the same overall composition shape across different beats -- if you've described several beats in a row as "a rectangle with a label," vary it: try a circular/radial layout, a growing line, a grid, a gauge, concentric shapes, instead of defaulting back to the same rectangle-plus-line pattern every time.

If you do add a secondary element (e.g. a chart plus an emoji accent), give it clearly different visual weight -- one primary, one small secondary -- not two equally-sized things competing for attention.

=== YOUR REQUIRED PROCESS, FOR EVERY BEAT ===
1. Read the beat's text and data fields. Identify the SINGLE underlying concept.
2. Does this beat have real data (has_data)? If yes, your default move is a chart matching that data's shape (see chart guidance above) -- don't reach past this for an abstract shape unless a chart genuinely doesn't fit.
3. If no data: is this a person/behavior beat where a simple figure fits, a pure transition that warrants minimal/no visual, or an abstract concept (risk, comparison, time) better shown with shape/motion?
4. Write the `concept` field as the real reasoning chain: "<what this beat is about> -> <why this specific visual represents that>".
5. Only then write the code.

=== WORKED EXAMPLES ===
- Beat: "you'd have about $34,000" (has_data, data_value=34000, data_unit=dollars) -> concept: a single grown quantity -> a bar or fill-meter rising to represent the value, with "$34,000" labeled at the top as it settles, because the number is the entire point and a chart shows scale better than a bare counter alone.
- Beat: "over 60% of gig workers underestimate" (has_data, data_value=60, data_unit=percent) -> concept: a portion of a population -> an icon-grid of ~10 small figures/circles where 6 are filled solid and 4 are outline-only, OR a donut pie at 60%, with "60%" labeled -- NOT a sentence on screen.
- Beat: "less than $250 monthly cash flow nationwide" vs "the $5,000 average unplanned repair" (comparison, compare_value set) -> concept: a small recurring amount dwarfed by an occasional large one -> two bars, one short one tall, both labeled with their values.
- Beat: "would you actually still be financially okay" (hook, no data) -> concept: a person facing uncertainty -> a simple seated/standing figure with a slumped or questioning gesture, OR if unsure about the figure, a single dimming/uncertain shape -- not the sentence as text.
- Beat: "let's raise the stakes with item five" (transition introducing a new list item, no real data) -> concept: a new item beginning -> a small "05" badge or outlined number appearing, minimal otherwise -- not a big visual, just the marker.
- Beat: "but here's the part nobody talks about" (pure transition, no data) -> concept: connective tissue -> minimal or nothing. Do not invent content to fill the frame.

=== WHAT YOU ARE WRITING ===
A single Python function per beat, exactly this signature:

def draw_beat(draw, t, w, h, np, math):
    # your code here

- `draw` is a PIL ImageDraw.Draw object on a transparent RGBA layer already sized to the canvas. Use draw.line(...), draw.ellipse(...), draw.polygon(...), draw.rectangle(...), draw.rounded_rectangle(...), draw.text(...), draw.arc(...), draw.pieslice(...) -- anything PIL's ImageDraw supports EXCEPT draw.textsize() (removed from modern Pillow -- use the text_size() helper below instead).
- `draw_emoji(emoji_char, cx, cy, size)` is also available in scope -- call it directly, don't redefine it.
- `get_font(size, style="bold")` returns a ready-to-use font object. Two styles available: "bold" (clean modern sans, good for general numbers/labels) and "black" (a heavier, condensed, more dramatic display face -- good for hook-style headline words or high-intensity warning beats). Pick whichever weight matches this beat's tone, don't default to the same one every time. There is no other way to get a font: `ImageFont` is NOT available in scope.
- `text_size(text, font)` returns (width, height) in pixels for a string rendered with a given font -- use this for centering/sizing text, not draw.textsize() which doesn't exist in this Pillow version.
- `t` is the number of seconds elapsed since THIS beat started (0.0 at beat start). Use it to animate -- compute positions/sizes/opacity as a function of t.
- `w`, `h` are the canvas pixel dimensions ({OUTPUT_WIDTH}x{OUTPUT_HEIGHT}).
- `np` is numpy, `math` is the math module. Note: there is no np.Font or any font-related attribute on numpy -- fonts only come from get_font(size).
- Colors are RGBA tuples, e.g. (255, 215, 130, 255). Always include alpha. Compute alpha from t for fade in/out.

=== WHAT YOU MAY NOT DO ===
No imports, no file/network access, no `open`, `exec`, `eval`, `os`, `sys`. Code that tries to do anything else will be rejected before it ever runs.

=== COMPLETENESS ===
Every `for`, `if`, `while`, `else`, `elif` must have a real statement on the next line(s), properly indented -- never leave a block with only a comment and no code, and never leave a block empty. The ENTIRE function body must be one single, complete, syntactically valid Python function with nothing left unfinished -- this is checked mechanically before your code ever runs, and an incomplete block fails that check and the whole beat renders blank, wasting the beat entirely. If you're unsure a more complex composition will come out complete and correct, write a simpler one you're confident is fully correct instead.

=== NO DOUBLE-DRAWING ===
Each frame is a SINGLE independent call to your function at one value of t -- draw.text() and other draw calls do not "replace" what's at that position, they layer on top. If your code computes a displayed number/label at more than one point in the function and draws it more than once (e.g. once for a settling animation and again for a "final" state), both will be visible simultaneously and overlap into unreadable garbled text. Compute each value ONCE per call, decide its current state from t, and draw it exactly ONCE.

=== COMMON MISTAKES THAT HAVE ACTUALLY HAPPENED -- AVOID THESE SPECIFICALLY ===
- draw.text(xy, text, font, fill=color) -- font is already the 3rd POSITIONAL argument in PIL's signature; passing it positionally AND then also writing fill= is fine, but writing draw.text(xy, text, color) and ALSO font= afterward, or duplicating any argument, throws "got multiple values for argument". Pass each argument exactly once, in this order: draw.text((x, y), text, font=get_font(size), fill=(r,g,b,a)).
- Pillow's drawing calls require integer pixel coordinates in many cases -- if you compute a coordinate with division (w / 2) it's a float; wrap coordinates in int(...) before passing them to draw.ellipse/rectangle/line/etc, especially anything derived from a fraction or a range/index.
- For draw.ellipse/draw.rectangle, the box is [x0, y0, x1, y1] and REQUIRES x1 >= x0 and y1 >= y0 -- a shape whose size shrinks toward or past zero as t changes can easily produce an invalid box this way, which is a real, repeated crash seen in production. Use the provided `safe_box(cx, cy, half_w, half_h)` helper INSTEAD OF computing [x0,y0,x1,y1] yourself by hand -- it takes a center point and half-width/half-height and always returns a valid, correctly-ordered box no matter what values you pass in (including zero or negative sizes), so this failure becomes impossible. Example: draw.ellipse(safe_box(cx, cy, radius, radius), fill=color) instead of draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], fill=color).

=== TRANSITIONS BETWEEN BEATS (vary these, don't always fade) ===
A plain fade-in/fade-out on every single beat, with nothing else, starts to feel monotonous across a whole video -- vary HOW each beat enters and exits, not just whether it fades. Pick whichever of these fits the beat's energy, and rotate between them across the video rather than using the same one repeatedly:
- Slide-in: the whole visual (or its main element) enters from off-screen (left/right/top/bottom) and decelerates into its final position -- compute position as a function of t with an ease-out curve (e.g. position = target + (start - target) * (1 - progress) ** 3), don't move at constant speed.
- Scale-in / pop: the visual starts slightly smaller (or larger) than its final size and scales to 100% with a touch of overshoot (scale briefly past 100% then settle back, using a damped oscillation or a simple overshoot-then-correct curve) -- good for numbers/counters that should feel like they "land" with impact.
- Wipe / reveal: use a clip mask or progressively draw more of the shape over time (e.g. a bar chart's bars rise from 0 height to full height in sequence rather than all appearing at once; a line chart draws its line left-to-right rather than all points appearing simultaneously).
- Quick cut with a flash: for a high-intensity beat (a stark number, a warning), a near-instant appearance (very short fade, under 0.1s) optionally preceded by one frame of a brief brightness flash -- use sparingly, only for beats that should feel sudden/jarring on purpose.
- Cross-element handoff: if a number from this beat directly relates to the previous beat's number (e.g. a gross amount becoming a net amount after deductions), consider having the new element originate FROM roughly where the old number would have been (same x/y region) rather than appearing in an unrelated part of the frame -- this creates a felt sense of continuity between beats even though each beat's code is independent.
Exit transitions deserve the same variety as entrances -- a shape can shrink away, slide off in a direction, or dissolve (fade while also drifting slightly), not just fade in place every time.

=== QUALITY BAR ===
- Smooth animation: compute continuous functions of t (easing, sine waves, interpolation), not instant jumps, unless an instant snap is specifically the right feeling.
- Every element should fade in and fade out, not just appear/disappear instantly -- compute an alpha from t (ramping 0 to 255 over the first ~15-20% of the beat, holding, ramping back down over the last ~15-20%) and apply it to every shape/text/figure you draw. This applies broadly, not just to the figure.
- Reach for real effect techniques, not just static shapes: a soft glow (draw the same shape 2-3 times at increasing size with decreasing alpha, behind the main shape), a pulse (modulate a size or alpha with a sine wave over t), a subtle drop shadow (draw a duplicate shape offset by a few pixels in dark low-alpha color, behind the main shape), a trail (draw the last few positions of a moving element at decreasing alpha). These add real production value over a flat unadorned shape.
- Legible at video scale: numbers need font size proportional to h (e.g. h*0.08 for a prominent number), shapes need enough size/contrast to read instantly.
- Use this exact brand color palette, consistently, across every beat -- do not invent other colors: white (245,247,250) for primary text/numbers, market green (56,217,150) for growth/gains/positive, warning red (255,77,77) for risk/loss/danger, gold (255,209,102) for highlights/attention/key numbers, muted gray (138,148,166) for secondary/de-emphasized elements, dark panel (17,26,36) for card/panel backgrounds. These are RGB values for draw.text/draw.rectangle/etc fill colors.
- Use t=0 as the entrance state and design toward a settled state by the end of the beat's duration.
- Keep code self-contained and correct for ALL t in [0, duration], including exactly t=0 and t=duration -- no index errors, no division by zero.
- Across consecutive beats with DIFFERENT concepts, vary the composition -- don't reuse the same chart type for every beat regardless of content.

Return ONLY valid JSON:
{{
  "beats": [
    {{
      "beat_index": <int>,
      "concept": "<the real reasoning chain: what this beat is about -> why this specific visual represents that. Required, specific, not a generic label.>",
      "code": "def draw_beat(draw, t, w, h, np, math):\n    ...\n"
    }}
    // ... exactly {len(beats)} entries, in order
  ]
}}

The "code" field is a STRING containing the full function definition, with \n for newlines, valid Python, nothing else in that string (no markdown fences, no commentary). If a beat genuinely warrants no visual (pure transition, no concrete content), set "code" to an empty string "" rather than inventing decoration -- this is a valid and often correct choice, not a failure."""

    BATCH_SIZE = dynamic_batch_size(len(beats), min_size=3, max_size=6)
    all_results = []
    batches = [beats[i:i+BATCH_SIZE] for i in range(0, len(beats), BATCH_SIZE)]

    def _run_batch(batch_idx, batch):
        start_beat = batch_idx * BATCH_SIZE
        print(f"  🎬 Visual-code batch {batch_idx+1}/{len(batches)}: beats {start_beat}-{start_beat+len(batch)-1}...")
        response = _call_with_retry(lambda: gpt4o_call(client,
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _build_visual_code_batch_prompt(batch)}
            ],
            response_format={"type": "json_object"},
            temperature=0.85,
            max_tokens=8000,
            timeout=120,
        ), label=f"Call 3 batch {batch_idx+1}")
        result = json.loads(response.choices[0].message.content)
        batch_results = result.get('beats', [])
        if len(batch_results) > len(batch):
            batch_results = batch_results[:len(batch)]
        elif len(batch_results) < len(batch):
            while len(batch_results) < len(batch):
                batch_results.append({"beat_index": start_beat + len(batch_results),
                                       "concept": "", "code": ""})
        print(f"  ✅ Visual-code batch {batch_idx+1} done: {len(batch_results)} beats")
        return batch_idx, batch_results

    results = [None] * len(batches)
    MAX_WORKERS = 3
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_run_batch, i, b): i for i, b in enumerate(batches)}
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                idx, batch_results = future.result()
                results[idx] = batch_results
            except Exception as e:
                print(f"  ❌ Visual-code batch {batch_idx+1} failed: {e}")
                raise

    for batch_results in results:
        all_results.extend(batch_results)

    print(f"  ✅ {len(all_results)} total beat visual codes generated")
    return all_results


def validate_visual_code(beat_codes: list, beats: list) -> list:
    """Static safety check only -- this is NOT an aesthetics gate. Each
    beat's code is parsed and checked for forbidden constructs before
    it's ever allowed to run. A beat that fails this check gets its
    code cleared (renders blank); a beat that passes still might fail
    later at actual execution time, which _render_beat_frames_worker
    (called via prerender_all_beat_visuals) handles.

    Also logs (never rejects on) a missing or suspiciously generic
    `concept` field -- this is the visible signal that the prompt's
    required reasoning step was skipped for a beat, so it's something
    to notice when reviewing output, not a safety concern."""
    print(f"  🔍 Static safety check on {len(beat_codes)} beat code blocks...")
    rejected = 0
    GENERIC_CONCEPT_PHRASES = {"growth visual", "decoration", "visual", "animation", "shape", ""}
    for i, entry in enumerate(beat_codes):
        if not isinstance(entry, dict):
            beat_codes[i] = {"beat_index": i, "concept": "", "code": ""}
            continue
        entry["beat_index"] = i

        concept = str(entry.get("concept", "")).strip()
        if concept.lower() in GENERIC_CONCEPT_PHRASES or (concept and "->" not in concept and len(concept) < 15):
            print(f"  ⚠ Beat {i}: concept reasoning looks generic/missing ('{concept}') -- worth a look when reviewing this beat's visual")

        code = entry.get("code", "")
        if not isinstance(code, str) or not code.strip():
            entry["code"] = ""
            rejected += 1
            continue
        ok, reason = _static_safety_check(code)
        if not ok:
            print(f"  ⚠ Beat {i}: rejected generated code -- {reason}")
            entry["code"] = ""
            rejected += 1
    print(f"  ✅ Safety check done, {rejected} beat(s) rejected (will render blank)")
    return beat_codes


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

    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    if luminance >= min_luminance:
        return hex_color

    if luminance < 1.0:
        return "#FFFFFF"

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

            if etype == "text":
                content = (el.get("content") or "").strip()
                if not content:
                    continue
                check_words = [w.strip('.,!?;:\'"()[]- ').lower()
                               for w in content.split()
                               if len(w.strip('.,!?;:\'"()[]- ')) > 2]
                if check_words and beat_words:
                    matches = sum(1 for w in check_words if w in beat_words)
                    if matches == 0 and len(check_words) > 0:
                        print(f"  ⚠ Scene {scene_pos}: dropped hallucinated text '{content[:30]}'")
                        fixed += 1
                        continue

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

            elif etype == "number_counter":
                try:
                    el["target_value"] = safe_float(el, "target_value", 0)
                except (TypeError, ValueError):
                    print(f"  ⚠ Scene {scene_pos}: dropped number_counter with bad target_value")
                    fixed += 1
                    continue
                el.setdefault("prefix", "")
                el.setdefault("suffix", "")
                el.setdefault("decimals", 0)
                el.setdefault("x", 0.5)
                el.setdefault("y", 0.42)
                el.setdefault("anchor", "center")
                el.setdefault("size", 180)
                el.setdefault("color", "#FFFFFF")
                el.setdefault("weight", "black")
                el.setdefault("outline", 5)
                el.setdefault("count_from", 0)
                el.setdefault("count_duration", 0.8)
                el.setdefault("start_offset", 0.0)
                el.setdefault("duration", None)
                el["color"] = _ensure_bright_color(el["color"])
                el["size"] = max(60, min(safe_int(el, "size", 180), 220))

            elif etype == "grid":
                glyph = str(el.get("glyph", "0")).strip()
                if not glyph:
                    print(f"  ⚠ Scene {scene_pos}: dropped grid with empty glyph")
                    fixed += 1
                    continue
                el["glyph"] = glyph[:3]
                el.setdefault("rows", 4)
                el.setdefault("cols", 10)
                el.setdefault("cell_size", 60)
                el.setdefault("color", "#FBC02D")
                el.setdefault("x", 0.5)
                el.setdefault("y", 0.55)
                el.setdefault("anim", "fill_sequential")
                el.setdefault("fill_duration", 1.2)
                el.setdefault("start_offset", 0.0)
                el.setdefault("duration", None)
                el["color"] = _ensure_bright_color(el["color"])
                rows = max(1, min(safe_int(el, "rows", 4), 10))
                cols = max(1, min(safe_int(el, "cols", 10), 16))
                while rows * cols > 80:
                    if cols > rows:
                        cols -= 1
                    else:
                        rows -= 1
                el["rows"], el["cols"] = rows, cols

            else:
                continue

            for k in ("x", "y", "x1", "y1", "x2", "y2", "w", "h", "radius"):
                if k in el and isinstance(el[k], (int, float)):
                    el[k] = max(0.0, min(1.0, float(el[k])))

            cleaned.append(el)

        if len(cleaned) > 4:
            print(f"  ⚠ Scene {scene_pos}: trimmed {len(cleaned)} elements to 4")
            cleaned = cleaned[:4]
            fixed += 1

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


def render_text_overlay_opencv(video_path: str, scenes: list, beats: list,
                               whisper_segments: list, output_path: str,
                               beat_visual_codes: list = None):
    print(f"🎨 Scene renderer v5: {len(scenes)} scenes...")
    beat_visual_codes = beat_visual_codes or []
    _visual_code_by_beat = {}
    for entry in beat_visual_codes:
        if isinstance(entry, dict):
            _visual_code_by_beat[entry.get("beat_index", -1)] = entry.get("code", "")

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

    def load_pil_font(path, size, weight="black"):
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
        rows, cols = frame.shape[:2]
        X = cv2.getGaussianKernel(cols, cols * 0.6)
        Y = cv2.getGaussianKernel(rows, rows * 0.6)
        mask = (Y * X.T) / (Y * X.T).max()
        out = frame.copy().astype(np.float32)
        for i in range(3): out[:,:,i] *= mask
        return np.clip(out, 0, 255).astype(np.uint8)

    def apply_warm_grade(frame):
        out = frame.copy().astype(np.float32)
        out[:,:,2] = np.clip(out[:,:,2] * 1.04, 0, 255)
        return out.astype(np.uint8)

    def to_pil(frame):
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def to_frame(pil_img):
        return cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)

    def composite_layer(frame, layer):
        pil = to_pil(frame).convert('RGBA')
        merged = Image.alpha_composite(pil, layer)
        return to_frame(merged)

    def _ffprobe_dur(path):
        try:
            r = subprocess.run(['ffprobe','-v','error','-show_entries','format=duration',
                                '-of','default=noprint_wrappers=1:nokey=1', path],
                               capture_output=True, text=True)
            return float(r.stdout.strip())
        except: return 0.0

    TARGET_FPS = 30.0
    vid_dur = _ffprobe_dur(video_path)

    cfr_video = output_path.replace(".mp4", "_cfr_tmp.mp4")
    subprocess.run(['ffmpeg','-y','-i',video_path,
                    '-vf',f'fps={TARGET_FPS:.0f}',
                    '-c:v','libx264','-preset','ultrafast','-crf','18',
                    '-an', cfr_video],
                   capture_output=True, check=True)

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

        beat_words = get_beat_whisper_words(beat_start, beat_end)

        resolved = []
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

                if len(words_in_content) > 1:
                    available2 = [w for i, w in enumerate(beat_words) if i not in used_indices]
                    last_match = match_word_in_list(words_in_content[-1], available2)
                    if last_match:
                        idx2 = beat_words.index(last_match)
                        used_indices.add(idx2)
                        el_end = max(el_end, last_match['end'])

                resolved.append((el_start, el_end, el))
            else:
                resolved.append((beat_start, beat_end, el))

        resolved.sort(key=lambda x: x[0])
        is_single_word = len(resolved) == 1

        for i, (ws, we_t, el) in enumerate(resolved):
            anim_start = clamp(ws, 0.0, vid_dur - 0.1)

            if is_single_word:
                impact_end = anim_start + 1.0
                if next_beat_start is not None:
                    impact_end = min(impact_end, next_beat_start)
                anim_end = clamp(impact_end, anim_start + 0.1, vid_dur)
                impact = True
            else:
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
                "anim_duration": 0.06 if impact else safe_float(el, "anim_duration", 0.10),
                "impact":        impact,
            })

        for el in other_els:
            timeline.append({
                "el":            el,
                "start":         beat_start,
                "end":           beat_end,
                "anim_duration": safe_float(el, "anim_duration", 0.2),
                "impact":        False,
            })

    timeline.sort(key=lambda x: x["start"])
    print(f"  📊 Timeline: {len(timeline)} elements")

    visual_code_timeline = []
    for scene_pos, beat in enumerate(beats):
        code = _visual_code_by_beat.get(scene_pos, "")
        if not code:
            continue
        beat_start = clamp(float(beat.get("start_time", 0.0)), 0, vid_dur - 0.1)
        beat_end   = clamp(float(beat.get("end_time", beat_start + 2.0)),
                           beat_start + 0.05, vid_dur)
        if scene_pos + 1 < len(beats):
            beat_end = min(beat_end, float(beats[scene_pos + 1].get("start_time", beat_end)))
        visual_code_timeline.append({"code": code, "start": beat_start, "end": beat_end,
                                       "beat_index": scene_pos, "_warned": False})

    visual_code_timeline.sort(key=lambda x: x["start"])
    print(f"  🎬 Visual code timeline: {len(visual_code_timeline)} beats with generated visuals")

    prerendered_beats = prerender_all_beat_visuals(visual_code_timeline, OUTPUT_WIDTH, OUTPUT_HEIGHT)

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

    def get_anim_progress(el_t, start, end, anim_dur):
        """Return (entrance_progress, exit_progress) both 0..1.
        entrance_progress: 0=not started, 1=fully appeared
        exit_progress: 1=visible, 0=fully gone (only at end)
        """
        if anim_dur <= 0:
            anim_dur = 0.001
        entrance = clamp((el_t - start) / anim_dur, 0.0, 1.0)
        return entrance

    def draw_text_element(layer, el, el_t, anim_t):
        """Draw a TEXT element with animation."""
        draw = ImageDraw.Draw(layer)
        content = el.get("content", "").upper().strip()
        if not content:
            return
        x_pct = safe_float(el, "x", 0.5)
        y_pct = safe_float(el, "y", 0.5)
        raw_size = safe_int(el, "size", 90)
        word_count = len(content.split())
        if el.get("_is_counter"):
            size_cap = 220
        else:
            size_cap = 160 if word_count == 1 else 110
        size = max(20, min(raw_size, size_cap))
        color = hex_to_rgb(el.get("color", "#FFFFFF"))
        weight = el.get("weight", "black")
        outline = max(0, min(safe_int(el, "outline", 4), 12))
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

        pad = 30
        max_x = OUTPUT_WIDTH - tw - pad
        max_y = OUTPUT_HEIGHT - th - pad
        base_x = max(pad, min(base_x, max_x))
        base_y = max(pad, min(base_y, max_y))

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

        if effect == "flicker":
            if el_t - 0 < 0.3:
                frame_no = int(el_t * 30)
                if frame_no % 2 == 1:
                    return
        elif effect == "shake":
            import random as _r
            draw_x += _r.randint(-3, 3)
            draw_y += _r.randint(-3, 3)

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

        draw_x = max(pad, min(draw_x, OUTPUT_WIDTH - tw - pad))
        draw_y = max(pad, min(draw_y, OUTPUT_HEIGHT - th - pad))

        a_int = max(0, min(int(255 * alpha), 255))
        if a_int < 5:
            return

        if outline > 0:
            for ox in range(-outline, outline + 1):
                for oy in range(-outline, outline + 1):
                    if ox * ox + oy * oy <= outline * outline:
                        if ox == 0 and oy == 0:
                            continue
                        draw.text((draw_x + ox, draw_y + oy), content,
                                  font=render_font, fill=(0, 0, 0, a_int))
        draw.text((draw_x, draw_y), content, font=render_font,
                  fill=(color[0], color[1], color[2], a_int))

    def draw_line_element(layer, el, el_t, anim_t):
        """Draw a LINE element with animation."""
        draw = ImageDraw.Draw(layer)
        x1 = int(OUTPUT_WIDTH * safe_float(el, "x1", 0.3))
        y1 = int(OUTPUT_HEIGHT * safe_float(el, "y1", 0.5))
        x2 = int(OUTPUT_WIDTH * safe_float(el, "x2", 0.7))
        y2 = int(OUTPUT_HEIGHT * safe_float(el, "y2", 0.5))
        thickness = max(1, safe_int(el, "thickness", 6))
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
        x = int(OUTPUT_WIDTH * safe_float(el, "x", 0.4))
        y = int(OUTPUT_HEIGHT * safe_float(el, "y", 0.4))
        w = int(OUTPUT_WIDTH * safe_float(el, "w", 0.2))
        h = int(OUTPUT_HEIGHT * safe_float(el, "h", 0.1))
        color = hex_to_rgb(el.get("color", "#FFFFFF"))
        filled = bool(el.get("filled", True))
        thickness = max(1, safe_int(el, "thickness", 3))
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
        cx = int(OUTPUT_WIDTH * safe_float(el, "x", 0.5))
        cy = int(OUTPUT_HEIGHT * safe_float(el, "y", 0.5))
        r = int(min(OUTPUT_WIDTH, OUTPUT_HEIGHT) * safe_float(el, "radius", 0.05))
        color = hex_to_rgb(el.get("color", "#FFFFFF"))
        filled = bool(el.get("filled", False))
        thickness = max(1, safe_int(el, "thickness", 4))
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

    def _format_counter_value(value, decimals, prefix, suffix):
        """Format a number with comma separators, fixed decimals, and
        prefix/suffix -- e.g. 400000 -> '$400,000', 23.5 -> '23.5%'."""
        if decimals > 0:
            text = f"{value:,.{decimals}f}"
        else:
            text = f"{int(round(value)):,}"
        return f"{prefix}{text}{suffix}"

    def draw_number_counter_element(layer, el, el_t, anim_t):
        """Draw a NUMBER_COUNTER element -- animates from count_from to
        target_value over count_duration, then holds at target_value.
        Reuses draw_text_element's rendering by building a synthetic
        text element each frame with the current counted value."""
        target = safe_float(el, "target_value", 0)
        count_from = safe_float(el, "count_from", 0)
        count_dur = max(0.05, safe_float(el, "count_duration", 0.8))
        decimals = max(0, safe_int(el, "decimals", 0))
        prefix = el.get("prefix", "")
        suffix = el.get("suffix", "")

        progress = clamp(el_t / count_dur, 0.0, 1.0)
        eased = 1.0 - (1.0 - progress) ** 3
        current_value = count_from + (target - count_from) * eased
        content = _format_counter_value(current_value, decimals, prefix, suffix)

        synthetic = dict(el)
        synthetic["type"] = "text"
        synthetic["content"] = content
        synthetic["_is_counter"] = True
        draw_text_element(layer, synthetic, el_t, 1.0 if progress > 0 else anim_t)

    def draw_grid_element(layer, el, el_t, anim_t):
        """Draw a GRID element -- rows x cols of a repeated glyph, either
        all at once (fade_in) or revealed cell-by-cell left-to-right,
        top-to-bottom (fill_sequential)."""
        draw = ImageDraw.Draw(layer)
        glyph = el.get("glyph", "0")
        rows = max(1, safe_int(el, "rows", 4))
        cols = max(1, safe_int(el, "cols", 10))
        cell = max(10, safe_int(el, "cell_size", 60))
        color = hex_to_rgb(el.get("color", "#FBC02D"))
        anim = el.get("anim", "fill_sequential")
        fill_dur = max(0.05, safe_float(el, "fill_duration", 1.2))

        cx = int(OUTPUT_WIDTH * safe_float(el, "x", 0.5))
        cy = int(OUTPUT_HEIGHT * safe_float(el, "y", 0.55))
        grid_w = cols * cell
        grid_h = rows * cell
        ox = cx - grid_w // 2
        oy = cy - grid_h // 2

        font = load_pil_font(get_primary_font_path(), int(cell * 0.8), "black")

        total_cells = rows * cols
        if anim == "fill_sequential":
            progress = clamp(el_t / fill_dur, 0.0, 1.0)
            visible_cells = int(total_cells * progress)
            cell_alpha = 1.0
        else:
            visible_cells = total_cells
            cell_alpha = anim_t

        a_int = max(0, min(int(255 * cell_alpha), 255))
        if a_int < 5:
            return

        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= visible_cells:
                    return
                gx = ox + c * cell + cell // 2
                gy = oy + r * cell + cell // 2
                try:
                    bbox = font.getbbox(glyph)
                    gw, gh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except Exception:
                    gw, gh = cell // 2, cell // 2
                draw.text((gx - gw // 2, gy - gh // 2), glyph, font=font,
                          fill=(color[0], color[1], color[2], a_int))
                idx += 1

    _warned_beats = set()

    def _lookup_prerendered_frame(item, t):
        beat_frames = prerendered_beats.get(item["beat_index"])
        if beat_frames is None:
            if item["beat_index"] not in _warned_beats:
                print(f"  ⚠ Beat {item['beat_index']}: no pre-rendered frames available -- rendering blank")
                _warned_beats.add(item["beat_index"])
            return None
        code_t = t - item["start"]
        frame_i = int(code_t * INTERNAL_VISUAL_FPS)
        frame_i = max(0, min(frame_i, beat_frames.shape[0] - 1))
        return beat_frames[frame_i]

    while True:
        ret, frame = cap.read()
        if not ret: break

        t = frame_idx / fps_vid
        frame = apply_vignette(frame)
        frame = apply_warm_grade(frame)

        active_code_item = next((item for item in visual_code_timeline
                                  if item["start"] <= t < item["end"]), None)
        if active_code_item is not None:
            arr = _lookup_prerendered_frame(active_code_item, t)
            if arr is not None:
                code_layer = Image.fromarray(arr)
                frame = composite_layer(frame, code_layer)

        raw_active = [item for item in timeline
                      if item["start"] <= t < item["end"]]

        seen_keys = {}
        for item in sorted(raw_active, key=lambda x: x["start"], reverse=True):
            el = item["el"]
            if el.get("type") == "text":
                content_key = (el.get("content", "").upper().strip(),
                               round(safe_float(el, "x", 0.5), 1),
                               round(safe_float(el, "y", 0.5), 1))
                if content_key not in seen_keys:
                    seen_keys[content_key] = item
            else:
                seen_keys[id(item["el"])] = item
        active = list(seen_keys.values())

        if active:
            frame = cv2.addWeighted(frame, 0.82, np.zeros_like(frame), 0.18, 0)

            layer = Image.new('RGBA', (OUTPUT_WIDTH, OUTPUT_HEIGHT), (0, 0, 0, 0))

            for item in active:
                el    = item["el"]
                el_t  = t - item["start"]
                el_dur = max(item["end"] - item["start"], 0.01)
                impact = item.get("impact", False)
                etype  = el.get("type", "text")

                if impact and etype == "text":
                    in_flash = (
                        (0.00 <= el_t < 0.08) or
                        (0.16 <= el_t < 0.24) or
                        (0.32 <= el_t < 0.40)
                    )
                    in_off = (0.08 <= el_t < 0.16) or (0.24 <= el_t < 0.32)
                    if in_off:
                        continue
                    if el_t >= 0.40:
                        fade_window = 0.15
                        time_left = el_dur - el_t
                        if time_left < fade_window:
                            anim_t = max(0.0, time_left / fade_window)
                        else:
                            anim_t = 1.0
                    else:
                        anim_t = 1.0
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
                        elif etype == "number_counter":
                            draw_number_counter_element(layer, el, el_t, anim_t)
                        elif etype == "grid":
                            draw_grid_element(layer, el, el_t, anim_t)
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
    'ffmpeg', '-y',
    '-i', temp_video,
    '-i', video_path,
    '-map', '0:v',
    '-map', '1:a',
    '-c:v', 'copy',
    '-c:a', 'aac',
    '-b:a', '192k',
    '-shortest',
    '-movflags', '+faststart',
    output_path
], capture_output=True)

    if os.path.exists(temp_video): os.remove(temp_video)
    if result.returncode != 0:
        raise Exception(f"Audio merge failed: {result.stderr.decode()[-200:]}")

    print(f"  ✅ Render complete: {output_path}")


FINANCE_EXPLAINER_CTA_TEXT = "Subscribe to Math Unlocked"
FINANCE_EXPLAINER_USE_BACKGROUND_MUSIC = False


class MathUnlockedGenerator:
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

        valid_top = []
        for folder in top:
            files = self.get_all_files_from_dir(folder)
            if files:
                valid_top.append(folder)
            else:
                print(f"  ⚠ Skipping empty/missing broll folder: {folder}")

        if not valid_top:
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
        all_folders = []
        for folder in self.broll_dirs.values():
            if self.get_all_files_from_dir(folder):
                all_folders.append(folder)
        if not all_folders:
            raise Exception("No broll clips found in any folder.")

        folder_pools = {}
        for folder in all_folders:
            folder_pools[folder] = list(self.get_all_files_from_dir(folder))

        broll_cat_to_folder = {
            'space':   self.broll_dirs.get('space',   'space_vids'),
            'ancient': self.broll_dirs.get('ancient', 'ancient_ruins_vids'),
            'cosmic':  self.broll_dirs.get('cosmic',  'cosmic_vids'),
            'sky':     self.broll_dirs.get('sky',     'dark_sky_vids'),
            'temple':  self.broll_dirs.get('temple',  'temple_vids'),
        }

        folder_idx = {f: 0 for f in all_folders}
        for f in all_folders:
            random.shuffle(folder_pools[f])

        base_dur   = 4.0
        n_segs     = max(int(duration / base_dur), 1)
        folder_rot = 0

        for i in range(n_segs):
            seg_dur = float(beats[i].get('clip_duration', base_dur)) if i < len(beats) else base_dur
            target_folder = all_folders[folder_rot % len(all_folders)]
            folder_rot += 1

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
        """Legacy b-roll fallback. Never output a pure black/void screen."""
        return _make_safe_still_fallback(
            output_file,
            title="Visual Pause",
            subtitle="The explanation continues",
            duration=dur,
            w=OUTPUT_WIDTH,
            h=OUTPUT_HEIGHT,
            fps=fps,
        )

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

        vf = [f"fps={fps}"] + vf
        vf += ["eq=brightness=0.02:contrast=1.05:saturation=1.1", "format=yuv420p"]
        cmd += ['-vf', ','.join(vf), '-c:v', 'libx264', '-preset', ENCODE_PRESET,
                '-crf', ENCODE_CRF, '-tune', 'animation', '-pix_fmt', 'yuv420p',
                '-r', str(fps), '-movflags', '+faststart', '-an', output_file]

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
            print(f"  ⚠ Using branded fallback filler ({dur:.2f}s)")
            return self._make_black_filler(output_file, dur, fps)

        return output_file

    def _add_section_navigation_overlay(self, video_input: str, output_path: str,
                                          sections: list, duration: float) -> str:
        if not sections or len(sections) < 2:
            print(f"  ⏭  No list structure detected, skipping navigation overlay")
            shutil.copy(video_input, output_path)
            return output_path

        print(f"  🗂  Burning in {len(sections)}-item navigation overlay...")
        total = len(sections)
        filters = []
        tmp_dir = tempfile.mkdtemp(prefix="navtext_")

        try:
            for i, sec in enumerate(sections):
                num   = sec["number"]
                title = sec["title"]
                start = max(min(sec["start_time"], duration - 0.1), 0.0)
                end   = max(min(sec["end_time"], duration), start + 0.1)
                card_end = min(start + 1.8, end)
                label = f"{num}/{total}: {title}"

                label_path = os.path.join(tmp_dir, f"label_{i}.txt")
                with open(label_path, "w", encoding="utf-8") as f:
                    f.write(label)
                label_path_ff = label_path.replace("\\", "/").replace(":", "\\:")

                filters.append(
                    f"drawtext=textfile='{label_path_ff}'"
                    f":fontcolor=white:fontsize=50:font=Arial"
                    f":box=1:boxcolor=black@0.6:boxborderw=18"
                    f":x=(w-text_w)/2:y=h*0.10"
                    f":enable='between(t\\,{start:.3f}\\,{card_end:.3f})'"
                )
                filters.append(
                    f"drawtext=textfile='{label_path_ff}'"
                    f":fontcolor=white:fontsize=24:font=Arial"
                    f":box=1:boxcolor=black@0.45:boxborderw=10"
                    f":x=36:y=36"
                    f":enable='between(t\\,{start:.3f}\\,{end:.3f})'"
                )
                dots = ''.join('\u25CF' if j == i else '\u25CB' for j in range(total))
                filters.append(
                    f"drawtext=text='{dots}'"
                    f":fontcolor=#FFD166:fontsize=30:font=Arial"
                    f":x=(w-text_w)/2:y=h*0.93"
                    f":enable='between(t\\,{start:.3f}\\,{end:.3f})'"
                )

            vf = ','.join(filters)
            result = subprocess.run(
                ['ffmpeg', '-y', '-i', video_input, '-vf', vf,
                 '-c:v', 'libx264', '-preset', ENCODE_PRESET, '-crf', ENCODE_CRF,
                 '-tune', 'animation', '-pix_fmt', 'yuv420p',
                 '-c:a', 'copy', '-movflags', '+faststart', output_path],
                capture_output=True
            )
            if result.returncode != 0:
                print(f"  ⚠ Section nav overlay failed, shipping without it: {result.stderr.decode(errors='replace')[-200:]}")
                subprocess.run(['ffmpeg', '-y', '-i', video_input, '-c', 'copy', output_path], capture_output=True)
            else:
                print(f"  ✅ Navigation overlay added")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return output_path

    def _add_cta_overlay(self, video_input: str, output_path: str, duration: float):
        end_time = round(max(duration - 4, 1), 3)
        vf = (
            f"drawtext=text='{FINANCE_EXPLAINER_CTA_TEXT}'"
            f":fontcolor=yellow:fontsize=42:font=Arial"
            f":borderw=2:bordercolor=black:shadowx=2:shadowy=2"
            f":x=(w-text_w)/2:y=h*0.91:enable='gt(t\\,{end_time})'"
        )
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', video_input, '-vf', vf,
             '-c:v', 'libx264', '-preset', ENCODE_PRESET, '-crf', ENCODE_CRF,
             '-tune', 'animation', '-pix_fmt', 'yuv420p',
             '-c:a', 'copy', '-movflags', '+faststart', output_path],
            capture_output=True
        )
        if result.returncode != 0:
            subprocess.run(['ffmpeg', '-y', '-i', video_input, '-c', 'copy', output_path],
                           check=True, capture_output=True)
        else:
            print(f"  ✨ CTA added")

    def create_math_video(self, bg_volume: float = 0.12, fps: int = 30) -> bool:
        import time
        t0 = time.time()

        print(f"\n{'='*70}")
        print(f"🧮  MATH UNLOCKED -- Manim renderer")
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
            topic_hint = list(self.broll_dirs.keys())[0] if self.broll_dirs else "finance"
            with ThreadPoolExecutor(max_workers=2) as _outer_pool:
                _beats_future    = _outer_pool.submit(analyze_story_beats, full_text, whisper_segments, topic_hint, duration)
                _sections_future = _outer_pool.submit(extract_section_outline, full_text, whisper_segments, duration)
                beats_result = _beats_future.result()
                try:
                    sections = _sections_future.result()
                except Exception as e:
                    print(f"  ⚠ Section outline failed, skipping navigation overlay: {e}")
                    sections = []
            topic = beats_result.get('topic', 'finance')
            beats = beats_result.get('beats', [])

            _whisper_words = build_whisper_word_list(whisper_segments)
            beats = realign_beat_times(beats, _whisper_words)
            print(f"  🎯 Realigned {len(beats)} beat timestamps to Whisper word boundaries")
        except Exception as e:
            raise Exception(f"STEP 3 FAILED: {e}")

        print(f"\n[STEP 4] Grouping beats into Manim chunks...")
        chunks = group_beats_into_manim_chunks(beats, target_chunk_seconds=4.5)
        print(f"  ✅ {len(beats)} beats -> {len(chunks)} chunk(s)")

        print(f"\n[STEP 5] GPT Call: Manim chunk code generation...")
        try:
            chunk_code_list = generate_manim_chunk_code(chunks, topic)
        except Exception as e:
            print(f"  ⚠ Manim chunk generation failed entirely, every chunk will fall back to a dashboard filler: {e}")
            chunk_code_list = [None] * len(chunks)

        print(f"\n[STEP 6] Music...")
        if FINANCE_EXPLAINER_USE_BACKGROUND_MUSIC:
            bg_music = MUSIC_MAP.get(beats_result.get("detected_tone", "default"), MUSIC_MAP['default'])
            if not os.path.exists(bg_music):
                bg_music = MUSIC_MAP['default']
                if not os.path.exists(bg_music):
                    for fname in (os.listdir('bg_musics') if os.path.exists('bg_musics') else []):
                        if fname.endswith('.mp3'):
                            bg_music = os.path.join('bg_musics', fname)
                            break
                    else:
                        bg_music = None
        else:
            bg_music = None
        print(f"  🎵 {bg_music if bg_music else 'disabled'}")

        temp_files    = []
        concat_output = "manim_concatenated.mp4"
        audio_output  = "audio_mixed.mp4"

        try:
            print(f"\n[STEP 7] Rendering Manim chunks (parallel) + concatenating...")
            clip_paths = render_all_manim_chunks(chunks, chunk_code_list,
                                                  w=OUTPUT_WIDTH, h=OUTPUT_HEIGHT, fps=fps)

            concat_manim_clips(clip_paths, concat_output)
            print(f"  ✅ {len(clip_paths)} chunks concatenated -> {concat_output}")

            print(f"\n[STEP 8] Audio mix...")
            sfx_events = _build_sfx_audio_inputs(clip_paths, chunk_code_list)
            print(f"  🔊 SFX events: {len(sfx_events)}")

            cmd = ['ffmpeg', '-y', '-i', concat_output, '-i', self.audio_path]
            n_inputs = 2

            if bg_music and os.path.exists(bg_music):
                cmd += ['-i', bg_music]
                bg_idx = n_inputs
                n_inputs += 1
            else:
                bg_idx = None

            sfx_indices = []
            for (_t, sfx_file, _vol) in sfx_events:
                cmd += ['-i', sfx_file]
                sfx_indices.append(n_inputs)
                n_inputs += 1

            fc_parts = []
            fc_parts.append(
                f'[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume=1.0[voice]'
            )
            mix_labels = ['[voice]']

            if bg_idx is not None:
                fc_parts.append(
                    f'[{bg_idx}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,'
                    f'volume={bg_volume},aloop=loop=-1:size=2e+09[bg]'
                )
                mix_labels.append('[bg]')

            for k, (t_start, _sfx_file, vol_db) in enumerate(sfx_events):
                idx = sfx_indices[k]
                vol_linear = 10 ** (vol_db / 20.0)
                label = f'[sfx{k}]'
                fc_parts.append(
                    f'[{idx}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,'
                    f'volume={vol_linear:.4f},adelay={int(t_start*1000)}|{int(t_start*1000)},apad{label}'
                )
                mix_labels.append(label)

            n_mix = len(mix_labels)
            mix_in = ''.join(mix_labels)
            fc_parts.append(
                f'{mix_in}amix=inputs={n_mix}:duration=first:dropout_transition=2:normalize=0,aresample=48000[aout]'
            )
            fc = ';'.join(fc_parts)

            cmd += ['-filter_complex', fc, '-map', '0:v', '-map', '[aout]']
            cmd += ['-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                    '-ar', '48000', '-ac', '2', '-t', str(duration), audio_output]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                raise Exception(f"Audio failed: {r.stderr.decode()[-400:]}")
            print(f"  ✅ Mixed ({n_mix} audio streams)")

            shutil.move(audio_output, self.output_path)

            print(f"\n[STEP 8.5] Section navigation overlay...")
            nav_output = self.output_path.replace(".mp4", "_nav.mp4")
            self._add_section_navigation_overlay(self.output_path, nav_output, sections, duration)
            self.output_path = nav_output

            print(f"\n[STEP 9] CTA...")
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
            print(f"🎭 {len(beats)} beats | 🎬 {len(chunks)} Manim chunks")
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
            for f in [concat_output, audio_output]:
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass
            for tmp in glob.glob("*TEMP_MPY*.mp4") + glob.glob("*_noaudio_tmp.mp4") + glob.glob("*_cfr_tmp.mp4") + glob.glob("temp_segment_*.mp4"):
                try: os.remove(tmp)
                except: pass


NICHE_TEMPLATES = {
    'finance': {
        'broll_dirs': {},
        'keyword_map': {}
    },
}


@app.get("/")
def root():
    return {"service": "Math Unlocked", "status": "running",
            "openai_key": bool(OPENAI_API_KEY)}

@app.post("/generate")
async def generate_video_api(background_tasks: BackgroundTasks, niche: str = "finance"):
    global current_job
    if current_job["status"] == "processing":
        return {"message": "Already processing", "status": "processing"}
    current_job = {"status": "processing", "progress": 0, "output": None,
                   "error": None, "started_at": datetime.now().isoformat(), "niche": niche}
    background_tasks.add_task(process_video, niche)
    return {"message": f"Started niche={niche}", "status": "processing"}

def process_video(niche: str = "finance"):
    global current_job
    try:
        current_job["progress"] = 5
        audio_url   = "https://raw.githubusercontent.com/RandomSci/Automation_For_Math_Niche/main/Audio_Voice/vaults_narration.mp3"
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
        niche_config = NICHE_TEMPLATES.get(niche, NICHE_TEMPLATES['finance'])
        gen = MathUnlockedGenerator(audio_path=audio_file, output_path=output_file,
                              niche_config=niche_config)

        current_job["progress"] = 20
        success = gen.create_math_video(bg_volume=0.12, fps=30)
        current_job["progress"] = 95

        final = gen.output_path
        if success and final and os.path.exists(final):
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
            "niche": current_job.get("niche", "finance")}

@app.get("/download")
def download_video():
    if current_job["status"] != "completed":
        raise HTTPException(400, f"Not ready: {current_job['status']}")
    if not current_job["output"] or not os.path.exists(current_job["output"]):
        raise HTTPException(404, "File not found")
    return FileResponse(current_job["output"], media_type="video/mp4",
                        filename=f"finance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")


MANIM_FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__",
    "os", "sys", "subprocess", "socket", "requests", "shutil",
    "globals", "locals", "vars", "input", "breakpoint", "exit", "quit",
    "DecimalNumber",
    "MarkupText", "Integer", "Variable", "BulletedList", "Title", "Paragraph",
    "BarChart", "SVGMobject", "ComplexPlane", "PolarPlane",
    "MathTex", "Tex", "SingleStringMathTex",
    "DashedLine", "DashedVMobject", "Ellipse",
    "scipy",
}
MANIM_FORBIDDEN_PATTERNS = [
    r'\bDecimalNumber\b',
    r'\bMarkupText\b', r'\bInteger\b', r'\bVariable\b', r'\bBulletedList\b',
    r'\bTitle\b', r'\bParagraph\b', r'\bBarChart\b',
    r'\bSVGMobject\b', r'\bComplexPlane\b', r'\bPolarPlane\b',
    r'\bMathTex\b', r'\bSingleStringMathTex\b', r'\bTex\b',
    r'\bDashedLine\b', r'\bDashedVMobject\b',
    r'\bEllipse\b',
    r'\bscipy\b',
    r'plot_line_graph',
    r'set_stroke\s*\([^)]*dash_length',
    r'get_graph\s*\([^)]*x_range',
    r'set_style\s*\([^)]*dash_length_ratio',
    r'\binterpolate_color\b',
    r'Arrow\s*\([^)]*\bleft\s*=',
    r'Arrow\s*\([^)]*\bright\s*=',
    r'GrowFromEdge\s*\([^)]*\bdirection\s*=',
]
MANIM_ALLOWED_IMPORT_MODULES = {"manim", "numpy", "math", "random"}
MANIM_FORBIDDEN_REPLACEMENT_HINTS = {
    "MathTex": "fm_formula",
    "Tex": "fm_formula",
    "SingleStringMathTex": "fm_formula",
    "SVGMobject": "fm_icon(name, size, color)",
    "BarChart": "fm_animate_bar_chart",
    "DecimalNumber": "always_redraw with a plain Text() that reads tracker.get_value()",
    "MarkupText": "Text()",
    "Integer": "Text() built from an f-string",
    "Variable": "a ValueTracker plus always_redraw with a plain Text()",
    "BulletedList": "separate Text() lines arranged with .arrange(DOWN)",
    "Title": "Text()",
    "Paragraph": "Text()",
    "ComplexPlane": "fm_animate_line_chart, or drop the axis entirely and just show the data",
    "PolarPlane": "fm_animate_line_chart, or drop the axis entirely and just show the data",
}


def safety_correction_hint(reason: str) -> str:
    """Turns a manim_static_safety_check rejection reason into a sharp,
    specific correction instruction for a single targeted GPT retry,
    instead of the generic system prompt the chunk already saw once and
    apparently didn't follow. Naming the exact banned class AND the
    exact replacement to call, in the same short sentence, is a much
    stronger signal than the system prompt's general-purpose ban list
    buried among hundreds of other rules."""
    for name, replacement in MANIM_FORBIDDEN_REPLACEMENT_HINTS.items():
        if f"'{name}'" in reason:
            return (
                f"CORRECTION REQUIRED: your previous attempt at this exact chunk used "
                f"'{name}', which is permanently banned in this environment and was "
                f"automatically rejected before rendering -- it will be rejected again if "
                f"you use it a second time. Rewrite this chunk's visual using {replacement} "
                f"instead. Do not use '{name}' anywhere in your replacement code, including "
                f"inside helper expressions or as part of a workaround."
            )
    return (
        f"CORRECTION REQUIRED: your previous attempt at this exact chunk was rejected "
        f"for this specific reason: {reason}. Fix exactly that problem and resubmit "
        f"valid code -- do not change anything else about the visual."
    )


_MANIM_AVAILABLE_NAMES = None


def _get_chunk_available_names() -> set:
    """Names a chunk can legitimately reference without defining them
    itself: everything from manim, every fm_* library function and
    BRAND_* constant, MathScene/MathScene3D, and Python
    builtins. Computed once by actually executing the same boilerplate
    every chunk gets prefixed with, so this can never drift out of
    sync with what real chunk code actually has access to."""
    global _MANIM_AVAILABLE_NAMES
    if _MANIM_AVAILABLE_NAMES is None:
        ns = {}
        try:
            exec(MATH_SCENE_MANIM_BOILERPLATE, ns)
        except Exception:
            pass
        names = set(ns.keys())
        names.discard("__builtins__")
        import builtins as _builtins_mod
        names |= set(dir(_builtins_mod))
        _MANIM_AVAILABLE_NAMES = names
    return _MANIM_AVAILABLE_NAMES


def _collect_bound_names(tree) -> set:
    """Every name a chunk's own code binds anywhere in it -- assignments,
    for-loop/with/comprehension targets, function and class names,
    function parameters, import aliases, except-as names, walrus
    targets. Deliberately not scope-precise (a name bound inside a
    nested function counts as bound everywhere): the goal is zero
    false positives, never flagging legitimately-scoped code, even at
    the cost of occasionally missing a real scope bug."""
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound.add(node.name)
            all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            for arg in all_args:
                bound.add(arg.arg)
            if node.args.vararg:
                bound.add(node.args.vararg.arg)
            if node.args.kwarg:
                bound.add(node.args.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    return bound


def manim_static_safety_check(code: str) -> tuple[bool, str]:
    """Parse GPT-generated Manim chunk code and reject anything unsafe
    BEFORE it's ever written to disk and executed via the `manim` CLI
    subprocess. Different shape than the PIL pipeline's safety check:
    Manim's whole API requires `from manim import *`, so imports can't
    be banned outright -- instead this allows only a fixed allowlist
    of modules (manim, numpy, math) and rejects anything else, plus
    the same dangerous-builtin-name check as the PIL pipeline. Also
    structurally verifies the code contains exactly one top-level
    class definition named per the expected pattern with a construct
    method, the same AST-based approach (not substring matching) used
    for draw_beat, for the same reason: a substring check would pass
    even with stray unsafe top-level statements alongside the real
    class definition.

    MANIM_FORBIDDEN_NAMES bans DecimalNumber specifically: it has its
    own update-cycle quirks unrelated to LaTeX and reliably misbehaves
    in this codebase, so it stays banned. MathTex/Tex/SingleStringMathTex
    were briefly removed from this list on the assumption that having
    texlive installed meant they were safe -- production logs proved
    that wrong. The toolchain compiles fine; the failure mode is GPT
    generating malformed LaTeX source inside Python raw strings (most
    commonly doubled backslashes like r"\\text{...}" instead of
    r"\text{...}", since a raw string already preserves a single
    backslash literally, plus stray literal "$" characters inside the
    tex string). That is not a fixable-by-better-prompting problem at
    the rate GPT gets it wrong, so MathTex/Tex/SingleStringMathTex are
    back on this list for good. Axes/NumberLine/NumberPlane are banned
    for the same reason: the prompt previously contradicted itself by
    listing them as banned-because-they-crash in one sentence and then
    un-banning them in the next, which is now fixed by keeping them
    banned consistently here and in the prompt text.
    Rejecting the remaining names here is instant and free, instead of
    paying for a manim subprocess spin-up just to watch it crash.

    Also rejects any bare name the chunk references that isn't bound
    anywhere in its own code and isn't available from manim, the fm_*
    library, or BRAND_* constants. Two real production crashes prompted
    this: GPT calling GrowFromBottom (which sounds plausible by analogy
    with GrowFromCenter/GrowFromEdge but isn't a real Manim class) and a
    plain variable-name typo (defining emerg_lbl, then referencing
    emer_lbl in a VGroup() call a few lines later). Both are NameError
    at runtime regardless of cause -- a hallucinated class and a typo
    are indistinguishable to Python -- so both are caught the same way,
    before the slow manim subprocess ever starts instead of after it
    crashes partway through rendering.
    """
    for pattern in MANIM_FORBIDDEN_PATTERNS:
        if re.search(pattern, code):
            bare = pattern.replace(r'\b', '')
            return False, f"regex pre-scan: forbidden name '{bare}' found in code"

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0]
                if root_module not in MANIM_ALLOWED_IMPORT_MODULES:
                    return False, f"imports disallowed module '{alias.name}'"
        if isinstance(node, ast.ImportFrom):
            root_module = (node.module or "").split(".")[0]
            if root_module not in MANIM_ALLOWED_IMPORT_MODULES:
                return False, f"imports disallowed module '{node.module}'"
        if isinstance(node, ast.Name) and node.id in MANIM_FORBIDDEN_NAMES:
            return False, f"references forbidden name '{node.id}'"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"references dunder attribute '{node.attr}'"

        
        if isinstance(node, ast.Attribute) and node.attr in {"p2c", "point_to_coords", "point_to_number", "normalized", "point_at_angle"}:
            return False, f"references unsafe coordinate reverse-transform '{node.attr}' -- use axes.c2p(x, y) or an fm_* chart helper instead"

        if isinstance(node, ast.Call):
            fn = node.func
            fn_name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")

            if fn_name in {"p2c", "point_to_coords", "point_to_number", "normalized"}:
                return False, f"calls unsafe coordinate reverse-transform '{fn_name}' -- use axes.c2p(x, y) or an fm_* chart helper instead"

            if isinstance(fn, ast.Attribute) and fn.attr.startswith("fm_"):
                return False, f"called self.{fn.attr}() as a method -- fm_* functions are module-level, not Scene methods. Correct: {fn.attr}(self, ...). Wrong: self.{fn.attr}(...)"

            if fn_name == "Polygon" and len(node.args) == 1 and isinstance(node.args[0], (ast.List, ast.Tuple)):
                return False, "Polygon received one nested list of vertices; use Polygon(*points) or avoid Polygon and use fm_icon/fm_card/fm_* helpers"

            if fn_name == "rotate" and any(
                (isinstance(kw.arg, str) and kw.arg == "run_time")
                for kw in node.keywords
            ):
                return False, "rotate() does not accept run_time kwarg -- pass run_time to self.play() instead: self.play(obj.animate.rotate(...), run_time=X)"

            _fm_animation_fns = {
                "fm_animate_gauge", "fm_animate_donut", "fm_animate_counter",
                "fm_animate_line_chart", "fm_animate_line_chart_multi",
                "fm_animate_single_value", "fm_animate_glow_reveal",
                "fm_animate_icon_grid", "fm_animate_comparison_bars",
                "fm_animate_bar_chart", "fm_animate_waterfall",
                "fm_animate_stacked_cards", "fm_animate_bullet_chart",
                "fm_animate_timeline", "fm_animate_text_reveal",
                "fm_animate_bell_curve", "fm_animate_vector", "fm_animate_matrix",
                "fm_animate_scatter", "fm_animate_probability_bar",
                "fm_animate_number_line",
            }

            if fn_name == "play":
                for arg in node.args:
                    if isinstance(arg, ast.Call):
                        inner_fn = arg.func
                        inner_name = inner_fn.id if isinstance(inner_fn, ast.Name) else (inner_fn.attr if isinstance(inner_fn, ast.Attribute) else "")
                        if inner_name in _fm_animation_fns:
                            return False, f"fm_animate_* functions handle their own self.play() internally -- do NOT wrap them in self.play(). Call {inner_name}(self, ...) directly."

            if fn_name in _fm_animation_fns:
                first_arg_is_self = (
                    len(node.args) > 0
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "self"
                )
                first_kwarg_scene = any(
                    isinstance(kw.arg, str) and kw.arg == "scene"
                    for kw in node.keywords
                )
                if not first_arg_is_self and not first_kwarg_scene:
                    return False, f"{fn_name}() is missing 'self' as its first argument -- correct: {fn_name}(self, ...). The 'scene' parameter must be passed as the first positional argument."
                for child in ast.walk(node):
                    if isinstance(child, ast.Subscript) and child is not node:
                        pass

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            val = node.value
            if isinstance(val, ast.Call):
                fn = val.func
                fn_name2 = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
                if fn_name2 in {"fm_animate_gauge", "fm_animate_donut", "fm_animate_counter",
                                "fm_animate_line_chart", "fm_animate_line_chart_multi",
                                "fm_animate_single_value", "fm_animate_glow_reveal",
                                "fm_animate_icon_grid", "fm_animate_comparison_bars",
                                "fm_animate_bar_chart", "fm_animate_waterfall",
                                "fm_animate_stacked_cards", "fm_animate_bullet_chart",
                                "fm_animate_timeline", "fm_animate_text_reveal",
                                "fm_animate_bell_curve", "fm_animate_vector", "fm_animate_matrix",
                                "fm_animate_scatter", "fm_animate_probability_bar",
                                "fm_animate_number_line"}:
                    return False, f"indexing the return value of {fn_name2}() is unsafe -- these functions return None or a fixed-arity tuple; store the return in a named variable and index that, checking the documented arity first"

    class_defs = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if len(class_defs) != 1:
        return False, f"expected exactly one top-level class definition, found {len(class_defs)}"

    cls = class_defs[0]
    has_construct = any(
        isinstance(n, ast.FunctionDef) and n.name == "construct" for n in cls.body
    )
    if not has_construct:
        return False, "class definition is missing a construct(self) method"

    non_class_non_import = [
        n for n in tree.body
        if not isinstance(n, (ast.ClassDef, ast.Import, ast.ImportFrom))
    ]
    if non_class_non_import:
        kinds = sorted({type(n).__name__ for n in non_class_non_import})
        return False, f"unexpected top-level statement(s) outside the class/imports: {kinds}"

    bound_names = _collect_bound_names(tree)
    available_names = _get_chunk_available_names()
    loaded_names = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    undefined = sorted(
        n for n in loaded_names
        if n not in bound_names and n not in available_names
    )
    if undefined:
        return False, f"references undefined name(s) not found anywhere in manim, the fm_* library, or this chunk's own code: {', '.join(undefined[:5])}"

    return True, ""


MANIM_CHUNK_TIMEOUT_SECONDS = 180
MANIM_CHUNK_CACHE_DIR = "/tmp/math_unlocked_manim_cache"
MANIM_CHUNK_MAX_DRIFT_RATIO = 0.45


MATH_SCENE_MANIM_BOILERPLATE = '''
from manim import *
import random as _fdb_random
from manim.utils.rate_functions import (
    ease_in_sine, ease_out_sine, ease_in_out_sine,
    ease_in_quad, ease_out_quad, ease_in_out_quad,
    ease_in_cubic, ease_out_cubic, ease_in_out_cubic,
    ease_in_bounce, ease_out_bounce, ease_in_out_bounce,
    ease_in_elastic, ease_out_elastic, ease_in_out_elastic,
)

config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 30
config.frame_height = 8.0


def _math_scene_background_group():
    fw = config.frame_width
    fh = config.frame_height
    grid = VGroup()
    step = fh / 8
    x = -fw / 2
    while x <= fw / 2 + 1e-6:
        grid.add(Line([x, -fh / 2, 0], [x, fh / 2, 0]))
        x += step
    y = -fh / 2
    while y <= fh / 2 + 1e-6:
        grid.add(Line([-fw / 2, y, 0], [fw / 2, y, 0]))
        y += step
    grid.set_stroke(color="#8A94A6", width=0.6, opacity=0.06)

    axis = VGroup(
        Line([-fw * 0.46, -fh * 0.42, 0], [-fw * 0.46, fh * 0.42, 0]),
        Line([-fw * 0.46, -fh * 0.42, 0], [fw * 0.46, -fh * 0.42, 0]),
    )
    axis.set_stroke(color="#8A94A6", width=1.2, opacity=0.12)

    symbols = VGroup()
    rng = _fdb_random.Random(42)
    math_samples = ["∇", "∑", "∫", "π", "σ", "μ", "λ", "∞", "∂", "∈", "⊗", "≈", "Δ", "θ"]
    for i in range(14):
        val = rng.choice(math_samples)
        label = Text(val, font_size=16, color="#8A94A6")
        label.set_opacity(0.09)
        label.move_to([-fw / 2 + (i + 0.5) * (fw / 14), fh / 2 - 0.34, 0])
        symbols.add(label)

    return VGroup(grid, axis, symbols)


_FDB_MIN_WAIT = 0.04


class MathScene(Scene):
    def setup(self):
        self.camera.background_color = "#060F1A"
        self.add(_math_scene_background_group())

    def wait(self, duration=1.0, stop_condition=None, frozen_frame=None):
        if duration < _FDB_MIN_WAIT:
            duration = _FDB_MIN_WAIT
        return super().wait(duration, stop_condition=stop_condition, frozen_frame=frozen_frame)

    def play(self, *args, **kwargs):
        try:
            targets = []
            for arg in args:
                _fm_collect_play_targets(arg, targets)
            if targets:
                fm_clamp_to_frame(*targets)
        except Exception:
            pass
        return super().play(*args, **kwargs)


class MathScene3D(ThreeDScene):
    def setup(self):
        self.camera.background_color = "#060F1A"
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES)
        self.add_fixed_in_frame_mobjects(_math_scene_background_group())

    def wait(self, duration=1.0, stop_condition=None, frozen_frame=None):
        if duration < _FDB_MIN_WAIT:
            duration = _FDB_MIN_WAIT
        return super().wait(duration, stop_condition=stop_condition, frozen_frame=frozen_frame)

    def play(self, *args, **kwargs):
        try:
            targets = []
            for arg in args:
                _fm_collect_play_targets(arg, targets)
            if targets:
                fm_clamp_to_frame(*targets)
        except Exception:
            pass
        return super().play(*args, **kwargs)


''' + _FUNCTIONS_MANIM_CODE + '\n\n'


def _manim_chunk_cache_key(code: str, w: int, h: int, fps: int) -> str:
    raw = f"{code}|{w}|{h}|{fps}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _ffprobe_duration_seconds(path: str) -> float:
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
           '-of', 'default=noprint_wrappers=1:nokey=1', path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def _lock_chunk_duration(raw_path: str, target_duration: float, out_path: str,
                          w: int = 1920, h: int = 1080, fps: int = 30) -> tuple:
    """Forces a rendered Manim chunk onto an exact target duration,
    independent of whether the GPT-authored run_time/wait math inside
    the chunk was correct -- never trusts the chunk's own animation
    timing to be the timeline truth. Measures the actual rendered
    length via ffprobe, then retimes the whole clip with setpts so it
    lands exactly on target. If the raw render is wildly off (beyond
    MANIM_CHUNK_MAX_DRIFT_RATIO), a speed change would look broken
    rather than subtle, so this rejects the chunk entirely instead of
    warping it -- the caller falls back to a dashboard filler."""
    try:
        actual = _ffprobe_duration_seconds(raw_path)
    except Exception as e:
        return None, f"could not probe rendered chunk duration: {e}"

    if actual <= 0:
        return None, "rendered chunk reported zero or negative duration"

    drift = abs(actual - target_duration) / target_duration
    if drift > MANIM_CHUNK_MAX_DRIFT_RATIO:
        import sys
        print(f"  ⚠ duration drift {drift*100:.0f}% exceeds {MANIM_CHUNK_MAX_DRIFT_RATIO*100:.0f}% limit — rejecting chunk, falling back to filler", file=sys.stderr)
        return None, f"rendered duration drifted {drift*100:.0f}% from target ({actual:.2f}s vs {target_duration:.2f}s)"

    speed_ratio = target_duration / actual
    vf = f"setpts=PTS*{speed_ratio:.6f}"
    cmd = [
        'ffmpeg', '-y', '-i', raw_path,
        '-vf', vf, '-r', str(fps),
        '-c:v', 'libx264', '-preset', ENCODE_PRESET, '-crf', ENCODE_CRF,
        '-tune', 'animation', '-pix_fmt', 'yuv420p',
        '-an', '-t', str(target_duration), '-movflags', '+faststart',
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        tail = result.stderr.decode(errors='replace')[-300:]
        return None, f"duration-lock ffmpeg pass failed: {tail}"

    return out_path, ""


def _pil_font(size: int, bold: bool = True):
    """Best-effort font loader for fallback cards."""
    try:
        from PIL import ImageFont
        path = get_primary_font_path(bold=bold)
        if path:
            return ImageFont.truetype(path, size=size)
    except Exception:
        pass
    from PIL import ImageFont
    return ImageFont.load_default()


def _wrap_text_for_width(draw, text: str, font, max_width: int, max_lines: int = 3) -> list:
    words = re.sub(r'\s+', ' ', (text or '').strip()).split()
    if not words:
        return []
    lines, cur = [], ''
    for word in words:
        trial = (cur + ' ' + word).strip()
        try:
            bbox = draw.textbbox((0, 0), trial, font=font)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = len(trial) * 18
        if tw <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
            if len(lines) >= max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if len(lines) == max_lines and len(' '.join(words)) > len(' '.join(lines)):
        lines[-1] = lines[-1].rstrip('.,;:') + '...'
    return lines


def _chunk_fallback_subtitle(chunk: dict) -> str:
    beats = chunk.get('beats') or []
    text = ' '.join((b.get('text') or '').strip() for b in beats if isinstance(b, dict)).strip()
    text = re.sub(r'\s+', ' ', text)
    if not text:
        return 'The explanation continues'
    return text[:210].rstrip()


def _make_safe_still_fallback(out_path: str, title: str, subtitle: str, duration: float,
                              w: int = 1920, h: int = 1080, fps: int = 30) -> str:
    """Creates a non-blank exact-duration fallback video.

    This is the anti-void safety net. When a GPT-authored Manim chunk crashes,
    the viewer still sees a branded finance card with the current idea instead
    of an empty navy/black screen. It intentionally does NOT mention the error.
    """
    from PIL import Image, ImageDraw

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    frame_path = os.path.join(os.path.dirname(out_path) or '.', f"_fallback_{os.path.basename(out_path)}.png")

    img = Image.new('RGB', (int(w), int(h)), '#0B111A')
    draw = ImageDraw.Draw(img)

    grid_color = (32, 43, 56)
    step = max(80, int(min(w, h) * 0.075))
    for x in range(0, int(w), step):
        draw.line([(x, 0), (x, h)], fill=grid_color, width=max(1, w // 1800))
    for y in range(0, int(h), step):
        draw.line([(0, y), (w, y)], fill=grid_color, width=max(1, h // 1000))

    card_w = int(w * 0.70)
    card_h = int(h * 0.38)
    x0 = (w - card_w) // 2
    y0 = (h - card_h) // 2
    x1 = x0 + card_w
    y1 = y0 + card_h
    radius = int(min(w, h) * 0.035)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill='#111A24', outline='#38D996', width=max(4, w // 420))

    title_font = _pil_font(max(54, int(h * 0.055)), bold=True)
    body_font = _pil_font(max(38, int(h * 0.038)), bold=True)
    tag_font = _pil_font(max(26, int(h * 0.026)), bold=True)

    title = title or 'Key Idea'
    subtitle = subtitle or 'The explanation continues'

    tb = draw.textbbox((0, 0), title, font=title_font)
    tx = (w - (tb[2] - tb[0])) // 2
    ty = y0 + int(card_h * 0.16)
    draw.text((tx, ty), title, fill='#FFD166', font=title_font)

    lines = _wrap_text_for_width(draw, subtitle, body_font, int(card_w * 0.82), max_lines=3)
    line_h = int(h * 0.055)
    start_y = y0 + int(card_h * 0.40)
    for j, line in enumerate(lines):
        bb = draw.textbbox((0, 0), line, font=body_font)
        lx = (w - (bb[2] - bb[0])) // 2
        draw.text((lx, start_y + j * line_h), line, fill='#F5F7FA', font=body_font)

    tag = 'continuing'
    bb = draw.textbbox((0, 0), tag, font=tag_font)
    pad_x = int(w * 0.018)
    pad_y = int(h * 0.010)
    pill = [x1 - (bb[2]-bb[0]) - pad_x*3, y1 - int(h*0.070), x1 - pad_x, y1 - int(h*0.020)]
    draw.rounded_rectangle(pill, radius=int(h*0.018), fill='#0B111A', outline='#8A94A6', width=max(2, w//900))
    draw.text((pill[0] + pad_x, pill[1] + pad_y), tag, fill='#8A94A6', font=tag_font)

    img.save(frame_path, quality=95)

    cmd = [
        'ffmpeg', '-y', '-loop', '1', '-i', frame_path,
        '-t', str(max(float(duration), 0.05)),
        '-vf', f'scale={int(w)}:{int(h)},fps={int(fps)},format=yuv420p',
        '-c:v', 'libx264', '-preset', ENCODE_PRESET, '-crf', ENCODE_CRF,
        '-tune', 'stillimage', '-pix_fmt', 'yuv420p',
        '-an', '-movflags', '+faststart', out_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    try:
        os.remove(frame_path)
    except Exception:
        pass
    if result.returncode != 0:
        raise Exception(f"safe fallback failed: {result.stderr.decode(errors='replace')[-300:]}")
    return out_path


def _make_chunk_fallback(out_path: str, chunk: dict, duration: float, w: int = 1920,
                         h: int = 1080, fps: int = 30, reason: str = "") -> str:
    return _make_safe_still_fallback(
        out_path,
        title="Key Idea",
        subtitle=_chunk_fallback_subtitle(chunk or {}),
        duration=duration,
        w=w, h=h, fps=fps,
    )


def _make_dashboard_filler(out_path: str, duration: float, w: int = 1920,
                            h: int = 1080, fps: int = 30) -> str:
    return _make_safe_still_fallback(
        out_path,
        title="Next Idea",
        subtitle="The explanation continues",
        duration=duration,
        w=w, h=h, fps=fps,
    )


def _make_held_frame_filler(prev_clip_path: str, out_path: str, duration: float,
                             w: int = 1920, h: int = 1080, fps: int = 30) -> str:
    """Fills a short silence-gap chunk by freezing the previous content
    chunk's actual final frame for the gap's duration, instead of
    cutting to blank background. This is what real editors do for a
    natural speech pause -- the visual holds, it doesn't vanish.
    Falls back to a plain dashboard filler if the previous clip can't
    be read for any reason."""
    frame_path = os.path.join(os.path.dirname(out_path), f"_lastframe_{os.path.basename(out_path)}.png")
    extract_cmd = ['ffmpeg', '-y', '-sseof', '-0.1', '-i', prev_clip_path, '-vframes', '1', frame_path]
    result = subprocess.run(extract_cmd, capture_output=True)
    if result.returncode != 0 or not os.path.exists(frame_path):
        return None
    cmd_loop = [
        'ffmpeg', '-y', '-loop', '1', '-i', frame_path, '-t', str(duration),
        '-vf', f'scale={w}:{h},fps={fps},format=yuv420p',
        '-c:v', 'libx264', '-preset', ENCODE_PRESET, '-crf', ENCODE_CRF,
        '-tune', 'stillimage', '-pix_fmt', 'yuv420p',
        '-an', '-movflags', '+faststart', out_path,
    ]
    result2 = subprocess.run(cmd_loop, capture_output=True)
    try:
        os.remove(frame_path)
    except Exception:
        pass
    if result2.returncode != 0:
        return None
    return out_path


MANIM_CHUNK_DEBUG_LOG_DIR = "/tmp/math_unlocked_manim_logs"


def _extract_manim_error_summary(stderr_text: str, stdout_text: str, max_chars: int = 2500) -> str:
    """Manim renders its tracebacks through `rich`, which wraps the
    actual exception in box-drawing characters and ANSI color codes --
    a naive [-500:] slice of that raw text usually lands inside the
    source-code context panel rather than the actual exception type
    and message, which is what actually matters for debugging. This
    strips the box-drawing/ANSI noise and explicitly hunts for the
    last `SomeError: message` style line so the surfaced summary is
    readable instead of a fragment of a rendered box border."""
    raw = (stderr_text or "") + "\n" + (stdout_text or "")
    clean = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', raw)
    clean = re.sub(r'[│┃┌┐└┘─━╭╮╰╯┏┓┗┛┣┫┳┻╋]', '', clean)
    lines = [ln.rstrip() for ln in clean.splitlines() if ln.strip()]

    exception_line = ""
    for ln in reversed(lines):
        if re.match(r'^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception)\b', ln.strip()):
            exception_line = ln.strip()
            break

    tail = "\n".join(lines)[-max_chars:]
    if exception_line and exception_line not in tail:
        tail = f"{exception_line}\n...\n{tail}"
    return tail or "manim subprocess failed with no captured output"


def _save_manim_failure_log(class_name: str, stderr_text: str, stdout_text: str) -> str:
    os.makedirs(MANIM_CHUNK_DEBUG_LOG_DIR, exist_ok=True)
    log_path = os.path.join(MANIM_CHUNK_DEBUG_LOG_DIR, f"{class_name}.log")
    try:
        with open(log_path, "w") as f:
            f.write("=== STDERR ===\n")
            f.write(stderr_text or "")
            f.write("\n\n=== STDOUT ===\n")
            f.write(stdout_text or "")
    except Exception:
        pass
    return log_path


MANIM_CHUNK_SOURCE_DEBUG_DIR = "/tmp/math_unlocked_manim_sources"


def render_manim_chunk(code: str, class_name: str, duration: float, w: int = 1920,
                        h: int = 1080, fps: int = 30) -> tuple:
    """Renders ONE Manim chunk to its own exact-duration MP4 clip via the
    real `manim` CLI as a subprocess (not an in-process exec, since
    Manim's render pipeline isn't a simple function call -- it writes
    its own output file via ffmpeg internally). The GPT-authored code
    is safety-checked on its own BEFORE MATH_SCENE_MANIM_BOILERPLATE
    is prepended, so the structural check still only ever has to trust
    the boilerplate this file controls, not anything GPT wrote. Returns
    (clip_path_or_None, error_str). On any failure (safety rejection,
    subprocess crash, timeout, missing output, duration drift past
    MANIM_CHUNK_MAX_DRIFT_RATIO), returns (None, reason) -- the caller
    is expected to fall back to a dashboard-color filler of the exact
    target duration, never crash the whole video over one bad chunk.
    Disk-cached by content hash (including duration, since the cached
    artifact is the final duration-locked clip, not the raw render) so
    re-running after an unrelated downstream failure doesn't re-spend
    the wall-clock cost of every chunk that already rendered fine.

    Resolution and frame rate are NOT passed as `manim` CLI flags --
    MATH_SCENE_MANIM_BOILERPLATE sets config.pixel_width,
    config.pixel_height, and config.frame_rate directly at module
    level instead, since that is a stable Manim Community mechanism
    across versions, whereas CLI flag names/spellings can drift and an
    unrecognized flag would fail every single chunk at once.

    The chunk's own GPT-generated code (not the boilerplate) is always
    written to MANIM_CHUNK_SOURCE_DEBUG_DIR/{class_name}.py before
    rendering, success or failure -- a real failure had a chunk render
    cleanly (no crash, no safety rejection, no duration drift) but
    still produce a visually broken result (overlapping title and
    content), and since work_dir is always deleted in the finally
    block below, there was no way to ever see what code actually
    produced it. Crash logs only exist for crashes; this exists for
    every chunk, so a visually-bad-but-technically-successful render
    is always diagnosable after the fact, not just a hard failure."""
    ok, reason = manim_static_safety_check(code)

    try:
        os.makedirs(MANIM_CHUNK_SOURCE_DEBUG_DIR, exist_ok=True)
        with open(os.path.join(MANIM_CHUNK_SOURCE_DEBUG_DIR, f"{class_name}.py"), "w") as f:
            f.write(code)
    except Exception:
        pass

    if not ok:
        return None, f"safety check rejected: {reason}"

    os.makedirs(MANIM_CHUNK_CACHE_DIR, exist_ok=True)
    cache_key = _manim_chunk_cache_key(f"{code}|{duration:.4f}", w, h, fps)
    cached_path = os.path.join(MANIM_CHUNK_CACHE_DIR, f"{cache_key}.mp4")
    if os.path.exists(cached_path):
        return cached_path, ""

    work_dir = tempfile.mkdtemp(prefix="manim_chunk_")
    script_path = os.path.join(work_dir, "chunk_scene.py")
    try:
        boilerplate = (
            MATH_SCENE_MANIM_BOILERPLATE
            .replace("config.pixel_width = 1920", f"config.pixel_width = {int(w)}")
            .replace("config.pixel_height = 1080", f"config.pixel_height = {int(h)}")
            .replace("config.frame_rate = 30", f"config.frame_rate = {int(fps)}")
        )
        with open(script_path, "w") as f:
            f.write(boilerplate)
            f.write(code)

        
        quality_flag = "-qk" if int(w) >= 3840 and int(h) >= 2160 else "-qh"
        cmd = [
            "manim", quality_flag, "--disable_caching",
            "--media_dir", work_dir,
            script_path, class_name,
        ]
        try:
            result = subprocess.run(
                cmd, cwd=work_dir, capture_output=True, text=True,
                timeout=MANIM_CHUNK_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return None, f"manim render timed out after {MANIM_CHUNK_TIMEOUT_SECONDS}s"

        if result.returncode != 0:
            log_path = _save_manim_failure_log(class_name, result.stderr, result.stdout)
            summary = _extract_manim_error_summary(result.stderr, result.stdout)
            return None, f"manim subprocess failed (exit {result.returncode}), full log: {log_path}\n{summary}"

        produced = glob.glob(os.path.join(work_dir, "**", f"{class_name}.mp4"), recursive=True)
        if not produced:
            return None, "manim reported success but no output .mp4 was found"

        locked_path, lock_err = _lock_chunk_duration(produced[0], duration, cached_path, w, h, fps)
        if not locked_path:
            return None, lock_err

        return locked_path, ""
    except Exception as e:
        return None, f"unexpected error rendering manim chunk: {e}"
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


GAP_CHUNK_MIN_SECONDS = 0.35


def group_beats_into_manim_chunks(beats: list, target_chunk_seconds: float = 4.5) -> list:
    """Groups consecutive beats into chunks of roughly target_chunk_seconds
    each, without splitting any single beat across two chunks -- a
    chunk boundary always falls between beats, never inside one, since
    each chunk becomes its own independent Manim Scene/clip. A beat
    longer than target_chunk_seconds on its own still gets its own
    whole chunk (never truncated).

    CRITICAL: every chunk is rendered and concatenated back-to-back
    with zero gap between clips (see concat_manim_clips), but real
    narration always has silence between sentences -- natural speech
    pauses between Whisper segments. If those gaps are not represented
    as their own chunks, the concatenated video's internal clock runs
    fractions of a second fast relative to the audio track every time
    a gap is skipped, and that loss compounds across the whole video:
    by minute one this can total several seconds, which is exactly
    why visuals appear to fire earlier and earlier as a video runs
    on -- they are not actually drifting, the silent gaps between them
    were simply never given their own screen time. Whenever there is
    a real gap between one chunk's end_time and the next chunk's
    start_time, an explicit silent gap chunk (is_gap=True, no beats)
    is inserted to occupy exactly that duration, so the video timeline
    always matches the audio timeline beat-for-beat, gap-for-gap."""
    raw_chunks = []
    current = []
    current_start = None
    current_end = None
    for beat in beats:
        b_start = beat.get("start_time", 0.0)
        b_end = beat.get("end_time", b_start + 1.0)
        if current and (b_end - current_start) > target_chunk_seconds:
            raw_chunks.append({"beats": current, "start_time": current_start, "end_time": current_end})
            current = []
            current_start = None
        if not current:
            current_start = b_start
        current.append(beat)
        current_end = b_end
    if current:
        raw_chunks.append({"beats": current, "start_time": current_start, "end_time": current_end})

    chunks = []
    prev_end = None
    for c in raw_chunks:
        if prev_end is not None:
            gap = c["start_time"] - prev_end
            if gap >= GAP_CHUNK_MIN_SECONDS:
                chunks.append({"beats": [], "start_time": prev_end, "end_time": c["start_time"], "is_gap": True})
        chunks.append(c)
        prev_end = c["end_time"]
    return chunks


def generate_manim_chunk_code(chunks: list, topic: str) -> list:
    """Manim-based replacement for the old PIL/cv2 generate_visual_code.
    Produces one Manim Scene class per chunk (each chunk is roughly
    target_chunk_seconds of consecutive beats, see
    group_beats_into_manim_chunks) instead of one draw_beat() function
    per beat. Uses the same shared adaptive rate limiter
    (gpt4o_call/_call_with_retry) and the same dynamic_batch_size
    scaling as the old PIL-based Call 3, since smaller multi-second
    chunks means MORE total API calls per video than the old
    pipeline's larger 3-6-beat batches -- without dynamic batching
    this would hit the shared 30k TPM ceiling far more easily, not
    less. Each batch asks GPT for several chunks' worth of Manim code
    in one call, same batching shape as the old PIL Call 3.

    Confirmed real bug, now fixed: results used to be slotted into
    the final ordered list by trusting GPT's own self-reported
    "chunk_index" field in its JSON response. In practice GPT would
    sometimes reset that count back to 0 for a later batch instead of
    continuing the real global count, which silently overwrote an
    earlier batch's correct chunks with a later batch's content (e.g.
    a beat from minute four landing at the start of the video) and
    left that batch's own true slot empty. The fix never trusts that
    field for indexing -- the global index is derived from each
    item's own position in the returned array, which is the same
    order the chunks were requested in, and the class name inside the
    code is force-renamed to match via regex. This makes correct
    ordering independent of GPT getting cross-batch arithmetic right."""
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not set.")

    print(f"  🎬 Manim Call: chunk code generation for {len(chunks)} chunks...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = """You are a generative motion graphics engineer using Manim (Manim Community edition) for a premium long-form math and machine learning education channel called Math Unlocked. For each chunk of narration (a few seconds, sometimes one beat, sometimes several consecutive beats grouped together) you write ONE complete Manim Scene class that animates that chunk's visual. This is a 3Blue1Brown-standard math explainer — the visual itself carries the idea, not a wall of words.

=== THINK IN MATHEMATICAL STRUCTURES — NOT TEXT LABELS ===
Every mathematical concept has a geometric or structural visual that IS the concept. Your job is to find that visual and animate it. The audio already says the words. You show the REALITY of the mathematics.

The pattern: audio names the concept → you animate the mathematical structure that concept represents → viewer understands without reading a single word.

=== THE 3BLUE1BROWN STANDARD ===
The visual language of this channel is axes, plotted curves, animated vectors, growing bars, transforming matrices, probability distributions, scatter plots, number lines, geometric diagrams, and actual mathematical structure made visible. Almost nothing in a 3blue1brown video is a literal pictogram — it is the STRUCTURE of the idea made visible: a curve bending, a bar growing, a vector rotating, a distribution spreading, a matrix transforming space. That is the standard here.

Default to a chart, plot, matrix, vector, distribution, number line, formula, or geometric construction for every beat that has a mathematical concept, a number, a trend, a transformation, or a comparison.

=== VISUAL METAPHOR MAPPING ===
- Probability / frequency / proportion → fm_animate_probability_bar, fm_animate_donut, fm_animate_icon_grid
- Distribution / bell curve / normal → fm_animate_bell_curve
- Trend / growth / sequence → fm_animate_line_chart, fm_animate_line_chart_multi
- Vector / direction / linear combination → fm_animate_vector
- Matrix / transformation / system → fm_animate_matrix
- Single statistic / key number → fm_animate_counter, fm_animate_single_value
- Comparison between values → fm_animate_comparison_bars, fm_two_cards
- Formula / equation / calculation → fm_formula
- Scatter data / correlation → fm_animate_scatter
- Number line / range / interval → fm_animate_number_line
- Concept names / taxonomy → fm_concept_pills
- Chapter / hook / major reveal → fm_animate_glow_reveal
- Step-by-step process → fm_animate_timeline

=== AXES AND GEOMETRY ARE ALLOWED AND ENCOURAGED ===
Unlike the previous pipeline, Axes, NumberLine, NumberPlane, and Rectangle ARE available and encouraged when the fm_* library does not cover the visual well. Use them correctly:
- Axes: always pass x_length and y_length explicitly. Use axis_config with color=BRAND_GRAY, stroke_opacity=0.45, include_tip=False. Place axes with .move_to(ORIGIN) or .move_to(position). Use axes.c2p(x, y) to convert coordinates to screen points.
- NumberLine: NumberLine(x_range=[min, max, step], length=9.0). Use .n2p(value) to get a point.
- Rectangle: Rectangle(width=w, height=h) is fine for bars, containers, grids. RoundedRectangle(corner_radius=0.1) preferred for cards and panels.
- When hand-building a coordinate plot, always draw axes first, then content on top.

=== WHAT TEXT IS ALLOWED ===
- Actual mathematical values: 0.68 | 95% | n=30 | p < 0.05
- Ultra-short axis labels: "x" | "f(x)" | "n" | "P(x)" (1-3 chars)
- Chapter title cards via fm_animate_glow_reveal ONLY (hook/concept beats)
- Formula lines via fm_formula (never raw Text for formulas)

WRONG: Text("Most people misunderstand variance because they confuse it with standard deviation")
RIGHT: fm_animate_bell_curve(self, label_text="Variance = spread", accent_color=BRAND_GOLD, duration=D)

WRONG: Text("The gradient points in the direction of steepest ascent")
RIGHT: fm_animate_vector(self, [0.6, 0.8], "gradient", accent_color=BRAND_GREEN, duration=D)

If your construct() has more than 2 Text() objects that are not numbers or 1-3 character labels, you are writing captions. STOP. Replace them with a chart, distribution, vector, matrix, or formula visual.

=== HARD STRUCTURAL RULES ===
- Your response for EACH chunk must be exactly: `from manim import *` on its own line, then exactly ONE class definition subclassing either MathScene (the normal case) or MathScene3D (only for a deliberate 3D establishing-shot tilt), with a `construct(self)` method, and nothing else at the top level.
- MathScene and MathScene3D are already defined for you before your code runs. Do not redefine them, do not set self.camera.background_color yourself, do not draw your own grid or background.
- The ONLY imports allowed in your own code are `manim`, `numpy`, `math` -- nothing else, ever.
- Never reference: open, exec, eval, compile, __import__, os, sys, subprocess, socket, requests, shutil, globals, locals, vars, input, breakpoint, exit, quit.
- MathTex, Tex, and SingleStringMathTex are BANNED -- they crash on GPT-generated LaTeX strings. Use fm_formula() for any formula or equation.
- DecimalNumber is BANNED -- use always_redraw with plain Text() instead.
- SVGMobject is BANNED -- use fm_icon() for icons.
- MarkupText, Integer, Variable, BulletedList, Title, Paragraph, BarChart, ComplexPlane, PolarPlane are BANNED.
- Triangle() takes NO vertex arguments -- use Polygon(p1, p2, p3) for custom triangles.
- RoundedRectangle uses corner_radius, NOT radius.
- Star's first positional argument is n (number of points), not a center point.
- NEVER write inline subtraction inside wait(): compute your timing as named variables first.
- NEVER use a bare _ to reference the previous line's result -- assign every mobject to a named variable.
- ONE full-screen animate primitive per chunk -- fm_animate_bell_curve, fm_animate_vector, fm_animate_matrix etc. fill the frame on their own. Never call two of them in the same construct().
- rotate() does NOT accept run_time as a kwarg -- run_time belongs in self.play().
- point_at_angle() does NOT exist on Arc -- use arc.point_from_proportion(t).
- NEVER call fm_* functions as self.fm_* -- they are module-level functions, not Scene methods. Correct: fm_animate_number_line(self, ...). Wrong: self.fm_animate_number_line(...). This will be caught by the safety checker and rejected before rendering.
- NEVER call fm_animate_* without passing self as the FIRST positional argument. Correct: fm_animate_bar_chart(self, values, names, colors, duration). Wrong: fm_animate_bar_chart(values=values, names=names, colors=colors). The scene parameter is required and must be self.
- NEVER wrap fm_animate_* calls inside self.play() -- they handle their own animation internally. Correct: fm_animate_comparison_bars(self, ...). Wrong: self.play(fm_animate_comparison_bars(self, ...)).
- NEVER pass opacity= as a constructor kwarg to Dot, Circle, Line, Arrow or any geometry -- use .set_opacity() AFTER construction. Real failure: Dot([0,0,0], opacity=0.35) crashes. Correct: d = Dot([0,0,0]); d.set_opacity(0.35).
- plot_line_graph IS BANNED -- it crashes with color kwarg conflicts. Use fm_animate_line_chart or fm_animate_line_chart_multi instead for any trend or curve. Never call axes.plot_line_graph() under any circumstance.
- axes.get_graph() returns a ParametricFunction -- NEVER call .color= or pass color as a kwarg directly to get_graph(). Set color after: curve = axes.get_graph(func, x_range=[a,b]); curve.set_stroke(BRAND_GREEN, width=4).
- axes.get_graph() does NOT accept x_range as a kwarg -- use axes.plot(func, x_range=[a,b]) instead. axes.get_graph(func) with no x_range is also valid.
- set_stroke() does NOT accept dash_length or any dashing kwargs -- dashing is impossible via set_stroke(). Remove dash_length entirely.
- set_style() does NOT accept dash_length_ratio or dash_offset -- these kwargs do not exist on VMobject in Manim Community v0.20.1. Remove them entirely.
- interpolate_color() is BANNED -- it crashes when passed hex strings. Use ManimColor directly or pick a fixed color constant instead.
- axes.get_area() accepts x_range and bounded_graph but NOT dash_length or any stroke dash kwargs -- dashing is impossible on a filled region.
- DashedLine, DashedVMobject, Ellipse, scipy are BANNED.
- fm_animate_scatter returns (dots_group, regression_line) -- a plain tuple. NEVER access .axes on the return value. NEVER do scatter.axes.c2p(...). If you need coordinate mapping after calling fm_animate_scatter, build your own Axes separately first.
- fm_animate_bell_curve returns (curve, fill_region, label_mob) -- a plain tuple. NEVER call .get_center() or .move_to() on the return value. Do not chain methods on it.
- fm_animate_timeline returns (dots, labels) -- a plain tuple. NEVER call .move_to() on the return value.
- ALL fm_animate_* functions return tuples or None -- NEVER call .move_to(), .get_center(), .shift(), .next_to() or any Mobject method on their return values directly.
- fm_animate_bell_curve mean_label and std_label must be strings or omitted -- NEVER pass None. If you want no label just omit the argument entirely.
- fm_animate_number_line value must be a number -- NEVER pass None. If you want to show a range without a moving dot, use fm_animate_line_chart instead.
- fm_animate_line_chart end_value_label can be None or a string -- both are safe.
- fm_animate_icon_grid label_text can be "" for no label -- NEVER pass None.
- fm_card_row() does NOT accept a buff= keyword argument. Its spacing is controlled by the spacing= parameter.
- fm_two_cards() accepts an optional buff= kwarg that is silently ignored. Use spacing= to control card gap.
- fm_animate_line_chart_multi series must be a list of dicts: [{"y_values": [...], "color": BRAND_GREEN, "label": "Series A"}, ...]. NOT a list of plain lists.
- When iterating over a list of floats/numbers: for i, val in enumerate(my_list) -- NOT for i, (a, b) in enumerate(my_list) which assumes tuples.
- NEVER reference a variable on the right side of its own assignment: bar = Rectangle(...).move_to([i, bar.height/2, 0]) is an UnboundLocalError because bar doesn't exist yet. Compute bar_h first, then set bar, then move it.
- Arrow() uses start= and end= parameters, NOT left= or right=. Correct: Arrow(start=LEFT*0.5, end=RIGHT*0.5). Wrong: Arrow(left=LEFT*0.5, right=RIGHT*0.5). This will crash.
- GrowFromEdge() takes edge as the SECOND POSITIONAL argument, not a keyword. Correct: GrowFromEdge(mob, DOWN). Wrong: GrowFromEdge(mob, direction=DOWN). This will crash.
- fm_animate_number_line tick_labels must be a list of strings/numbers or None -- NEVER pass True or False.
- fm_formula() lines must be PLAIN TEXT ONLY -- no backslashes, curly braces, \varepsilon, \frac, subscript notation like X_{obs}. Write it as plain readable text: "X_obs = X_process + epsilon". fm_formula renders with Text() not LaTeX.
- VGroup() only accepts VMobject instances -- NEVER pass a plain Python list or tuple. If fm_animate_line_chart returns (axes, line, dot), unpack it: axes, line, dot = fm_animate_line_chart(self, ...).
- import random is ALLOWED in chunk code. Use it freely.
- 16:9 FRAME AWARENESS: The frame is 16 units wide × 9 units tall (frame_width=14.22, frame_height=8.0). Never position objects beyond x=±6.5 or y=±3.8. Use fm_clamp_to_frame() for complex layouts.
- DURATION RULE: ONE self.play() call or ONE fm_animate_* call per chunk. If your chunk has more than one self.play() call AND one fm_animate_* call, you are over-filling the chunk duration and it WILL drift. Pick one visual action per chunk.

=== BRAND PALETTE ===
White: "#F5F7FA". Green (positive/growth): "#38D996". Red (warning/error): "#FF4D4D". Gold (neutral highlight): "#FFD166". Gray (secondary/de-emphasized): "#8A94A6". Panel/card bg: "#0D1B2A". Background: "#060F1A".
Brand constants in scope: BRAND_WHITE, BRAND_GREEN, BRAND_RED, BRAND_GOLD, BRAND_GRAY, BRAND_PANEL, BRAND_BG, BRAND_NAVY. No others exist.

=== LIBRARY REFERENCE ===
fm_animate_vector(scene, direction, label_text, accent_color, duration, origin, scale, show_components)
  direction=[dx,dy]. Draws an Arrow from origin in that direction, labeled. show_components draws dashed x/y lines.

fm_animate_matrix(scene, rows_data, label_text, accent_color, duration, position, cell_size, font_size)
  rows_data=list of lists of strings/numbers. Draws bracket notation with cells fading in row by row.

fm_animate_bell_curve(scene, label_text, accent_color, duration, position, show_std_regions, mean_label, std_label)
  Draws a normal distribution curve with shaded 1-sigma region. show_std_regions=True by default.

fm_animate_scatter(scene, points, label_text, accent_color, duration, position, show_regression, x_label, y_label)
  points=list of (x,y) tuples. show_regression=True draws best-fit line in BRAND_RED.

fm_animate_probability_bar(scene, outcomes, label_text, accent_color, duration, position)
  outcomes=list of (label_str, probability_float). Bar heights = probability values 0-1.

fm_animate_number_line(scene, value, min_val, max_val, label_text, accent_color, duration, position, line_length, tick_labels)
  Animated dot moving to target value on a number line. tick_labels must be a list or None -- never True/False.

fm_formula(scene, lines, font_size, color, duration, position)
  lines: single string or list of strings. PLAIN TEXT ONLY -- no backslashes, no curly braces, no LaTeX syntax.
  Write readable text: "x^2 + y^2 = r^2" or "P(A|B) = P(B|A) x P(A) / P(B)".
  NEVER write LaTeX like "\varepsilon_{noise}" -- those render as literal characters.

fm_animate_counter(scene, start_val, end_val, label_text, accent_color, prefix, suffix, duration, position, value_size, label_size)
  prefix and suffix default to empty string (not "$"). Pass prefix="$" only if showing currency.

fm_animate_bar_chart(scene, values, names, colors, duration, title_text)
fm_animate_line_chart(scene, y_values, end_value_label, accent_color, x_labels, duration, title_text)
  end_value_label: string label shown at the end dot. Pass None or "" for no label.
fm_animate_line_chart_multi(scene, series, duration, title_text)
  series MUST be: [{"y_values": [1,2,3], "color": BRAND_GREEN, "label": "A"}, {"y_values": [2,3,4], "color": BRAND_GOLD, "label": "B"}]
  NOT a list of plain lists. Each dict MUST have "y_values" key.
fm_animate_gauge(scene, value, max_val, label_text, accent_color, duration, position, radius)
fm_animate_donut(scene, percentage, label_text, accent_color, duration, position)
fm_animate_comparison_bars(scene, items, duration, title_text, show_net)
fm_animate_glow_reveal(scene, text_str, accent_color, duration, font_size, subtitle, subtitle_color)
fm_animate_text_reveal(scene, lines, colors, duration, sizes)
fm_animate_icon_grid(scene, total, filled, label_text, accent_color, duration, cols, position, icon_radius)
  label_text: use "" for no label. NEVER pass None.
fm_animate_timeline(scene, events, accent_color, duration, show_index)
fm_animate_single_value(scene, value_str, label_text, accent_color, duration, value_size, label_size, sublabel, sublabel_color)
fm_animate_waterfall(scene, steps, duration)
fm_animate_bullet_chart(scene, actual, target, range_low, range_high, label_text, accent_color, duration, position, bar_length)
fm_animate_stacked_cards(scene, items, duration)
fm_card(label_text, value_text, accent_color, panel_color, text_color, label_size, value_size, buff)
fm_two_cards(left_label, left_val, left_color, right_label, right_val, right_color, ...)
fm_card_row(items, panel_color, text_color, label_size, value_size, spacing)
  items=[(label, value, color), ...]. Uses spacing= NOT buff= to control card gaps.
fm_stacked_cards(items, ...)
fm_concept_pills(labels, colors, panel_color, text_color, font_size, direction, spacing)
fm_glow_around(mobject, color, n_layers) -- returns VGroup(glow_layers, original)
fm_clamp_to_frame(*mobjects, margin_x, margin_y) -- call LAST before self.play()
fm_icon(name, size, color) -- names: sigma, integral, pi_sym, infinity, gradient, neuron, matrix_sym, derivative, dollar, coin, house, person, clock, arrow_up, arrow_down, warning, checkmark, fire

=== VISUAL VARIETY — REQUIRED ===
Every 4 consecutive chunks must use a DIFFERENT primary visual type. Bell curve → number line → scatter → bar chart is good. Bell curve → bell curve → bell curve → bell curve is unacceptable and WILL be flagged. Rotate through: Axes plots, fm_animate_vector, fm_animate_matrix, fm_animate_scatter, fm_animate_number_line, fm_animate_probability_bar, fm_animate_counter, fm_animate_comparison_bars, fm_animate_bar_chart, fm_animate_line_chart, fm_formula, fm_animate_glow_reveal. Bell curve should appear AT MOST once per 6 chunks.

=== QUALITY BAR ===
- Use Create() for curves, lines, axes. Use FadeIn() for cards, text, dots. Use GrowFromEdge() for bars. Use GrowFromCenter() for dots/icons. Use Write() for end labels.
- Fill not just outline: bars, arcs, cards must be solid fills in brand colors.
- Glow/emphasis: use fm_glow_around() or layer 2-3 concentric copies at decreasing opacity.
- Real proportionality: if a shape represents a quantity, its size must scale with that quantity.
- Vary the visual type across chunks -- if three consecutive chunks all use the same primitive, pick a different one.

=== ESTABLISHING SHOTS ===
Occasionally for a chapter-opening beat subclass MathScene3D so a hero object tilts in from an angle and settles flat. Build the hero as a single VGroup, give it a starting rotation (hero.rotate(60 * DEGREES, axis=UP)), then settle it with self.play(Rotate(hero, angle=-60 * DEGREES, axis=UP, run_time=...)). Use sparingly.

Return your response as a JSON object: {"chunks": [{"chunk_index": 0, "class_name": "Chunk0", "code": "from manim import *\\n\\nclass Chunk0(MathScene):\\n    def construct(self):\\n        ..."}, ...]}. The "code" field must be the complete, final Python source for that chunk as a single string with real newlines escaped as \\n."""

    def _build_user_prompt(batch_items):
        lines = []
        for global_idx, chunk in batch_items:
            chunk_text = " ".join(b.get("text", "") for b in chunk["beats"])
            duration = round(chunk["end_time"] - chunk["start_time"], 2)
            lines.append(
                f'Chunk {global_idx}: duration={duration}s, class_name="Chunk{global_idx}", '
                f'narration="{chunk_text}"'
            )
        return f"Topic: {topic}\n\n" + "\n".join(lines)

    def _gpt_call_for_prompt(user_prompt):
        def _do():
            return gpt4o_call(
                client,
                model="gpt-4.1",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "manim_chunks",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "chunks": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "chunk_index": {"type": "integer"},
                                            "class_name":  {"type": "string"},
                                            "code":        {"type": "string"},
                                        },
                                        "required": ["chunk_index", "class_name", "code"],
                                        "additionalProperties": False,
                                    },
                                }
                            },
                            "required": ["chunks"],
                            "additionalProperties": False,
                        },
                    },
                },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        return _call_with_retry(_do, label="Manim chunk call")

    def _parse_chunks_from_raw(raw_text: str) -> list:
        """Three-layer output parser. Layer 1: clean json.loads. Layer 2:
        JSONDecoder.raw_decode walk through the chunks array, extracting
        each object individually and stopping at the first one that fails
        rather than discarding everything. Layer 3: give up and return
        whatever was salvaged so the caller can retry missing indices."""
        raw = raw_text.strip()

        try:
            return json.loads(raw).get("chunks", [])
        except json.JSONDecodeError:
            pass

        chunks_out = []
        arr_start = raw.find('"chunks"')
        if arr_start < 0:
            return chunks_out
        bracket = raw.find('[', arr_start)
        if bracket < 0:
            return chunks_out

        decoder = json.JSONDecoder()
        pos = bracket + 1
        while pos < len(raw):
            stripped = raw[pos:]
            lstripped = stripped.lstrip(' \t\n\r,')
            if not lstripped or lstripped[0] in (']', '}'):
                break
            offset = len(stripped) - len(lstripped)
            try:
                obj, end = decoder.raw_decode(lstripped)
                if isinstance(obj, dict) and "code" in obj and "class_name" in obj:
                    chunks_out.append(obj)
                pos += offset + end
                while pos < len(raw) and raw[pos] in (' ', '\t', '\n', '\r', ','):
                    pos += 1
            except json.JSONDecodeError:
                break

        return chunks_out

    def _slot_items(returned_items, batch_global_indices):
        """Rename classes and slot parsed items into all_results by position."""
        for i, item in enumerate(returned_items):
            if i >= len(batch_global_indices):
                break
            global_idx = batch_global_indices[i]
            if global_idx >= len(chunks):
                continue
            expected_class_name = f"Chunk{global_idx}"
            raw_code = item.get("code", "") or ""
            renamed_code, _ = re.subn(
                r'^class\s+\w+\(', f'class {expected_class_name}(',
                raw_code, count=1, flags=re.MULTILINE,
            )
            item["code"] = renamed_code
            item["class_name"] = expected_class_name
            item["chunk_index"] = global_idx
            all_results[global_idx] = item

    gap_indices = {i for i, c in enumerate(chunks) if c.get("is_gap")}
    codegen_chunks = [c for i, c in enumerate(chunks) if i not in gap_indices]
    codegen_to_global = [i for i in range(len(chunks)) if i not in gap_indices]

    all_results = {i: {"is_gap": True} for i in gap_indices}
    if gap_indices:
        print(f"  ⏸  {len(gap_indices)} silence gap chunk(s) skip codegen entirely")

    chunk_batch_size = dynamic_batch_size(len(codegen_chunks), min_size=2, max_size=4)
    n_batches = max(1, math.ceil(len(codegen_chunks) / chunk_batch_size)) if codegen_chunks else 0
    print(f"  🎬 Manim chunks: {n_batches} batch(es) of ~{chunk_batch_size} chunks each...")

    for batch_idx in range(n_batches):
        local_batch = codegen_chunks[batch_idx * chunk_batch_size: (batch_idx + 1) * chunk_batch_size]
        if not local_batch:
            continue
        local_indices = list(range(batch_idx * chunk_batch_size, batch_idx * chunk_batch_size + len(local_batch)))
        batch = local_batch
        batch_global_indices = [codegen_to_global[i] for i in local_indices]

        print(f"  🎬 Manim chunk batch {batch_idx + 1}/{n_batches}: {len(batch)} chunks...")

        try:
            response = _gpt_call_for_prompt(_build_user_prompt(list(zip(batch_global_indices, batch))))
            returned = _parse_chunks_from_raw(response.choices[0].message.content)
            _slot_items(returned, batch_global_indices)
            missing = [idx for idx in batch_global_indices if idx not in all_results]
            if missing:
                print(f"  ⚠ Batch {batch_idx + 1}: parser recovered {len(returned)}/{len(batch)} chunks, retrying {len(missing)} individually...")
                for idx in missing:
                    local_i = batch_global_indices.index(idx)
                    try:
                        r2 = _gpt_call_for_prompt(_build_user_prompt([(idx, batch[local_i])]))
                        recovered = _parse_chunks_from_raw(r2.choices[0].message.content)
                        _slot_items(recovered, [idx])
                    except Exception as e2:
                        print(f"    ⚠ Single-chunk retry for Chunk{idx} failed: {e2}")
            n_done = sum(1 for idx in batch_global_indices if idx in all_results)
            print(f"  ✅ Manim chunk batch {batch_idx + 1} done: {n_done}/{len(batch)} chunks")
        except Exception as e:
            print(f"  ⚠ Manim chunk batch {batch_idx + 1} failed entirely ({e}), retrying one by one...")
            for idx, chunk in zip(batch_global_indices, batch):
                try:
                    r = _gpt_call_for_prompt(_build_user_prompt([(idx, chunk)]))
                    recovered = _parse_chunks_from_raw(r.choices[0].message.content)
                    _slot_items(recovered, [idx])
                except Exception as e2:
                    print(f"    ⚠ Single-chunk retry for Chunk{idx} failed: {e2}")

    repair_targets = []
    for idx in codegen_to_global:
        item = all_results.get(idx)
        if not item or not item.get("code"):
            continue
        ok, reason = manim_static_safety_check(item["code"])
        if not ok:
            repair_targets.append((idx, reason))

    if repair_targets:
        print(f"  🔧 {len(repair_targets)} chunk(s) failed the safety check, attempting one targeted repair each...")
        for idx, reason in repair_targets:
            print(f"    🔧 Chunk{idx} original failure: {reason}")
            local_i = codegen_to_global.index(idx)
            chunk = codegen_chunks[local_i]
            hint = safety_correction_hint(reason)
            corrected_prompt = _build_user_prompt([(idx, chunk)]) + "\n\n" + hint
            try:
                r = _gpt_call_for_prompt(corrected_prompt)
                recovered = _parse_chunks_from_raw(r.choices[0].message.content)
                if recovered:
                    candidate = recovered[0]
                    candidate_code = candidate.get("code", "") or ""
                    renamed_code, _ = re.subn(
                        r'^class\s+\w+\(', f'class Chunk{idx}(',
                        candidate_code, count=1, flags=re.MULTILINE,
                    )
                    ok2, reason2 = manim_static_safety_check(renamed_code)
                    if ok2:
                        candidate["code"] = renamed_code
                        candidate["class_name"] = f"Chunk{idx}"
                        candidate["chunk_index"] = idx
                        all_results[idx] = candidate
                        print(f"    ✅ Chunk{idx} repaired")
                    else:
                        print(f"    ⚠ Chunk{idx} repair attempt still failed (original: {reason} -- retry: {reason2}), keeping original code for the renderer to reject the same way")
                else:
                    print(f"    ⚠ Chunk{idx} repair attempt returned no parseable code, keeping original")
            except Exception as e:
                print(f"    ⚠ Chunk{idx} repair attempt errored: {e}, keeping original")

    ordered = [all_results.get(i) for i in range(len(chunks))]
    print(f"  ✅ {sum(1 for r in ordered if r)} / {len(chunks)} total manim chunks generated")
    return ordered


MATH_UNLOCKED_INTRO_QUOTES = [
    ("All models are wrong, but some are useful.", "George Box"),
    ("Statistics is the grammar of science.", "Karl Pearson"),
    ("It is easy to lie with statistics, but easier to lie without them.", "Frederick Mosteller"),
    ("Probability is the most important concept in modern science.", "Bertrand Russell"),
    ("The introduction of numbers as coordinates is an act of violence.", "Hermann Weyl"),
    ("Mathematics is the art of giving the same name to different things.", "Henri Poincare"),
    ("Pure mathematics is, in its way, the poetry of logical ideas.", "Albert Einstein"),
    ("Algebra is the offer made by the devil to the mathematician.", "Michael Atiyah"),
    ("A year spent in artificial intelligence is enough to make one believe in God.", "Alan Perlis"),
    ("The question of whether a computer can think is no more interesting than whether a submarine can swim.", "Edsger Dijkstra"),
    ("The purpose of computing is insight, not numbers.", "Richard Hamming"),
    ("An algorithm must be seen to be believed.", "Donald Knuth"),
    ("The calculus is the greatest aid we have to the appreciation of physical truth.", "W.B. Smith"),
    ("Geometry is knowledge of the eternally existent.", "Pythagoras"),
    ("Geometry is the archetype of the beauty of the world.", "Johannes Kepler"),
    ("The shortest path between two truths in the real domain passes through the complex domain.", "Jacques Hadamard"),
    ("Probability is common sense reduced to calculation.", "Pierre-Simon Laplace"),
    ("Statistical thinking will one day be as necessary for efficient citizenship as the ability to read and write.", "H.G. Wells"),
    ("The goal of statistics is to gain understanding from data.", "David Freedman"),
    ("Any sufficiently advanced technology is indistinguishable from magic.", "Arthur C. Clarke"),
    ("Intelligence is the ability to adapt to change.", "Stephen Hawking"),
    ("The measure of intelligence is the ability to change.", "Albert Einstein"),
    ("Beware of bugs in the above code; I have only proved it correct, not tried it.", "Donald Knuth"),
    ("Simplicity is the ultimate sophistication.", "Leonardo da Vinci"),
    ("First, solve the problem. Then, write the code.", "John Johnson"),
    ("We can only see a short distance ahead, but we can see plenty there that needs to be done.", "Alan Turing"),
    ("A computer would deserve to be called intelligent if it could deceive a human into believing it was human.", "Alan Turing"),
    ("Mathematics is not about numbers, equations, or algorithms. It is about understanding.", "William Paul Thurston"),
    ("The essence of mathematics lies in its freedom.", "Georg Cantor"),
    ("The only way to learn mathematics is to do mathematics.", "Paul Halmos"),
    ("A mathematician is a device for turning coffee into theorems.", "Paul Erdos"),
    ("The moving power of mathematical invention is not reasoning but imagination.", "Augustus De Morgan"),
    ("Small differences in initial conditions produce very great ones in the final phenomena.", "Henri Poincare"),
    ("Space is not simply the background to events, it is a physical entity.", "Roger Penrose"),
    ("If your experiment needs statistics, you ought to have done a better experiment.", "Ernest Rutherford"),
    ("Do not worry about your difficulties in mathematics. I can assure you mine are still greater.", "Albert Einstein"),
    ("Mathematics reveals its secrets only to those who approach it with pure love.", "Archimedes"),
    ("In mathematics the art of asking questions is more valuable than solving problems.", "Georg Cantor"),
    ("The brain is wider than the sky.", "Emily Dickinson"),
    ("It has become appallingly obvious that our technology has exceeded our humanity.", "Albert Einstein"),
    ("The question is not whether machines can think, but whether men do.", "B.F. Skinner"),
    ("Learning is not attained by chance. It must be sought for with ardor and attended to with diligence.", "Abigail Adams"),
    ("Mathematics is the language in which God has written the universe.", "Galileo Galilei"),
    ("To ask the right question is harder than to answer it.", "Georg Cantor"),
    ("Nature does not hurry, yet everything is accomplished.", "Lao Tzu"),
    ("The theory of probabilities is at bottom nothing but common sense reduced to calculus.", "Pierre-Simon Laplace"),
    ("What we cannot compute, we cannot understand.", "Richard Hamming"),
    ("The computer was born to solve problems that did not exist before.", "Bill Gates"),
    ("Imagination is more important than knowledge.", "Albert Einstein"),
    ("If people do not believe that mathematics is simple, it is only because they do not realize how complicated life is.", "John von Neumann"),
]


def render_intro_clip(w: int = 1920, h: int = 1080, fps: int = 30,
                      duration: float = 5.0) -> str:
    import random as _random
    quote_text, quote_author = _random.choice(MATH_UNLOCKED_INTRO_QUOTES)

    _t_draw  = 1.2
    _t_glow  = 0.5
    _t_quote = 0.8
    _t_fade  = 0.5
    _t_hold  = max(duration - _t_draw - _t_glow - _t_quote - _t_fade, 0.8)

    intro_code = f'''from manim import *
import math

BRAND_WHITE = "#F5F7FA"
BRAND_GOLD  = "#FFD166"
BRAND_PANEL = "#111A24"
BRAND_BG    = "#0B1628"

class MathUnlockedIntro(Scene):
    def construct(self):
        self.camera.background_color = BRAND_BG

        p1 = [-1.4, 1.4, 0]
        p2 = [1.4, 1.4, 0]
        p3 = [0.0, -1.4, 0]
        left_line  = Line(p1, p3, stroke_width=9).set_stroke(color=BRAND_GOLD, opacity=1.0)
        right_line = Line(p2, p3, stroke_width=9).set_stroke(color=BRAND_GOLD, opacity=1.0)
        top_line   = Line(p1, p2, stroke_width=9).set_stroke(color=BRAND_GOLD, opacity=1.0)
        logo = VGroup(top_line, left_line, right_line)

        glow_layers = VGroup()
        for i in range(4, 0, -1):
            gl = logo.copy()
            gl.set_stroke(color=BRAND_GOLD, width=9 + i * 6,
                          opacity=max(0.06 - i * 0.012, 0.01))
            glow_layers.add(gl)
        logo_group = VGroup(glow_layers, logo)
        logo_group.move_to(ORIGIN + UP * 0.6)

        raw_quote = {repr(quote_text)}
        raw_author = {repr(quote_author)}
        max_w = config.frame_width * 0.78

        quote_mob = Text(
            raw_quote, font_size=36, color=BRAND_WHITE,
            weight=NORMAL, line_spacing=1.4,
        )
        if quote_mob.width > max_w:
            quote_mob.scale_to_fit_width(max_w)

        author_mob = Text(
            "— " + raw_author, font_size=30, color=BRAND_GOLD,
        )
        if author_mob.width > max_w:
            author_mob.scale_to_fit_width(max_w)

        text_group = VGroup(quote_mob, author_mob).arrange(DOWN, buff=0.32)
        text_group.move_to(ORIGIN + DOWN * 2.2)

        t_draw   = {_t_draw}
        t_glow   = {_t_glow}
        t_quote  = {_t_quote}
        t_hold   = {_t_hold}
        t_fade   = {_t_fade}

        self.play(
            LaggedStart(
                Create(top_line),
                Create(left_line),
                Create(right_line),
                lag_ratio=0.25,
            ),
            run_time=t_draw, rate_func=smooth,
        )
        self.play(FadeIn(glow_layers), run_time=t_glow, rate_func=smooth)
        self.play(
            LaggedStart(
                FadeIn(quote_mob, shift=UP * 0.18),
                FadeIn(author_mob, shift=UP * 0.12),
                lag_ratio=0.35,
            ),
            run_time=t_quote, rate_func=smooth,
        )
        self.wait(t_hold)
        self.play(
            FadeOut(VGroup(logo_group, text_group)),
            run_time=t_fade, rate_func=smooth,
        )
'''

    os.makedirs(MANIM_CHUNK_CACHE_DIR, exist_ok=True)
    intro_path = os.path.join(MANIM_CHUNK_CACHE_DIR, "intro_clip.mp4")
    script_path = os.path.join(MANIM_CHUNK_CACHE_DIR, "intro_scene.py")

    with open(script_path, "w") as f:
        f.write(intro_code)

    cmd = [
        "manim", "render", script_path, "MathUnlockedIntro",
        "--format=mp4",
        f"--resolution={w},{h}",
        f"--frame_rate={fps}",
        "--media_dir", MANIM_CHUNK_CACHE_DIR,
        "--output_file", "intro_clip",
        "-q", "h",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        print(f"  ⚠ Intro clip render failed: {result.stderr.decode(errors='replace')[-300:]}")
        return None

    for root, dirs, files in os.walk(MANIM_CHUNK_CACHE_DIR):
        for fname in files:
            if fname == "intro_clip.mp4" and root != MANIM_CHUNK_CACHE_DIR:
                found = os.path.join(root, fname)
                shutil.copy(found, intro_path)
                return intro_path

    if os.path.exists(intro_path):
        return intro_path

    print("  ⚠ Intro clip file not found after render")
    return None


def concat_manim_clips(clip_paths: list, output_path: str) -> str:
    """Concatenates every rendered (or filler) chunk clip into one
    continuous silent video via ffmpeg concat. Every clip going into
    this -- both real Manim renders and dashboard fillers -- is encoded
    with the same libx264/yuv420p/fps settings, so a fast stream-copy
    concat should always apply; the re-encode path only exists as a
    safety net in case any clip's stream parameters drifted."""
    concat_list_path = output_path + "_concat_list.txt"
    with open(concat_list_path, "w") as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")

    cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
           '-i', concat_list_path, '-c', 'copy', output_path]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        cmd_reencode = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                         '-i', concat_list_path,
                         '-c:v', 'libx264', '-preset', ENCODE_PRESET, '-crf', ENCODE_CRF,
                         '-tune', 'animation', '-pix_fmt', 'yuv420p',
                         '-movflags', '+faststart', output_path]
        result2 = subprocess.run(cmd_reencode, capture_output=True)
        if result2.returncode != 0:
            raise Exception(f"chunk concat failed: {result2.stderr.decode(errors='replace')[-300:]}")

    try:
        os.remove(concat_list_path)
    except Exception:
        pass
    return output_path


def _render_or_fill_one_chunk(chunk_index: int, chunk: dict, item, w: int, h: int, fps: int) -> tuple:
    target_duration = round(max(chunk["end_time"] - chunk["start_time"], 0.05), 3)
    fallback_path = os.path.join(MANIM_CHUNK_CACHE_DIR, f"filler_{chunk_index:04d}_{target_duration:.3f}.mp4")

    if not item or not item.get("code") or not item.get("class_name"):
        _make_chunk_fallback(fallback_path, chunk, target_duration, w, h, fps, "no chunk code generated")
        return chunk_index, fallback_path, "no chunk code generated"

    clip_path, err = render_manim_chunk(item["code"], item["class_name"],
                                         target_duration, w, h, fps)
    if not clip_path:
        _make_chunk_fallback(fallback_path, chunk, target_duration, w, h, fps, err)
        return chunk_index, fallback_path, err
    return chunk_index, clip_path, ""


def render_all_manim_chunks(chunks: list, chunk_code_list: list, w: int = 1920,
                             h: int = 1080, fps: int = 30, max_workers: int = None) -> list:
    """Renders every CONTENT chunk's Manim code in parallel via a
    process pool, each chunk being its own independent `manim` CLI
    subprocess. Any content chunk that fails safety check, crashes,
    times out, or drifts past MANIM_CHUNK_MAX_DRIFT_RATIO gets a
    dashboard-color filler of its exact target duration instead, so
    total video length always matches the audio regardless of how
    many chunks failed.

    Gap (silence) chunks are deliberately excluded from this parallel
    pass and handled afterward in a sequential second pass (see the
    end of this function) -- a gap chunk needs to read the ACTUAL
    final frame of the content chunk immediately before it, which is
    only guaranteed to exist once the parallel pass above has fully
    finished, since parallel workers can complete in any order.

    Default worker count is deliberately smaller than
    prerender_all_beat_visuals' cpu_count()-1 pattern: each worker here
    spawns a full `manim` subprocess (its own Python interpreter, Cairo
    renderer, and an ffmpeg mux at the end), which is much heavier per
    worker than the old in-process PIL beat renderer, so a full
    cpu_count()-1 pool risks thrashing the machine rather than helping."""
    os.makedirs(MANIM_CHUNK_CACHE_DIR, exist_ok=True)
    n = len(chunks)
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() // 2)
    clip_paths = [None] * n
    content_indices = [i for i in range(n) if not chunks[i].get("is_gap")]
    gap_indices = [i for i in range(n) if chunks[i].get("is_gap")]
    print(f"  🎬 Rendering {len(content_indices)} content chunk(s) across up to {max_workers} parallel workers...")

    import time as _render_time
    _render_start = _render_time.time()

    def _fmt_ts(seconds):
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"

    def _chunk_video_ts(chunk_idx):
        try:
            t = float(chunks[chunk_idx].get("start_time", 0.0))
            return f"~{_fmt_ts(t)} in video"
        except Exception:
            return ""

    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_render_or_fill_one_chunk, i, chunks[i],
                        chunk_code_list[i] if i < len(chunk_code_list) else None, w, h, fps): i
            for i in content_indices
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                idx, path, err = future.result()
            except Exception as e:
                target_duration = round(max(chunks[i]["end_time"] - chunks[i]["start_time"], 0.05), 3)
                fallback_path = os.path.join(MANIM_CHUNK_CACHE_DIR, f"filler_{i:04d}_{target_duration:.3f}.mp4")
                _make_chunk_fallback(fallback_path, chunks[i], target_duration, w, h, fps, f"worker exception: {e}")
                idx, path, err = i, fallback_path, f"worker exception: {e}"

            clip_paths[idx] = path
            done += 1
            elapsed = _render_time.time() - _render_start
            if err:
                vts = _chunk_video_ts(idx)
                print(f"  ⚠ Chunk {idx} ({vts}): {err[:160]} -- filler [{done}/{len(content_indices)}] | elapsed {_fmt_ts(elapsed)}")
            if done % 10 == 0 or done == len(content_indices):
                print(f"  ⚙️  Manim chunk progress: {done}/{len(content_indices)} | elapsed {_fmt_ts(elapsed)}")

    if gap_indices:
        print(f"  ⏸  Filling {len(gap_indices)} silence gap(s) by holding the previous frame...")
        held = 0
        for i in gap_indices:
            target_duration = round(max(chunks[i]["end_time"] - chunks[i]["start_time"], 0.05), 3)
            gap_path = os.path.join(MANIM_CHUNK_CACHE_DIR, f"gap_{i:04d}_{target_duration:.3f}.mp4")
            prev_path = clip_paths[i - 1] if i > 0 else None
            held_path = None
            if prev_path and os.path.exists(prev_path):
                held_path = _make_held_frame_filler(prev_path, gap_path, target_duration, w, h, fps)
            if held_path:
                clip_paths[i] = held_path
                held += 1
            else:
                fallback_path = os.path.join(MANIM_CHUNK_CACHE_DIR, f"filler_{i:04d}_{target_duration:.3f}.mp4")
                _make_dashboard_filler(fallback_path, target_duration, w, h, fps)
                clip_paths[i] = fallback_path
        print(f"  ✅ {held}/{len(gap_indices)} silence gap(s) held on previous frame, {len(gap_indices) - held} used blank fallback")

    print(f"  ✅ All {n} chunks accounted for ({sum(1 for p in clip_paths if p)} clips ready)")
    return clip_paths


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Math Unlocked (Manim renderer) on :{port} | Key: {'set' if OPENAI_API_KEY else 'MISSING'}")
    uvicorn.run(app, host="0.0.0.0", port=port)