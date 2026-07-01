# ============================================================
# CHARACTER MOTION TEST - FastAPI service
# Same pattern as Vaults v3 (background task -> /status -> /download).
# Renders the 4 AI-generated expression frames (character_test/*.png)
# with code-driven idle motion (breathing/sway/rotation) + crossfade
# between expressions on a fixed beat schedule.
# ============================================================

import os
import math
import subprocess
import traceback
from datetime import datetime

import numpy as np
import cv2
from PIL import Image, ImageFilter

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Character Motion Test")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

current_job = {"status": "idle", "progress": 0, "output": None, "error": None, "started_at": None}

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "character_test")
OUTPUT_FILE = os.path.join(BASE_DIR, "character_motion_output.mp4")

W, H = 1280, 720
FPS = 30
DURATION = 12.0
N_FRAMES = int(DURATION * FPS)

BEATS = [
    (0.0, 3.0, "neutral"),
    (3.0, 6.0, "happy"),
    (6.0, 9.0, "surprised"),
    (9.0, 12.0, "turned"),
]
CROSSFADE = 0.25  # seconds

# Mouth/lower-face edit region on 01_neutral.png (1024x1024) -- used to
# generate an inpaint mask for the image-edit API so "talk" mouth
# variants stay locked to the base character.
MOUTH_MASK_BOX = (380, 440, 640, 620)  # x0, y0, x1, y1
MOUTH_MASK_FEATHER = 8
MOUTH_MASK_FILE = "mouth_mask.png"

# Procedural talking-mouth overlay -- used until AI-generated mouth
# variants exist. Only applied to expressions with a closed/neutral
# mouth (neutral, turned); happy/surprised already have an open mouth
# baked into the generated art.
MOUTH_CENTER = (510, 503)
MOUTH_HALF_WIDTH = 36
MOUTH_MIN_HEIGHT = 5
MOUTH_MAX_HEIGHT = 46
MOUTH_TALK_THRESHOLD = 0.05
TALKING_EXPRESSIONS = ("neutral", "turned")


# ── asset loading ───────────────────────────────────────────────
def load_rgba(path):
    im = Image.open(path).convert("RGBA")
    arr = np.array(im)
    if arr[:, :, 3].min() == 255:
        rgb = arr[:, :, :3].astype(np.int16)
        near_white = np.all(rgb > 245, axis=2)
        arr[near_white, 3] = 0
    return arr


def load_frames():
    return {
        "neutral":   load_rgba(os.path.join(ASSETS_DIR, "01_neutral.png")),
        "happy":     load_rgba(os.path.join(ASSETS_DIR, "02_happy.png")),
        "surprised": load_rgba(os.path.join(ASSETS_DIR, "03_surprised.png")),
        "turned":    load_rgba(os.path.join(ASSETS_DIR, "04_turned.png")),
    }


# ── mouth-edit mask (for image-edit/inpaint API: alpha=0 = edit here) ──
def generate_mouth_mask():
    base = Image.open(os.path.join(ASSETS_DIR, "01_neutral.png")).convert("RGBA")
    w, h = base.size
    x0, y0, x1, y1 = MOUTH_MASK_BOX

    mask = np.full((h, w), 255, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 0
    mask_img = Image.fromarray(mask, mode="L").filter(
        ImageFilter.GaussianBlur(MOUTH_MASK_FEATHER))

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 3] = np.array(mask_img)
    out_path = os.path.join(ASSETS_DIR, MOUTH_MASK_FILE)
    Image.fromarray(rgba, mode="RGBA").save(out_path)
    return out_path


# ── beat lookup ─────────────────────────────────────────────────
def active_pair(t):
    for i, (start, end, name) in enumerate(BEATS):
        if start <= t < end:
            if t - start < CROSSFADE and i > 0:
                prev_name = BEATS[i - 1][2]
                blend = (t - start) / CROSSFADE
                return prev_name, name, blend
            return name, name, 0.0
    return BEATS[-1][2], BEATS[-1][2], 0.0


# ── background (simple gradient, no extra assets) ─────────────────
def make_background():
    top    = np.array([28, 22, 16], dtype=np.float32)   # dark warm brown (BGR)
    bottom = np.array([70, 48, 30], dtype=np.float32)   # lighter warm brown (BGR)
    grad = np.linspace(0, 1, H, dtype=np.float32).reshape(H, 1, 1)
    bg = top.reshape(1, 1, 3) * (1 - grad) + bottom.reshape(1, 1, 3) * grad
    return np.repeat(bg, W, axis=1).astype(np.uint8)


# ── idle motion (continuous, every frame) ──────────────────────────
def idle_transform(t):
    breathe = math.sin(2 * math.pi * t / 2.6)
    scale = 1.0 + 0.015 * breathe
    dy = -6 * breathe
    dx = 4 * math.sin(2 * math.pi * t / 4.3 + 1.0)
    angle = 1.5 * math.sin(2 * math.pi * t / 5.0)
    return scale, dx, dy, angle


# ── procedural talking mouth (placeholder until AI talk-frame edits exist) ─
def talk_amplitude(t):
    """0..1 'how open is the mouth right now' -- sum of a few sines at
    speech-like rates, clamped to >=0 so there are natural closed gaps
    between syllable bursts instead of a smooth breathing-style wave."""
    raw = (math.sin(2 * math.pi * t * 5.5)
           + 0.6 * math.sin(2 * math.pi * t * 8.3 + 1.7)
           + 0.4 * math.sin(2 * math.pi * t * 12.1 + 0.4))
    return max(0.0, raw) / 2.0


def apply_talking_mouth(frame, amplitude):
    """Draw an open-mouth ellipse over a closed-mouth frame. frame is
    HxWx4 uint8 RGBA. Returns a modified copy."""
    if amplitude < MOUTH_TALK_THRESHOLD:
        return frame
    out = frame.copy()
    half_h = max(MOUTH_MIN_HEIGHT,
                  int(MOUTH_MIN_HEIGHT + (MOUTH_MAX_HEIGHT - MOUTH_MIN_HEIGHT) * amplitude))
    cx, cy = MOUTH_CENTER
    axes = (MOUTH_HALF_WIDTH, half_h)
    cv2.ellipse(out, (cx, cy), axes, 0, 0, 360, (90, 40, 45, 255), -1)   # mouth interior
    cv2.ellipse(out, (cx, cy), axes, 0, 0, 360, (25, 15, 18, 255), 3)    # outline
    return out


# ── frame compositing ──────────────────────────────────────────────
def render_frame(t, frames, bg, char_w, char_h, base_x, base_y):
    name_a, name_b, blend = active_pair(t)

    frame_a = frames[name_a]
    if name_a in TALKING_EXPRESSIONS:
        frame_a = apply_talking_mouth(frame_a, talk_amplitude(t))

    if blend == 0.0:
        char = frame_a.astype(np.float32)
    else:
        frame_b = frames[name_b]
        if name_b in TALKING_EXPRESSIONS:
            frame_b = apply_talking_mouth(frame_b, talk_amplitude(t))
        char = frame_a.astype(np.float32) * (1 - blend) + frame_b.astype(np.float32) * blend

    scale, dx, dy, angle = idle_transform(t)
    size = int(char_w * scale)

    resized = cv2.resize(char, (size, size), interpolation=cv2.INTER_LINEAR)
    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    rotated = cv2.warpAffine(resized, M, (size, size),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    frame = bg.copy()
    x0 = base_x + (char_w - size) // 2 + int(dx)
    y0 = base_y + (char_h - size) // 2 + int(dy)

    src_x0, src_y0 = max(0, -x0), max(0, -y0)
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1 = min(W, x0 + size)
    dst_y1 = min(H, y0 + size)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return frame

    sprite_rgb = rotated[src_y0:src_y1, src_x0:src_x1, :3][:, :, ::-1]  # RGB->BGR
    sprite_a = rotated[src_y0:src_y1, src_x0:src_x1, 3:4] / 255.0

    roi = frame[dst_y0:dst_y1, dst_x0:dst_x1, :].astype(np.float32)
    blended = sprite_rgb * sprite_a + roi * (1 - sprite_a)
    frame[dst_y0:dst_y1, dst_x0:dst_x1, :] = blended.astype(np.uint8)
    return frame


# ── full render ────────────────────────────────────────────────────
def render_character_video(output_path: str, progress_cb=None):
    frames = load_frames()
    bg = make_background()

    char_h = int(H * 0.95)
    char_w = char_h  # source frames are square
    base_x = (W - char_w) // 2
    base_y = H - char_h + int(H * 0.06)

    raw_path = output_path.replace(".mp4", "_raw.mp4")
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    for f in range(N_FRAMES):
        t = f / FPS
        frame = render_frame(t, frames, bg, char_w, char_h, base_x, base_y)
        writer.write(frame)
        if progress_cb and f % 30 == 0:
            progress_cb(10 + int(85 * f / N_FRAMES))
    writer.release()

    subprocess.run([
        "ffmpeg", "-y", "-i", raw_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", output_path
    ], check=True, capture_output=True)
    os.remove(raw_path)


# ── background task ────────────────────────────────────────────────
def process_character_test():
    global current_job
    try:
        current_job["progress"] = 5
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)

        def cb(p):
            current_job["progress"] = p

        render_character_video(OUTPUT_FILE, progress_cb=cb)

        if not os.path.exists(OUTPUT_FILE):
            raise Exception("Output missing")

        current_job.update({"status": "completed", "progress": 100, "output": OUTPUT_FILE})
        print(f"\n✅ DONE: {OUTPUT_FILE}")

    except Exception as e:
        current_job.update({"status": "error", "error": str(e), "progress": 0})
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()


# ── FastAPI routes (same shape as Vaults v3) ──────────────────────
@app.get("/")
def root():
    return {"service": "Character Motion Test", "status": "running",
            "assets_dir": ASSETS_DIR}

@app.post("/generate-mouth-mask")
def generate_mouth_mask_endpoint():
    path = generate_mouth_mask()
    return {"status": "ok", "mask_path": path, "box": MOUTH_MASK_BOX}

@app.get("/mouth-mask")
def get_mouth_mask():
    path = os.path.join(ASSETS_DIR, MOUTH_MASK_FILE)
    if not os.path.exists(path):
        generate_mouth_mask()
    return FileResponse(path, media_type="image/png", filename=MOUTH_MASK_FILE)

@app.post("/generate")
async def generate(background_tasks: BackgroundTasks):
    global current_job
    if current_job["status"] == "processing":
        return {"message": "Already processing", "status": "processing"}
    current_job = {"status": "processing", "progress": 0, "output": None,
                   "error": None, "started_at": datetime.now().isoformat()}
    background_tasks.add_task(process_character_test)
    return {"message": "Started", "status": "processing"}

@app.get("/status")
def check_status():
    return {**current_job, "ready": current_job["status"] == "completed"}

@app.get("/download")
def download():
    if current_job["status"] != "completed":
        raise HTTPException(400, f"Not ready: {current_job['status']}")
    if not current_job["output"] or not os.path.exists(current_job["output"]):
        raise HTTPException(404, "File not found")
    return FileResponse(current_job["output"], media_type="video/mp4",
                         filename="character_motion_test.mp4")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    print(f"🚀 Character Motion Test on :{port} | assets: {ASSETS_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=port)