import math
import torch
from src.ansi_renderer import AnsiRenderer
import cv2 as cv
import os
import subprocess
import numpy as np
from src.config import Config

video_path = r"/home/user/python/ANSI_Renderer/SLAUGHTERHOUSE - FULL [UNNERFED] [7W5bZJY2IPI].mp4"


def open_software_capture(path: str) -> cv.VideoCapture:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel;none")
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "quiet")

    cap = cv.VideoCapture(path, cv.CAP_FFMPEG)
    if hasattr(cv, "CAP_PROP_HW_ACCELERATION") and hasattr(
        cv, "VIDEO_ACCELERATION_NONE"
    ):
        cap.set(cv.CAP_PROP_HW_ACCELERATION, cv.VIDEO_ACCELERATION_NONE)
    return cap


def probe_video_stream(path: str) -> tuple[int, int, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd, text=True).strip().splitlines()
    width = int(out[0])
    height = int(out[1])
    rate_text = out[2]
    if "/" in rate_text:
        num, den = rate_text.split("/", 1)
        fps = float(num) / max(float(den), 1.0)
    else:
        fps = float(rate_text)
    return width, height, fps


def get_video_fps(path: str) -> float:
    try:
        _, _, fps = probe_video_stream(path)
        return fps
    except Exception:
        cap = open_software_capture(path)
        fps = cap.get(cv.CAP_PROP_FPS)
        cap.release()
        return fps


def get_target_render_size() -> tuple[int, int]:
    try:
        term_size = os.get_terminal_size()
        cols = max(80, term_size.columns - 1)
        rows = max(24, term_size.lines - 1)
    except OSError:
        cols, rows = 160, 48

    return cols, rows


def ffmpeg_frame_generator(path: str):
    width, height, _ = probe_video_stream(path)
    frame_bytes = width * height * 3
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        if proc.stdout is None:
            return
        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            frame_np = (
                np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
            )
            frame_rgb = torch.from_numpy(frame_np)
            yield frame_rgb.to(torch.device("cuda"))
    finally:
        proc.terminate()
        proc.wait()


def get_frame_generator(video_path: str):
    cap = open_software_capture(video_path)
    if not cap.isOpened():
        yield from ffmpeg_frame_generator(video_path)
        return

    ret, frame = cap.read()
    if not ret:
        cap.release()
        yield from ffmpeg_frame_generator(video_path)
        return

    frame_rgb = torch.from_numpy(frame[:, :, [2, 1, 0]]).to(torch.device("cuda"))
    yield frame_rgb

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = torch.from_numpy(frame[:, :, [2, 1, 0]]).to(
            torch.device("cuda")
        )  # Keep as (H, W, C)
        yield frame_rgb
    cap.release()


fps = get_video_fps(video_path)
config = Config(
    width=1280,
    height=720,
    device=torch.device("cuda"),
    fps=fps,
    audio_path=video_path,
    render_mode="octant",
    quadrant_cell_divisor=2,
    color_mode="truecolor",
    quant_mask=0xF8,
    diff_thresh=12,
    run_color_diff_thresh=12,
    adaptive_quality=True,
    adaptive_quant_masks=(0xF8, 0xF0, 0xE0),
    adaptive_diff_thresh_offsets=(0, 4, 10),
    adaptive_run_color_diff_offsets=(0, 6, 14),
    adaptive_ema_alpha=0.15,
    target_frame_bytes=3_000_000,
    frame_byte_buffer_frames=8,
    max_frame_bytes=3_500_000,
    relative_cursor_moves=True,
    use_rep=True,
    rep_min_run=4,
    sync_output=True,
    prefer_writev=True,
    write_chunk_size=2_097_152,
    queue_size=8,
    buffer_pool_size=10,
    initial_buffer_size=12 * 1024 * 1024,
    async_copy_stream=True,
    pacing_render_lead=True,
    pacing_render_alpha=0.25,
    timing_enabled=False,
    timing_file="timing_object.csv",
)
renderer = AnsiRenderer(frame_generator=get_frame_generator(video_path), config=config)

try:
    for ansi, frame_idx in renderer.get_next_ansi_sequence():
        renderer.render_frame(ansi, frame_idx)
except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting...")
except Exception as e:
    print(f"Main thread caught exception: {e}")
    import traceback

    traceback.print_exc()
    print("Program exiting due to thread crash")
