import os
import os
import subprocess
import traceback
from typing import Generator

import cv2 as cv
import numpy as np
import torch

from src.ansi_renderer import AnsiRenderer
from src.config import Config

DEFAULT_VIDEO_PATH = (
    r"(Extreme Demon) ''Tidal Wave'' by OniLinkGD ｜ Geometry Dash [YbyfDYChIYU].mp4"
)
TIMING_FILE = "timing_object.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_video_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


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
        try:
            fps = cap.get(cv.CAP_PROP_FPS)
        finally:
            cap.release()
        return fps if fps > 0 else 30.0


def get_target_render_size() -> tuple[int, int]:
    return 1280, 720
    try:
        term_size = os.get_terminal_size()
        cols = max(80, term_size.columns - 1)
        rows = max(24, term_size.lines - 1)
    except OSError:
        cols, rows = 80, 24

    return cols * 2, rows * 4


def ffmpeg_frame_generator(
    path: str, device: torch.device
) -> Generator[torch.Tensor, None, None]:
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
            yield torch.from_numpy(frame_np).to(device)
    finally:
        proc.terminate()
        proc.wait()


def get_frame_generator(
    path: str, device: torch.device
) -> Generator[torch.Tensor, None, None]:
    cap = open_software_capture(path)
    if not cap.isOpened():
        yield from ffmpeg_frame_generator(path, device)
        return

    ret, frame = cap.read()
    if not ret:
        cap.release()
        yield from ffmpeg_frame_generator(path, device)
        return

    try:
        frame_rgb = torch.from_numpy(frame[:, :, [2, 1, 0]])
        yield frame_rgb.to(device)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = torch.from_numpy(frame[:, :, [2, 1, 0]])
            yield frame_rgb.to(device)
    finally:
        cap.release()


def build_config(video_path: str) -> Config:
    width, height = get_target_render_size()
    fps = get_video_fps(video_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return Config(
        width=width,
        height=height,
        device=device,
        fps=fps,
        audio_path=video_path,
        render_mode="quadrant",
        quadrant_cell_divisor=2,
        quant_mask=0xFF,
        diff_thresh=0,
        run_color_diff_thresh=0,
        adaptive_quality=False,
        adaptive_quant_masks=(0xFF,),
        adaptive_diff_thresh_offsets=(0,),
        adaptive_run_color_diff_offsets=(0,),
        adaptive_ema_alpha=0.12,
        target_frame_bytes=0,
        frame_byte_buffer_frames=8,
        max_frame_bytes=0,
        relative_cursor_moves=True,
        use_rep=False,
        rep_min_run=12,
        sync_output=True,
        prefer_writev=True,
        write_chunk_size=2_097_152,
        queue_size=12,
        buffer_pool_size=14,
        initial_buffer_size=16 * 1024 * 1024,
        async_copy_stream=True,
        pacing_render_lead=True,
        pacing_render_alpha=0.18,
        timing_enabled=True,
        timing_file=TIMING_FILE,
    )


def main() -> None:
    video_path = resolve_video_path(DEFAULT_VIDEO_PATH)
    config = build_config(video_path)
    renderer = AnsiRenderer(
        frame_generator=get_frame_generator(video_path, config.device),
        config=config,
    )

    try:
        for ansi, frame_idx in renderer.get_next_ansi_sequence():
            renderer.render_frame(ansi, frame_idx)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    except Exception as exc:
        print(f"Main thread caught exception: {exc}")
        traceback.print_exc()
        print("Program exiting due to thread crash")


if __name__ == "__main__":
    main()
