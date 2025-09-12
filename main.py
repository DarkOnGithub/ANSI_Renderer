import os
import sys
import argparse
import time
import threading
import queue
import signal

import cv2

import torch

from config import Config, DEVICE, RESET_VALS, SHOW_CURSOR
from utils import write_all, setup_lookup, writer_task
from video_processing import process_frame
from ansi_generator import ansi_generate
from video_capture import read_video_from_file, read_video_from_screen
from typing import Optional, Tuple


exit_event = None
writer_thread = None
prod_thread = None


def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) signal"""
    print("\nInterrupted by user. Cleaning up...", file=sys.stderr)
    if exit_event:
        exit_event.set()

    write_all(sys.stdout.fileno(), bytes(RESET_VALS))
    write_all(sys.stdout.fileno(), SHOW_CURSOR)
    sys.exit(0)


def frame_producer(path: Optional[str], capture_screen: bool, cfg: Config,
                   lookup_vals: torch.Tensor, lookup_lens: torch.Tensor,
                   out_queue: queue.Queue,
                   region: Optional[Tuple[int,int,int,int]], fps: int,
                   color_mode: str, exit_event: threading.Event, args: argparse.Namespace) -> None:
    prev_gray = None
    source = None
    current_cursor_pos = (0, 0)
    stream = torch.cuda.Stream(device=cfg.device)

    # Memory optimization: pre-allocate commonly used tensors
    prev_gray = None
    skipped_frames = 0
    max_skip_frames = 3  # Skip up to 3 identical frames
    frame_similarity_threshold = 0.95  # 95% similarity to skip frame

    # Adaptive diff threshold parameters
    adaptive_diff_enabled = args.adaptive_diff
    base_diff_thresh = args.diff_thresh

    try:
        if path:
            source = read_video_from_file(path, cfg, stream, exit_event)
        else:
            source = read_video_from_screen(cfg, stream, region, fps, exit_event)
        i = 0
        # Prefetch next frame for parallel processing
        rgb_iter = iter(source)
        try:
            rgb = next(rgb_iter)
            next_rgb = next(rgb_iter)
        except StopIteration:
            next_rgb = None

        while rgb is not None:
            if exit_event.is_set():
                break

            # Convert to grayscale for frame comparison (before full processing)
            current_gray = (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]).round_().to(torch.uint8)

            # Frame skipping logic for very similar frames
            should_skip = False
            if prev_gray is not None and skipped_frames < max_skip_frames:
                # Calculate frame similarity
                diff = (current_gray - prev_gray).abs()
                similarity = 1.0 - (diff.float().mean() / 255.0)
                if similarity > frame_similarity_threshold:
                    should_skip = True
                    skipped_frames += 1
                else:
                    skipped_frames = 0

            if should_skip:
                # Send empty frame to maintain timing
                empty_ansi = torch.tensor(RESET_VALS, dtype=torch.uint8, device=cfg.device)
                out_queue.put(empty_ansi.cpu())
                prev_gray = current_gray  # Update prev_gray for next comparison
                i += 1

                # Advance to next frame with prefetching
                rgb = next_rgb
                try:
                    next_rgb = next(rgb_iter)
                except StopIteration:
                    next_rgb = None

                if rgb is None:
                    break
                continue

            # Adaptive diff threshold calculation
            current_diff_thresh = base_diff_thresh
            if adaptive_diff_enabled and prev_gray is not None:
                # Calculate frame complexity (variance) to adjust threshold
                if prev_gray.numel() > 0:
                    variance = torch.var(prev_gray.float())
                    # Higher variance = more detail = lower threshold for better quality
                    # Lower variance = less detail = higher threshold for more compression
                    complexity_factor = min(1.0, variance / 1000.0)  # Normalize variance
                    current_diff_thresh = int(base_diff_thresh * (0.5 + complexity_factor))

            processing_start = time.perf_counter()
            with torch.cuda.stream(stream):
                xs, ys, cols, gray = process_frame(prev_gray, rgb, current_diff_thresh, color_mode)
                prev_gray = gray
                ansi, current_cursor_pos = ansi_generate(xs, ys, cols, lookup_vals, lookup_lens, cfg, current_cursor_pos, color_mode)
                # Move to CPU asynchronously to overlap with next frame processing
                ansi_cpu = ansi.cpu()
            stream.synchronize()
            processing_time = time.perf_counter() - processing_start

            out_queue.put(ansi_cpu)
            skipped_frames = 0  # Reset skip counter on successful frame
            i += 1

            # if i <= 5 or i % 100 == 0:
            # print(f"Frame {i}: Processing time: {processing_time:.4f}s; fps: {1/processing_time:.2f}", file=sys.stderr)

            # Advance to next frame with prefetching
            rgb = next_rgb
            try:
                next_rgb = next(rgb_iter)
            except StopIteration:
                next_rgb = None

            if rgb is None:
                break
    except Exception as e:
        print(f"\nError in frame producer: {e}", file=sys.stderr)
    finally:
        out_queue.put(None)


def main() -> None:
    global exit_event, writer_thread, prod_thread

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="ANSI video terminal with buffering")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-v', '--video', help='Path to video file')
    group.add_argument('-s', '--screen', action='store_true', help='Capture screen')
    parser.add_argument('-r', '--region', help='x,y,w,h for screen capture', default=None)
    parser.add_argument('-f', '--fps', type=int, default=60)
    parser.add_argument('--diff-thresh', type=int, default=0, help='Base pixel difference threshold for delta encoding (higher = more compression)')
    parser.add_argument('--adaptive-diff', action='store_true', default=True, help='Use adaptive diff threshold based on frame content')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--queue-size', type=int, default=100)
    parser.add_argument('--colors', choices=['256', 'full'], default='full',
                       help='Color mode: 256 for ANSI 256-color palette, full for RGB (default: full)')
    args = parser.parse_args()

    cols, rows = os.get_terminal_size() if sys.stdout.isatty() else (80, 24)
    cols, rows = 1280, 720
    # cols, rows = 1920, 1080
    cfg = Config(width=cols-1, height=rows-1, device=DEVICE,
                 gpu_id=args.gpu_id, diff_thresh=args.diff_thresh)

    fps = args.fps
    if args.video:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()  

    lookup_vals, lookup_lens = setup_lookup(max(cfg.width+1, cfg.height+1, 256), DEVICE)
    out_q = queue.Queue(maxsize=max(1, args.queue_size))

    # Memory optimization: pre-allocate CUDA stream and enable memory pooling
    torch.cuda.set_per_process_memory_fraction(0.8)  # Reserve some memory for system
    torch.cuda.empty_cache()  # Clear any existing cache

    # Enable pinned memory for faster CPU-GPU transfers
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.gpu_id)

    exit_event = threading.Event()
    writer_thread = threading.Thread(target=writer_task, args=(out_q, fps, exit_event), daemon=True)
    writer_thread.start()
    region = tuple(map(int, args.region.split(','))) if args.region else None
    prod_thread = threading.Thread(
        target=frame_producer,
        args=(args.video, args.screen, cfg, lookup_vals, lookup_lens,
              out_q, region, args.fps, args.colors, exit_event, args), daemon=True)
    prod_thread.start()

    try:
        while True:
            time.sleep(1)  
    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...", file=sys.stderr)
        exit_event.set()
        write_all(sys.stdout.fileno(), bytes(RESET_VALS))
        write_all(sys.stdout.fileno(), SHOW_CURSOR)
        sys.exit(0)

if __name__ == '__main__':
    main()
