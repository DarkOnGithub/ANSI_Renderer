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
                   color_mode: str, exit_event: threading.Event) -> None:
    prev_gray = None
    source = None
    current_cursor_pos = (0, 0)  
    stream = torch.cuda.Stream(device=cfg.device)

    try:
        if path:
            source = read_video_from_file(path, cfg, stream, exit_event)
        else:
            source = read_video_from_screen(cfg, stream, region, fps, exit_event)
        i = 0
        for rgb in source:
            if exit_event.is_set():
                break

            processing_start = time.perf_counter()
            with torch.cuda.stream(stream):
                xs, ys, cols, gray = process_frame(prev_gray, rgb, cfg.diff_thresh, color_mode)
                prev_gray = gray
                ansi, current_cursor_pos = ansi_generate(xs, ys, cols, lookup_vals, lookup_lens, cfg, current_cursor_pos, color_mode)
            stream.synchronize()
            processing_time = time.perf_counter() - processing_start

            out_queue.put(ansi.cpu())
            i += 1

            if i <= 5 or i % 100 == 0:
                print(f"Frame {i}: Processing time: {processing_time:.4f}s", file=sys.stderr)
            if exit_event.is_set():
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
    parser.add_argument('--diff-thresh', type=int, default=0)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--queue-size', type=int, default=100)
    parser.add_argument('--colors', choices=['256', 'full'], default='full',
                       help='Color mode: 256 for ANSI 256-color palette, full for RGB (default: full)')
    args = parser.parse_args()

    cols, rows = os.get_terminal_size() if sys.stdout.isatty() else (80, 24)
    cfg = Config(width=cols-1, height=rows-1, device=DEVICE,
                 gpu_id=args.gpu_id, diff_thresh=args.diff_thresh)

    fps = args.fps
    if args.video:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()  

    lookup_vals, lookup_lens = setup_lookup(max(cfg.width+1, cfg.height+1, 256), DEVICE)
    out_q = queue.Queue(maxsize=max(1, args.queue_size))

    exit_event = threading.Event()
    writer_thread = threading.Thread(target=writer_task, args=(out_q, fps, exit_event), daemon=True)
    writer_thread.start()
    region = tuple(map(int, args.region.split(','))) if args.region else None
    prod_thread = threading.Thread(
        target=frame_producer,
        args=(args.video, args.screen, cfg, lookup_vals, lookup_lens,
              out_q, region, args.fps, args.colors, exit_event), daemon=True)
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
