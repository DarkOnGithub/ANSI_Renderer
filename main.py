import os
import sys
import argparse
import time
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Tuple, Generator

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import cv2

try:
    import PyNvVideoCodec as nvc
    _has_pynvcodec = True
except ImportError:
    _has_pynvcodec = False
    nvc = None

try:
    import dxcam
    _has_dxcam = True
except ImportError:
    _has_dxcam = False
    dxcam = None

torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
torch.backends.cuda.matmul.allow_tf32 = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu':
    sys.exit("CUDA is required for this application.")


ESC_VALS = (27, 91)  
SEP_VAL = 59  
CUR_END_VAL = 72  
COL_PREF_VALS = (27, 91, 52, 56, 59, 50, 59)  
COL_SUF_VALS = (109, 32)  
RESET_VALS = (27, 91, 48, 109)  
HIDE_CURSOR = b"\033[?25l"
SHOW_CURSOR = b"\033[?25h"
CLEAR_SCREEN = b"\033[2J"
CELL_ASPECT = 0.5

@dataclass
class Config:
    width: int
    height: int
    device: torch.device
    gpu_id: int = 0
    diff_thresh: int = 0


def write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    total = 0
    while total < len(data):
        written = os.write(fd, view[total:])
        if not written:
            break
        total += written


def writer_task(out_queue: queue.Queue, target_fps: float, exit_event: threading.Event) -> None:
    fd = sys.stdout.fileno()
    write_all(fd, CLEAR_SCREEN)
    write_all(fd, HIDE_CURSOR)
    interval = 1.0 / target_fps
    last_time = time.monotonic()
    i = 0
    try:
        while not exit_event.is_set():
            try:
                
                item = out_queue.get(timeout=0.1)
                if item is None:
                    break
                data = item.numpy().tobytes() if isinstance(item, torch.Tensor) else item
                write_all(fd, data)
                
                i += 1
                now = time.monotonic()
                elapsed = now - last_time
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_time = time.monotonic()
            except queue.Empty:
                continue  
    except KeyboardInterrupt:
        pass  
    finally:
        write_all(fd, bytes(RESET_VALS))
        write_all(fd, SHOW_CURSOR)

def setup_lookup(max_val: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ascii_bytes = [str(i).encode() for i in range(max_val)]
    max_len = max(len(x) for x in ascii_bytes)
    buf = torch.zeros((max_val, max_len), dtype=torch.uint8, device=device)
    lens = torch.zeros(max_val, dtype=torch.int64, device=device)
    for i, bs in enumerate(ascii_bytes):
        buf[i, :len(bs)] = torch.tensor(list(bs), dtype=torch.uint8)
        lens[i] = len(bs)
    return buf, lens


def nv12_to_rgb(frame: torch.Tensor, W: int, H: int) -> torch.Tensor:
    y = frame[:H].float() - 16.0
    uv = (
        frame[H:].reshape(H//2, W//2, 2)
        .permute(2, 0, 1).float() - 128.0
    )
    u_full = F.interpolate(uv[0:1].unsqueeze(0), (H, W), mode='nearest').squeeze()
    v_full = F.interpolate(uv[1:2].unsqueeze(0), (H, W), mode='nearest').squeeze()
    r = (1.164 * y + 1.596 * v_full).clamp(0, 255)
    g = (1.164 * y - 0.391 * u_full - 0.813 * v_full).clamp(0, 255)
    b = (1.164 * y + 2.018 * u_full).clamp(0, 255)
    return torch.stack([r, g, b], -1).to(torch.uint8)


def resize_aspect(img: torch.Tensor, tgt_h: int, tgt_w: int) -> torch.Tensor:
    h, w = img.shape[:2]
    eff_w = tgt_w * CELL_ASPECT
    eff_h = tgt_h
    scale = min(eff_w / w, eff_h / h)
    new_w = max(1, int(round(w * scale / CELL_ASPECT)))
    new_h = max(1, int(round(h * scale)))
    t = img.permute(2, 0, 1).unsqueeze(0).float()
    out = F.interpolate(t, (new_h, new_w), mode='bilinear', align_corners=False)
    return out.squeeze(0).permute(1, 2, 0).to(img.dtype)


@torch.compile
def process_frame(prev_gray: Optional[torch.Tensor], rgb: torch.Tensor, diff_thresh: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gray = (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2])
    gray = gray.round().to(torch.uint8)
    if prev_gray is None:
        ys, xs = torch.meshgrid(
            torch.arange(gray.size(0), device=gray.device),
            torch.arange(gray.size(1), device=gray.device),
            indexing='ij'
        )
        ys, xs = ys.flatten(), xs.flatten()
    else:
        diff = (gray.to(torch.int16) - prev_gray.to(torch.int16)).abs()
        mask = diff > diff_thresh
        if not mask.any():
            return torch.empty(0, device=rgb.device), torch.empty(0, device=rgb.device), torch.empty((0,3), device=rgb.device), gray
        ys, xs = mask.nonzero(as_tuple=True)
    colors = rgb[ys, xs]
    return xs, ys, colors, gray

def resize_with_aspect_ratio(img: torch.Tensor, target_height: int, target_width: int, cell_aspect: float = CELL_ASPECT) -> torch.Tensor:
    orig_h, orig_w = img.shape[:2]
    eff_w = target_width * cell_aspect
    eff_h = target_height * 1.0
    scale = min(eff_w / orig_w, eff_h / orig_h)
    new_w = max(1, int(round(orig_w * scale / cell_aspect)))
    new_h = max(1, int(round(orig_h * scale)))
    img_t = img.permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(img_t, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0).to(img.dtype)

@triton.jit
def ansi_run_kernel(
    run_xs_ptr, run_ys_ptr,
    run_r_ptr, run_g_ptr, run_b_ptr,
    run_lengths_ptr, offsets_ptr,
    row_lookup_bytes_ptr, row_lookup_lens_ptr,
    col_lookup_bytes_ptr, col_lookup_lens_ptr,
    r_lookup_bytes_ptr, r_lookup_lens_ptr,
    g_lookup_bytes_ptr, g_lookup_lens_ptr,
    b_lookup_bytes_ptr, b_lookup_lens_ptr,
    out_ptr,
    N,
    LOOKUP_MAX_LEN: tl.constexpr,
    ESC0: tl.constexpr, ESC1: tl.constexpr,
    SEP: tl.constexpr,
    CUR_END: tl.constexpr,
    COL_PREF0: tl.constexpr, COL_PREF1: tl.constexpr, COL_PREF2: tl.constexpr,
    COL_PREF3: tl.constexpr, COL_PREF4: tl.constexpr, COL_PREF5: tl.constexpr,
    COL_PREF6: tl.constexpr,
    SPACE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    
    x_start = tl.load(run_xs_ptr + pid)
    y = tl.load(run_ys_ptr + pid)
    r_val = tl.load(run_r_ptr + pid)
    g_val = tl.load(run_g_ptr + pid)
    b_val = tl.load(run_b_ptr + pid)
    length = tl.load(run_lengths_ptr + pid)
    offset = tl.load(offsets_ptr + pid)

    ptr = out_ptr + offset

    
    tl.store(ptr, ESC0); ptr += 1
    tl.store(ptr, ESC1); ptr += 1
    row_idx = y + 1
    row_len = tl.load(row_lookup_lens_ptr + row_idx)
    row_bytes_base = row_lookup_bytes_ptr + row_idx * BLOCK_SIZE
    for i in tl.static_range(BLOCK_SIZE):
        if i < row_len:
            byte = tl.load(row_bytes_base + i)
            tl.store(ptr + i, byte)
    ptr += row_len
    tl.store(ptr, SEP); ptr += 1
    col_idx = x_start + 1
    col_len = tl.load(col_lookup_lens_ptr + col_idx)
    col_bytes_base = col_lookup_bytes_ptr + col_idx * BLOCK_SIZE
    for i in tl.static_range(BLOCK_SIZE):
        if i < col_len:
            byte = tl.load(col_bytes_base + i)
            tl.store(ptr + i, byte)
    ptr += col_len
    tl.store(ptr, CUR_END); ptr += 1

    
    tl.store(ptr + 0, COL_PREF0)
    tl.store(ptr + 1, COL_PREF1)
    tl.store(ptr + 2, COL_PREF2)
    tl.store(ptr + 3, COL_PREF3)
    tl.store(ptr + 4, COL_PREF4)
    tl.store(ptr + 5, COL_PREF5)
    tl.store(ptr + 6, COL_PREF6)
    ptr += 7

    r_len = tl.load(r_lookup_lens_ptr + r_val)
    r_bytes_base = r_lookup_bytes_ptr + r_val * BLOCK_SIZE
    for i in tl.static_range(BLOCK_SIZE):
        if i < r_len:
            byte = tl.load(r_bytes_base + i)
            tl.store(ptr + i, byte)
    ptr += r_len
    tl.store(ptr, SEP); ptr += 1
    g_len = tl.load(g_lookup_lens_ptr + g_val)
    g_bytes_base = g_lookup_bytes_ptr + g_val * BLOCK_SIZE
    for i in tl.static_range(BLOCK_SIZE):
        if i < g_len:
            byte = tl.load(g_bytes_base + i)
            tl.store(ptr + i, byte)
    ptr += g_len
    tl.store(ptr, SEP); ptr += 1
    b_len = tl.load(b_lookup_lens_ptr + b_val)
    b_bytes_base = b_lookup_bytes_ptr + b_val * BLOCK_SIZE
    for i in tl.static_range(BLOCK_SIZE):
        if i < b_len:
            byte = tl.load(b_bytes_base + i)
            tl.store(ptr + i, byte)
    ptr += b_len
    tl.store(ptr, 109); ptr += 1  

    
    for i in range(length):
        tl.store(ptr, SPACE)
        ptr += 1
        
@torch.compile
def ansi_generate(xs: torch.Tensor, ys: torch.Tensor, colors: torch.Tensor, byte_vals: torch.Tensor, byte_lens: torch.Tensor, cfg: Config) -> torch.Tensor:
    N = xs.numel()
    if N == 0:
        return torch.tensor(RESET_VALS, dtype=torch.uint8, device=cfg.device)

    device = cfg.device
    key = ys * cfg.width + xs
    sort_idx = torch.argsort(key)
    xs_sorted = xs[sort_idx]
    ys_sorted = ys[sort_idx]
    colors_sorted = colors[sort_idx]

    y_diff = ys_sorted[1:] != ys_sorted[:-1]
    x_diff = xs_sorted[1:] - xs_sorted[:-1]
    color_diff = (colors_sorted[1:] != colors_sorted[:-1]).any(dim=1)
    is_new_run = torch.zeros(N, dtype=torch.bool, device=device)
    is_new_run[0] = True
    is_new_run[1:] = y_diff | ((~y_diff) & ((x_diff != 1) | color_diff))
    run_start_indices = torch.where(is_new_run)[0]
    run_lengths = torch.diff(run_start_indices, append=torch.tensor([N], device=device))

    num_runs = run_start_indices.size(0)
    if num_runs == 0:  
        return torch.tensor(RESET_VALS, dtype=torch.uint8, device=device)
    
    run_xs = xs_sorted[run_start_indices]
    run_ys = ys_sorted[run_start_indices]
    run_colors = colors_sorted[run_start_indices]
    run_r = run_colors[:, 0].contiguous()
    run_g = run_colors[:, 1].contiguous()
    run_b = run_colors[:, 2].contiguous()
    row_idx = run_ys + 1
    col_idx = run_xs + 1
    r_idx = run_r.to(torch.long)
    g_idx = run_g.to(torch.long)
    b_idx = run_b.to(torch.long)
    row_lens = byte_lens[row_idx]
    col_lens = byte_lens[col_idx]
    r_lens = byte_lens[r_idx]
    g_lens = byte_lens[g_idx]
    b_lens = byte_lens[b_idx]
    cursor_part = 3 + row_lens + 1 + col_lens + 1  
    color_part = 7 + r_lens + 1 + g_lens + 1 + b_lens   
    total_lens = cursor_part + color_part + run_lengths

    
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64, device=device), total_lens.cumsum(dim=0)[:-1]), dim=0)
    reset_len = len(RESET_VALS)
    out_buffer = torch.empty(total_lens.sum() + reset_len, dtype=torch.uint8, device=device)

    
    LOOKUP_MAX_LEN = byte_vals.size(1)
    grid = (num_runs,)
    ansi_run_kernel[grid](
        run_xs, run_ys,
        run_r.to(torch.int64), run_g.to(torch.int64), run_b.to(torch.int64),  
        run_lengths, offsets,
        byte_vals, byte_lens,
        byte_vals, byte_lens,
        byte_vals, byte_lens,
        byte_vals, byte_lens,
        byte_vals, byte_lens,
        out_buffer,
        num_runs,
        LOOKUP_MAX_LEN,
        ESC_VALS[0], ESC_VALS[1],
        SEP_VAL,
        CUR_END_VAL,
        COL_PREF_VALS[0], COL_PREF_VALS[1], COL_PREF_VALS[2], COL_PREF_VALS[3],
        COL_PREF_VALS[4], COL_PREF_VALS[5], COL_PREF_VALS[6],
        32,  
        BLOCK_SIZE=LOOKUP_MAX_LEN,
        num_warps=4
    )

    
    reset_tensor = torch.tensor(RESET_VALS, dtype=torch.uint8, device=device)
    out_buffer[total_lens.sum():] = reset_tensor

    return out_buffer

def read_video_from_screen(cfg: Config, stream: torch.cuda.Stream, region: Optional[Tuple[int, int, int, int]], target_fps: int) -> Generator[torch.Tensor, None, None]:
    if not _has_dxcam:
        raise ImportError("DXCam not installed. Cannot capture screen.")
    camera = dxcam.create(output_color="RGB", output_idx=0)
    if not camera:
        raise RuntimeError("Failed to initialize DXCam.")
    capture_region = region if region else camera.region
    print(f"Screen capture starting: Region={capture_region}, Target FPS={target_fps}")
    camera.start(region=capture_region, target_fps=target_fps)
    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.001)
                continue
            with torch.cuda.stream(stream):
                rgb_captured = torch.from_numpy(frame).to(cfg.device)
                rgb_resized = resize_with_aspect_ratio(rgb_captured, cfg.height, cfg.width)
                yield rgb_resized

    finally:
        camera.stop()

def read_video_from_file(filename: str, cfg: Config, stream: torch.cuda.Stream) -> Generator[torch.Tensor, None, None]:
    if not _has_pynvcodec:
        raise ImportError("PyNvVideoCodec not installed. Cannot play video file.")
    demuxer = nvc.CreateDemuxer(filename=filename)
    codec = demuxer.GetNvCodecId()
    decoder = nvc.CreateDecoder(gpuid=cfg.gpu_id, codec=codec, cudacontext=0, cudastream=0, usedevicememory=True)
    if not decoder:
        raise RuntimeError("Failed to create NVCUVID decoder.")
    for packet in demuxer:
        for frame in decoder.Decode(packet):
            with torch.cuda.stream(stream):
                nv12_tensor = torch.from_dlpack(frame).to(cfg.device)
                frame_h_nv12 = nv12_tensor.shape[0]
                frame_w = nv12_tensor.shape[1]
                frame_h = int(frame_h_nv12 * 2 / 3)
                rgb_full = nv12_to_rgb(nv12_tensor, frame_w, frame_h)
                rgb_resized = resize_with_aspect_ratio(rgb_full, cfg.height, cfg.width)
                yield rgb_resized
            
def frame_producer(path: Optional[str], capture_screen: bool, cfg: Config,
                   lookup_vals: torch.Tensor, lookup_lens: torch.Tensor,
                   stream: torch.cuda.Stream, out_queue: queue.Queue,
                   region: Optional[Tuple[int,int,int,int]], fps: int,
                   exit_event: threading.Event) -> None:
    prev_gray = None
    source = None
    try:
        if path:
            source = read_video_from_file(path, cfg, stream)
        else:
            source = read_video_from_screen(cfg, stream, region, fps)
        i = 0
        for rgb in source:
            if exit_event.is_set():
                break
            with torch.cuda.stream(stream):   
                xs, ys, cols, gray = process_frame(prev_gray, rgb, cfg.diff_thresh)
                prev_gray = gray
                ansi = ansi_generate(xs, ys, cols, lookup_vals, lookup_lens, cfg)
                
            stream.synchronize()
            out_queue.put(ansi.cpu())
            i += 1
    except KeyboardInterrupt:
        pass  
    except Exception as e:
        print(f"\nError in frame producer: {e}", file=sys.stderr)
    finally:
        out_queue.put(None)  

def main() -> None:
    parser = argparse.ArgumentParser(description="ANSI video terminal with buffering")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-v', '--video', help='Path to video file')
    group.add_argument('-s', '--screen', action='store_true', help='Capture screen')
    parser.add_argument('-r', '--region', help='x,y,w,h for screen capture', default=None)
    parser.add_argument('-f', '--fps', type=int, default=60)
    parser.add_argument('--diff-thresh', type=int, default=0)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--queue-size', type=int, default=100)
    args = parser.parse_args()

    cols, rows = os.get_terminal_size() if sys.stdout.isatty() else (80, 24)
    cfg = Config(width=cols-1, height=rows-1, device=DEVICE,
                 gpu_id=args.gpu_id, diff_thresh=args.diff_thresh)

    lookup_vals, lookup_lens = setup_lookup(max(cfg.width+1, cfg.height+1, 256), DEVICE)
    stream = torch.cuda.Stream(device=DEVICE)
    fps = args.fps
    if args.video:
        fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS)
    out_q = queue.Queue(maxsize=max(1, args.queue_size))

    exit_event = threading.Event()
    writer = threading.Thread(target=writer_task, args=(out_q, fps, exit_event), daemon=True)
    writer.start()
    region = tuple(map(int, args.region.split(','))) if args.region else None
    prod = threading.Thread(
        target=frame_producer,
        args=(args.video, args.screen, cfg, lookup_vals, lookup_lens,
              stream, out_q, region, args.fps, exit_event),
        daemon=True)
    prod.start()

    try:
        prod.join()
        writer.join()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...", file=sys.stderr)
        exit_event.set()  
        write_all(sys.stdout.fileno(), bytes(RESET_VALS))
        write_all(sys.stdout.fileno(), SHOW_CURSOR)
        time.sleep(0.5)
        sys.exit(0)

if __name__ == '__main__':
    main()
