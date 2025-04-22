import os
import sys
import argparse
import time
import multiprocessing
from dataclasses import dataclass
from typing import Generator, Tuple, Optional

import torch
import torch._inductor.config
import torch.nn.functional as F
import triton
import triton.language as tl
import time

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
torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
torch.backends.cuda.matmul.allow_tf32 = True



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu':
    exit("CUDA is required for this application to run.") 


ESC_VALS = (27, 91)
SEP_VAL = 59
CUR_END_VAL = 72
COL_PREF_VALS = (27, 91, 52, 56, 59, 50, 59) 
COL_SUF_VALS = (109, 32) 
RESET_VALS = (27, 91, 48, 109) 
RESET_BYTES = bytes(RESET_VALS) 
HIDE_CURSOR_BYTES = b"\033[?25l"
SHOW_CURSOR_BYTES = b"\033[?25h"
CLEAR_SCREEN_BYTES = b"\033[2J"

@dataclass
class Config:
    width: int
    height: int
    device: torch.device
    gpu_id: int = 0
    diff_thresh: int = 1


def write_bytes(fd: int, data: bytes) -> None:
    total = 0
    view = memoryview(data)
    while total < len(data):
        try:
            written = os.write(fd, view[total:])
            if written == 0:
                break
            total += written
        except OSError as e:            
            print(f"Warning: os.write() error: {e}", file=sys.stderr)
            break 

def writer_task(q: multiprocessing.Queue) -> None:
    stdout_fd = sys.stdout.fileno()
    try:
        write_bytes(stdout_fd, CLEAR_SCREEN_BYTES)
        write_bytes(stdout_fd, HIDE_CURSOR_BYTES)
        while True:
            chunk = q.get()
            if chunk is None: 
                break
            if isinstance(chunk, torch.Tensor):
                write_bytes(stdout_fd, chunk.numpy().tobytes())
            elif isinstance(chunk, bytes):
                 write_bytes(stdout_fd, chunk)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Writer process error: {e}", file=sys.stderr)
    finally:
        try:
            write_bytes(stdout_fd, RESET_BYTES)
            write_bytes(stdout_fd, SHOW_CURSOR_BYTES)
        except OSError as e:
             print(f"Writer process cleanup error: {e}", file=sys.stderr)

def setup_ascii_lookup(max_val: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ascii_vals = [str(i).encode('ascii') for i in range(max_val)]
    max_len = max(len(s) for s in ascii_vals)
    bytes_buf = torch.zeros((max_val, max_len), dtype=torch.uint8, device=device)
    lengths = torch.zeros(max_val, dtype=torch.int64, device=device)
    for i, s in enumerate(ascii_vals):
        lengths[i] = len(s)
        bytes_buf[i, :len(s)] = torch.tensor(list(s), dtype=torch.uint8)
    return bytes_buf, lengths


def nv12_to_rgb(tensor_nv12: torch.Tensor, W: int, H: int) -> torch.Tensor:
    y = tensor_nv12[:H, :] 
    uv = tensor_nv12[H:, :].reshape(H // 2, W // 2, 2) 
    y_f = y.float()
    u_f = uv[..., 0].float() 
    v_f = uv[..., 1].float() 
    u_full = F.interpolate(u_f.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest').squeeze() 
    v_full = F.interpolate(v_f.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest').squeeze() 
    y_f = y_f - 16.0
    u_f = u_full - 128.0
    v_f = v_full - 128.0
    r = 1.164 * y_f + 1.596 * v_f
    g = 1.164 * y_f - 0.813 * v_f - 0.391 * u_f
    b = 1.164 * y_f + 2.018 * u_f
    rgb = torch.stack([r, g, b], dim=-1)
    return torch.clamp(rgb, 0, 255).to(torch.uint8)

def resize_preserving_aspect_ratio(
    img: torch.Tensor, target_height: int, target_width: int, fill_value: int = 0
) -> torch.Tensor:
    if not (img.dim() == 3 and img.shape[2] in (1, 3, 4)):
         raise ValueError(f"Input image must be HWC, got shape {img.shape}")
    orig_h, orig_w = img.shape[:2]
    if orig_h == 0 or orig_w == 0:
        return torch.full((target_height, target_width, img.shape[2]), fill_value,
                          dtype=img.dtype, device=img.device)

    scale = min(target_width / orig_w, target_height / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    img_t = img.permute(2, 0, 1).unsqueeze(0).float() 
    resized = F.interpolate(img_t, size=(new_h, new_w), mode='bilinear', align_corners=False)
    resized = resized.squeeze(0) 
    pad_w = target_width - new_w
    pad_h = target_height - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    
    padded = F.pad(
        resized,
        pad=(pad_left, pad_right, pad_top, pad_bottom),
        mode='constant',
        value=fill_value,
    ) 
    out = padded.permute(1, 2, 0).to(dtype=img.dtype) 
    return out

@torch.compile
def process_frame_torch(
    prev_gray: Optional[torch.Tensor], rgb: torch.Tensor, diff_thresh: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).round().to(torch.uint8)
    if prev_gray is None:
        h, w = rgb.shape[:2]
        ys = torch.arange(h, device=rgb.device).unsqueeze(1).expand(-1, w).flatten()
        xs = torch.arange(w, device=rgb.device).unsqueeze(0).expand(h, -1).flatten()
    else:
        diff = torch.abs(gray.to(torch.int16) - prev_gray.to(torch.int16))
        mask = diff > diff_thresh

        if not mask.any():
            return (
                torch.empty(0, dtype=torch.long, device=rgb.device), 
                torch.empty(0, dtype=torch.long, device=rgb.device), 
                torch.empty((0, 3), dtype=torch.uint8, device=rgb.device), 
                gray
            )

        ys, xs = mask.nonzero(as_tuple=True)
    colors = rgb[ys, xs]
    return xs, ys, colors, gray

@triton.jit
def ansi_kernel(
    xs_ptr, ys_ptr,
    r_ptr, g_ptr, b_ptr,
    row_lookup_bytes_ptr, row_lookup_lens_ptr,
    col_lookup_bytes_ptr, col_lookup_lens_ptr,
    r_lookup_bytes_ptr, r_lookup_lens_ptr,
    g_lookup_bytes_ptr, g_lookup_lens_ptr,
    b_lookup_bytes_ptr, b_lookup_lens_ptr,
    offsets_ptr,
    out_ptr,
    N, 
    LOOKUP_MAX_LEN: tl.constexpr, 
    ESC0: tl.constexpr, ESC1: tl.constexpr,
    SEP: tl.constexpr,
    CUR_END: tl.constexpr,
    COL_PREF0: tl.constexpr, COL_PREF1: tl.constexpr, COL_PREF2: tl.constexpr,
    COL_PREF3: tl.constexpr, COL_PREF4: tl.constexpr, COL_PREF5: tl.constexpr,
    COL_PREF6: tl.constexpr,
    COL_SUF0: tl.constexpr, COL_SUF1: tl.constexpr,
    BLOCK_SIZE: tl.constexpr 
):
    pid = tl.program_id(0)
    if pid >= N:
        return
    
    x = tl.load(xs_ptr + pid) 
    y = tl.load(ys_ptr + pid) 
    r_val = tl.load(r_ptr + pid) 
    g_val = tl.load(g_ptr + pid) 
    b_val = tl.load(b_ptr + pid) 

    row_idx = y + 1 
    col_idx = x + 1 

    current_offset = tl.load(offsets_ptr + pid) 
    ptr = out_ptr + current_offset 

    
    tl.store(ptr, ESC0); ptr += 1
    tl.store(ptr, ESC1); ptr += 1

    row_len = tl.load(row_lookup_lens_ptr + row_idx) 
    row_bytes_base = row_lookup_bytes_ptr + row_idx * BLOCK_SIZE 
    for i in tl.static_range(BLOCK_SIZE):
        if i < row_len:
            byte = tl.load(row_bytes_base + i) 
            tl.store(ptr + i, byte) 
    ptr += row_len

    tl.store(ptr, SEP); ptr += 1

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

    
    r_idx = r_val.to(tl.int64)
    g_idx = g_val.to(tl.int64)
    b_idx = b_val.to(tl.int64)

    r_len = tl.load(r_lookup_lens_ptr + r_idx)
    r_bytes_base = r_lookup_bytes_ptr + r_idx * BLOCK_SIZE
    for i in tl.static_range(BLOCK_SIZE):
         if i < r_len:
            byte = tl.load(r_bytes_base + i)
            tl.store(ptr + i, byte)
    ptr += r_len
    tl.store(ptr, SEP); ptr += 1

    g_len = tl.load(g_lookup_lens_ptr + g_idx)
    g_bytes_base = g_lookup_bytes_ptr + g_idx * BLOCK_SIZE
    for i in tl.static_range(BLOCK_SIZE):
         if i < g_len:
            byte = tl.load(g_bytes_base + i)
            tl.store(ptr + i, byte)
    ptr += g_len
    tl.store(ptr, SEP); ptr += 1

    b_len = tl.load(b_lookup_lens_ptr + b_idx)
    b_bytes_base = b_lookup_bytes_ptr + b_idx * BLOCK_SIZE
    for i in tl.static_range(BLOCK_SIZE):
         if i < b_len:
            byte = tl.load(b_bytes_base + i)
            tl.store(ptr + i, byte)
    ptr += b_len 

    tl.store(ptr + 0, COL_SUF0)
    tl.store(ptr + 1, COL_SUF1)
    
@torch.compile()
def ansi_generation(
    xs: torch.Tensor,        
    ys: torch.Tensor,        
    colors: torch.Tensor,    
    byte_vals: torch.Tensor, 
    byte_lens: torch.Tensor, 
    cfg: Config
) -> torch.Tensor:
    N = xs.numel()
    if N == 0:
        return torch.tensor(RESET_VALS, dtype=torch.uint8, device=cfg.device)

    device = cfg.device
    rows_idx = (ys + 1).long()  
    cols_idx = (xs + 1).long()  
    r_idx = colors[:, 0].long() 
    g_idx = colors[:, 1].long()
    b_idx = colors[:, 2].long()
    row_lens = byte_lens[rows_idx]
    col_lens = byte_lens[cols_idx]
    r_lens = byte_lens[r_idx]
    g_lens = byte_lens[g_idx]
    b_lens = byte_lens[b_idx]
    
    per_pixel_len = (
        2 + row_lens + 1 + col_lens + 1 +  
        7 + r_lens + 1 + g_lens + 1 + b_lens + 2  
    )  
    total_len = per_pixel_len.sum()
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int64, device=device), per_pixel_len.cumsum(dim=0)[:-1]),
        dim=0
    )
    
    reset_len = len(RESET_VALS)
    out_buffer = torch.empty(total_len + reset_len, dtype=torch.uint8, device=device)
    r_contig = colors[:, 0].contiguous()  
    g_contig = colors[:, 1].contiguous()
    b_contig = colors[:, 2].contiguous()

    LOOKUP_MAX_LEN = byte_vals.size(1)  
    grid = (N,)  

    ansi_kernel[grid](
        xs, ys,    
        r_contig, g_contig, b_contig,  
        byte_vals, byte_lens,  
        byte_vals, byte_lens,  
        byte_vals, byte_lens,  
        byte_vals, byte_lens,  
        byte_vals, byte_lens,  
        offsets,      
        out_buffer,   
        N,            
        LOOKUP_MAX_LEN,  
        ESC_VALS[0], ESC_VALS[1],
        SEP_VAL,
        CUR_END_VAL,
        COL_PREF_VALS[0], COL_PREF_VALS[1], COL_PREF_VALS[2], COL_PREF_VALS[3],
        COL_PREF_VALS[4], COL_PREF_VALS[5], COL_PREF_VALS[6],
        COL_SUF_VALS[0], COL_SUF_VALS[1],
        BLOCK_SIZE=LOOKUP_MAX_LEN,  
        num_warps=4  
    )
    reset_tensor = torch.tensor(RESET_VALS, dtype=torch.uint8, device=device)
    out_buffer[total_len:] = reset_tensor
    return out_buffer


def read_video_from_screen(cfg: Config, stream: torch.cuda.Stream, region: Optional[Tuple[int, int, int, int]], target_fps: int) -> Generator[torch.Tensor, None, None]:
    if not _has_dxcam:
        raise ImportError("DXCam not installed. Cannot capture screen.")
    if not dxcam:
        raise RuntimeError("DXCam failed to initialize.") 
    camera = None
    try:
        camera = dxcam.create(output_color="RGB", output_idx=0) 
        capture_region = region if region else camera.region 
        print(f"Screen capture starting: Region={capture_region}, Target FPS={target_fps}")
        camera.start(region=capture_region, target_fps=target_fps) 
        last_frame_time = time.monotonic() 
        while True:
            frame = camera.get_latest_frame() 
            current_time = time.monotonic()
            if frame is None:
                time.sleep(0.001) 
                continue
            
            with torch.cuda.stream(stream):
                rgb_captured = torch.from_numpy(frame).to(cfg.device) 
                rgb_resized = resize_preserving_aspect_ratio(
                    rgb_captured, cfg.height, cfg.width, fill_value=0
                )
                yield rgb_resized
            
            elapsed = current_time - last_frame_time
            wait_time = (1.0 / target_fps) - elapsed
            if wait_time > 0:
                 time.sleep(wait_time) 
            last_frame_time = time.monotonic() 

    except Exception as e:
        print(f"Error during screen capture: {e}", file=sys.stderr)
        raise 
    finally:
        if camera:
            print("Stopping screen capture...")
            camera.stop()
        print("Screen capture finished.")


def read_video_from_file(filename: str, cfg: Config, stream: torch.cuda.Stream) -> Generator[torch.Tensor, None, None]:
    if not _has_pynvcodec:
        raise ImportError("PyNvVideoCodec not installed. Cannot play video file.")
    if not nvc:
        raise RuntimeError("PyNvVideoCodec failed to initialize.")

    demuxer = None
    decoder = None
    try:
        print(f"Opening video file: {filename}")
        demuxer = nvc.CreateDemuxer(filename=filename)
        if not demuxer:
             raise RuntimeError(f"Failed to create demuxer for {filename}")

        fps = demuxer.FrameRate()
        if fps <= 0:
             print("Warning: Could not determine video FPS, defaulting to 30.")
             fps = 30.0
        codec = demuxer.GetNvCodecId()
        print(f"Video Info: Codec={codec}, FPS={fps:.2f}, Size={demuxer.Width()}x{demuxer.Height()}")
        frame_interval = 1.0 / fps
        last_frame_time = time.monotonic()
        print(f"Creating decoder for GPU {cfg.gpu_id}...")
        decoder = nvc.CreateDecoder(
            gpuid=cfg.gpu_id, codec=codec,
            cudacontext=0, cudastream=0, usedevicememory=True
        )
        if not decoder:
             raise RuntimeError("Failed to create NVCUVID decoder.")
        total_frames = 0
        for packet in demuxer: 
            for frame in decoder.Decode(packet):
                current_time = time.monotonic()
                elapsed = current_time - last_frame_time
                wait_time = frame_interval - elapsed
                if wait_time > 0:
                    time.sleep(wait_time)
                
                with torch.cuda.stream(stream):
                    nv12_tensor = torch.from_dlpack(frame).to(cfg.device) 
                    frame_h_nv12 = nv12_tensor.shape[0] 
                    frame_w = nv12_tensor.shape[1]
                    frame_h = int(frame_h_nv12 * 2 / 3) 
                
                    rgb_full = nv12_to_rgb(nv12_tensor, frame_w, frame_h)
                    rgb_resized = resize_preserving_aspect_ratio(
                        rgb_full, cfg.height, cfg.width, fill_value=0
                    )
                    yield rgb_resized 
                    total_frames += 1

                last_frame_time = time.monotonic() 
    except Exception as e:
        raise
    finally:
        decoder = None
        demuxer = None



def main():
    parser = argparse.ArgumentParser(
        description='Play video or capture screen in the terminal using ANSI escape codes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-v', '--video', type=str, help='Path to video file to play.')
    group.add_argument('-s', '--screen', action='store_true', help='Capture screen instead of playing video.')

    parser.add_argument('-r', '--region', type=str, default=None,
                        help='Screen region "x,y,width,height" (e.g., "0,0,1920,1080"). Only used with --screen.')
    parser.add_argument('-f', '--fps', type=int, default=60,
                        help='Target FPS for screen capture or approximate playback speed limit.')
    parser.add_argument('--diff-thresh', type=int, default=5,
                        help='Pixel difference threshold (0-255) to trigger update. Higher values mean fewer updates.')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use for decoding and processing.')
    parser.add_argument('--queue-size', type=int, default=5, help='Max size of the frame output queue.')

    args = parser.parse_args()
    
    if DEVICE.type == 'cpu':
        print("CUDA is required for this application to run.")
        sys.exit(1)
    if args.video and not _has_pynvcodec:
        print("PyNvVideoCodec not installed. Cannot play video file.")
        sys.exit(1)
    if args.screen and not _has_dxcam:
        print("DXCam not installed. Cannot capture screen.")
        sys.exit(1)

    
    region = None
    if args.screen and args.region:
        try:
            x, y, w, h = map(int, args.region.split(','))
            if w <= 0 or h <= 0: raise ValueError("Width and height must be positive")
            region = (x, y, x + w, y + h) 
        except Exception as e:
            print(f"Error: Invalid region format '{args.region}'. Use 'x,y,width,height'. {e}")
            sys.exit(1)

    queue = multiprocessing.Queue(maxsize=max(1, args.queue_size)) 
    writer_process = multiprocessing.Process(target=writer_task, args=(queue,), daemon=True)
    writer_process.start()

    try:
        cols, rows = os.get_terminal_size()
    except OSError:
        print("Warning: Could not get terminal size. Defaulting to 80x24.")
        cols, rows = 80, 24 
    
    term_w = max(1, cols -1) 
    term_h = max(1, rows -1) 

    cfg = Config(width=term_w, height=term_h, device=DEVICE, gpu_id=args.gpu_id, diff_thresh=args.diff_thresh)
    print(f"Terminal size detected: {cols}x{rows}. Using render area: {cfg.width}x{cfg.height}")

    max_lookup_val = max(cfg.width + 1, cfg.height + 1, 256)
    byte_vals, byte_lens = setup_ascii_lookup(max_lookup_val, cfg.device)
    stream = torch.cuda.Stream(device=cfg.device)
    prev_gray: Optional[torch.Tensor] = None 
    try:
        
        if args.screen:
            frame_source = read_video_from_screen(cfg, stream, region=region, target_fps=args.fps)
        else: 
            frame_source = read_video_from_file(args.video, cfg, stream)    
        with torch.inference_mode(): 
            for rgb_frame in frame_source:
                if rgb_frame.device != cfg.device:
                     rgb_frame = rgb_frame.to(cfg.device)

                with torch.cuda.stream(stream):
                    xs, ys, colors, current_gray = process_frame_torch(
                        prev_gray, rgb_frame, cfg.diff_thresh
                    )
                    prev_gray = current_gray 
                    if xs.numel() > 0:
                        ansi_bytes_gpu = ansi_generation(
                            xs, ys, colors, byte_vals, byte_lens, cfg
                        )
                    else:
                        ansi_bytes_gpu = torch.empty(0, dtype=torch.uint8, device=cfg.device)
                
                stream.synchronize() 
                if ansi_bytes_gpu.numel() > 0:
                    ansi_bytes_cpu = ansi_bytes_gpu.cpu() 
                    queue.put(ansi_bytes_cpu) 

    except KeyboardInterrupt:
        print("\nInterrupt received, stopping...")
    except Exception as e:
        import traceback
        print(f"\n--- An error occurred ---", file=sys.stderr)
        print(f"Error type: {type(e).__name__}", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        print("Traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("--------------------------", file=sys.stderr)
    finally:
        queue.put(None)
        writer_process.join(timeout=2) 
        if writer_process.is_alive():
            writer_process.terminate()

if __name__ == '__main__':
    main()
