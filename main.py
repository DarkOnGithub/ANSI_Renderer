import os
import sys
from dataclasses import dataclass

import torch
import torch._inductor.config
import torch.nn.functional as F
import PyNvVideoCodec as nvc
import time
import multiprocessing
import triton
import triton.language as tl
from typing import Generator
import dxcam
import argparse

from typing import Tuple

torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None


DEVICE   = torch.device('cuda')
ESC      = torch.tensor([27,91],      dtype=torch.uint8, device=DEVICE)
SEP      = torch.tensor([59],         dtype=torch.uint8, device=DEVICE)   
CUR_END  = torch.tensor([72],         dtype=torch.uint8, device=DEVICE)   
COL_PREF = torch.tensor([27,91,52,56,59,50,59], dtype=torch.uint8, device=DEVICE)  
COL_SUF  = torch.tensor([109,32],     dtype=torch.uint8, device=DEVICE)  
RESET    = torch.tensor([27,91,48,109],dtype=torch.uint8, device=DEVICE)  


@dataclass
class Config:
    width: int
    height: int
    device: torch.device
    gpu_id: int = 0
    diff_thresh: int = 0
    
def write(fd: int, data: bytes) -> None:
    total = 0
    while total < len(data):
        written = os.write(fd, data[total:])
        if written == 0:
            raise RuntimeError("os.write() returned 0 bytes written")
        total += written

def writer_task(q: multiprocessing.Queue):
    try:
        while True:
            chunk = q.get()            
            if chunk is None:      
                break            
            write(1, chunk)        
    
    except KeyboardInterrupt:
        pass
    
    finally:
        os.write(1, b"\033[0m\033[?25h")
         
def setup_ascii_lookup(max_val: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ascii_vals = [str(i).encode('ascii') for i in range(max_val)]
    max_len = max(len(s) for s in ascii_vals)
    bytes_buf = torch.zeros((max_val, max_len), dtype=torch.uint8, device=device)
    lengths = torch.zeros(max_val, dtype=torch.int64, device=device)
    for i, s in enumerate(ascii_vals):
        lengths[i] = len(s)
        bytes_buf[i, :len(s)] = torch.tensor(list(s), dtype=torch.uint8)
    return bytes_buf, lengths

@torch.compile
def process_frame(prev_gray: torch.Tensor, rgb: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gray = (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]).round().to(torch.uint8)
    if prev_gray is None:
        ys, xs = torch.meshgrid(
            torch.arange(cfg.height, device=cfg.device),
            torch.arange(cfg.width, device=cfg.device), indexing='ij'
        )
        ys, xs = ys.flatten(), xs.flatten()
    else:
        diff = (gray.to(torch.int16) - prev_gray.to(torch.int16)).abs()
        ys, xs = torch.nonzero(diff > cfg.diff_thresh, as_tuple=True)
    colors = rgb[ys, xs]
    return xs, ys, colors, gray

@triton.jit
def ansi_kernel(
    rows_ptr, row_bytes_ptr, row_lens_ptr,
    cols_ptr, col_bytes_ptr, col_lens_ptr,
    r_bytes_ptr, r_lens_ptr,
    g_bytes_ptr, g_lens_ptr,
    b_bytes_ptr, b_lens_ptr,
    offsets_ptr, out_ptr,
    N,
    esc0, esc1, sep, cur_end,
    col_pref0, col_pref1, col_pref2, col_pref3, col_pref4, col_pref5, col_pref6,
    col_suf0, col_suf1,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    base_off = tl.load(offsets_ptr + pid)
    ptr = out_ptr + base_off
    
    tl.store(ptr + 0, esc0)
    tl.store(ptr + 1, esc1)
    ptr += 2
    
    row_len = tl.load(row_lens_ptr + pid)
    row_base = pid * BLOCK_SIZE
    for i in range(BLOCK_SIZE):
        b = tl.load(row_bytes_ptr + row_base + i)
        mask = i < row_len
        tl.store(ptr + i, tl.where(mask, b, 0))
    ptr += row_len
    tl.store(ptr, sep)
    ptr += 1
    
    col_len = tl.load(col_lens_ptr + pid)
    col_base = pid * BLOCK_SIZE
    for i in range(BLOCK_SIZE):
        b = tl.load(col_bytes_ptr + col_base + i)
        mask = i < col_len
        tl.store(ptr + i, tl.where(mask, b, 0))
    ptr += col_len
    tl.store(ptr, cur_end)
    ptr += 1    
    
    tl.store(ptr + 0, col_pref0)
    tl.store(ptr + 1, col_pref1)
    tl.store(ptr + 2, col_pref2)
    tl.store(ptr + 3, col_pref3)
    tl.store(ptr + 4, col_pref4)
    tl.store(ptr + 5, col_pref5)
    tl.store(ptr + 6, col_pref6)
    ptr += 7
    
    r_len = tl.load(r_lens_ptr + pid)
    r_base = pid * BLOCK_SIZE
    for i in range(BLOCK_SIZE):
        b = tl.load(r_bytes_ptr + r_base + i)
        mask = i < r_len
        tl.store(ptr + i, tl.where(mask, b, 0))
    ptr += r_len
    tl.store(ptr, sep)
    ptr += 1
    
    g_len = tl.load(g_lens_ptr + pid)
    g_base = pid * BLOCK_SIZE
    for i in range(BLOCK_SIZE):
        b = tl.load(g_bytes_ptr + g_base + i)
        mask = i < g_len
        tl.store(ptr + i, tl.where(mask, b, 0))
    ptr += g_len
    tl.store(ptr, sep)
    ptr += 1
    
    b_len = tl.load(b_lens_ptr + pid)
    b_base = pid * BLOCK_SIZE
    b_len = tl.maximum(b_len, 1)
    for i in range(BLOCK_SIZE):
        b = tl.load(b_bytes_ptr + b_base + i)
        mask = i < b_len
        if i < b_len:
            tl.store(ptr + i, b)
    ptr += b_len
    
    tl.store(ptr + 0, col_suf0)
    tl.store(ptr + 1, col_suf1)

@torch.compile
def fused_changes_to_ansi(xs: torch.Tensor,
                          ys: torch.Tensor,
                          colors: torch.Tensor,
                          byte_vals: torch.Tensor,
                          byte_lens: torch.Tensor,
                          cfg: Config) -> torch.Tensor:
    device = xs.device
    N = xs.numel()
    if N == 0:
        return RESET.clone()

    
    rows = (ys + 1).to(torch.long)
    cols = (xs + 1).to(torch.long)
    row_bytes = byte_vals[rows]    
    row_lens  = byte_lens[rows]
    col_bytes = byte_vals[cols]
    col_lens  = byte_lens[cols]

    r_idx, g_idx, b_idx = colors[:,0].long(), colors[:,1].long(), colors[:,2].long()
    r_bytes, r_lens = byte_vals[r_idx], byte_lens[r_idx]
    g_bytes, g_lens = byte_vals[g_idx], byte_lens[g_idx]
    b_bytes, b_lens = byte_vals[b_idx], byte_lens[b_idx]

    seg_lens = torch.stack([
        torch.full((N,), 2, device=device),   
        row_lens,
        torch.full((N,), 1, device=device),   
        col_lens,
        torch.full((N,), 1, device=device),   
        torch.full((N,), 7, device=device),   
        r_lens,
        torch.full((N,), 1, device=device),
        g_lens,
        torch.full((N,), 1, device=device),
        b_lens,
        torch.full((N,), 2, device=device),   
    ], dim=1)  
    per_pixel_len = seg_lens.sum(dim=1)      
    total_len = per_pixel_len.sum()
    offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.int64),
                           per_pixel_len.cumsum(dim=0)[:-1]], dim=0)

    max_dig = byte_vals.size(1)
    def flatten(buf): return buf.contiguous().view(-1)
    row_bytes_f = flatten(row_bytes)
    col_bytes_f = flatten(col_bytes)
    r_bytes_f = flatten(r_bytes)
    g_bytes_f = flatten(g_bytes)
    b_bytes_f = flatten(b_bytes)
    
    out = torch.zeros(total_len + len(RESET), dtype=torch.uint8, device=device)

    BLOCK_SIZE = max_dig
    grid = lambda _: (N,)
    ansi_kernel[grid](
        rows, row_bytes_f, row_lens,
        cols, col_bytes_f, col_lens,
        r_bytes_f, r_lens,
        g_bytes_f, g_lens,
        b_bytes_f, b_lens,
        offsets, out,        
        N,
        27, 91,            
        59,                
        72,                
        27, 91, 52, 56, 59, 50, 59,  
        109, 32,           
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
        
    out[total_len:] = RESET
    return out

def nv12_to_rgb(tensor_nv12: torch.Tensor, cfg: Config) -> torch.Tensor:
    H, W = cfg.height, cfg.width
    y = tensor_nv12[:H, :].float()
    uv = tensor_nv12[H:, :].reshape(H//2, W//2, 2).float()
    U = uv[..., 0]; V = uv[..., 1]
    U_full = F.interpolate(U.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze()
    V_full = F.interpolate(V.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze()
    C = y - 16; D = U_full - 128; E = V_full - 128
    R = (1.164 * C + 1.793 * E).clamp(0,255)
    G = (1.164 * C - 0.213 * D - 0.533 * E).clamp(0,255)
    B = (1.164 * C + 2.112 * D).clamp(0,255)
    rgb = torch.stack([R, G, B], dim=-1).to(torch.uint8)
    return rgb

def read_video_from_screen(cfg: Config, stream, region=None, target_fps=60) -> Generator[torch.Tensor, None, None]:
    camera = dxcam.create(output_color="RGB", output_idx=0)
    
    if region is None:
        region = (0, 0, 1920, 1080)
    
    camera.start(region=region, target_fps=target_fps)
    last_time = time.time()
    frame_interval = 1.0 / target_fps
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - last_time
            if elapsed < frame_interval:
                time.sleep(max(0, frame_interval - elapsed))
            frame = camera.get_latest_frame()
            if frame is None:
                continue
            last_time = time.time()
            with torch.cuda.stream(stream):
                rgb = torch.from_numpy(frame).to(cfg.device)
                rgb_resized = F.interpolate(rgb.permute(2, 0, 1).unsqueeze(0).float(), 
                                          size=(cfg.height, cfg.width), 
                                          mode='bilinear', 
                                          align_corners=False).squeeze(0).permute(1, 2, 0).byte()
                yield rgb_resized
    finally:
        camera.stop()
        
def read_video_from_file(filename: str, cfg: Config, stream) -> Generator[torch.Tensor, None, None]:
    demux = nvc.CreateDemuxer(filename=filename)
    fps = demux.FrameRate()
    frame_interval = 1.0 / fps
    last_frame_time = time.time()
    dec = nvc.CreateDecoder(
        gpuid=cfg.gpu_id, codec=demux.GetNvCodecId(),
        cudacontext=0, cudastream=0, usedevicememory=True
    )
    for packet in demux:
        for frame in dec.Decode(packet):
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            with torch.cuda.stream(stream):
                nv12 = torch.from_dlpack(frame)
                video_height = int(2 * nv12.numel() // (3 * nv12.shape[1]))
                video_width = nv12.shape[1]
                nv12 = nv12.reshape(video_height * 3 // 2, video_width)
                video_cfg = Config(width=video_width, height=video_height, device=cfg.device, gpu_id=cfg.gpu_id, diff_thresh=cfg.diff_thresh)
                rgb_full = nv12_to_rgb(nv12, video_cfg)
                rgb_resized = F.interpolate(rgb_full.permute(2, 0, 1).unsqueeze(0).float(), 
                                        size=(cfg.height, cfg.width), 
                                        mode='bilinear', 
                                        align_corners=False).squeeze(0).permute(1, 2, 0).byte()
                last_frame_time = time.time()
                yield rgb_resized

def main():
    parser = argparse.ArgumentParser(description='Terminal Video Player/Screen Recorder')
    parser.add_argument('-v', '--video', type=str, help='Path to video file to play')
    parser.add_argument('-s', '--screen', action='store_true', help='Capture screen instead of playing video')
    parser.add_argument('-r', '--region', type=str, help='Screen region to capture (x,y,width,height)')
    parser.add_argument('-f', '--fps', type=int, default=60, help='Target FPS for screen capture')
    
    args = parser.parse_args()
    capture_screen = args.screen
    region = None
    if args.region:
        try:
            x, y, width, height = map(int, args.region.split(','))
            region = (x, y, width, height)
        except:
            print("Invalid region format. Use x,y,width,height (e.g. '0,0,1920,1080')")
            return
    queue_size = 1 if capture_screen else 100
    queue = multiprocessing.Queue(maxsize=queue_size)
    writer = multiprocessing.Process(target=writer_task, args=(queue,), daemon=True)
    writer.start()
    
    w, h = os.get_terminal_size()
    cfg = Config(width=w, height=h, device=torch.device('cuda'), gpu_id=0)
    byte_vals, byte_lens = setup_ascii_lookup(10000, cfg.device)
    stream = torch.cuda.Stream(device=cfg.device)
    prev_gray = None
    queue.put(b"\033[2J\033[?25l")  
    try:
        if capture_screen:
            print("Capturing screen...")
            frame_source = read_video_from_screen(cfg, stream, region=region, target_fps=args.fps)
        else:
            video_path = args.video or sys.argv[1]
            print(f"Playing video: {video_path}")
            frame_source = read_video_from_file(video_path, cfg, stream)
        
        for frame in frame_source:
            xs, ys, cols, prev_gray = process_frame(prev_gray, frame, cfg)
            
            if xs.numel():
                ansi = fused_changes_to_ansi(xs, ys, cols, byte_vals, byte_lens, cfg)
                torch.cuda.synchronize()
                torch.cuda.current_stream().synchronize()
                queue.put(ansi.cpu().numpy().tobytes())
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:    
        queue.put(None)
        writer.join()
        
if __name__ == '__main__':
    main()