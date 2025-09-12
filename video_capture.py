import time
import threading
import torch

from config import Config, _has_dxcam, _has_pynvcodec, dxcam, nvc
from video_processing import nv12_to_rgb, resize_with_aspect_ratio
from typing import Generator, Optional, Tuple


def read_video_from_screen(cfg: Config, stream: torch.cuda.Stream, region: Optional[Tuple[int, int, int, int]], target_fps: int, exit_event: threading.Event) -> Generator[torch.Tensor, None, None]:
    if not _has_dxcam:
        raise ImportError("DXCam not installed. Cannot capture screen.")
    camera = dxcam.create(output_color="RGB", output_idx=0)
    if not camera:
        raise RuntimeError("Failed to initialize DXCam.")
    capture_region = region if region else camera.region
    print(f"Screen capture starting: Region={capture_region}, Target FPS={target_fps}")
    camera.start(region=capture_region, target_fps=target_fps)
    try:
        while not exit_event.is_set():
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


def read_video_from_file(filename: str, cfg: Config, stream: torch.cuda.Stream, exit_event: threading.Event) -> Generator[torch.Tensor, None, None]:
    if not _has_pynvcodec:
        raise ImportError("PyNvVideoCodec not installed. Cannot play video file.")
    demuxer = nvc.CreateDemuxer(filename=filename)
    codec = demuxer.GetNvCodecId()
    decoder = nvc.CreateDecoder(gpuid=cfg.gpu_id, codec=codec, cudacontext=0, cudastream=0, usedevicememory=True)
    if not decoder:
        raise RuntimeError("Failed to create NVCUVID decoder.")
    for packet in demuxer:
        if exit_event.is_set():
            break
        for frame in decoder.Decode(packet):
            if exit_event.is_set():
                break
            with torch.cuda.stream(stream):
                nv12_tensor = torch.from_dlpack(frame).to(cfg.device)
                frame_h_nv12 = nv12_tensor.shape[0]
                frame_w = nv12_tensor.shape[1]
                frame_h = int(frame_h_nv12 * 2 / 3)
                rgb_full = nv12_to_rgb(nv12_tensor, frame_w, frame_h)
                rgb_resized = resize_with_aspect_ratio(rgb_full, cfg.height, cfg.width)
                yield rgb_resized
