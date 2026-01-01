from typing import Optional, Tuple
import torch
from .utils import resize_frame_keep_aspect
from .config import Config


def pre_process_frame(
    previous_frame: Optional[torch.Tensor], 
    frame: torch.Tensor, 
    config: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    resized_frame = resize_frame_keep_aspect(frame, config.height, config.width)
    device = resized_frame.device
    
    if previous_frame is None or previous_frame.shape != resized_frame.shape:
        height, width = resized_frame.shape[:2]
        ys = torch.arange(height, device=device).repeat_interleave(width)
        xs = torch.arange(width, device=device).repeat(height)
    else:
        if config.diff_thresh <= 0:
            mask = (resized_frame != previous_frame).any(dim=-1)
        else:
            thresh = int(config.diff_thresh)
            mask = torch.any(torch.abs(resized_frame.to(torch.int16) - previous_frame.to(torch.int16)) > thresh, dim=-1)

        if not mask.any():
            return (
                torch.empty(0, device=device, dtype=torch.int64),
                torch.empty(0, device=device, dtype=torch.int64),
                torch.empty(0, device=device, dtype=torch.uint8),
                resized_frame,
            )
        ys, xs = mask.nonzero(as_tuple=True)
    colors_rgb = resized_frame[ys, xs]
    return xs, ys, colors_rgb, resized_frame
