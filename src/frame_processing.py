from typing import Optional, Tuple
import torch
from utils import resize_frame_keep_aspect
from config import Config


"""
Process frame with pixel-level differencing and delta encoding.
Note: the frame is resized to the config.height and config.width.
Returns:
    xs: torch.Tensor - x coordinates of pixels to render
    ys: torch.Tensor - y coordinates of pixels to render
    colors_rgb: torch.Tensor - colors in RGB format of pixels to render
    resized_frame: torch.Tensor - resized RGB frame
args:
    previous_frame: Optional[torch.Tensor] - previous frame, has to be RGB
    frame: torch.Tensor - current frame, has to be RGB
    config: Config - configuration
"""
def pre_process_frame(
    previous_frame: Optional[torch.Tensor], 
    frame: torch.Tensor, 
    config: Config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    resized_frame = resize_frame_keep_aspect(frame, config.height, config.width)
    device = resized_frame.device
    if previous_frame is None:
        height, width = resized_frame.shape[:2]
        ys = torch.arange(height, device=device).repeat_interleave(width)
        xs = torch.arange(width, device=device).repeat(height)
    else:
        if config.diff_thresh <= 0:
            mask = (resized_frame != previous_frame).any(dim=-1)
        else:
            thresh = int(config.diff_thresh)
            rf0 = resized_frame[..., 0].to(torch.int16)
            pf0 = previous_frame[..., 0].to(torch.int16)
            mask = (rf0 - pf0).abs_() > thresh
            if not mask.all():
                rf1 = resized_frame[..., 1].to(torch.int16)
                pf1 = previous_frame[..., 1].to(torch.int16)
                mask |= (rf1 - pf1).abs_() > thresh
                if not mask.all():
                    rf2 = resized_frame[..., 2].to(torch.int16)
                    pf2 = previous_frame[..., 2].to(torch.int16)
                    mask |= (rf2 - pf2).abs_() > thresh

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
