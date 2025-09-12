import torch
import torch.nn.functional as F

from config import CELL_ASPECT, DEVICE, ANSI_COLORS
from utils import quantize_colors
from typing import Optional, Tuple


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


def process_frame(prev_gray: Optional[torch.Tensor], rgb: torch.Tensor, diff_thresh: int, color_mode: str = 'full') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process frame with pixel-level differencing and optimized data structures.
    color_mode: '256' for ANSI 256-color palette, 'full' for RGB colors
    """
    gray = (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]).round_().to(torch.uint8).contiguous()

    if prev_gray is None:
        height, width = gray.shape
        total_pixels = height * width
        ys = torch.arange(height, device=gray.device).repeat_interleave(width)
        xs = torch.arange(width, device=gray.device).repeat(height)
    else:
        diff = (gray.to(torch.int16) - prev_gray.to(torch.int16)).abs_()
        mask = diff > diff_thresh
        if not mask.any():
            return torch.empty(0, device=rgb.device), torch.empty(0, device=rgb.device), torch.empty(0, device=rgb.device, dtype=torch.uint8), gray
        ys, xs = mask.nonzero(as_tuple=True)

    ys = ys.contiguous()
    xs = xs.contiguous()

    colors_rgb = rgb[ys, xs].contiguous()
    if color_mode == '256':
        colors_quantized = quantize_colors(colors_rgb, ANSI_COLORS().to(colors_rgb.device))
        return xs, ys, colors_quantized, gray
    else:
        return xs, ys, colors_rgb, gray


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
