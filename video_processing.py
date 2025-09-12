import torch
import torch.nn.functional as F

from config import CELL_ASPECT, ANSI_COLORS
from utils import quantize_colors
from typing import Optional, Tuple

# Memory pool for tensor reuse to reduce allocation overhead
_tensor_pool = {}

def get_pooled_tensor(shape, dtype, device):
    """Get a tensor from pool or create new one"""
    key = (shape, dtype, device)
    if key in _tensor_pool:
        tensor = _tensor_pool.pop(key)
        tensor.zero_()  # Clear the tensor
        return tensor
    return torch.zeros(shape, dtype=dtype, device=device)

def return_to_pool(tensor):
    """Return tensor to pool for reuse"""
    if tensor.numel() <= 1000000:  # Only pool small tensors to avoid memory bloat
        key = (tensor.shape, tensor.dtype, tensor.device)
        _tensor_pool[key] = tensor


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
    Process frame with optimized pixel-level differencing and delta encoding.
    color_mode: '256' for ANSI 256-color palette, 'full' for RGB colors
    """
    gray = (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]).round_().to(torch.uint8).contiguous()

    if prev_gray is None:
        # First frame - process all pixels but with spatial optimization
        height, width = gray.shape
        ys = torch.arange(height, device=gray.device).repeat_interleave(width)
        xs = torch.arange(width, device=gray.device).repeat(height)
    else:
        # Delta frame - only process changed pixels for massive compression
        diff = (gray.to(torch.int16) - prev_gray.to(torch.int16)).abs_()
        mask = diff > diff_thresh
        if not mask.any():
            # No changes detected - return empty for maximum compression
            return torch.empty(0, device=rgb.device), torch.empty(0, device=rgb.device), torch.empty(0, device=rgb.device, dtype=torch.uint8), gray

        # Get changed pixel coordinates
        ys, xs = mask.nonzero(as_tuple=True)

        # Spatial optimization: sort by row-major order for better cache locality
        indices = ys * rgb.shape[1] + xs
        sort_idx = torch.argsort(indices)
        ys = ys[sort_idx]
        xs = xs[sort_idx]

    ys = ys.contiguous()
    xs = xs.contiguous()

    colors_rgb = rgb[ys, xs].contiguous()

    # Ultra-aggressive color quantization for maximum compression and speed
    num_pixels = colors_rgb.numel() // 3  # Number of actual pixels

    if num_pixels > 10000:  # Massive frames - extreme quantization
        QUANTIZE_LEVEL = 16  # Every 16th value (16 values per component)
    elif num_pixels > 5000:  # Very large frames - maximum quantization
        QUANTIZE_LEVEL = 12  # Every 12th value (21 values per component)
    elif num_pixels > 2000:  # Large frames - very aggressive quantization
        QUANTIZE_LEVEL = 8  # Every 8th value (32 values per component)
    elif num_pixels > 500:  # Medium frames - aggressive quantization
        QUANTIZE_LEVEL = 6  # Every 6th value (43 values per component)
    else:  # Small frames - moderate quantization
        QUANTIZE_LEVEL = 4  # Every 4th value (64 values per component)

    # Quantize colors with optimized rounding
    colors_rgb = colors_rgb.float()
    colors_rgb.div_(QUANTIZE_LEVEL).round_().mul_(QUANTIZE_LEVEL)
    colors_rgb = colors_rgb.clamp_(0, 255).to(torch.uint8)

    if color_mode == '256':
        colors_quantized = quantize_colors(colors_rgb, ANSI_COLORS().to(colors_rgb.device))
        return xs, ys, colors_quantized, gray
    else:
        return xs, ys, colors_rgb, gray


def resize(img: torch.Tensor, target_height: int, target_width: int, cell_aspect: float = CELL_ASPECT) -> torch.Tensor:
    img_t = img.permute(2, 0, 1).unsqueeze(0).float()  # (1, C, H, W)
    resized = F.interpolate(img_t, size=(target_height, target_width), mode='bilinear', align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0).to(img.dtype)