from typing import Optional, Tuple
import torch
from .utils import resize_frame_keep_aspect
from .config import Config

_QUADRANT_MASKS = (
    (0, 0, 0, 0),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (1, 1, 0, 0),
    (0, 0, 1, 0),
    (1, 0, 1, 0),
    (0, 1, 1, 0),
    (1, 1, 1, 0),
    (0, 0, 0, 1),
    (1, 0, 0, 1),
    (0, 1, 0, 1),
    (1, 1, 0, 1),
    (0, 0, 1, 1),
    (1, 0, 1, 1),
    (0, 1, 1, 1),
    (1, 1, 1, 1),
)

_quadrant_mask_cache: dict[torch.device, torch.Tensor] = {}

# Precomputed 256-color palette RGB values (indices 16–255)
# Cached per-device for fast nearest-neighbor lookup
_palette_cache: dict[torch.device, torch.Tensor] = {}

# Grayscale ramp: indices 232–255 map to luminances 8, 18, 28, …, 238
_GRAY_VALS = tuple(8 + 10 * i for i in range(24))


def _get_palette_rgb(device: torch.device) -> torch.Tensor:
    """Return (256, 3) uint8 tensor of the standard 256-color palette."""
    pal = _palette_cache.get(device)
    if pal is not None:
        return pal

    rgb = torch.zeros((256, 3), dtype=torch.uint8, device=device)
    # 6x6x6 color cube (indices 16–231)
    cube_steps = torch.tensor([0, 95, 135, 175, 215, 255], dtype=torch.uint8, device=device)
    idx = 16
    for ri in range(6):
        for gi in range(6):
            for bi in range(6):
                rgb[idx, 0] = cube_steps[ri]
                rgb[idx, 1] = cube_steps[gi]
                rgb[idx, 2] = cube_steps[bi]
                idx += 1
    # grayscale ramp (indices 232–255)
    for i, v in enumerate(_GRAY_VALS):
        rgb[232 + i] = v

    _palette_cache[device] = rgb
    return rgb


def rgb_to_256(frame: torch.Tensor) -> torch.Tensor:
    """Map an (H, W, 3) uint8 RGB frame to (H, W) uint8 palette indices.

    Uses a fast vectorised approach:
    - Compute the 6x6x6 cube index for every pixel
    - Compute the nearest grayscale ramp index for every pixel
    - Pick whichever has the lower squared error
    """
    device = frame.device
    shape = frame.shape[:2]  # (H, W)
    r = frame[..., 0].to(torch.int32)
    g = frame[..., 1].to(torch.int32)
    b = frame[..., 2].to(torch.int32)

    # --- 6x6x6 cube quantisation ---
    # Map 0–255 → 0–5.  The cube steps are [0, 95, 135, 175, 215, 255].
    # Boundaries (midpoints): 47.5, 115, 155, 195, 235
    bounds = torch.tensor([48, 115, 155, 195, 235], dtype=torch.int32, device=device)
    cube_steps_i32 = torch.tensor([0, 95, 135, 175, 215, 255], dtype=torch.int32, device=device)

    ri = torch.bucketize(r, bounds)
    gi = torch.bucketize(g, bounds)
    bi = torch.bucketize(b, bounds)

    cube_idx = (16 + 36 * ri + 6 * gi + bi).to(torch.int64)

    # Reconstruct RGB of chosen cube colour for error calculation
    cube_r = cube_steps_i32[ri]
    cube_g = cube_steps_i32[gi]
    cube_b = cube_steps_i32[bi]
    cube_err = (r - cube_r) ** 2 + (g - cube_g) ** 2 + (b - cube_b) ** 2

    # --- Grayscale ramp quantisation ---
    lum = (r + g + b) // 3  # rough luminance
    # Ramp values: 8, 18, 28, …, 238  →  nearest index = clamp(round((lum - 8) / 10))
    gray_i = ((lum - 8 + 5) // 10).clamp(0, 23)
    gray_val = (8 + 10 * gray_i).to(torch.int32)
    gray_err = (r - gray_val) ** 2 + (g - gray_val) ** 2 + (b - gray_val) ** 2
    gray_idx = (232 + gray_i).to(torch.int64)

    # Pick whichever has less error
    use_gray = gray_err < cube_err
    result = torch.where(use_gray, gray_idx, cube_idx).to(torch.uint8)
    return result


def _get_quadrant_masks(device: torch.device) -> torch.Tensor:
    masks = _quadrant_mask_cache.get(device)
    if masks is None:
        masks = torch.tensor(_QUADRANT_MASKS, dtype=torch.float32, device=device)
        _quadrant_mask_cache[device] = masks
    return masks


def _encode_quadrant_cells(frame: torch.Tensor) -> torch.Tensor:
    if frame.shape[0] % 2 == 1:
        frame = torch.cat([frame, frame[-1:, :, :]], dim=0)
    if frame.shape[1] % 2 == 1:
        frame = torch.cat([frame, frame[:, -1:, :]], dim=1)

    tl = frame[0::2, 0::2]
    tr = frame[0::2, 1::2]
    bl = frame[1::2, 0::2]
    br = frame[1::2, 1::2]

    pixels = torch.stack((tl, tr, bl, br), dim=2).to(torch.float32)
    masks = _get_quadrant_masks(frame.device)
    inv_masks = 1.0 - masks

    pixels_exp = pixels.unsqueeze(2)
    masks_exp = masks.view(1, 1, 16, 4, 1)
    inv_masks_exp = inv_masks.view(1, 1, 16, 4, 1)

    fg_counts = masks.sum(dim=1, keepdim=False).view(1, 1, 16, 1).clamp_min(1.0)
    bg_counts = inv_masks.sum(dim=1, keepdim=False).view(1, 1, 16, 1).clamp_min(1.0)

    fg = (pixels_exp * masks_exp).sum(dim=3) / fg_counts
    bg = (pixels_exp * inv_masks_exp).sum(dim=3) / bg_counts

    assigned = torch.where(masks_exp.bool(), fg.unsqueeze(3), bg.unsqueeze(3))
    errors = ((pixels_exp - assigned) ** 2).sum(dim=(3, 4))

    glyph_idx = errors.argmin(dim=2)
    gather_idx = glyph_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)

    fg_best = fg.gather(dim=2, index=gather_idx).squeeze(2)
    bg_best = bg.gather(dim=2, index=gather_idx).squeeze(2)

    styles = torch.empty(
        (glyph_idx.shape[0], glyph_idx.shape[1], 7),
        dtype=torch.uint8,
        device=frame.device,
    )
    styles[..., 0:3] = fg_best.round().clamp(0, 255).to(torch.uint8)
    styles[..., 3:6] = bg_best.round().clamp(0, 255).to(torch.uint8)
    styles[..., 6] = glyph_idx.to(torch.uint8)
    return styles


def _encode_quadrant_cells_256(frame: torch.Tensor) -> torch.Tensor:
    """Encode quadrant cells for 256-color mode.

    Input frame is (H, W, 3) uint8 RGB. Output is (H/2, W/2, 3) uint8
    where channels are [fg_palette_idx, bg_palette_idx, glyph_idx].
    """
    if frame.shape[0] % 2 == 1:
        frame = torch.cat([frame, frame[-1:, :, :]], dim=0)
    if frame.shape[1] % 2 == 1:
        frame = torch.cat([frame, frame[:, -1:, :]], dim=1)

    tl = frame[0::2, 0::2]
    tr = frame[0::2, 1::2]
    bl = frame[1::2, 0::2]
    br = frame[1::2, 1::2]

    pixels = torch.stack((tl, tr, bl, br), dim=2).to(torch.float32)
    masks = _get_quadrant_masks(frame.device)
    inv_masks = 1.0 - masks

    pixels_exp = pixels.unsqueeze(2)
    masks_exp = masks.view(1, 1, 16, 4, 1)
    inv_masks_exp = inv_masks.view(1, 1, 16, 4, 1)

    fg_counts = masks.sum(dim=1, keepdim=False).view(1, 1, 16, 1).clamp_min(1.0)
    bg_counts = inv_masks.sum(dim=1, keepdim=False).view(1, 1, 16, 1).clamp_min(1.0)

    fg = (pixels_exp * masks_exp).sum(dim=3) / fg_counts
    bg = (pixels_exp * inv_masks_exp).sum(dim=3) / bg_counts

    assigned = torch.where(masks_exp.bool(), fg.unsqueeze(3), bg.unsqueeze(3))
    errors = ((pixels_exp - assigned) ** 2).sum(dim=(3, 4))

    glyph_idx = errors.argmin(dim=2)
    gather_idx = glyph_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)

    fg_best = fg.gather(dim=2, index=gather_idx).squeeze(2)
    bg_best = bg.gather(dim=2, index=gather_idx).squeeze(2)

    # Quantize the fg/bg mean RGB to 256-color indices
    fg_rgb = fg_best.round().clamp(0, 255).to(torch.uint8)
    bg_rgb = bg_best.round().clamp(0, 255).to(torch.uint8)

    fg_idx = rgb_to_256(fg_rgb)  # (H/2, W/2)
    bg_idx = rgb_to_256(bg_rgb)  # (H/2, W/2)

    styles = torch.empty(
        (glyph_idx.shape[0], glyph_idx.shape[1], 3),
        dtype=torch.uint8,
        device=frame.device,
    )
    styles[..., 0] = fg_idx
    styles[..., 1] = bg_idx
    styles[..., 2] = glyph_idx.to(torch.uint8)
    return styles


# ---------------------------------------------------------------------------
# Octant (4×2) cell encoding
# ---------------------------------------------------------------------------
_OCTANT_MASKS = tuple(
    tuple((pattern >> bit) & 1 for bit in range(8))
    for pattern in range(256)
)

_octant_mask_cache: dict[torch.device, torch.Tensor] = {}


def _get_octant_masks(device: torch.device) -> torch.Tensor:
    masks = _octant_mask_cache.get(device)
    if masks is None:
        masks = torch.tensor(_OCTANT_MASKS, dtype=torch.float32, device=device)
        _octant_mask_cache[device] = masks
    return masks


def _octant_pixels(frame: torch.Tensor):
    """Pad frame and extract 8 sub-pixels per octant cell.

    Returns pixels (cH, cW, 8, 3) float32 and cell dims (cH, cW).
    """
    if frame.shape[0] % 4 != 0:
        pad_rows = 4 - (frame.shape[0] % 4)
        frame = torch.cat([frame] + [frame[-1:, :, :]] * pad_rows, dim=0)
    if frame.shape[1] % 2 == 1:
        frame = torch.cat([frame, frame[:, -1:, :]], dim=1)

    H, W = frame.shape[:2]
    cH, cW = H // 4, W // 2

    # Row-major: pos 0=(r0,c0), 1=(r0,c1), 2=(r1,c0), ..., 7=(r3,c1)
    p = [frame[r::4, c::2] for r in range(4) for c in range(2)]
    pixels = torch.stack(p, dim=2).to(torch.float32)  # (cH, cW, 8, 3)
    return pixels, cH, cW


def _octant_best_fg_bg(pixels: torch.Tensor, masks: torch.Tensor):
    """Vectorized 2-means for all 256 octant patterns simultaneously.

    Uses the identity: MSE_c = total_sq - ||fg_sum_c||²/n_fg - ||bg_sum_c||²/n_bg
    so we maximise the last two terms.  All 256 patterns evaluated in one einsum.

    Returns (best_glyph, best_fg, best_bg) — int64, float32, float32.
    """
    cH, cW = pixels.shape[:2]
    device = pixels.device

    # fg_sum[h,w,p,c] = sum_i masks[p,i] * pixels[h,w,i,c]  (256 patterns at once)
    fg_sum = torch.einsum("pi,hwic->hwpc", masks, pixels)   # (cH, cW, 256, 3)

    total_sum = pixels.sum(dim=2)                            # (cH, cW, 3)
    bg_sum = total_sum.unsqueeze(2) - fg_sum                 # (cH, cW, 256, 3)

    n_fg = masks.sum(dim=1).clamp_min(1.0)                  # (256,)
    n_bg = (8.0 - masks.sum(dim=1)).clamp_min(1.0)          # (256,)

    # score to maximise (higher = lower MSE)
    score = (
        (fg_sum ** 2).sum(dim=-1) / n_fg                    # (cH, cW, 256)
        + (bg_sum ** 2).sum(dim=-1) / n_bg
    )

    best_glyph = score.argmax(dim=2)                         # (cH, cW)

    gi = best_glyph.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
    n_fg_best = n_fg[best_glyph].unsqueeze(-1)               # (cH, cW, 1)
    n_bg_best = n_bg[best_glyph].unsqueeze(-1)

    best_fg = fg_sum.gather(2, gi).squeeze(2) / n_fg_best   # (cH, cW, 3)
    best_bg = bg_sum.gather(2, gi).squeeze(2) / n_bg_best

    return best_glyph, best_fg, best_bg


def _encode_octant_cells(frame: torch.Tensor) -> torch.Tensor:
    """Encode octant (4×2) cells for truecolor mode.

    Input:  (H, W, 3) uint8 RGB.
    Output: (H/4, W/2, 7) uint8 — [fg_r, fg_g, fg_b, bg_r, bg_g, bg_b, glyph].
    """
    pixels, cH, cW = _octant_pixels(frame)
    masks = _get_octant_masks(frame.device)
    best_glyph, best_fg, best_bg = _octant_best_fg_bg(pixels, masks)

    styles = torch.empty((cH, cW, 7), dtype=torch.uint8, device=frame.device)
    styles[..., 0:3] = best_fg.round().clamp(0, 255).to(torch.uint8)
    styles[..., 3:6] = best_bg.round().clamp(0, 255).to(torch.uint8)
    styles[..., 6] = best_glyph.to(torch.uint8)
    return styles


def _encode_octant_cells_256(frame: torch.Tensor) -> torch.Tensor:
    """Encode octant (4×2) cells for 256-color mode.

    Input:  (H, W, 3) uint8 RGB.
    Output: (H/4, W/2, 3) uint8 — [fg_palette_idx, bg_palette_idx, glyph].
    """
    pixels, cH, cW = _octant_pixels(frame)
    masks = _get_octant_masks(frame.device)
    best_glyph, best_fg, best_bg = _octant_best_fg_bg(pixels, masks)

    fg_rgb = best_fg.round().clamp(0, 255).to(torch.uint8)
    bg_rgb = best_bg.round().clamp(0, 255).to(torch.uint8)

    styles = torch.empty((cH, cW, 3), dtype=torch.uint8, device=frame.device)
    styles[..., 0] = rgb_to_256(fg_rgb)
    styles[..., 1] = rgb_to_256(bg_rgb)
    styles[..., 2] = best_glyph.to(torch.uint8)
    return styles


def pre_process_frame(
    previous_frame: Optional[torch.Tensor],
    frame: torch.Tensor,
    config: Config,
    quant_mask: Optional[int] = None,
    diff_thresh_override: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    render_mode = str(getattr(config, "render_mode", "pixel")).lower()
    color_mode = str(getattr(config, "color_mode", "truecolor")).lower()
    is_256 = color_mode == "256"

    if render_mode == "quadrant":
        quadrant_cell_divisor = max(1, int(getattr(config, "quadrant_cell_divisor", 2)))
        cell_height = max(1, int(config.height) // quadrant_cell_divisor)
        cell_width = max(1, int(config.width) // quadrant_cell_divisor)
        target_height = max(1, cell_height * 2)
        target_width = max(1, cell_width * 2)
    elif render_mode == "octant":
        quadrant_cell_divisor = max(1, int(getattr(config, "quadrant_cell_divisor", 2)))
        cell_height = max(1, int(config.height) // quadrant_cell_divisor)
        cell_width = max(1, int(config.width) // quadrant_cell_divisor)
        target_height = max(1, cell_height * 4)
        target_width = max(1, cell_width * 2)
    else:
        target_height = int(config.height)
        target_width = int(config.width)

    resized_frame = resize_frame_keep_aspect(frame, target_height, target_width)

    quant_mask_value = config.quant_mask if quant_mask is None else int(quant_mask)
    quant_mask_value = max(0, min(quant_mask_value, 0xFF))
    if quant_mask_value != 0xFF:
        resized_frame = resized_frame & quant_mask_value

    if render_mode in ("quadrant", "octant"):
        if render_mode == "octant":
            if is_256:
                cell_styles = _encode_octant_cells_256(resized_frame)
            else:
                cell_styles = _encode_octant_cells(resized_frame)
        else:
            if is_256:
                cell_styles = _encode_quadrant_cells_256(resized_frame)
            else:
                cell_styles = _encode_quadrant_cells(resized_frame)
        device = cell_styles.device
        style_channels = cell_styles.shape[-1]  # 3 for 256, 7 for truecolor

        if previous_frame is None or previous_frame.shape != cell_styles.shape:
            height, width = cell_styles.shape[:2]
            ys = torch.arange(height, device=device).repeat_interleave(width)
            xs = torch.arange(width, device=device).repeat(height)
            styles = cell_styles[ys, xs]
            return xs, ys, styles, cell_styles

        diff_thresh = (
            config.diff_thresh
            if diff_thresh_override is None
            else int(diff_thresh_override)
        )
        if is_256:
            # For 256-color quadrant: channels are [fg_idx, bg_idx, glyph_idx]
            diff_mask = (cell_styles != previous_frame).any(dim=-1)
        else:
            glyph_diff = cell_styles[..., 6] != previous_frame[..., 6]
            if diff_thresh <= 0:
                color_diff = (cell_styles[..., :6] != previous_frame[..., :6]).any(dim=-1)
            else:
                color_diff = torch.abs(
                    cell_styles[..., :6].to(torch.int16)
                    - previous_frame[..., :6].to(torch.int16)
                ).amax(dim=-1) > int(diff_thresh)
            diff_mask = glyph_diff | color_diff

        if not diff_mask.any():
            return (
                torch.empty(0, device=device, dtype=torch.int64),
                torch.empty(0, device=device, dtype=torch.int64),
                torch.empty((0, style_channels), device=device, dtype=torch.uint8),
                previous_frame,
            )

        ys, xs = diff_mask.nonzero(as_tuple=True)
        styles = cell_styles[ys, xs]
        previous_frame[ys, xs] = styles
        return xs, ys, styles, previous_frame

    # --- Pixel mode ---
    device = resized_frame.device

    if is_256:
        # Quantize to 256-color palette indices (H, W) uint8
        quantized = rgb_to_256(resized_frame)

        if previous_frame is None or previous_frame.shape != quantized.shape:
            height, width = quantized.shape[:2]
            ys = torch.arange(height, device=device).repeat_interleave(width)
            xs = torch.arange(width, device=device).repeat(height)
            colors = quantized[ys, xs]
            return xs, ys, colors, quantized

        diff_mask = quantized != previous_frame
        if not diff_mask.any():
            return (
                torch.empty(0, device=device, dtype=torch.int64),
                torch.empty(0, device=device, dtype=torch.int64),
                torch.empty(0, device=device, dtype=torch.uint8),
                previous_frame,
            )

        ys, xs = diff_mask.nonzero(as_tuple=True)
        colors = quantized[ys, xs]
        previous_frame[ys, xs] = colors
        return xs, ys, colors, previous_frame

    # --- Truecolor pixel mode (original) ---
    if previous_frame is None or previous_frame.shape != resized_frame.shape:
        height, width = resized_frame.shape[:2]
        ys = torch.arange(height, device=device).repeat_interleave(width)
        xs = torch.arange(width, device=device).repeat(height)
        colors_rgb = resized_frame[ys, xs]
        return xs, ys, colors_rgb, resized_frame

    diff_thresh = (
        config.diff_thresh
        if diff_thresh_override is None
        else int(diff_thresh_override)
    )
    if diff_thresh <= 0:
        diff_mask = (resized_frame != previous_frame).any(dim=-1)
    else:
        thresh = int(diff_thresh)
        diff_mask = torch.any(
            torch.abs(resized_frame.to(torch.int16) - previous_frame.to(torch.int16))
            > thresh,
            dim=-1,
        )

    if not diff_mask.any():
        return (
            torch.empty(0, device=device, dtype=torch.int64),
            torch.empty(0, device=device, dtype=torch.int64),
            torch.empty(0, device=device, dtype=torch.uint8),
            previous_frame,
        )

    ys, xs = diff_mask.nonzero(as_tuple=True)
    colors_rgb = resized_frame[ys, xs]
    previous_frame[ys, xs] = colors_rgb
    return xs, ys, colors_rgb, previous_frame
