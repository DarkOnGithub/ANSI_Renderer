from typing import Optional, Tuple

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .config import Config
from .glyph_tables import OCTANT_GLYPH_SWAP
from .utils import resize_frame_keep_aspect

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
_octant_swap_cache: dict[torch.device, torch.Tensor] = {}


def _get_quadrant_masks(device: torch.device) -> torch.Tensor:
    masks = _quadrant_mask_cache.get(device)
    if masks is None:
        masks = torch.tensor(_QUADRANT_MASKS, dtype=torch.float32, device=device)
        _quadrant_mask_cache[device] = masks
    return masks


def _get_octant_swap_flags(device: torch.device) -> torch.Tensor:
    swap_flags = _octant_swap_cache.get(device)
    if swap_flags is None:
        swap_flags = torch.tensor(OCTANT_GLYPH_SWAP, dtype=torch.bool, device=device)
        _octant_swap_cache[device] = swap_flags
    return swap_flags


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

    fg_counts = masks.sum(dim=1).view(1, 1, 16, 1).clamp_min(1.0)
    bg_counts = inv_masks.sum(dim=1).view(1, 1, 16, 1).clamp_min(1.0)

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


def _encode_octant_cells(frame: torch.Tensor) -> torch.Tensor:
    if frame.shape[0] % 4:
        pad_rows = 4 - (frame.shape[0] % 4)
        frame = torch.cat([frame, frame[-1:, :, :].expand(pad_rows, -1, -1)], dim=0)
    if frame.shape[1] % 2:
        frame = torch.cat([frame, frame[:, -1:, :]], dim=1)

    slices = (
        frame[0::4, 0::2],
        frame[0::4, 1::2],
        frame[1::4, 0::2],
        frame[1::4, 1::2],
        frame[2::4, 0::2],
        frame[2::4, 1::2],
        frame[3::4, 0::2],
        frame[3::4, 1::2],
    )
    pixels = torch.stack(slices, dim=2).to(torch.float32)
    height, width = pixels.shape[:2]
    cells = pixels.view(-1, 8, 3)
    device = frame.device

    luminance = 0.299 * cells[:, :, 0] + 0.587 * cells[:, :, 1] + 0.114 * cells[:, :, 2]
    min_vals = luminance.argmin(dim=1)
    max_vals = luminance.argmax(dim=1)
    cell_indices = torch.arange(cells.size(0), device=device)

    bg_centroid = cells[cell_indices, min_vals]
    fg_centroid = cells[cell_indices, max_vals]

    cluster_mask = torch.zeros((cells.size(0), 8), dtype=torch.bool, device=device)
    for _ in range(2):
        fg_dist = ((cells - fg_centroid.unsqueeze(1)) ** 2).sum(dim=2)
        bg_dist = ((cells - bg_centroid.unsqueeze(1)) ** 2).sum(dim=2)
        cluster_mask = fg_dist <= bg_dist

        empty_fg = ~cluster_mask.any(dim=1)
        if empty_fg.any():
            cluster_mask[empty_fg, max_vals[empty_fg]] = True

        empty_bg = cluster_mask.all(dim=1)
        if empty_bg.any():
            cluster_mask[empty_bg, min_vals[empty_bg]] = False

        fg_counts = cluster_mask.sum(dim=1, keepdim=True).clamp_min(1)
        bg_mask = ~cluster_mask
        bg_counts = bg_mask.sum(dim=1, keepdim=True).clamp_min(1)

        fg_centroid = (cells * cluster_mask.unsqueeze(-1)).sum(dim=1) / fg_counts
        bg_centroid = (cells * bg_mask.unsqueeze(-1)).sum(dim=1) / bg_counts

    fg_luma = (
        0.299 * fg_centroid[:, 0]
        + 0.587 * fg_centroid[:, 1]
        + 0.114 * fg_centroid[:, 2]
    )
    bg_luma = (
        0.299 * bg_centroid[:, 0]
        + 0.587 * bg_centroid[:, 1]
        + 0.114 * bg_centroid[:, 2]
    )
    swap_clusters = fg_luma < bg_luma

    fg_mask = cluster_mask.clone()
    fg = fg_centroid.clone()
    bg = bg_centroid.clone()
    if swap_clusters.any():
        fg_mask[swap_clusters] = ~cluster_mask[swap_clusters]
        fg[swap_clusters] = bg_centroid[swap_clusters]
        bg[swap_clusters] = fg_centroid[swap_clusters]

    bit_weights = torch.tensor(
        (1, 2, 4, 8, 16, 32, 64, 128), dtype=torch.int64, device=device
    )
    mask_idx = (fg_mask.to(torch.int64) * bit_weights).sum(dim=1)

    swap_flags = _get_octant_swap_flags(device).index_select(0, mask_idx)
    fg_out = fg.clone()
    bg_out = bg.clone()
    if swap_flags.any():
        fg_out[swap_flags] = bg[swap_flags]
        bg_out[swap_flags] = fg[swap_flags]

    styles = torch.empty((cells.size(0), 7), dtype=torch.uint8, device=device)
    styles[:, 0:3] = fg_out.round().clamp(0, 255).to(torch.uint8)
    styles[:, 3:6] = bg_out.round().clamp(0, 255).to(torch.uint8)
    styles[:, 6] = mask_idx.to(torch.uint8)
    return styles.view(height, width, 7)


def _set_block_source_cache(
    config: Config, render_mode: str, frame: torch.Tensor
) -> None:
    cached = getattr(config, "_block_source_cache_frame", None)
    if cached is None or cached.shape != frame.shape or cached.device != frame.device:
        setattr(config, "_block_source_cache_frame", frame.clone())
    else:
        cached.copy_(frame)
    setattr(config, "_block_source_cache_mode", render_mode)


def _empty_block_update(
    previous_frame: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty(0, device=device, dtype=torch.int64),
        torch.empty(0, device=device, dtype=torch.int64),
        torch.empty((0, 7), device=device, dtype=torch.uint8),
        previous_frame,
    )


def pre_process_frame(
    previous_frame: Optional[torch.Tensor],
    frame: torch.Tensor,
    config: Config,
    quant_mask: Optional[int] = None,
    diff_thresh_override: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    render_mode = str(getattr(config, "render_mode", "pixel")).lower()

    if render_mode not in ("pixel", "quadrant", "octant"):
        raise ValueError(
            f"Unsupported render mode '{render_mode}'. Supported modes: pixel, quadrant, octant"
        )

    if render_mode == "quadrant":
        quadrant_cell_divisor = max(1, int(getattr(config, "quadrant_cell_divisor", 2)))
        cell_height = max(1, int(config.height) // quadrant_cell_divisor)
        cell_width = max(1, int(config.width) // quadrant_cell_divisor)
        target_height = max(1, cell_height * 2)
        target_width = max(1, cell_width * 2)
    elif render_mode == "octant":
        width_divisor = max(1, int(getattr(config, "octant_cell_width_divisor", 2)))
        height_divisor = max(1, int(getattr(config, "octant_cell_height_divisor", 4)))
        cell_height = max(1, int(config.height) // height_divisor)
        cell_width = max(1, int(config.width) // width_divisor)
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
        if render_mode == "quadrant":
            encode_cells = _encode_quadrant_cells
            source_cell_height = 2
            source_cell_width = 2
        else:
            encode_cells = _encode_octant_cells
            source_cell_height = 4
            source_cell_width = 2

        device = resized_frame.device
        cell_shape = (
            (resized_frame.shape[0] + source_cell_height - 1) // source_cell_height,
            (resized_frame.shape[1] + source_cell_width - 1) // source_cell_width,
            7,
        )
        cached_source = getattr(config, "_block_source_cache_frame", None)
        cache_matches = (
            cached_source is not None
            and getattr(config, "_block_source_cache_mode", None) == render_mode
            and cached_source.shape == resized_frame.shape
            and cached_source.device == resized_frame.device
        )

        if (
            previous_frame is None
            or previous_frame.shape != cell_shape
            or not cache_matches
        ):
            cell_styles = encode_cells(resized_frame)
            _set_block_source_cache(config, render_mode, resized_frame)
            height, width = cell_styles.shape[:2]
            ys = torch.arange(height, device=device).repeat_interleave(width)
            xs = torch.arange(width, device=device).repeat(height)
            styles = cell_styles[ys, xs]
            return xs, ys, styles, cell_styles

        source_diff = (resized_frame != cached_source).any(dim=-1)
        if not source_diff.any():
            return _empty_block_update(previous_frame, device)

        dirty_cell_mask = (
            F.max_pool2d(
                source_diff.to(torch.float32).unsqueeze(0).unsqueeze(0),
                kernel_size=(source_cell_height, source_cell_width),
                stride=(source_cell_height, source_cell_width),
                ceil_mode=True,
            )
            .squeeze(0)
            .squeeze(0)
            .to(torch.bool)
        )
        dirty_cell_ys, dirty_cell_xs = dirty_cell_mask.nonzero(as_tuple=True)
        cell_y0 = int(dirty_cell_ys.min().item())
        cell_y1 = int(dirty_cell_ys.max().item()) + 1
        cell_x0 = int(dirty_cell_xs.min().item())
        cell_x1 = int(dirty_cell_xs.max().item()) + 1

        src_y0 = cell_y0 * source_cell_height
        src_y1 = min(cell_y1 * source_cell_height, resized_frame.shape[0])
        src_x0 = cell_x0 * source_cell_width
        src_x1 = min(cell_x1 * source_cell_width, resized_frame.shape[1])

        cell_styles = encode_cells(resized_frame[src_y0:src_y1, src_x0:src_x1])
        previous_slice = previous_frame[cell_y0:cell_y1, cell_x0:cell_x1]

        diff_thresh = (
            config.diff_thresh
            if diff_thresh_override is None
            else int(diff_thresh_override)
        )

        glyph_diff = cell_styles[..., 6] != previous_slice[..., 6]
        if diff_thresh <= 0:
            color_diff = (cell_styles[..., :6] != previous_slice[..., :6]).any(dim=-1)
        else:
            color_diff = torch.abs(
                cell_styles[..., :6].to(torch.int16)
                - previous_slice[..., :6].to(torch.int16)
            ).amax(dim=-1) > int(diff_thresh)
        diff_mask = glyph_diff | color_diff

        _set_block_source_cache(config, render_mode, resized_frame)

        if not diff_mask.any():
            return _empty_block_update(previous_frame, device)

        ys, xs = diff_mask.nonzero(as_tuple=True)
        ys = ys + cell_y0
        xs = xs + cell_x0
        styles = cell_styles[ys - cell_y0, xs - cell_x0]
        previous_frame[ys, xs] = styles
        return xs, ys, styles, previous_frame

    device = resized_frame.device

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
