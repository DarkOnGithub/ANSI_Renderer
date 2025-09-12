import torch
import triton
import triton.language as tl
from typing import Optional, Tuple

from config import Config, ESC_VALS, SEP_VAL, CUR_END_VAL, COL_PREF_VALS, COL_256_PREF_VALS, RESET_VALS, CUR_UP_PREFIX, CUR_DOWN_PREFIX, CUR_RIGHT_PREFIX, CUR_LEFT_PREFIX, CUR_UP_SUFFIX, CUR_DOWN_SUFFIX, CUR_RIGHT_SUFFIX, CUR_LEFT_SUFFIX


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


@triton.jit
def ansi_256_run_kernel(
    run_xs_ptr, run_ys_ptr,
    run_color_idx_ptr,
    run_lengths_ptr, offsets_ptr,
    row_lookup_bytes_ptr, row_lookup_lens_ptr,
    col_lookup_bytes_ptr, col_lookup_lens_ptr,
    color_lookup_bytes_ptr, color_lookup_lens_ptr,
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
    color_idx = tl.load(run_color_idx_ptr + pid)
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

    color_len = tl.load(color_lookup_lens_ptr + color_idx)
    color_bytes_base = color_lookup_bytes_ptr + color_idx * BLOCK_SIZE
    for i in tl.static_range(BLOCK_SIZE):
        if i < color_len:
            byte = tl.load(color_bytes_base + i)
            tl.store(ptr + i, byte)
    ptr += color_len
    tl.store(ptr, 109); ptr += 1  # 'm'

    for i in range(length):
        tl.store(ptr, SPACE)
        ptr += 1


@torch.compile
def ansi_generate_256(xs: torch.Tensor, ys: torch.Tensor, color_indices: torch.Tensor, byte_vals: torch.Tensor, byte_lens: torch.Tensor, cfg: Config) -> torch.Tensor:
    N = xs.numel()
    if N == 0:
        return torch.tensor(RESET_VALS, dtype=torch.uint8, device=cfg.device)

    device = cfg.device
    key = ys * cfg.width + xs
    sort_idx = torch.argsort(key)
    xs_sorted = xs[sort_idx]
    ys_sorted = ys[sort_idx]
    color_indices_sorted = color_indices[sort_idx]

    y_diff = ys_sorted[1:] != ys_sorted[:-1]
    x_diff = xs_sorted[1:] - xs_sorted[:-1]
    color_diff = color_indices_sorted[1:] != color_indices_sorted[:-1]
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
    run_color_indices = color_indices_sorted[run_start_indices]
    row_idx = run_ys + 1
    col_idx = run_xs + 1
    color_idx = run_color_indices.to(torch.long)
    row_lens = byte_lens[row_idx]
    col_lens = byte_lens[col_idx]
    color_lens = byte_lens[color_idx]
    cursor_part = 3 + row_lens + 1 + col_lens + 1
    color_part = 7 + color_lens 
    total_lens = cursor_part + color_part + run_lengths

    offsets = torch.cat((torch.zeros(1, dtype=torch.int64, device=device), total_lens.cumsum(dim=0)[:-1]), dim=0)
    reset_len = len(RESET_VALS)
    out_buffer = torch.empty(total_lens.sum() + reset_len, dtype=torch.uint8, device=device)

    LOOKUP_MAX_LEN = byte_vals.size(1)
    grid = (num_runs,)
    ansi_256_run_kernel[grid](
        run_xs, run_ys,
        run_color_indices.to(torch.int64),
        run_lengths, offsets,
        byte_vals, byte_lens,
        byte_vals, byte_lens,
        byte_vals, byte_lens,
        out_buffer,
        num_runs,
        LOOKUP_MAX_LEN,
        ESC_VALS[0], ESC_VALS[1],
        SEP_VAL,
        CUR_END_VAL,
        COL_256_PREF_VALS[0], COL_256_PREF_VALS[1], COL_256_PREF_VALS[2],
        COL_256_PREF_VALS[3], COL_256_PREF_VALS[4], COL_256_PREF_VALS[5],
        COL_256_PREF_VALS[6],
        32,
        BLOCK_SIZE=LOOKUP_MAX_LEN,
        num_warps=4
    )

    reset_tensor = torch.tensor(RESET_VALS, dtype=torch.uint8, device=device)
    out_buffer[total_lens.sum():] = reset_tensor

    return out_buffer


@torch.compile
def ansi_generate(xs: torch.Tensor, ys: torch.Tensor, colors: torch.Tensor, byte_vals: torch.Tensor, byte_lens: torch.Tensor, cfg: Config, current_pos: Optional[tuple[int, int]] = None, color_mode: str = 'full') -> tuple[torch.Tensor, tuple[int, int]]:
    if color_mode == '256':
        return ansi_generate_256(xs, ys, colors, byte_vals, byte_lens, cfg), current_pos or (0, 0)
    else:
        return ansi_generate_rgb(xs, ys, colors, byte_vals, byte_lens, cfg), current_pos or (0, 0)


@torch.compile
def ansi_generate_rgb(xs: torch.Tensor, ys: torch.Tensor, colors: torch.Tensor, byte_vals: torch.Tensor, byte_lens: torch.Tensor, cfg: Config) -> torch.Tensor:
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
