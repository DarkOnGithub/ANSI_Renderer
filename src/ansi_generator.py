import torch
import triton
import triton.language as tl
from .config import Config, ESC_VALS, SEP_VAL, CUR_END_VAL, COL_PREF_VALS, RESET_VALS

@triton.jit
def ansi_run_kernel(
    run_xs_ptr, run_ys_ptr,
    run_r_ptr, run_g_ptr, run_b_ptr,
    run_lengths_ptr, offsets_ptr,
    needs_move_ptr,
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
    needs_move = tl.load(needs_move_ptr + pid)

    ptr = out_ptr + offset

    if needs_move:
        tl.store(ptr, ESC0)
        ptr += 1
        tl.store(ptr, ESC1)
        ptr += 1
        row_idx = y + 1
        row_len = tl.load(row_lookup_lens_ptr + row_idx)
        row_bytes_base = row_lookup_bytes_ptr + row_idx * BLOCK_SIZE
        for i in tl.static_range(BLOCK_SIZE):
            if i < row_len:
                byte = tl.load(row_bytes_base + i)
                tl.store(ptr + i, byte)
        ptr += row_len
        tl.store(ptr, SEP)
        ptr += 1
        col_idx = x_start + 1
        col_len = tl.load(col_lookup_lens_ptr + col_idx)
        col_bytes_base = col_lookup_bytes_ptr + col_idx * BLOCK_SIZE
        for i in tl.static_range(BLOCK_SIZE):
            if i < col_len:
                byte = tl.load(col_bytes_base + i)
                tl.store(ptr + i, byte)
        ptr += col_len
        tl.store(ptr, CUR_END)
        ptr += 1

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
    tl.store(ptr, SEP)
    ptr += 1
    g_len = tl.load(g_lookup_lens_ptr + g_val)
    g_bytes_base = g_lookup_bytes_ptr + g_val * BLOCK_SIZE
    for i in tl.static_range(BLOCK_SIZE):
        if i < g_len:
            byte = tl.load(g_bytes_base + i)
            tl.store(ptr + i, byte)
    ptr += g_len
    tl.store(ptr, SEP)
    ptr += 1
    b_len = tl.load(b_lookup_lens_ptr + b_val)
    b_bytes_base = b_lookup_bytes_ptr + b_val * BLOCK_SIZE
    for i in tl.static_range(BLOCK_SIZE):
        if i < b_len:
            byte = tl.load(b_bytes_base + i)
            tl.store(ptr + i, byte)
    ptr += b_len
    tl.store(ptr, 109)
    ptr += 1
    
    for i in range(0, 2048, 32):
        if i < length:
            off = i + tl.arange(0, 32)
            tl.store(ptr + off, SPACE, mask=off < length)

def ansi_generate(xs: torch.Tensor, ys: torch.Tensor, colors: torch.Tensor, byte_vals: torch.Tensor, byte_lens: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, tuple[int, int]]:
    result = ansi_generate_rgb(xs, ys, colors, byte_vals, byte_lens, cfg)
    return result


def ansi_generate_rgb(xs: torch.Tensor, ys: torch.Tensor, colors: torch.Tensor, byte_vals: torch.Tensor, byte_lens: torch.Tensor, cfg: Config) -> torch.Tensor:
    N = xs.numel()
    if N == 0:
        return torch.tensor(RESET_VALS, dtype=torch.uint8, device=cfg.device)

    device = cfg.device
    
    if not hasattr(cfg, '_ansi_out_buffer'):
        cfg._ansi_out_buffer = torch.empty(10 * 1024 * 1024, dtype=torch.uint8, device=device)

    spatial_key = ys.to(torch.int64) * cfg.width + xs.to(torch.int64)
    sort_idx = torch.argsort(spatial_key)
    xs_sorted = xs[sort_idx]
    ys_sorted = ys[sort_idx]
    colors_sorted = colors[sort_idx]

    y_diff = ys_sorted[1:] != ys_sorted[:-1]
    x_diff = xs_sorted[1:] - xs_sorted[:-1]
    color_diff = (colors_sorted[1:] != colors_sorted[:-1]).any(dim=1)

    is_new_run = torch.zeros(N, dtype=torch.bool, device=device)
    is_new_run[0] = True
    is_new_run[1:] = y_diff | (x_diff != 1) | color_diff
    run_start_indices = torch.where(is_new_run)[0]
    run_lengths = torch.diff(run_start_indices, append=torch.tensor([N], device=device))

    num_runs = run_start_indices.size(0)
    if num_runs == 0:
        return torch.tensor(RESET_VALS, dtype=torch.uint8, device=device)

    run_xs = xs_sorted[run_start_indices]
    run_ys = ys_sorted[run_start_indices]
    
    needs_move = torch.ones(num_runs, dtype=torch.bool, device=device)
    if num_runs > 1:
        same_row = (run_ys[1:] == run_ys[:-1])
        continuous_x = (run_xs[1:] == (run_xs[:-1] + run_lengths[:-1]))
        needs_move[1:] = ~(same_row & continuous_x)

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
    
    cursor_part = torch.where(needs_move, 2 + row_lens + 1 + col_lens + 1, torch.zeros(num_runs, dtype=torch.int64, device=device))
    color_part = 7 + r_lens + 1 + g_lens + 1 + b_lens + 1
    total_lens = cursor_part + color_part + run_lengths

    total_len_all = total_lens.sum()
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64, device=device), total_lens.cumsum(dim=0)[:-1]), dim=0)
    reset_len = len(RESET_VALS)
    
    total_size_needed_val = total_len_all.item() + reset_len
    if cfg._ansi_out_buffer.size(0) < total_size_needed_val:
        cfg._ansi_out_buffer = torch.empty(int(total_size_needed_val * 1.2), dtype=torch.uint8, device=device)
    
    out_buffer = cfg._ansi_out_buffer[:total_size_needed_val]

    LOOKUP_MAX_LEN = byte_vals.size(1)
    grid = (num_runs,)
    block_size = min(LOOKUP_MAX_LEN, 64)

    num_warps_raw = min(8, max(1, num_runs // 32))
    num_warps = 1
    if num_warps_raw >= 8:
        num_warps = 8
    elif num_warps_raw >= 4:
        num_warps = 4
    elif num_warps_raw >= 2:
        num_warps = 2

    ansi_run_kernel[grid](
        run_xs, run_ys,
        run_r.to(torch.int64), run_g.to(torch.int64), run_b.to(torch.int64),
        run_lengths, offsets,
        needs_move,
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
        BLOCK_SIZE=block_size,
        num_warps=num_warps
    )

    reset_tensor = torch.tensor(RESET_VALS, dtype=torch.uint8, device=device)
    out_buffer[total_len_all.item():] = reset_tensor

    return out_buffer