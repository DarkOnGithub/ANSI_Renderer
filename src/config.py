import sys
import torch
from dataclasses import dataclass

torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
torch.backends.cuda.matmul.allow_tf32 = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ESC_VALS = (27, 91)
SEP_VAL = 59
CUR_END_VAL = 72
FG_COL_PREF_VALS = (27, 91, 51, 56, 59, 50, 59)  # \e[38;2;
COL_PREF_VALS = (27, 91, 52, 56, 59, 50, 59)  # \e[48;2;
COL_SUF_VALS = (109, 32)
RESET_VALS = (27, 91, 48, 109)
CUR_UP_PREFIX = (27, 91)  # \e[
CUR_DOWN_PREFIX = (27, 91)  # \e[
CUR_RIGHT_PREFIX = (27, 91)  # \e[
CUR_LEFT_PREFIX = (27, 91)  # \e[
CUR_UP_SUFFIX = (65,)  # A
CUR_DOWN_SUFFIX = (66,)  # B
CUR_RIGHT_SUFFIX = (67,)  # C
CUR_LEFT_SUFFIX = (68,)  # D
HIDE_CURSOR = b"\033[?25l"
SHOW_CURSOR = b"\033[?25h"
CLEAR_SCREEN = b"\033[2J"
ENABLE_ALT_BUFFER = b"\033[?1049h"
DISABLE_ALT_BUFFER = b"\033[?1049l"
SYNC_OUTPUT_BEGIN = b"\033[?2026h"
SYNC_OUTPUT_END = b"\033[?2026l"
CELL_ASPECT = 0.5
QUADRANT_GLYPHS = (
    " ",
    "▘",
    "▝",
    "▀",
    "▖",
    "▌",
    "▞",
    "▛",
    "▗",
    "▚",
    "▐",
    "▜",
    "▄",
    "▙",
    "▟",
    "█",
)


@dataclass
class Config:
    width: int
    height: int
    device: torch.device
    fps: float = 30.0
    audio_path: str = None
    diff_thresh: int = 0
    quant_mask: int = 0xFF
    run_color_diff_thresh: int = 0
    relative_cursor_moves: bool = True
    use_rep: bool = False
    rep_min_run: int = 12
    output_fd: int = sys.stdout.fileno()
    timing_file: str = "timing.csv"
    timing_enabled: bool = False
    timing_flush_interval: int = 30
    write_chunk_size: int = 262144
    prefer_writev: bool = True
    queue_size: int = 4
    buffer_pool_size: int = 6
    initial_buffer_size: int = 10 * 1024 * 1024
    async_copy_stream: bool = True
    sync_output: bool = True
    pacing_render_lead: bool = True
    pacing_render_alpha: float = 0.2
    print_delay: float = 0.0
    audio_delay: float = 0.0

    adaptive_quality: bool = False
    adaptive_quant_masks: tuple[int, ...] = (0xFF, 0xF8, 0xF0)
    adaptive_diff_thresh_offsets: tuple[int, ...] = (0, 1, 2)
    adaptive_run_color_diff_offsets: tuple[int, ...] = (0, 2, 5)
    adaptive_ema_alpha: float = 0.15
    target_frame_bytes: int = 0
    frame_byte_buffer_frames: int = 4
    max_frame_bytes: int = 0
    cap_staleness_weight: float = 8.0
    cap_staleness_max: int = 255
    cap_density_weight: float = 24.0
    render_mode: str = "pixel"
    quadrant_cell_divisor: int = 2
    octant_cell_width_divisor: int = 2
    octant_cell_height_divisor: int = 4
