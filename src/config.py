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
COL_256_PREF_VALS = (27, 91, 51, 56, 59, 53, 59)  # \e[38;5;
BG_COL_256_PREF_VALS = (27, 91, 52, 56, 59, 53, 59)  # \e[48;5;
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
# Octant glyphs: 256 entries for 4x2 sub-pixel patterns (bit i = position i+1).
# Grid:  1|2 / 3|4 / 5|6 / 7|8.  Uses Unicode 16.0 Block Octant chars + pre-existing block elements.
OCTANT_GLYPHS = (
    " ", "\U0001FB00", "\U0001FB01", "\U0001FB82", "\U0001CD00", "\u2598", "\U0001CD01", "\U0001CD02",
    "\U0001CD03", "\U0001CD04", "\u259D", "\U0001CD05", "\U0001CD06", "\U0001CD07", "\U0001CD08", "\u2580",
    "\U0001CD09", "\U0001CD0A", "\U0001CD0B", "\U0001CD0C", "\U0001FBE6", "\U0001CD0D", "\U0001CD0E", "\U0001CD0F",
    "\U0001CD10", "\U0001CD11", "\U0001CD12", "\U0001CD13", "\U0001CD14", "\U0001CD15", "\U0001CD16", "\U0001CD17",
    "\U0001CD18", "\U0001CD19", "\U0001CD1A", "\U0001CD1B", "\U0001CD1C", "\U0001CD1D", "\U0001CD1E", "\U0001CD1F",
    "\U0001FBE7", "\U0001CD20", "\U0001CD21", "\U0001CD22", "\U0001CD23", "\U0001CD24", "\U0001CD25", "\U0001CD26",
    "\U0001CD27", "\U0001CD28", "\U0001CD29", "\U0001CD2A", "\U0001CD2B", "\U0001CD2C", "\U0001CD2D", "\U0001CD2E",
    "\U0001CD2F", "\U0001CD30", "\U0001CD31", "\U0001CD32", "\U0001CD33", "\U0001CD34", "\U0001CD35", "\U0001FB84",
    "\U0001FB0F", "\U0001CD36", "\U0001CD37", "\U0001CD38", "\U0001CD39", "\U0001CD3A", "\U0001CD3B", "\U0001CD3C",
    "\U0001CD3D", "\U0001CD3E", "\U0001CD3F", "\U0001CD40", "\U0001CD41", "\U0001CD42", "\U0001CD43", "\U0001CD44",
    "\u2596", "\U0001CD45", "\U0001CD46", "\U0001CD47", "\U0001CD48", "\u258C", "\U0001CD49", "\U0001CD4A",
    "\U0001CD4B", "\U0001CD4C", "\u259E", "\U0001CD4D", "\U0001CD4E", "\U0001CD4F", "\U0001CD50", "\u259B",
    "\U0001CD51", "\U0001CD52", "\U0001CD53", "\U0001CD54", "\U0001CD55", "\U0001CD56", "\U0001CD57", "\U0001CD58",
    "\U0001CD59", "\U0001CD5A", "\U0001CD5B", "\U0001CD5C", "\U0001CD5D", "\U0001CD5E", "\U0001CD5F", "\U0001CD60",
    "\U0001CD61", "\U0001CD62", "\U0001CD63", "\U0001CD64", "\U0001CD65", "\U0001CD66", "\U0001CD67", "\U0001CD68",
    "\U0001CD69", "\U0001CD6A", "\U0001CD6B", "\U0001CD6C", "\U0001CD6D", "\U0001CD6E", "\U0001CD6F", "\U0001CD70",
    "\U0001FB1E", "\U0001CD71", "\U0001CD72", "\U0001CD73", "\U0001CD74", "\U0001CD75", "\U0001CD76", "\U0001CD77",
    "\U0001CD78", "\U0001CD79", "\U0001CD7A", "\U0001CD7B", "\U0001CD7C", "\U0001CD7D", "\U0001CD7E", "\U0001CD7F",
    "\U0001CD80", "\U0001CD81", "\U0001CD82", "\U0001CD83", "\U0001CD84", "\U0001CD85", "\U0001CD86", "\U0001CD87",
    "\U0001CD88", "\U0001CD89", "\U0001CD8A", "\U0001CD8B", "\U0001CD8C", "\U0001CD8D", "\U0001CD8E", "\U0001CD8F",
    "\u2597", "\U0001CD90", "\U0001CD91", "\U0001CD92", "\U0001CD93", "\u259A", "\U0001CD94", "\U0001CD95",
    "\U0001CD96", "\U0001CD97", "\u2590", "\U0001CD98", "\U0001CD99", "\U0001CD9A", "\U0001CD9B", "\u259C",
    "\U0001CD9C", "\U0001CD9D", "\U0001CD9E", "\U0001CD9F", "\U0001CDA0", "\U0001CDA1", "\U0001CDA2", "\U0001CDA3",
    "\U0001CDA4", "\U0001CDA5", "\U0001CDA6", "\U0001CDA7", "\U0001CDA8", "\U0001CDA9", "\U0001CDAA", "\U0001CDAB",
    "\u2582", "\U0001CDAC", "\U0001CDAD", "\U0001CDAE", "\U0001CDAF", "\U0001CDB0", "\U0001CDB1", "\U0001CDB2",
    "\U0001CDB3", "\U0001CDB4", "\U0001CDB5", "\U0001CDB6", "\U0001CDB7", "\U0001CDB8", "\U0001CDB9", "\U0001CDBA",
    "\U0001CDBB", "\U0001CDBC", "\U0001CDBD", "\U0001CDBE", "\U0001CDBF", "\U0001CDC0", "\U0001CDC1", "\U0001CDC2",
    "\U0001CDC3", "\U0001CDC4", "\U0001CDC5", "\U0001CDC6", "\U0001CDC7", "\U0001CDC8", "\U0001CDC9", "\U0001CDCA",
    "\U0001CDCB", "\U0001CDCC", "\U0001CDCD", "\U0001CDCE", "\U0001CDCF", "\U0001CDD0", "\U0001CDD1", "\U0001CDD2",
    "\U0001CDD3", "\U0001CDD4", "\U0001CDD5", "\U0001CDD6", "\U0001CDD7", "\U0001CDD8", "\U0001CDD9", "\U0001CDDA",
    "\u2584", "\U0001CDDB", "\U0001CDDC", "\U0001CDDD", "\U0001CDDE", "\u2599", "\U0001CDDF", "\U0001CDE0",
    "\U0001CDE1", "\U0001CDE2", "\u259F", "\U0001CDE3", "\u2586", "\U0001CDE4", "\U0001CDE5", "\u2588",
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
    color_mode: str = "truecolor"
