import sys
import torch

try:
    import PyNvVideoCodec as nvc
    _has_pynvcodec = True
except ImportError:
    _has_pynvcodec = False
    nvc = None

try:
    import dxcam
    _has_dxcam = True
except ImportError:
    _has_dxcam = False
    dxcam = None

torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
torch.backends.cuda.matmul.allow_tf32 = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu':
    sys.exit("CUDA is required for this application.")

ESC_VALS = (27, 91)
SEP_VAL = 59
CUR_END_VAL = 72
COL_PREF_VALS = (27, 91, 52, 56, 59, 50, 59)
COL_256_PREF_VALS = (27, 91, 51, 56, 59, 53, 59)  # \e[38;5;
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
CELL_ASPECT = 0.5

_ANSI_COLORS = None

def get_ansi_colors():
    global _ANSI_COLORS
    if _ANSI_COLORS is None:
        _ANSI_COLORS = torch.zeros((256, 3), dtype=torch.uint8)
        for i in range(216):
            r = (i // 36) * 51
            g = ((i % 36) // 6) * 51
            b = (i % 6) * 51
            _ANSI_COLORS[i] = torch.tensor([r, g, b], dtype=torch.uint8)
        for i in range(24):
            val = 8 + i * 10
            _ANSI_COLORS[232 + i] = torch.tensor([val, val, val], dtype=torch.uint8)

        basic_colors = torch.tensor([
            [0, 0, 0],       # 0: black
            [128, 0, 0],     # 1: red
            [0, 128, 0],     # 2: green
            [128, 128, 0],   # 3: yellow
            [0, 0, 128],     # 4: blue
            [128, 0, 128],   # 5: magenta
            [0, 128, 128],   # 6: cyan
            [192, 192, 192], # 7: white
            [128, 128, 128], # 8: bright black
            [255, 0, 0],     # 9: bright red
            [0, 255, 0],     # 10: bright green
            [255, 255, 0],   # 11: bright yellow
            [0, 0, 255],     # 12: bright blue
            [255, 0, 255],   # 13: bright magenta
            [0, 255, 255],   # 14: bright cyan
            [255, 255, 255], # 15: bright white
        ], dtype=torch.uint8)
        _ANSI_COLORS[:16] = basic_colors
    return _ANSI_COLORS

ANSI_COLORS = get_ansi_colors

from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Config:
    width: int
    height: int
    device: torch.device
    gpu_id: int = 0
    diff_thresh: int = 0  # 
