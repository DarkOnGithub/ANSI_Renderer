import torch
import torch.nn.functional as F

_resize_cache = {}

def grayscale_frame(frame: torch.Tensor) -> torch.Tensor:
    return ((76 * frame[...,0] + 150 * frame[...,1] + 29 * frame[...,2] + 128) // 255).to(torch.uint8)

def setup_lookup(max_val: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ascii_bytes = [str(i).encode() for i in range(max_val)]
    max_len = max(len(x) for x in ascii_bytes)
    buf = torch.zeros((max_val, max_len), dtype=torch.uint8, device=device)
    lens = torch.zeros(max_val, dtype=torch.int64, device=device)
    for i, bs in enumerate(ascii_bytes):
        buf[i, :len(bs)] = torch.tensor(list(bs), dtype=torch.uint8)
        lens[i] = len(bs)
    return buf, lens


def resize_frame(frame: torch.Tensor, height: int, width: int) -> torch.Tensor:
    frame_chw = frame.permute(2, 0, 1).unsqueeze(0).float()
    resized_chw = F.interpolate(frame_chw, size=(height, width), mode='bilinear', align_corners=False)
    resized = resized_chw.squeeze(0).permute(1, 2, 0).to(frame.dtype)
    return resized


def resize_frame_keep_aspect(frame: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    orig_height, orig_width = frame.shape[:2]

    if orig_height <= target_height and orig_width <= target_width:
        return frame

    orig_aspect_num = orig_width * target_height
    target_aspect_num = target_width * orig_height

    if orig_aspect_num > target_aspect_num:
        new_width = target_width
        new_height = (orig_height * target_width) // orig_width
    else:
        new_height = target_height
        new_width = (orig_width * target_height) // orig_height

    new_width = min(new_width, target_width)
    new_height = min(new_height, target_height)

    frame_chw = frame.permute(2, 0, 1).unsqueeze(0).float()
    resized_chw = F.interpolate(frame_chw, size=(new_height, new_width), mode='bilinear', align_corners=False)
    resized = resized_chw.squeeze(0).permute(1, 2, 0).to(frame.dtype)
    return resized