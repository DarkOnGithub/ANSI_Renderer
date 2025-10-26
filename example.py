import torch
from src.ansi_renderer import AnsiRenderer
import cv2 as cv
from src.config import Config

video_path = "./test.mp4"

def get_frame_generator(video_path: str):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = torch.from_numpy(frame[:, :, [2, 1, 0]]).to(torch.device('cuda'))  # Keep as (H, W, C)
        yield frame_rgb
    cap.release()

terminal_size = {
    "columns": 854,
    "lines": 480,
}
config = Config(width=terminal_size["columns"], height=terminal_size["lines"], device=torch.device('cuda'))
renderer = AnsiRenderer(frame_generator=get_frame_generator(video_path), config=config)
try:
    while True:
        ansi = next(renderer.get_next_ansi_sequence())
        data = ansi.cpu().numpy().tobytes() if isinstance(ansi, torch.Tensor) else ansi
        renderer.render_frame(data)
except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting...")
except Exception as e:
    print(f"Main thread caught exception: {e}")
    print(f"Exception type: {type(e)}")
    print("Program exiting due to thread crash")