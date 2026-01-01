import torch
from src.ansi_renderer import AnsiRenderer
import cv2 as cv
from src.config import Config

video_path = "ChatGPT ｜ The Intelligence Age [kIhb5pEo_j0].mp4"

def get_video_fps(path: str) -> float:
    cap = cv.VideoCapture(path)
    fps = cap.get(cv.CAP_PROP_FPS)
    cap.release()
    return fps

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

fps = get_video_fps(video_path)
config = Config(
    width=1920, 
    height=1080, 
    device=torch.device('cuda'),
    fps=fps,
    audio_path=video_path
)
renderer = AnsiRenderer(frame_generator=get_frame_generator(video_path), config=config)

try:
    for ansi, frame_idx in renderer.get_next_ansi_sequence():
        renderer.render_frame(ansi, frame_idx)
except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting...")
except Exception as e:
    print(f"Main thread caught exception: {e}")
    import traceback
    traceback.print_exc()
    print("Program exiting due to thread crash")