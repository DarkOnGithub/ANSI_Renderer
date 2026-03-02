# ANSI Renderer

A GPU-accelerated, terminal-based video player and screen recorder that renders output using ANSI escape sequences.  
Capable of rendering at **1080p**, though some stuttering may still occur depending on terminal performance and hardware.  
Recommended to run on GPU accelerated terminals such as Alacritty  
> ⚠️ Requires an **NVIDIA GPU** because of hardware acceleration.
---

## Examples

[![Project Demo](demo1.gif)](https://youtu.be/eTm7I9le77I)
[![Project Demo](demo2.gif)](https://youtu.be/Exscan4-mHE)
[![Project Demo](demo3.gif)](https://youtu.be/DE2veokYNXE)

## Installation

Install the following dependencies manually:

- [**PyTorch**](https://pytorch.org/get-started/locally/)
- [**PyNvVideoCodec**](https://docs.nvidia.com/video-technologies/pynvvideocodec/)
- [**Triton**]  
  - [Linux installation](https://triton-lang.org/main/getting-started/installation.html)  
  - [Windows (unofficial)](https://github.com/woct0rdho/triton-windows)

Then, install the required Python package:

```bash
pip install dxcam
```
---

## Usage

```bash
python main.py [options]
```

### Play a Video File
```bash
python main.py -v path/to/video.mp4
```

### Capture Screen Region
```bash
python main.py -s -r x,y,width,height -f 30
```
- `-s`: Enable screen capture (default: video playback).
- `-r`: Region to capture (e.g., `0,0,1920,1080`). If omitted, full primary monitor is used.
- `-f`: Target frame rate for screen capture (default: 60 fps).

---
## Command-Line Arguments

| Option                | Description                                                       | Default     |
|-----------------------|-------------------------------------------------------------------|-------------|
| `-v`, `--video <str>` | Path to a video file to play.                                     | _None_*     |
| `-s`, `--screen`      | Capture screen instead of playing video.                          | `False`     |
| `-r`, `--region <str>`| `x,y,width,height` region for screen capture.                     | Full screen |
| `-f`, `--fps <int>`   | Target frames per second for screen capture.                      | `60`        |

_*If neither `-v` nor `-s` is specified, the first positional argument is treated as the video path._

---

