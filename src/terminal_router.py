import os
from typing import Generator, Iterable

import torch

from .ansi_renderer import AnsiRenderer
from .config import DISABLE_ALT_BUFFER, SHOW_CURSOR, Config
from .multi_pane import MultiPaneOptions, MultiPaneRenderer


def _frame_generator(
    frames: Iterable[torch.Tensor],
    device: torch.device,
) -> Generator[torch.Tensor, None, None]:
    for frame in frames:
        if isinstance(frame, torch.Tensor):
            yield frame.to(device=device, dtype=torch.uint8)
            continue
        yield torch.as_tensor(frame, dtype=torch.uint8, device=device)


def cleanup_renderer(renderer: AnsiRenderer) -> None:
    if renderer.audio_process is not None:
        renderer.audio_process.terminate()
        renderer.audio_process.wait()
        renderer.audio_process = None

    if renderer._output_initialized:
        os.write(renderer.config.output_fd, SHOW_CURSOR + DISABLE_ALT_BUFFER)
        renderer._output_initialized = False


def render_single_terminal(
    frame_generator: Iterable[torch.Tensor],
    config: Config,
) -> None:
    renderer = AnsiRenderer(_frame_generator(frame_generator, config.device), config)
    try:
        for ansi, frame_idx in renderer.get_next_ansi_sequence():
            renderer.render_frame(ansi, frame_idx)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup_renderer(renderer)


def render_multi_terminal(
    frame_generator: Iterable[torch.Tensor],
    config: Config,
    options: MultiPaneOptions | None = None,
) -> None:
    try:
        with MultiPaneRenderer(config, options) as renderer:
            renderer.render_frames(frame_generator)
    except KeyboardInterrupt:
        pass


def render_with_terminal_mode(
    frame_generator: Iterable[torch.Tensor],
    config: Config,
    terminal_mode: str,
    multi_pane_options: MultiPaneOptions | None = None,
) -> None:
    normalized_mode = terminal_mode.strip().lower()
    if normalized_mode == "single":
        render_single_terminal(frame_generator, config)
        return
    if normalized_mode == "multi":
        render_multi_terminal(frame_generator, config, multi_pane_options)
        return
    raise ValueError(f"Unsupported terminal mode: {terminal_mode}")
