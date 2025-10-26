import queue
import threading
from typing import Generator
import torch
import os
import sys
from ansi_generator import ansi_generate
from config import Config
from frame_processing import pre_process_frame
from utils import setup_lookup

class AnsiRenderer:
    def __init__(self,
                 frame_generator: Generator[torch.Tensor, None, None],
                 config: Config):
        self.frame_generator = frame_generator
        self.config = config
        self.ansi_queue = queue.Queue()
        self.lookup_vals, self.lookup_lens = setup_lookup(max(self.config.width+1, self.config.height+1, 256), self.config.device)
        self.generator_thread = threading.Thread(target=self._generator_thread, daemon=True)
        self.thread_crashed = threading.Event()
        self.thread_exception = None
        self.generator_thread.start()
        
    def render_frame(self, frame: bytes) -> None:
        if not frame:
            return

        BUFFER_SIZE = 65536 * 10
        view = memoryview(frame)
        fd = self.config.output_fd

        while view:
            written = os.write(fd, view[:BUFFER_SIZE])
            view = view[written:]
        sys.stdout.flush()
    def get_next_ansi_sequence(self) -> Generator[torch.Tensor, None, None]:
        while True:
            if self.thread_crashed.is_set():
                if self.thread_exception:
                    raise self.thread_exception
                else:
                    raise RuntimeError("Generator thread crashed without exception details")
            try:
                ansi = self.ansi_queue.get(timeout=0.1)
            except queue.Empty:
                continue 
            if ansi is None:
                break
            yield ansi
    
    def _generator_thread(self) -> None:
        try:
            frame = next(self.frame_generator, None)
            previous_frame = None
            while frame is not None:
                xs, ys, colors_rgb, previous_frame = pre_process_frame(previous_frame, frame, self.config)
                ansi = ansi_generate(xs, ys, colors_rgb, self.lookup_vals, self.lookup_lens, self.config)
                self.ansi_queue.put(ansi)
                frame = next(self.frame_generator, None)
            self.ansi_queue.put(None)
        except Exception as e:
            print(f"Error in generator thread: {e}")
            self.thread_exception = e
            self.thread_crashed.set()
            self.ansi_queue.put(None)
            raise
    