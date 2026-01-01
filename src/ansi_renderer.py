import queue
import threading
import time
from typing import Generator
import torch
import os
import subprocess
from .ansi_generator import ansi_generate
from .config import Config
from .frame_processing import pre_process_frame
from .utils import setup_lookup

class AnsiRenderer:
    def __init__(self,
                 frame_generator: Generator[torch.Tensor, None, None],
                 config: Config):
        self.frame_generator = frame_generator
        self.config = config
        self.ansi_queue = queue.Queue(maxsize=2)
        
        self.free_buffers = queue.Queue()
        for _ in range(3): 
            self.free_buffers.put(torch.empty(10 * 1024 * 1024, dtype=torch.uint8, pin_memory=True))
            
        self.lookup_vals, self.lookup_lens = setup_lookup(max(self.config.width+1, self.config.height+1, 256), self.config.device)
        self.generator_thread = threading.Thread(target=self._generator_thread, daemon=True)
        self.thread_crashed = threading.Event()
        self.thread_exception = None
        
        self.start_time = None
        self.rendered_frames = 0
        self.audio_process = None
        
        if self.config.timing_enabled:
            self.timing_f = open(self.config.timing_file, 'w')
            self.timing_f.write("frame_idx,gen_time,fetch_time,render_time,total_time\n")
            self.frame_idx = 0
        else:
            self.timing_f = None

        self.generator_thread.start()
        
    def render_frame(self, frame: torch.Tensor | bytes, frame_idx: int) -> None:
        if frame is None:
            return

        if self.start_time is None:
            if self.config.audio_path:
                self.audio_process = subprocess.Popen(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", self.config.audio_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            os.write(self.config.output_fd, b"\033[?1049h\033[2J\033[?25l\033[H")
            self.start_time = time.perf_counter()

        target_time = self.start_time + (frame_idx / self.config.fps) + self.config.audio_delay
        now = time.perf_counter()
        wait_time = target_time - now
        
        if wait_time > 0:
            time.sleep(wait_time)
        
        start_render = time.perf_counter()
        
        if isinstance(frame, torch.Tensor):
            try:
                data = memoryview(frame.untyped_storage())[:frame.nbytes]
            except Exception:
                try:
                    data = memoryview(frame.storage())[:frame.nbytes]
                except Exception:
                    data = bytes(frame.view(-1).tolist())
        else:
            data = frame

        BUFFER_SIZE = 131072 
        view = memoryview(data)
        fd = self.config.output_fd

        while view:
            written = os.write(fd, view[:BUFFER_SIZE])
            view = view[written:]
        
        self.rendered_frames += 1
        
        if self.config.timing_enabled:
            render_time = time.perf_counter() - start_render
            gen_time = getattr(self, '_last_gen_time', 0.0)
            fetch_time = getattr(self, '_last_fetch_time', 0.0)
            total_time = gen_time + fetch_time + render_time
            self.timing_f.write(f"{self.frame_idx},{gen_time:.6f},{fetch_time:.6f},{render_time:.6f},{total_time:.6f}\n")
            self.frame_idx += 1
            self.timing_f.flush()

    def get_next_ansi_sequence(self) -> Generator[tuple[torch.Tensor, int], None, None]:
        while True:
            if self.thread_crashed.is_set():
                if self.thread_exception:
                    raise self.thread_exception
                else:
                    raise RuntimeError("Generator thread crashed without exception details")
            try:
                item = self.ansi_queue.get(timeout=0.1)
            except queue.Empty:
                continue 
            if item is None:
                break
            
            if self.config.timing_enabled:
                ansi, frame_idx, gen_time, fetch_time, buffer_to_release = item
                self._last_gen_time = gen_time
                self._last_fetch_time = fetch_time
                yield ansi, frame_idx
                self.free_buffers.put(buffer_to_release)
            else:
                ansi, frame_idx, buffer_to_release = item
                yield ansi, frame_idx
                self.free_buffers.put(buffer_to_release)
    
    def _generator_thread(self) -> None:
        try:
            previous_frame = None
            current_frame_idx = 0
            while True:
                # Catch up logic: if we are behind the wall clock, skip frames in the generator
                if self.start_time is not None:
                    elapsed = time.perf_counter() - self.start_time
                    # Aim to stay ~2 frames ahead for buffering, but skip if we fall behind
                    target_idx = int((elapsed - self.config.audio_delay) * self.config.fps)
                    
                    while current_frame_idx < target_idx:
                        frame = next(self.frame_generator, None)
                        if frame is None:
                            break
                        current_frame_idx += 1

                start_fetch = time.perf_counter()
                frame = next(self.frame_generator, None)
                fetch_time = time.perf_counter() - start_fetch
                
                if frame is None:
                    break
                
                old_shape = previous_frame.shape if previous_frame is not None else None
                xs, ys, colors_rgb, previous_frame = pre_process_frame(previous_frame, frame, self.config)
                if xs.numel() == 0:
                    current_frame_idx += 1
                    continue
                
                start_gen = time.perf_counter()
                ansi_gpu = ansi_generate(xs, ys, colors_rgb, self.lookup_vals, self.lookup_lens, self.config)
                
                if old_shape is not None and old_shape != previous_frame.shape:
                    clear_seq = torch.tensor(list(b"\033[2J\033[H"), dtype=torch.uint8, device=self.config.device)
                    ansi_gpu = torch.cat([clear_seq, ansi_gpu])
                
                cpu_buf = self.free_buffers.get()
                
                if cpu_buf.size(0) < ansi_gpu.size(0):
                    cpu_buf = torch.empty(int(ansi_gpu.size(0) * 1.2), dtype=torch.uint8, pin_memory=True)
                
                cpu_view = cpu_buf[:ansi_gpu.size(0)]
                cpu_view.copy_(ansi_gpu, non_blocking=True)
                
                torch.cuda.current_stream().synchronize()
                
                gen_time = time.perf_counter() - start_gen
                
                if self.config.timing_enabled:
                    self.ansi_queue.put((cpu_view, current_frame_idx, gen_time, fetch_time, cpu_buf))
                else:
                    self.ansi_queue.put((cpu_view, current_frame_idx, cpu_buf))
                
                current_frame_idx += 1
            self.ansi_queue.put(None)
        except Exception as e:
            print(f"Error in generator thread: {e}")
            self.thread_exception = e
            self.thread_crashed.set()
            self.ansi_queue.put(None)
            raise
    
    def __del__(self):
        if hasattr(self, 'audio_process') and self.audio_process:
            self.audio_process.terminate()
            self.audio_process.wait()
        if hasattr(self, 'timing_f') and self.timing_f:
            self.timing_f.close()

        if hasattr(self, 'config'):
            os.write(self.config.output_fd, b"\033[?25h\033[?1049l")
    