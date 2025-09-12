import os
import sys
import time
import threading
import queue
import torch

try:
    import lz4.frame
    _has_lz4 = True
except ImportError:
    _has_lz4 = False

from config import CLEAR_SCREEN, HIDE_CURSOR, SHOW_CURSOR, RESET_VALS


def write_all(fd: int, data: bytes) -> None:
    """Write all data to fd, handling partial writes efficiently."""
    if not data:
        return
    BUFFER_SIZE = 65536  
    view = memoryview(data)
    total = 0
    remaining = len(data)

    while remaining > 0:
        chunk_size = min(remaining, BUFFER_SIZE)
        written = os.write(fd, view[total:total + chunk_size])
        if not written:
            break
        total += written
        remaining -= written


def setup_lookup(max_val: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ascii_bytes = [str(i).encode() for i in range(max_val)]
    max_len = max(len(x) for x in ascii_bytes)
    buf = torch.zeros((max_val, max_len), dtype=torch.uint8, device=device)
    lens = torch.zeros(max_val, dtype=torch.int64, device=device)
    for i, bs in enumerate(ascii_bytes):
        buf[i, :len(bs)] = torch.tensor(list(bs), dtype=torch.uint8)
        lens[i] = len(bs)
    return buf, lens


def quantize_colors(rgb_colors: torch.Tensor, ansi_colors: torch.Tensor) -> torch.Tensor:
    """
    Quantize RGB colors to nearest ANSI 256-color palette indices.
    Uses Euclidean distance in RGB space with optimized operations.
    """
    rgb_colors = rgb_colors.contiguous()
    ansi_colors = ansi_colors.contiguous()

    rgb_float = rgb_colors.float()
    ansi_float = ansi_colors.float()
    diff = rgb_float.unsqueeze(1) - ansi_float.unsqueeze(0)  
    distances = (diff ** 2).sum(dim=2)  # (N, 256)
    return distances.argmin(dim=1).contiguous()


def writer_task(out_queue, target_fps: float, exit_event: threading.Event) -> None:
    fd = sys.stdout.fileno()
    write_all(fd, CLEAR_SCREEN)
    write_all(fd, HIDE_CURSOR)
    interval = 1.0 / target_fps
    last_time = time.monotonic()
    i = 0

    # Increased buffer size for better batching and reduced system calls
    BUFFER_SIZE = 262144  # 256KB buffer for better performance
    buffer = bytearray(BUFFER_SIZE)
    buffer_pos = 0

    # LZ4 compression context for better compression ratios
    compressor = lz4.frame.LZ4FrameCompressor() if _has_lz4 else None

    def flush_buffer():
        nonlocal buffer_pos
        if buffer_pos > 0:
            data_to_write = buffer[:buffer_pos]
            if compressor and len(data_to_write) > 1024:  # Only compress larger chunks
                try:
                    compressed_data = compressor.compress(data_to_write)
                    compressed_data += compressor.flush()  # Finalize compression
                    if len(compressed_data) < len(data_to_write):  # Only use if smaller
                        # Prepend compression marker (simple approach)
                        write_all(fd, b'\x01' + compressed_data)
                        buffer_pos = 0
                        return
                except Exception:
                    pass  # Fall back to uncompressed on compression error

            write_all(fd, data_to_write)
            buffer_pos = 0

    try:
        while not exit_event.is_set():
            try:
                item = out_queue.get(timeout=0.1)
                if item is None:
                    flush_buffer()
                    break
                data = item.numpy().tobytes() if isinstance(item, torch.Tensor) else item
                # Only log data size for first few frames and every 100th frame
                # print(f"Frame {i}: Data length: {len(data) / 1024:.2f} KB", file=sys.stderr)

                # Optimized buffer management with immediate flush for large data
                if len(data) >= BUFFER_SIZE // 2:  # If data is > 50% of buffer size
                    flush_buffer()  # Flush existing buffer first
                    write_all(fd, data)  # Write large data directly
                elif buffer_pos + len(data) <= BUFFER_SIZE:
                    buffer[buffer_pos:buffer_pos + len(data)] = data
                    buffer_pos += len(data)
                else:
                    flush_buffer()
                    buffer[buffer_pos:buffer_pos + len(data)] = data
                    buffer_pos += len(data)

                i += 1
                now = time.monotonic()
                elapsed = now - last_time
                sleep_time = interval - elapsed

                # Performance logging commented out for reduced overhead
                # render_time = time.perf_counter() - render_start
                # if i <= 5 or i % 100 == 0:
                #     print(f"Frame {i}: Render time: {render_time:.4f}s", file=sys.stderr)

                if sleep_time > 0:
                    flush_buffer()
                    time.sleep(sleep_time)
                last_time = time.monotonic()
            except queue.Empty:
                continue
    finally:
        flush_buffer()
        write_all(fd, bytes(RESET_VALS))
        write_all(fd, SHOW_CURSOR)
