import os
import queue
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Generator

import torch
import torch.nn.functional as F

from .ansi_generator import ansi_generate
from .config import Config, SYNC_OUTPUT_BEGIN, SYNC_OUTPUT_END
from .frame_processing import pre_process_frame
from .utils import setup_lookup


@dataclass
class GpuBuildTiming:
    preprocess_segments: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=list
    )
    gen_segments: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=list
    )

    def synchronize(self) -> None:
        for _, end_event in self.preprocess_segments:
            end_event.synchronize()
        for _, end_event in self.gen_segments:
            end_event.synchronize()

    def preprocess_ms(self) -> float:
        return sum(
            float(start_event.elapsed_time(end_event))
            for start_event, end_event in self.preprocess_segments
        )

    def gen_ms(self) -> float:
        return sum(
            float(start_event.elapsed_time(end_event))
            for start_event, end_event in self.gen_segments
        )

    def total_ms(self) -> float:
        return self.preprocess_ms() + self.gen_ms()


class AnsiRenderer:
    def __init__(
        self,
        frame_generator: Generator[torch.Tensor, None, None],
        config: Config,
        autostart: bool = True,
    ):
        self.frame_generator = frame_generator
        self.config = config
        self.ansi_queue = queue.Queue(maxsize=max(2, int(self.config.queue_size)))
        self.cuda_enabled = (
            self.config.device.type == "cuda" and torch.cuda.is_available()
        )
        self.copy_stream = None
        if self.cuda_enabled and self.config.async_copy_stream:
            self.copy_stream = torch.cuda.Stream(device=self.config.device)
        self._pending_copy_done_event = None
        self._has_writev = hasattr(os, "writev")
        self._sync_begin_view = memoryview(SYNC_OUTPUT_BEGIN)
        self._sync_end_view = memoryview(SYNC_OUTPUT_END)

        self.free_buffers = queue.Queue()
        initial_buffer_size = max(1024, int(self.config.initial_buffer_size))
        for _ in range(max(2, int(self.config.buffer_pool_size))):
            self.free_buffers.put(
                torch.empty(
                    initial_buffer_size,
                    dtype=torch.uint8,
                    pin_memory=self.cuda_enabled,
                )
            )

        self.lookup_vals, self.lookup_lens = setup_lookup(
            max(self.config.width + 1, self.config.height + 1, 256), self.config.device
        )
        self.generator_thread = threading.Thread(
            target=self._generator_thread, daemon=True
        )
        self.thread_crashed = threading.Event()
        self.thread_exception = None

        self.start_time = None
        self.rendered_frames = 0
        self.audio_process = None
        self._render_time_ema = 0.0
        self._output_initialized = False

        self._quality_level = 0
        self._producer_time_ema = None
        self._pressure_frames = 0
        self._recovery_frames = 0
        (
            self._adaptive_quant_masks,
            self._adaptive_diff_thresholds,
            self._adaptive_run_color_diff_thresholds,
        ) = self._build_adaptive_profiles()
        self._stale_age = None
        target_frame_bytes = max(0, int(getattr(self.config, "target_frame_bytes", 0)))
        if target_frame_bytes > 0:
            buffer_frames = max(
                1, int(getattr(self.config, "frame_byte_buffer_frames", 4))
            )
            self._byte_budget_capacity = float(target_frame_bytes * buffer_frames)
            self._byte_budget_tokens = self._byte_budget_capacity
        else:
            self._byte_budget_capacity = 0.0
            self._byte_budget_tokens = 0.0

        if self.config.timing_enabled:
            self.timing_f = open(self.config.timing_file, "w")
            self.timing_f.write(
                "frame_idx,gen_time,fetch_time,preprocess_time,producer_time,queue_wait_time,copy_wait_time,render_time,consumer_time,pipeline_time,total_time,sleep_time,end_to_end_time,datasize,quality_level,quant_mask,diff_thresh,frame_start_time,frame_end_time\n"
            )
            self.frame_idx = 0
            self._timing_rows_since_flush = 0
        else:
            self.timing_f = None

        if autostart:
            self.generator_thread.start()

    def build_frame_payload(
        self,
        previous_frame: torch.Tensor | None,
        frame: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor,
        float,
        float,
        int,
        int,
        int,
        int,
        GpuBuildTiming | None,
    ]:
        return self._build_frame_payload(previous_frame, frame)

    def _build_adaptive_profiles(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        base_mask = int(self.config.quant_mask) & 0xFF

        raw_masks = tuple(int(x) & 0xFF for x in self.config.adaptive_quant_masks)
        if not raw_masks:
            raw_masks = (base_mask,)
        if raw_masks[0] != base_mask:
            raw_masks = (base_mask,) + tuple(x for x in raw_masks if x != base_mask)

        raw_offsets = tuple(int(x) for x in self.config.adaptive_diff_thresh_offsets)
        if not raw_offsets:
            raw_offsets = (0,)
        if raw_offsets[0] != 0:
            raw_offsets = (0,) + tuple(x for x in raw_offsets if x != 0)

        raw_run_offsets = tuple(
            int(x) for x in self.config.adaptive_run_color_diff_offsets
        )
        if not raw_run_offsets:
            raw_run_offsets = (0,)
        if raw_run_offsets[0] != 0:
            raw_run_offsets = (0,) + tuple(x for x in raw_run_offsets if x != 0)

        levels = max(1, min(len(raw_masks), len(raw_offsets), len(raw_run_offsets)))
        masks = raw_masks[:levels]
        diff_thresholds = tuple(
            max(0, int(self.config.diff_thresh) + x) for x in raw_offsets[:levels]
        )
        run_color_thresholds = tuple(
            max(0, int(self.config.run_color_diff_thresh) + x)
            for x in raw_run_offsets[:levels]
        )
        return masks, diff_thresholds, run_color_thresholds

    def _quality_settings_for_level(self, level: int) -> tuple[int, int, int]:
        if not self.config.adaptive_quality:
            return (
                int(self.config.quant_mask),
                int(self.config.diff_thresh),
                int(self.config.run_color_diff_thresh),
            )

        clamped_level = min(max(int(level), 0), len(self._adaptive_quant_masks) - 1)
        return (
            self._adaptive_quant_masks[clamped_level],
            self._adaptive_diff_thresholds[clamped_level],
            self._adaptive_run_color_diff_thresholds[clamped_level],
        )

    def _current_quality_settings(self) -> tuple[int, int, int]:
        return self._quality_settings_for_level(self._quality_level)

    def _update_adaptive_quality(
        self,
        producer_time: float,
        current_frame_idx: int,
        frame_bytes: int | None = None,
    ) -> None:
        if not self.config.adaptive_quality:
            return

        max_level = len(self._adaptive_quant_masks) - 1
        if max_level <= 0:
            return

        alpha = min(max(float(self.config.adaptive_ema_alpha), 0.01), 1.0)
        if self._producer_time_ema is None:
            self._producer_time_ema = producer_time
        else:
            self._producer_time_ema = (
                1.0 - alpha
            ) * self._producer_time_ema + alpha * producer_time

        budget = 1.0 / max(float(self.config.fps), 1e-6)
        queue_size = self.ansi_queue.qsize()

        lag_frames = 0
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            target_idx = int((elapsed - self.config.audio_delay) * self.config.fps)
            lag_frames = max(0, target_idx - current_frame_idx)

        max_frame_bytes = max(0, int(getattr(self.config, "max_frame_bytes", 0)))
        target_frame_bytes = max(0, int(getattr(self.config, "target_frame_bytes", 0)))
        byte_budget_ref = max_frame_bytes if max_frame_bytes > 0 else target_frame_bytes
        byte_pressure = (
            frame_bytes is not None
            and byte_budget_ref > 0
            and int(frame_bytes) > int(byte_budget_ref * 1.10)
        )
        byte_recovery = (
            frame_bytes is not None
            and byte_budget_ref > 0
            and int(frame_bytes) <= int(byte_budget_ref * 0.75)
        )

        pressure = (
            lag_frames > 0
            or self._producer_time_ema > budget * 1.05
            or (self.start_time is not None and queue_size == 0)
            or byte_pressure
        )

        recovery = (
            lag_frames == 0
            and self._producer_time_ema < budget * 0.82
            and queue_size >= max(1, self.ansi_queue.maxsize - 1)
            and (byte_budget_ref <= 0 or byte_recovery)
        )

        if pressure:
            self._pressure_frames += 1
            self._recovery_frames = 0
            if self._pressure_frames >= 2 and self._quality_level < max_level:
                self._quality_level += 1
                self._pressure_frames = 0
        elif recovery:
            self._recovery_frames += 1
            self._pressure_frames = 0
            if self._recovery_frames >= 8 and self._quality_level > 0:
                self._quality_level -= 1
                self._recovery_frames = 0
        else:
            self._pressure_frames = max(0, self._pressure_frames - 1)
            self._recovery_frames = 0

    def _begin_frame_budget(self) -> int:
        hard_cap = max(0, int(getattr(self.config, "max_frame_bytes", 0)))
        target_frame_bytes = max(0, int(getattr(self.config, "target_frame_bytes", 0)))

        if target_frame_bytes <= 0:
            return hard_cap

        self._byte_budget_tokens = min(
            self._byte_budget_capacity,
            self._byte_budget_tokens + float(target_frame_bytes),
        )
        token_budget = int(self._byte_budget_tokens)
        if hard_cap > 0:
            return min(hard_cap, token_budget)
        return token_budget

    def _consume_frame_budget(self, frame_bytes: int) -> None:
        target_frame_bytes = max(0, int(getattr(self.config, "target_frame_bytes", 0)))
        if target_frame_bytes <= 0:
            return
        self._byte_budget_tokens = max(
            0.0,
            self._byte_budget_tokens - float(max(0, int(frame_bytes))),
        )

    def _ensure_stale_age(
        self, frame_shape: tuple[int, int], device: torch.device
    ) -> torch.Tensor:
        if (
            self._stale_age is None
            or self._stale_age.shape != frame_shape
            or self._stale_age.device != device
        ):
            self._stale_age = torch.zeros(
                frame_shape,
                dtype=torch.uint8,
                device=device,
            )
        return self._stale_age

    def _mark_updates_sent(
        self, ys: torch.Tensor, xs: torch.Tensor, frame_shape: tuple[int, int]
    ) -> None:
        if ys.numel() == 0:
            return
        stale_age = self._ensure_stale_age(frame_shape, ys.device)
        stale_age[ys, xs] = 0

    def _cap_frame_payload(
        self,
        ansi_gpu: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
        colors_rgb: torch.Tensor,
        base_previous: torch.Tensor | None,
        updated_previous: torch.Tensor,
        shape_changed: bool,
        max_frame_bytes: int,
        run_color_diff_thresh: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        frame_bytes = int(ansi_gpu.size(0))
        if max_frame_bytes <= 0 or frame_bytes <= max_frame_bytes:
            return ansi_gpu, updated_previous, frame_bytes

        total_updates = int(xs.numel())
        if total_updates <= 0:
            return ansi_gpu, updated_previous, frame_bytes

        frame_shape = (int(updated_previous.shape[0]), int(updated_previous.shape[1]))
        stale_age = self._ensure_stale_age(frame_shape, xs.device)
        stale_weight = float(
            max(0.0, getattr(self.config, "cap_staleness_weight", 0.0))
        )
        stale_max = max(0, int(getattr(self.config, "cap_staleness_max", 255)))
        density_weight = float(
            max(0.0, getattr(self.config, "cap_density_weight", 0.0))
        )

        use_uniform_selection = base_previous is None
        if use_uniform_selection:
            priority = torch.arange(
                total_updates, device=xs.device, dtype=torch.float32
            )
        else:
            old_values = base_previous[ys, xs]
            if colors_rgb.size(1) >= 7 and old_values.size(1) >= 7:
                color_delta = torch.abs(
                    colors_rgb[:, :6].to(torch.int16)
                    - old_values[:, :6].to(torch.int16)
                ).amax(dim=1)
                glyph_delta = (colors_rgb[:, 6] != old_values[:, 6]).to(
                    torch.int16
                ) * 255
                priority = color_delta + glyph_delta
            else:
                compare_channels = min(colors_rgb.size(1), old_values.size(1))
                priority = torch.abs(
                    colors_rgb[:, :compare_channels].to(torch.int16)
                    - old_values[:, :compare_channels].to(torch.int16)
                ).amax(dim=1)

            if stale_weight > 0.0:
                priority = priority.to(torch.float32) + (
                    stale_age[ys, xs].to(torch.float32) * stale_weight
                )
            else:
                priority = priority.to(torch.float32)

            if density_weight > 0.0:
                occupancy = torch.zeros(
                    frame_shape, dtype=torch.float32, device=xs.device
                )
                occupancy[ys, xs] = 1.0
                density = (
                    F.avg_pool2d(
                        occupancy.unsqueeze(0).unsqueeze(0),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
                priority = priority + (density[ys, xs] * density_weight)

        ranked_idx = torch.argsort(priority, descending=True)

        def build_payload(
            top_k: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
            if use_uniform_selection:
                step = max(1, total_updates // max(1, top_k))
                chosen = torch.arange(0, total_updates, step, device=xs.device)
                if chosen.numel() > top_k:
                    chosen = chosen[:top_k]
            else:
                chosen = ranked_idx[:top_k]
                chosen, _ = torch.sort(chosen)

            chosen_xs = xs.index_select(0, chosen)
            chosen_ys = ys.index_select(0, chosen)
            chosen_colors = colors_rgb.index_select(0, chosen)

            payload = ansi_generate(
                chosen_xs,
                chosen_ys,
                chosen_colors,
                self.lookup_vals,
                self.lookup_lens,
                self.config,
                run_color_diff_thresh_override=run_color_diff_thresh,
            )

            if shape_changed:
                clear_seq = torch.tensor(
                    list(b"\033[2J\033[H"),
                    dtype=torch.uint8,
                    device=self.config.device,
                )
                payload = torch.cat([clear_seq, payload])

            return chosen_xs, chosen_ys, chosen_colors, payload, int(payload.size(0))

        low = 1
        high = total_updates
        best_result = None

        for _ in range(8):
            if low > high:
                break
            mid = (low + high) // 2
            chosen_xs, chosen_ys, chosen_colors, payload, candidate_bytes = (
                build_payload(mid)
            )
            if candidate_bytes <= max_frame_bytes:
                best_result = (
                    chosen_xs,
                    chosen_ys,
                    chosen_colors,
                    payload,
                    candidate_bytes,
                )
                low = mid + 1
            else:
                high = mid - 1

        if best_result is None:
            selected_xs, selected_ys, selected_colors, capped_ansi, capped_bytes = (
                build_payload(1)
            )
        else:
            selected_xs, selected_ys, selected_colors, capped_ansi, capped_bytes = (
                best_result
            )

        if base_previous is None:
            capped_previous = torch.zeros_like(updated_previous)
        else:
            capped_previous = base_previous.clone()
        capped_previous[selected_ys, selected_xs] = selected_colors

        stale_candidates = torch.clamp(
            stale_age[ys, xs].to(torch.int16) + 1,
            min=0,
            max=stale_max,
        ).to(torch.uint8)
        stale_age[ys, xs] = stale_candidates
        stale_age[selected_ys, selected_xs] = 0

        return capped_ansi, capped_previous, capped_bytes

    def _build_frame_payload(
        self,
        previous_frame: torch.Tensor | None,
        frame: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor,
        float,
        float,
        int,
        int,
        int,
        int,
        GpuBuildTiming | None,
    ]:
        old_shape = previous_frame.shape if previous_frame is not None else None
        max_level = len(self._adaptive_quant_masks) - 1
        start_level = min(max(int(self._quality_level), 0), max_level)

        hard_cap = max(0, int(getattr(self.config, "max_frame_bytes", 0)))
        target_frame_bytes = max(0, int(getattr(self.config, "target_frame_bytes", 0)))
        budgeting_enabled = hard_cap > 0 or target_frame_bytes > 0
        frame_budget_bytes = self._begin_frame_budget()
        if budgeting_enabled and frame_budget_bytes <= 0:
            frame_budget_bytes = 1
        can_retry_for_budget = (
            budgeting_enabled and self.config.adaptive_quality and max_level > 0
        )

        preprocess_time = 0.0
        gen_time = 0.0
        level = start_level
        gpu_timing = GpuBuildTiming() if self.cuda_enabled else None

        while True:
            quant_mask, diff_thresh, run_color_diff_thresh = (
                self._quality_settings_for_level(level)
            )
            if can_retry_for_budget and previous_frame is not None:
                prev_candidate = previous_frame.clone()
            else:
                prev_candidate = previous_frame

            pre_start_event = None
            pre_end_event = None
            if gpu_timing is not None:
                pre_start_event = torch.cuda.Event(enable_timing=True)
                pre_end_event = torch.cuda.Event(enable_timing=True)
                pre_start_event.record(
                    torch.cuda.current_stream(device=self.config.device)
                )
            start_pre = time.perf_counter()
            xs, ys, colors_rgb, updated_previous = pre_process_frame(
                prev_candidate,
                frame,
                self.config,
                quant_mask=quant_mask,
                diff_thresh_override=diff_thresh,
            )
            if pre_end_event is not None and pre_start_event is not None:
                pre_end_event.record(
                    torch.cuda.current_stream(device=self.config.device)
                )
                timing_ref = gpu_timing
                if timing_ref is not None:
                    timing_ref.preprocess_segments.append(
                        (pre_start_event, pre_end_event)
                    )
            preprocess_time += time.perf_counter() - start_pre

            if xs.numel() == 0:
                self._consume_frame_budget(0)
                return (
                    None,
                    updated_previous,
                    preprocess_time,
                    gen_time,
                    level,
                    quant_mask,
                    diff_thresh,
                    0,
                    gpu_timing,
                )

            gen_start_event = None
            gen_end_event = None
            if gpu_timing is not None:
                gen_start_event = torch.cuda.Event(enable_timing=True)
                gen_end_event = torch.cuda.Event(enable_timing=True)
                gen_start_event.record(
                    torch.cuda.current_stream(device=self.config.device)
                )
            start_gen = time.perf_counter()
            ansi_gpu = ansi_generate(
                xs,
                ys,
                colors_rgb,
                self.lookup_vals,
                self.lookup_lens,
                self.config,
                run_color_diff_thresh_override=run_color_diff_thresh,
            )

            shape_changed = (
                old_shape is not None and old_shape != updated_previous.shape
            )
            if shape_changed:
                clear_seq = torch.tensor(
                    list(b"\033[2J\033[H"),
                    dtype=torch.uint8,
                    device=self.config.device,
                )
                ansi_gpu = torch.cat([clear_seq, ansi_gpu])

            if gen_end_event is not None and gen_start_event is not None:
                gen_end_event.record(
                    torch.cuda.current_stream(device=self.config.device)
                )
                timing_ref = gpu_timing
                if timing_ref is not None:
                    timing_ref.gen_segments.append((gen_start_event, gen_end_event))
            gen_time += time.perf_counter() - start_gen
            frame_bytes = int(ansi_gpu.size(0))
            capped_for_budget = False

            if budgeting_enabled and frame_bytes > frame_budget_bytes:
                if can_retry_for_budget and level < max_level:
                    level += 1
                    continue

                ansi_gpu, updated_previous, frame_bytes = self._cap_frame_payload(
                    ansi_gpu,
                    xs,
                    ys,
                    colors_rgb,
                    previous_frame,
                    updated_previous,
                    shape_changed,
                    frame_budget_bytes,
                    run_color_diff_thresh,
                )
                capped_for_budget = True

            if not capped_for_budget:
                self._mark_updates_sent(
                    ys,
                    xs,
                    (int(updated_previous.shape[0]), int(updated_previous.shape[1])),
                )

            if level > self._quality_level:
                self._quality_level = level
                self._pressure_frames = 0
                self._recovery_frames = 0
            self._consume_frame_budget(frame_bytes)
            return (
                ansi_gpu,
                updated_previous,
                preprocess_time,
                gen_time,
                level,
                quant_mask,
                diff_thresh,
                frame_bytes,
                gpu_timing,
            )

    def _write_all(self, fd: int, view: memoryview, chunk_size: int) -> None:
        bounded_chunk = max(4096, int(chunk_size)) if chunk_size > 0 else 0
        while view:
            try:
                if bounded_chunk > 0 and view.nbytes > bounded_chunk:
                    written = os.write(fd, view[:bounded_chunk])
                else:
                    written = os.write(fd, view)
            except InterruptedError:
                continue

            if written <= 0:
                raise RuntimeError("Short write on terminal output")
            view = view[written:]

    def _writev_all(self, fd: int, segments: list[memoryview]) -> None:
        filtered = [segment for segment in segments if segment.nbytes > 0]
        if not filtered:
            return

        index = 0
        offset = 0
        while index < len(filtered):
            head = filtered[index]
            if offset:
                head = head[offset:]
            iovecs = [head, *filtered[index + 1 :]]

            try:
                written = os.writev(fd, iovecs)
            except InterruptedError:
                continue

            if written <= 0:
                raise RuntimeError("Short writev on terminal output")

            remaining = written
            head_len = head.nbytes
            if remaining < head_len:
                offset += remaining
                continue

            remaining -= head_len
            index += 1
            offset = 0

            while remaining > 0 and index < len(filtered):
                current_len = filtered[index].nbytes
                if remaining < current_len:
                    offset = remaining
                    remaining = 0
                else:
                    remaining -= current_len
                    index += 1

    def render_frame(self, frame: Any, frame_idx: int) -> None:
        if frame is None:
            return

        consumer_start_time = time.perf_counter()
        if self.start_time is None:
            if self.config.audio_path:
                self.audio_process = subprocess.Popen(
                    [
                        "ffplay",
                        "-nodisp",
                        "-autoexit",
                        "-loglevel",
                        "quiet",
                        "-vn",
                        "-sn",
                        self.config.audio_path,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            os.write(self.config.output_fd, b"\033[?1049h\033[2J\033[?25l\033[H")
            self.start_time = time.perf_counter()
            consumer_start_time = self.start_time
            self._output_initialized = True

        target_time = (
            self.start_time + (frame_idx / self.config.fps) + self.config.audio_delay
        )
        now = time.perf_counter()
        lead = 0.0
        if self.config.pacing_render_lead and self._render_time_ema > 0.0:
            lead = self._render_time_ema
        wait_time = target_time - now - lead

        sleep_time = 0.0
        # if wait_time > 0:
        #     sleep_start = time.perf_counter()
        #     # time.sleep(wait_time)
        #     sleep_time = time.perf_counter() - sleep_start

        pending_event = self._pending_copy_done_event
        copy_wait_time = 0.0
        if pending_event is not None:
            copy_wait_start = time.perf_counter()
            pending_event.synchronize()
            copy_wait_time = time.perf_counter() - copy_wait_start
            self._pending_copy_done_event = None

        start_render = time.perf_counter()

        if isinstance(frame, torch.Tensor):
            if frame.device.type != "cpu":
                data = bytes(frame.view(-1).tolist())
            else:
                data = memoryview(frame.numpy())
        else:
            if isinstance(frame, (bytes, bytearray, memoryview)):
                data = frame
            else:
                try:
                    data = memoryview(frame)
                except TypeError:
                    data = bytes(frame)

        buffer_size = int(self.config.write_chunk_size)
        view = memoryview(data)
        datasize = view.nbytes
        fd = self.config.output_fd
        sync_output = bool(self.config.sync_output)
        use_writev = (
            sync_output and bool(self.config.prefer_writev) and self._has_writev
        )

        if use_writev:
            self._writev_all(fd, [self._sync_begin_view, view, self._sync_end_view])
        elif sync_output:
            self._write_all(fd, self._sync_begin_view, 0)
            try:
                self._write_all(fd, view, buffer_size)
            finally:
                self._write_all(fd, self._sync_end_view, 0)
        else:
            self._write_all(fd, view, buffer_size)

        self.rendered_frames += 1
        render_time = time.perf_counter() - start_render

        if self.config.pacing_render_lead:
            alpha = min(max(float(self.config.pacing_render_alpha), 0.01), 1.0)
            if self._render_time_ema <= 0.0:
                self._render_time_ema = render_time
            else:
                self._render_time_ema = (
                    1.0 - alpha
                ) * self._render_time_ema + alpha * render_time

        if self.config.timing_enabled:
            timing_f = self.timing_f
            if timing_f is None:
                return
            gen_time = getattr(self, "_last_gen_time", 0.0)
            fetch_time = getattr(self, "_last_fetch_time", 0.0)
            preprocess_time = getattr(self, "_last_preprocess_time", 0.0)
            queue_wait_time = getattr(self, "_last_queue_wait_time", 0.0)
            quality_level = getattr(self, "_last_quality_level", 0)
            quant_mask = getattr(self, "_last_quant_mask", int(self.config.quant_mask))
            diff_thresh = getattr(
                self, "_last_diff_thresh", int(self.config.diff_thresh)
            )

            producer_time = fetch_time + preprocess_time + gen_time
            consumer_time = copy_wait_time + render_time
            pipeline_time = max(producer_time, consumer_time)
            total_time = pipeline_time
            end_to_end_time = queue_wait_time + sleep_time + consumer_time
            frame_start_time = max(0.0, consumer_start_time - self.start_time)
            frame_end_time = max(
                frame_start_time, time.perf_counter() - self.start_time
            )
            timing_f.write(
                f"{self.frame_idx},{gen_time:.6f},{fetch_time:.6f},{preprocess_time:.6f},{producer_time:.6f},{queue_wait_time:.6f},{copy_wait_time:.6f},{render_time:.6f},{consumer_time:.6f},{pipeline_time:.6f},{total_time:.6f},{sleep_time:.6f},{end_to_end_time:.6f},{datasize},{quality_level},{quant_mask},{diff_thresh},{frame_start_time:.6f},{frame_end_time:.6f}\n"
            )
            self.frame_idx += 1
            self._timing_rows_since_flush += 1

            flush_interval = max(1, int(self.config.timing_flush_interval))
            if self._timing_rows_since_flush >= flush_interval:
                timing_f.flush()
                self._timing_rows_since_flush = 0

    def get_next_ansi_sequence(self) -> Generator[tuple[Any, int], None, None]:
        while True:
            if self.thread_crashed.is_set():
                if self.thread_exception:
                    raise self.thread_exception
                else:
                    raise RuntimeError(
                        "Generator thread crashed without exception details"
                    )
            queue_wait_time = 0.0
            while True:
                wait_start = time.perf_counter()
                try:
                    item = self.ansi_queue.get(timeout=0.1)
                    queue_wait_time += time.perf_counter() - wait_start
                    break
                except queue.Empty:
                    queue_wait_time += time.perf_counter() - wait_start
                    if self.thread_crashed.is_set():
                        if self.thread_exception:
                            raise self.thread_exception
                        raise RuntimeError(
                            "Generator thread crashed without exception details"
                        )
            if item is None:
                break

            if self.config.timing_enabled:
                (
                    ansi,
                    frame_idx,
                    gen_time,
                    fetch_time,
                    preprocess_time,
                    quality_level,
                    quant_mask,
                    diff_thresh,
                    copy_done_event,
                    buffer_to_release,
                ) = item
                self._last_gen_time = gen_time
                self._last_fetch_time = fetch_time
                self._last_preprocess_time = preprocess_time
                self._last_queue_wait_time = queue_wait_time
                self._last_quality_level = quality_level
                self._last_quant_mask = quant_mask
                self._last_diff_thresh = diff_thresh

                self._pending_copy_done_event = copy_done_event
                yield ansi, frame_idx
                if (
                    self._pending_copy_done_event is copy_done_event
                    and copy_done_event is not None
                ):
                    copy_done_event.synchronize()
                self._pending_copy_done_event = None
                self.free_buffers.put(buffer_to_release)
            else:
                ansi, frame_idx, copy_done_event, buffer_to_release = item
                self._pending_copy_done_event = copy_done_event
                yield ansi, frame_idx
                if (
                    self._pending_copy_done_event is copy_done_event
                    and copy_done_event is not None
                ):
                    copy_done_event.synchronize()
                self._pending_copy_done_event = None
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
                    target_idx = int(
                        (elapsed - self.config.audio_delay) * self.config.fps
                    )

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

                (
                    ansi_gpu,
                    previous_frame,
                    preprocess_time,
                    gen_time,
                    quality_level_used,
                    quant_mask,
                    diff_thresh,
                    frame_bytes,
                    _,
                ) = self._build_frame_payload(previous_frame, frame)

                if ansi_gpu is None:
                    producer_time = fetch_time + preprocess_time + gen_time
                    self._update_adaptive_quality(
                        producer_time,
                        current_frame_idx,
                        frame_bytes=frame_bytes,
                    )
                    current_frame_idx += 1
                    continue

                cpu_buf = self.free_buffers.get()

                if cpu_buf.size(0) < ansi_gpu.size(0):
                    cpu_buf = torch.empty(
                        int(ansi_gpu.size(0) * 1.2),
                        dtype=torch.uint8,
                        pin_memory=self.cuda_enabled,
                    )

                cpu_view = cpu_buf[: ansi_gpu.size(0)]

                copy_done_event = None
                if self.cuda_enabled:
                    if self.copy_stream is not None:
                        current_stream = torch.cuda.current_stream(
                            device=self.config.device
                        )
                        with torch.cuda.stream(self.copy_stream):
                            self.copy_stream.wait_stream(current_stream)
                            cpu_view.copy_(ansi_gpu, non_blocking=True)
                            copy_done_event = torch.cuda.Event()
                            copy_done_event.record(self.copy_stream)
                    else:
                        cpu_view.copy_(ansi_gpu, non_blocking=True)
                        copy_done_event = torch.cuda.Event()
                        copy_done_event.record(
                            torch.cuda.current_stream(device=self.config.device)
                        )
                else:
                    cpu_view.copy_(ansi_gpu, non_blocking=False)

                cpu_payload = cpu_view.numpy()
                producer_time = fetch_time + preprocess_time + gen_time
                self._update_adaptive_quality(
                    producer_time,
                    current_frame_idx,
                    frame_bytes=frame_bytes,
                )

                if self.config.timing_enabled:
                    self.ansi_queue.put(
                        (
                            cpu_payload,
                            current_frame_idx,
                            gen_time,
                            fetch_time,
                            preprocess_time,
                            quality_level_used,
                            quant_mask,
                            diff_thresh,
                            copy_done_event,
                            cpu_buf,
                        )
                    )
                else:
                    self.ansi_queue.put(
                        (cpu_payload, current_frame_idx, copy_done_event, cpu_buf)
                    )

                current_frame_idx += 1
            self.ansi_queue.put(None)
        except Exception as e:
            print(f"Error in generator thread: {e}")
            self.thread_exception = e
            self.thread_crashed.set()
            self.ansi_queue.put(None)
            raise

    def __del__(self):
        try:
            if hasattr(self, "audio_process") and self.audio_process:
                self.audio_process.terminate()
                self.audio_process.wait()
            if hasattr(self, "timing_f") and self.timing_f:
                self.timing_f.flush()
                self.timing_f.close()

            if hasattr(self, "config") and getattr(self, "_output_initialized", False):
                os.write(self.config.output_fd, b"\033[?25h\033[?1049l")
        except Exception:
            pass
