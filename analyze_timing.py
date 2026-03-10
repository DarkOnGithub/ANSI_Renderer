import csv
import os
import statistics
import sys

METRIC_ORDER = (
    "gen_time",
    "fetch_time",
    "preprocess_time",
    "producer_time",
    "queue_wait_time",
    "copy_wait_time",
    "render_time",
    "consumer_time",
    "pipeline_time",
    "total_time",
    "sleep_time",
    "end_to_end_time",
    "datasize",
)


def correlation(x, y):
    n = len(x)
    if n < 2:
        return 0.0
    mu_x = statistics.mean(x)
    mu_y = statistics.mean(y)
    stdev_x = statistics.stdev(x)
    stdev_y = statistics.stdev(y)
    if stdev_x == 0 or stdev_y == 0:
        return 0.0
    return sum((xi - mu_x) * (yi - mu_y) for xi, yi in zip(x, y)) / (
        (n - 1) * stdev_x * stdev_y
    )


def percentile(sorted_values, pct):
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int((len(sorted_values) - 1) * pct))
    return sorted_values[idx]


def numeric_rows(file_path):
    rows = []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            numeric = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    numeric[key] = float(value)
                except ValueError:
                    continue
            if numeric:
                rows.append(numeric)
    return rows


def collect_metric(rows, key):
    return [row[key] for row in rows if key in row]


def mean_fps(values):
    if not values:
        return 0.0
    mean_value = statistics.mean(values)
    if mean_value <= 0:
        return 0.0
    return 1.0 / mean_value


def print_metric_table(rows):
    print(
        f"{'Metric':<15} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Median':<10} | {'95th':<10} | {'99th':<10}"
    )
    print("-" * 85)

    for key in METRIC_ORDER:
        values = collect_metric(rows, key)
        if not values:
            continue

        v_min = min(values)
        v_max = max(values)
        v_mean = statistics.mean(values)
        v_median = statistics.median(values)
        sorted_values = sorted(values)
        p95 = percentile(sorted_values, 0.95)
        p99 = percentile(sorted_values, 0.99)

        if key == "datasize":
            print(
                f"{key:<15} | {v_min:<10.0f} | {v_max:<10.0f} | {v_mean:<10.0f} | {v_median:<10.0f} | {p95:<10.0f} | {p99:<10.0f}"
            )
        else:
            print(
                f"{key:<15} | {v_min:<10.4f} | {v_max:<10.4f} | {v_mean:<10.4f} | {v_median:<10.4f} | {p95:<10.4f} | {p99:<10.4f}"
            )


def print_summary(rows):
    frame_count = len(rows)
    print(f"\nTotal frames: {frame_count}")

    frame_start_times = collect_metric(rows, "frame_start_time")
    frame_end_times = collect_metric(rows, "frame_end_time")
    if frame_start_times and frame_end_times:
        wall_duration = frame_end_times[-1] - frame_start_times[0]
        if wall_duration > 0:
            print(f"Wall clock time: {wall_duration:.2f} seconds")
            print(f"True wall-clock FPS: {frame_count / wall_duration:.2f}")

    producer_values = collect_metric(rows, "producer_time")
    if producer_values:
        print(f"Estimated producer FPS: {mean_fps(producer_values):.2f}")

    consumer_values = collect_metric(rows, "consumer_time")
    if consumer_values:
        print(f"Estimated consumer FPS: {mean_fps(consumer_values):.2f}")

    pipeline_values = collect_metric(rows, "pipeline_time")
    if pipeline_values:
        pipeline_duration = sum(pipeline_values)
        print(f"Estimated pipeline time: {pipeline_duration:.2f} seconds")
        print(f"Estimated pipeline FPS: {frame_count / pipeline_duration:.2f}")
    else:
        total_values = collect_metric(rows, "total_time")
        if total_values:
            total_duration = sum(total_values)
            print(f"Legacy total time: {total_duration:.2f} seconds")
            print(f"Legacy FPS estimate: {frame_count / total_duration:.2f}")

    end_to_end_values = collect_metric(rows, "end_to_end_time")
    if end_to_end_values:
        print(f"Consumer loop FPS estimate: {mean_fps(end_to_end_values):.2f}")


def print_correlations(rows):
    datasize = collect_metric(rows, "datasize")
    if not datasize:
        return

    pipeline_values = collect_metric(rows, "pipeline_time")
    if pipeline_values and len(datasize) == len(pipeline_values):
        print(
            f"\nCorrelation between Datasize and Pipeline Time: {correlation(datasize, pipeline_values):.4f}"
        )
    else:
        total_values = collect_metric(rows, "total_time")
        if total_values and len(datasize) == len(total_values):
            print(
                f"\nCorrelation between Datasize and Total Time: {correlation(datasize, total_values):.4f}"
            )

    render_values = collect_metric(rows, "render_time")
    if render_values and len(datasize) == len(render_values):
        print(
            f"Correlation between Datasize and Render Time: {correlation(datasize, render_values):.4f}"
        )

    producer_values = collect_metric(rows, "producer_time")
    if producer_values and len(datasize) == len(producer_values):
        print(
            f"Correlation between Datasize and Producer Time: {correlation(datasize, producer_values):.4f}"
        )
    else:
        gen_values = collect_metric(rows, "gen_time")
        if gen_values and len(datasize) == len(gen_values):
            print(
                f"Correlation between Datasize and Gen Time: {correlation(datasize, gen_values):.4f}"
            )


def print_section(rows, title):
    print(f"\n{title}")
    print_metric_table(rows)
    print_summary(rows)
    print_correlations(rows)


def analyze_csv(file_path, warmup_frames=1):
    try:
        rows = numeric_rows(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    if not rows:
        print("No data found.")
        return

    print_section(rows, "All Frames")

    if len(rows) > warmup_frames:
        steady_rows = rows[warmup_frames:]
        print_section(
            steady_rows,
            f"Steady State (Skipping First {warmup_frames} Frame{'s' if warmup_frames != 1 else ''})",
        )


if __name__ == "__main__":
    csv_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "timing_object.csv"
    )
    warmup = 1
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        warmup = max(0, int(sys.argv[2]))
    analyze_csv(csv_file, warmup_frames=warmup)
