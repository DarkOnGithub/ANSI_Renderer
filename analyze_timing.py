import csv
import os
import statistics
import sys

METRIC_ORDER = (
    "gen_time",
    "fetch_time",
    "preprocess_time",
    "render_time",
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


def print_section(rows, title):
    print(f"\n{title}")
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

    total_values = collect_metric(rows, "total_time")
    if total_values:
        total_duration = sum(total_values)
        print("\nTotal frames:", len(total_values))
        print(f"Total processing time: {total_duration:.2f} seconds")
        print(
            f"Average FPS (processing only): {len(total_values) / total_duration:.2f}"
        )

    end_to_end_values = collect_metric(rows, "end_to_end_time")
    if end_to_end_values:
        end_duration = sum(end_to_end_values)
        print(f"Average FPS (with pacing): {len(end_to_end_values) / end_duration:.2f}")

    datasize = collect_metric(rows, "datasize")
    if datasize and total_values and len(datasize) == len(total_values):
        print(
            f"\nCorrelation between Datasize and Total Time: {correlation(datasize, total_values):.4f}"
        )

    render_values = collect_metric(rows, "render_time")
    if datasize and render_values and len(datasize) == len(render_values):
        print(
            f"Correlation between Datasize and Render Time: {correlation(datasize, render_values):.4f}"
        )

    gen_values = collect_metric(rows, "gen_time")
    if datasize and gen_values and len(datasize) == len(gen_values):
        print(
            f"Correlation between Datasize and Gen Time: {correlation(datasize, gen_values):.4f}"
        )


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
