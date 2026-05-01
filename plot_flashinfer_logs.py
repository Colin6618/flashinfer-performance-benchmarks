import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator


FIG_DPI = 220
COLORS = {
    "blue": "#2563eb",
    "green": "#059669",
    "purple": "#7c3aed",
    "orange": "#d97706",
    "red": "#dc2626",
    "cyan": "#0891b2",
    "gray": "#64748b",
    "grid": "#d7dee8",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate report-ready FlashInfer benchmark figures."
    )
    parser.add_argument("logfile", type=Path, help="Telemetry JSONL file.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def number(value, default=0.0):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def savefig(fig, out_dir, stem):
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def style_axes(ax):
    ax.grid(True, axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


def event_summary(records):
    grouped = OrderedDict()
    for record in records:
        event = record.get("event", "unknown")
        if number(record.get("duration_ms")) > 0:
            grouped.setdefault(event, []).append(record)

    rows = []
    total_duration = 0.0
    for event, items in grouped.items():
        duration = sum(number(item.get("duration_ms")) for item in items)
        total_duration += duration
        rows.append(
            {
                "event": event,
                "timed_count": len(items),
                "duration": duration,
            }
        )
    for row in rows:
        row["percent"] = row["duration"] / total_duration * 100.0 if total_duration else 0.0
    return sorted(rows, key=lambda row: row["duration"], reverse=True)


def short_spec(spec):
    if not spec:
        return "unknown"
    if spec.startswith("single_decode_with_kv_cache"):
        prefix = "single decode"
    elif spec.startswith("batch_decode_with_kv_cache"):
        prefix = "batch decode"
    else:
        prefix = spec.split("_dtype", 1)[0].replace("_", " ")

    head_dim = "?"
    parts = spec.split("_")
    for i, token in enumerate(parts):
        if token == "qk" and i + 1 < len(parts):
            head_dim = parts[i + 1]
            break
    return f"{prefix}\nhead_dim={head_dim}"


def plot_event_time_breakdown(records, out_dir):
    rows = [row for row in event_summary(records) if row["duration"] > 0]
    rows = rows[:5]
    labels = [row["event"] for row in rows][::-1]
    durations = [row["duration"] for row in rows][::-1]
    percents = [row["percent"] for row in rows][::-1]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colors = [
        COLORS["blue"] if label.startswith("jit") else COLORS["green"]
        for label in labels
    ]
    bars = ax.barh(labels, durations, color=colors, height=0.62)
    ax.set_xlabel("Total event-scope duration (ms)")
    ax.set_title("Inclusive Telemetry Event Durations")
    style_axes(ax)

    xmax = max(durations) * 1.25 if durations else 1
    ax.set_xlim(0, xmax)
    for bar, value, pct in zip(bars, durations, percents):
        ax.text(
            bar.get_width() + xmax * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,.1f} ms ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=8.5,
        )
    fig.text(
        0.01,
        0.035,
        "Timed event scopes can overlap; percentages are shares of summed event durations.",
        fontsize=8,
        color=COLORS["gray"],
    )
    fig.subplots_adjust(bottom=0.20)
    savefig(fig, out_dir, "event_time_breakdown")


def plot_jit_build_by_specialization(records, out_dir):
    cache_hit = {}
    builds = []
    for record in records:
        spec = record.get("spec_name")
        if record.get("event") == "jit_cache_hit" and spec:
            cache_hit[spec] = bool(record.get("hit"))
        if record.get("event") == "jit_build":
            builds.append(record)

    if not builds:
        return

    labels = [short_spec(row.get("spec_name")) for row in builds]
    seconds = [number(row.get("duration_ms")) / 1000.0 for row in builds]
    colors = [
        COLORS["cyan"] if cache_hit.get(row.get("spec_name"), False) else COLORS["red"]
        for row in builds
    ]

    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    x = list(range(len(builds)))
    bars = ax.bar(x, seconds, color=colors, width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel("JIT build path time (s)")
    ax.set_title("JIT Cost by Kernel Specialization")
    style_axes(ax)
    ymax = max(seconds) * 1.22
    ax.set_ylim(0, ymax)
    for bar, value in zip(bars, seconds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.025,
            f"{value:.1f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.scatter([], [], color=COLORS["red"], label="cache miss metadata")
    ax.scatter([], [], color=COLORS["cyan"], label="cache hit metadata")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    savefig(fig, out_dir, "jit_build_by_specialization")


def workflow_rows(results_dir):
    path = results_dir / "workflow_results.csv"
    if not path.exists():
        return []
    return read_csv(path)


def plot_workflow_phase_latency(results_dir, out_dir):
    rows = workflow_rows(results_dir)
    if not rows:
        return

    labels = []
    values = []
    for row in rows:
        phase = row.get("phase", "")
        metric = row.get("metric", "")
        label = phase.replace("_", "\n")
        if metric == "avg_call_ms":
            label += "\navg"
        labels.append(label)
        values.append(number(row.get("value_ms")))

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    bars = ax.bar(
        range(len(values)),
        values,
        color=[COLORS["blue"], COLORS["green"], COLORS["purple"], COLORS["orange"]][
            : len(values)
        ],
        width=0.62,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_title("Workflow Phase Latency")
    style_axes(ax)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.35,
            f"{value:,.4f} ms",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.text(
        0.5,
        0.035,
        "Single steady and batch run values are per-call averages; cold and plan values are one-call timings.",
        ha="center",
        fontsize=8,
        color=COLORS["gray"],
    )
    fig.subplots_adjust(bottom=0.30)
    savefig(fig, out_dir, "workflow_phase_latency")


def plot_jit_first_call_overhead(results_dir, out_dir):
    workflow = {row.get("phase"): row for row in workflow_rows(results_dir)}
    if "single_decode_cold" in workflow and "single_decode_steady" in workflow:
        first_ms = number(workflow["single_decode_cold"].get("value_ms"))
        steady_ms = number(workflow["single_decode_steady"].get("value_ms"))
        ratio = first_ms / steady_ms if steady_ms else 0.0
    else:
        first_ms = steady_ms = ratio = 0.0

    path = results_dir / "jit_overhead_results.csv"
    if first_ms <= 0 and not path.exists():
        return
    if first_ms <= 0:
        rows = read_csv(path)
        if not rows:
            return
        row = rows[0]
        first_ms = number(row.get("first_call_ms"))
        steady_ms = number(row.get("steady_avg_ms"))
        ratio = number(row.get("first_vs_steady_ratio"))

    values = [("cold call", first_ms), ("steady avg", steady_ms)]
    labels = [item[0] for item in values]
    y = [item[1] for item in values]

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    bars = ax.bar(labels, y, color=[COLORS["blue"], COLORS["green"]], width=0.55)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_title("First-Call JIT Overhead")
    style_axes(ax)
    for bar, value in zip(bars, y):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.35,
            f"{value:,.4f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.text(
        0.5,
        0.02,
        f"First call / steady-state ratio: {ratio:,.0f}x",
        ha="center",
        fontsize=9,
        color=COLORS["gray"],
    )
    savefig(fig, out_dir, "jit_first_call_overhead")


def plot_decode_run_distribution(records, out_dir):
    values = [
        number(record.get("duration_ms"))
        for record in records
        if record.get("event") == "decode_run" and number(record.get("duration_ms")) > 0
    ]
    if not values:
        return

    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.plot(range(1, len(values) + 1), values, marker="o", linewidth=1.8, color=COLORS["green"])
    ax.set_xlabel("decode_run sample")
    ax.set_ylabel("CPU time (ms)")
    ax.set_title("Batch Decode Run Latency Samples")
    style_axes(ax)
    ax.axhline(sum(values) / len(values), color=COLORS["orange"], linestyle="--", linewidth=1.4, label=f"mean={sum(values)/len(values):.4f} ms")
    ax.legend(frameon=False, fontsize=8)
    savefig(fig, out_dir, "decode_run_samples")


def main():
    args = parse_args()
    if not args.logfile.exists():
        raise SystemExit(f"File not found: {args.logfile}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.logfile)
    plot_event_time_breakdown(records, args.out_dir)
    plot_workflow_phase_latency(args.results_dir, args.out_dir)
    plot_jit_build_by_specialization(records, args.out_dir)
    plot_jit_first_call_overhead(args.results_dir, args.out_dir)
    plot_decode_run_distribution(records, args.out_dir)


if __name__ == "__main__":
    main()
