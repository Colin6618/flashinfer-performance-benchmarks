import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path


def numeric_values(records, key):
    return [r[key] for r in records if isinstance(r.get(key), (int, float))]


def fmt(value):
    if value is None:
        return "nan"
    return f"{value:.2f}"


def summarize(values):
    if not values:
        return None, None, None, 0.0
    total = sum(values)
    return total / len(values), min(values), max(values), total


# Usage: python analyze_flashinfer_logs.py logs/obs_xxx.jsonl
if len(sys.argv) < 2:
    print("Usage: python analyze_flashinfer_logs.py <logfile.jsonl>")
    sys.exit(1)

logfile = Path(sys.argv[1])
if not logfile.exists():
    print(f"File not found: {logfile}")
    sys.exit(1)

records = []
with open(logfile, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

if not records:
    print(f"No records found in: {logfile}")
    sys.exit(1)

events = list(OrderedDict.fromkeys(r.get("event", "<missing>") for r in records))
print("Available event types:", events)

for event in events:
    sub = [r for r in records if r.get("event") == event]
    cpu_mean, cpu_min, cpu_max, _ = summarize(numeric_values(sub, "duration_ms"))
    gpu_mean, gpu_min, gpu_max, _ = summarize(numeric_values(sub, "gpu_duration_ms"))

    print(f"\n==== {event} ====")
    if cpu_mean is not None:
        print(
            "CPU time (ms): mean=%s, min=%s, max=%s"
            % (fmt(cpu_mean), fmt(cpu_min), fmt(cpu_max))
        )
    if gpu_mean is not None:
        print(
            "GPU time (ms): mean=%s, min=%s, max=%s"
            % (fmt(gpu_mean), fmt(gpu_min), fmt(gpu_max))
        )
    print(f"Count: {len(sub)}")
    hits = [r.get("hit") for r in sub if r.get("hit") is not None]
    if hits:
        print(f"Hit count: {sum(1 for h in hits if h)} / {len(hits)}")

total_inclusive_cpu_ms = sum(numeric_values(records, "duration_ms"))
summary_rows = []
for event in events:
    sub = [r for r in records if r.get("event") == event]
    cpu_values = numeric_values(sub, "duration_ms")
    gpu_values = numeric_values(sub, "gpu_duration_ms")
    cpu_mean, cpu_min, cpu_max, cpu_total = summarize(cpu_values)
    gpu_mean, gpu_min, gpu_max, gpu_total = summarize(gpu_values)
    summary_rows.append(
        {
            "event": event,
            "count": len(sub),
            "timed_count": len(cpu_values),
            "inclusive_total_ms": cpu_total,
            "mean_duration_ms": cpu_mean if cpu_mean is not None else 0.0,
            "min_duration_ms": cpu_min if cpu_min is not None else 0.0,
            "max_duration_ms": cpu_max if cpu_max is not None else 0.0,
            "inclusive_sum_percent": (
                cpu_total / total_inclusive_cpu_ms * 100.0
                if total_inclusive_cpu_ms > 0
                else 0.0
            ),
            "total_gpu_ms": gpu_total,
            "mean_gpu_ms": gpu_mean if gpu_mean is not None else 0.0,
            "min_gpu_ms": gpu_min if gpu_min is not None else 0.0,
            "max_gpu_ms": gpu_max if gpu_max is not None else 0.0,
        }
    )

summary_rows.sort(key=lambda r: r["inclusive_total_ms"], reverse=True)

print("\n==== Inclusive Event Table ====")
print("Durations are full event-scope measurements. Shares are over summed event durations.")
headers = [
    "event",
    "count",
    "timed_count",
    "inclusive_total_ms",
    "mean_duration_ms",
    "min_duration_ms",
    "max_duration_ms",
    "inclusive_sum_percent",
    "total_gpu_ms",
    "mean_gpu_ms",
]
print("\t".join(headers))
for row in summary_rows:
    print(
        "\t".join(
            str(row[h]) if h in ("event", "count", "timed_count") else f"{row[h]:.4f}"
            for h in headers
        )
    )

all_fields = []
seen_fields = set()
for preferred in ["event", "module", "duration_ms", "gpu_duration_ms", "hit", "is_aot", "ts"]:
    if any(preferred in r for r in records):
        all_fields.append(preferred)
        seen_fields.add(preferred)
for record in records:
    for key in record:
        if key not in seen_fields:
            all_fields.append(key)
            seen_fields.add(key)

out_csv = logfile.with_suffix(".summary.csv")
with open(out_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_fields)
    writer.writeheader()
    writer.writerows(records)

out_agg = logfile.with_suffix(".event_summary.csv")
with open(out_agg, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"\nRaw data exported to: {out_csv}")
print(f"Aggregated event summary exported to: {out_agg}")
