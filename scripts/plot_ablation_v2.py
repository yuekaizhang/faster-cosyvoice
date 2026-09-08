#!/usr/bin/env python3
"""Build the compact C1/C8 TTFP and audio-xRT ablation figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

CONFIGS = (
    ("A0", "Baseline\nvLLM + Torch"),
    ("A1", "+ DSpark"),
    ("A2", "+ FlashInfer"),
    ("A3", "+ Packed\nbatching"),
    ("A4", "+ Deadline\nscheduler"),
    ("A5", "+ Flow\nCUDA Graph"),
    ("A6", "+ HiFT\nCUDA Graph"),
    ("A7", "+ CUDA MPS"),
)
CONCURRENCIES = (1, 8)


@dataclass(frozen=True)
class Point:
    config: str
    label: str
    concurrency: int
    ttfp_p50_ms: float
    ttfp_p95_ms: float
    received_audio_xrt: float
    actual_rps: float
    started_requests: int
    successful_requests: int
    pcm_complete_requests: int
    underrun_requests: int
    first_chunk_samples: int
    first_chunk_p50_ms: float
    first_chunk_p95_ms: float
    clean: bool


def percentile(values: list[float], quantile: float) -> float:
    """Return the benchmark contract's nearest-rank percentile."""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def first_chunk_durations_ms(events_path: Path) -> list[float]:
    """Extract each measured request's first response-audio chunk duration."""
    first_by_request: dict[str, tuple[int, float]] = {}
    with events_path.open() as events:
        for line in events:
            event = json.loads(line)
            if event.get("phase") != "measurement":
                continue
            if event.get("type") != "response.body_chunk":
                continue
            media_duration_ns = event.get("data", {}).get("media_duration_ns")
            if media_duration_ns is None:
                continue
            request_id = str(event["request_id"])
            elapsed_ns = int(event["elapsed_ns"])
            previous = first_by_request.get(request_id)
            if previous is None or elapsed_ns < previous[0]:
                first_by_request[request_id] = (
                    elapsed_ns,
                    float(media_duration_ns) / 1_000_000.0,
                )
    return [value[1] for value in first_by_request.values()]


def load_points(
    root: Path,
    seed: int,
    expected_first_chunk_ms: float,
    first_chunk_tolerance_ms: float,
    concurrencies: tuple[int, ...] = CONCURRENCIES,
) -> list[Point]:
    points: list[Point] = []
    for config, label in CONFIGS:
        for concurrency in concurrencies:
            run_dir = root / f"{config}-c{concurrency}-seed-{seed}"
            status = json.loads((run_dir / "status.json").read_text())
            if status.get("state") != "complete":
                raise ValueError(f"incomplete run: {run_dir}")
            summary = json.loads((run_dir / "summary.json").read_text())
            if summary.get("load_model") != "closed_loop_fixed_concurrency":
                raise ValueError(f"unexpected load model: {run_dir}")
            if int(summary["fixed_concurrency"]) != concurrency:
                raise ValueError(f"concurrency mismatch: {run_dir}")
            first_chunks = first_chunk_durations_ms(run_dir / "events.jsonl")
            if not first_chunks:
                raise ValueError(f"no measured first chunks: {run_dir}")
            outside_tolerance = [
                value
                for value in first_chunks
                if abs(value - expected_first_chunk_ms) > first_chunk_tolerance_ms
            ]
            if outside_tolerance:
                raise ValueError(
                    f"unaligned first chunks in {run_dir}: expected "
                    f"{expected_first_chunk_ms:.3f} +/- {first_chunk_tolerance_ms:.3f} ms, "
                    f"observed {min(first_chunks):.3f}..{max(first_chunks):.3f} ms"
                )
            metric = summary["first_playable_ms"]
            started_requests = int(summary["started_requests"])
            successful_requests = int(summary["successful_requests"])
            pcm_complete_requests = int(summary["pcm_complete_requests"])
            underrun_requests = int(summary["underrun_requests"])
            if len(first_chunks) != pcm_complete_requests:
                raise ValueError(
                    f"first-chunk/complete-PCM mismatch in {run_dir}: "
                    f"{len(first_chunks)} != {pcm_complete_requests}"
                )
            points.append(
                Point(
                    config=config,
                    label=label,
                    concurrency=concurrency,
                    ttfp_p50_ms=float(metric["p50"]),
                    ttfp_p95_ms=float(metric["p95"]),
                    received_audio_xrt=float(summary["received_audio_xrt"]),
                    actual_rps=float(summary["actual_rps"]),
                    started_requests=started_requests,
                    successful_requests=successful_requests,
                    pcm_complete_requests=pcm_complete_requests,
                    underrun_requests=underrun_requests,
                    first_chunk_samples=len(first_chunks),
                    first_chunk_p50_ms=percentile(first_chunks, 0.50),
                    first_chunk_p95_ms=percentile(first_chunks, 0.95),
                    clean=(
                        successful_requests == started_requests
                        and pcm_complete_requests == started_requests
                        and underrun_requests == 0
                    ),
                )
            )
    return points


def by_concurrency(points: list[Point], concurrency: int) -> list[Point]:
    return [point for point in points if point.concurrency == concurrency]


def style_axis(axis, title: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", fontsize=13, fontweight="bold")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#D9DEE7", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def annotate(axis, bars, values, fmt: str) -> None:
    for bar, value in zip(bars, values, strict=True):
        axis.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_latency(points: list[Point], output: Path, percentile: str) -> None:
    field = f"ttfp_{percentile}_ms"
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    colors = ("#4C78A8", "#E45756")
    for axis, concurrency, color in zip(axes, CONCURRENCIES, colors, strict=True):
        cohort = by_concurrency(points, concurrency)
        values = [getattr(point, field) for point in cohort]
        bars = axis.bar(np.arange(len(cohort)), values, color=color, width=0.72)
        for bar, point in zip(bars, cohort, strict=True):
            if not point.clean:
                bar.set_hatch("///")
                bar.set_edgecolor("#5E2B2B")
        axis.set_xticks(np.arange(len(cohort)), [point.label for point in cohort], fontsize=8)
        style_axis(axis, f"Fixed concurrency C{concurrency}", f"TTFP {percentile} (ms) · lower is better")
        if max(values) / max(min(values), 1e-9) >= 8:
            axis.set_yscale("log")
            axis.set_ylim(bottom=max(min(values) * 0.7, 1))
        annotate(axis, bars, values, "{:.0f}")
        if any(not point.clean for point in cohort):
            axis.legend(
                handles=[
                    Patch(
                        facecolor=color,
                        edgecolor="#5E2B2B",
                        hatch="///",
                        label="Underrun / not benchmark-clean",
                    )
                ],
                loc="upper right",
                bbox_to_anchor=(1.0, 1.08),
                frameon=False,
                fontsize=8,
            )
    fig.suptitle(
        f"Faster CosyVoice3 cumulative ablation · TTFP {percentile}",
        fontsize=16,
        fontweight="bold",
    )
    for suffix in ("png", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_xrt(points: list[Point], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    x = np.arange(len(CONFIGS))
    width = 0.36
    c1 = by_concurrency(points, 1)
    c8 = by_concurrency(points, 8)
    values1 = [point.received_audio_xrt for point in c1]
    values8 = [point.received_audio_xrt for point in c8]
    bars1 = axis.bar(x - width / 2, values1, width, label="C1", color="#4C78A8")
    bars8 = axis.bar(x + width / 2, values8, width, label="C8", color="#F2A541")
    for bar, point in zip(bars1, c1, strict=True):
        if not point.clean:
            bar.set_hatch("///")
            bar.set_edgecolor("#5E2B2B")
    for bar, point in zip(bars8, c8, strict=True):
        if not point.clean:
            bar.set_hatch("///")
            bar.set_edgecolor("#5E2B2B")
    axis.set_xticks(x, [label for _, label in CONFIGS], fontsize=9)
    style_axis(axis, "Fixed-concurrency aggregate audio throughput", "Received audio xRT · higher is better")
    axis.legend(
        handles=[
            Patch(facecolor="#4C78A8", label="C1"),
            Patch(facecolor="#F2A541", label="C8"),
            Patch(
                facecolor="#F2A541",
                edgecolor="#5E2B2B",
                hatch="///",
                label="Underrun / not benchmark-clean",
            ),
        ],
        frameon=False,
        ncols=3,
    )
    annotate(axis, bars1, values1, "{:.1f}")
    annotate(axis, bars8, values8, "{:.1f}")
    fig.suptitle(
        "Faster CosyVoice3 cumulative ablation · audio throughput",
        fontsize=16,
        fontweight="bold",
    )
    for suffix in ("png", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_c1_ttfp(points: list[Point], output: Path) -> None:
    """Plot the final C1-only grouped p50/p95 TTFP figure."""
    cohort = by_concurrency(points, 1)
    if len(cohort) != len(CONFIGS):
        raise ValueError(f"expected {len(CONFIGS)} C1 points, got {len(cohort)}")
    if any(not point.clean for point in cohort):
        raise ValueError("C1 final figure requires benchmark-clean points")

    figure_bg = "#FAF9F5"
    fig, axis = plt.subplots(figsize=(15, 7), constrained_layout=True, facecolor=figure_bg)
    axis.set_facecolor(figure_bg)
    x = np.arange(len(cohort))
    width = 0.36
    p50 = [point.ttfp_p50_ms for point in cohort]
    p95 = [point.ttfp_p95_ms for point in cohort]
    bars50 = axis.bar(x - width / 2, p50, width, label="p50", color="#4C78A8")
    bars95 = axis.bar(x + width / 2, p95, width, label="p95", color="#F2A541")

    axis.set_xticks(x, [point.label for point in cohort], fontsize=9)
    style_axis(axis, "Fixed concurrency C1 · single H100 80GB", "TTFP (ms) · lower is better")
    axis.set_ylim(0, max(p95) * 1.18)
    axis.legend(frameon=False, ncols=2, loc="upper right", fontsize=11)
    annotate(axis, bars50, p50, "{:.1f}")
    annotate(axis, bars95, p95, "{:.1f}")
    fig.suptitle(
        "Faster CosyVoice3 cumulative ablation · C1 TTFP",
        fontsize=17,
        fontweight="bold",
    )
    for suffix in ("png", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(points: list[Point], output: Path) -> None:
    fields = tuple(Point.__dataclass_fields__)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in points:
            row = {field: getattr(point, field) for field in fields}
            row["label"] = str(row["label"]).replace("\n", " ")
            writer.writerow(row)


def write_c1_csv(points: list[Point], output: Path) -> None:
    fields = ("config", "label", "ttfp_p50_ms", "ttfp_p95_ms")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in by_concurrency(points, 1):
            writer.writerow(
                {
                    "config": point.config,
                    "label": point.label.replace("\n", " "),
                    "ttfp_p50_ms": point.ttfp_p50_ms,
                    "ttfp_p95_ms": point.ttfp_p95_ms,
                }
            )


def pct_delta(before: float, after: float, higher_is_better: bool) -> str:
    if before == 0 or not math.isfinite(before) or not math.isfinite(after):
        return "n/a"
    value = ((after / before) - 1) * 100
    if not higher_is_better:
        value = -value
    return f"{value:+.1f}%"


def write_readme(points: list[Point], output: Path, root: Path) -> None:
    lookup = {(point.config, point.concurrency): point for point in points}
    a0_c1 = lookup[("A0", 1)]
    a0_c8 = lookup[("A0", 8)]
    a3_c8 = lookup[("A3", 8)]
    a6_c1 = lookup[("A6", 1)]
    a6_c8 = lookup[("A6", 8)]
    a7_c1 = lookup[("A7", 1)]
    a7_c8 = lookup[("A7", 8)]
    lines = [
        "# Faster CosyVoice3 cumulative ablation",
        "",
        "Single NVIDIA H100 80GB (physical GPU 0); NeMo RL container; measured 2026-09-02. "
        "Every point uses the same registered voice and Seed-TTS evaluation text sequence, "
        "15 s warm-up plus a 60 s measurement window, closed-loop fixed concurrency, no "
        "leading-silence trim, and uniform-20 streaming chunks. All measured first chunks "
        "were exactly 760 ms. A0–A6 use MPS OFF; A7 differs from A6 only by MPS ON.",
        "",
        f"Raw benchmark root: `{root}`",
        "",
        "Figures: [TTFP p50](ablation-ttfp-p50.png), "
        "[TTFP p95](ablation-ttfp-p95.png), and "
        "[received-audio xRT](ablation-audio-xrt.png). "
        "Machine-readable values are in [ablation-data.csv](ablation-data.csv).",
        "",
        "## Configurations",
        "",
        "| Step | Cumulative configuration | MPS |",
        "|---|---|---:|",
        "| A0 | vLLM target-only LLM + Torch Flow + serial token2wav + legacy scheduler + eager HiFT | OFF |",
        "| A1 | A0 + DSpark speculative LLM decoding | OFF |",
        "| A2 | A1 + FlashInfer Flow estimator | OFF |",
        "| A3 | A2 + packed token2wav batching | OFF |",
        "| A4 | A3 + deadline-aware scheduler | OFF |",
        "| A5 | A4 + Flow CUDA Graph | OFF |",
        "| A6 | A5 + HiFT CUDA Graph | OFF |",
        "| A7 | A6 + CUDA MPS | ON |",
        "",
        "## Results",
        "",
        "| Step | C1 TTFP p50 / p95 (ms) | C1 xRT | C8 TTFP p50 / p95 (ms) | C8 xRT | C1 p50 gain vs previous | C8 xRT gain vs previous |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    previous_c1 = None
    previous_c8 = None
    for config, label in CONFIGS:
        c1 = lookup[(config, 1)]
        c8 = lookup[(config, 8)]
        gain_c1 = "—" if previous_c1 is None else pct_delta(previous_c1.ttfp_p50_ms, c1.ttfp_p50_ms, False)
        gain_c8 = "—" if previous_c8 is None else pct_delta(previous_c8.received_audio_xrt, c8.received_audio_xrt, True)
        lines.append(
            f"| {config} {label.replace(chr(10), ' ')} | {c1.ttfp_p50_ms:.1f} / {c1.ttfp_p95_ms:.1f} | "
            f"{c1.received_audio_xrt:.2f} | {c8.ttfp_p50_ms:.1f} / {c8.ttfp_p95_ms:.1f} | "
            f"{c8.received_audio_xrt:.2f} | {gain_c1} | {gain_c8} |"
        )
        previous_c1, previous_c8 = c1, c8
    lines.extend(
        (
            "",
            "### Headline deltas",
            "",
            f"- A0 → A7 at C1: TTFP p50 {a0_c1.ttfp_p50_ms:.1f} → "
            f"{a7_c1.ttfp_p50_ms:.1f} ms ({pct_delta(a0_c1.ttfp_p50_ms, a7_c1.ttfp_p50_ms, False)}), "
            f"p95 {a0_c1.ttfp_p95_ms:.1f} → {a7_c1.ttfp_p95_ms:.1f} ms "
            f"({pct_delta(a0_c1.ttfp_p95_ms, a7_c1.ttfp_p95_ms, False)}), and xRT "
            f"{a0_c1.received_audio_xrt:.2f} → {a7_c1.received_audio_xrt:.2f} "
            f"({pct_delta(a0_c1.received_audio_xrt, a7_c1.received_audio_xrt, True)}).",
            f"- A0 → A7 at C8: TTFP p50 {a0_c8.ttfp_p50_ms:.1f} → "
            f"{a7_c8.ttfp_p50_ms:.1f} ms ({pct_delta(a0_c8.ttfp_p50_ms, a7_c8.ttfp_p50_ms, False)}), "
            f"p95 {a0_c8.ttfp_p95_ms:.1f} → {a7_c8.ttfp_p95_ms:.1f} ms "
            f"({pct_delta(a0_c8.ttfp_p95_ms, a7_c8.ttfp_p95_ms, False)}), and raw received-audio "
            f"xRT {a0_c8.received_audio_xrt:.2f} → {a7_c8.received_audio_xrt:.2f} "
            f"({pct_delta(a0_c8.received_audio_xrt, a7_c8.received_audio_xrt, True)}).",
            f"- Last-step MPS effect (A6 → A7): C1 p50 "
            f"{pct_delta(a6_c1.ttfp_p50_ms, a7_c1.ttfp_p50_ms, False)}, C1 xRT "
            f"{pct_delta(a6_c1.received_audio_xrt, a7_c1.received_audio_xrt, True)}; "
            f"C8 p50 {pct_delta(a6_c8.ttfp_p50_ms, a7_c8.ttfp_p50_ms, False)}, C8 xRT "
            f"{pct_delta(a6_c8.received_audio_xrt, a7_c8.received_audio_xrt, True)}.",
            f"- On the first fully clean C8 configuration, A3 → A7 raises xRT "
            f"{a3_c8.received_audio_xrt:.2f} → {a7_c8.received_audio_xrt:.2f} "
            f"({pct_delta(a3_c8.received_audio_xrt, a7_c8.received_audio_xrt, True)}).",
            "",
            "### Health and interpretation",
            "",
            "| Step | C1 complete / success / underrun | C8 complete / success / underrun | First chunk p50 / p95 (ms) |",
            "|---|---:|---:|---:|",
        )
    )
    for config, _label in CONFIGS:
        c1 = lookup[(config, 1)]
        c8 = lookup[(config, 8)]
        lines.append(
            f"| {config} | {c1.pcm_complete_requests}/{c1.successful_requests}/{c1.underrun_requests} "
            f"of {c1.started_requests} | {c8.pcm_complete_requests}/{c8.successful_requests}/"
            f"{c8.underrun_requests} of {c8.started_requests} | "
            f"{c8.first_chunk_p50_ms:.0f} / {c8.first_chunk_p95_ms:.0f} |"
        )
    lines.extend(
        (
            "",
            "TTFP is Nari `first_playable_ms`; xRT is aggregate PCM duration received inside the "
            "measurement window divided by wall-clock measurement duration. Incremental gains are "
            "order-dependent and must not be summed.",
            "",
            "A0–A2 at C8 completed every PCM stream but had underruns and fewer benchmark-successful "
            "requests, so their raw xRT values are retained for transparency rather than treated as "
            "quality-safe capacity. A3–A7 are clean at both C1 and C8. The deadline scheduler's "
            "benefit is intentionally small in this steady closed-loop C1/C8 view; bursty/open-loop "
            "traffic is the workload that exposes its tail-latency protection.",
            "",
            "Semantic speech quality was not evaluated in this performance ablation.",
            "",
        )
    )
    output.write_text("\n".join(lines))


def write_c1_readme(points: list[Point], output: Path, root: Path) -> None:
    cohort = by_concurrency(points, 1)
    baseline = cohort[0]
    final = cohort[-1]
    lines = [
        "# Faster CosyVoice3 C1 TTFP ablation",
        "",
        "Final presentation view containing only fixed concurrency C1 TTFP. The two bars "
        "for each cumulative configuration are p50 and p95.",
        "",
        "Protocol: single NVIDIA H100 80GB; fixed registered voice; Seed-TTS evaluation "
        "text sequence; 15 s warm-up and 60 s measurement; no leading-silence trim; "
        "uniform-20 streaming chunks. Every measured first chunk was exactly 760 ms, and "
        "all C1 requests completed successfully with zero underruns. A0–A6 use MPS OFF; "
        "A7 differs from A6 only by MPS ON.",
        "",
        f"Raw benchmark root: `{root}`",
        "",
        "Figure: [C1 TTFP p50 and p95](c1-ttfp-p50-p95.png). "
        "Machine-readable values: [c1-ttfp.csv](c1-ttfp.csv).",
        "",
        "| Step | Cumulative configuration | TTFP p50 (ms) | TTFP p95 (ms) |",
        "|---|---|---:|---:|",
    ]
    for point in cohort:
        lines.append(
            f"| {point.config} | {point.label.replace(chr(10), ' ')} | "
            f"{point.ttfp_p50_ms:.1f} | {point.ttfp_p95_ms:.1f} |"
        )
    lines.extend(
        (
            "",
            f"A0 → A7 reduces p50 from {baseline.ttfp_p50_ms:.1f} to "
            f"{final.ttfp_p50_ms:.1f} ms "
            f"({pct_delta(baseline.ttfp_p50_ms, final.ttfp_p50_ms, False)}) and p95 from "
            f"{baseline.ttfp_p95_ms:.1f} to {final.ttfp_p95_ms:.1f} ms "
            f"({pct_delta(baseline.ttfp_p95_ms, final.ttfp_p95_ms, False)}).",
            "",
            "TTFP is the Nari benchmark's `first_playable_ms`. No C8 or xRT values are "
            "included in this final view.",
            "",
        )
    )
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets_v2"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-first-chunk-ms", type=float, default=760.0)
    parser.add_argument("--first-chunk-tolerance-ms", type=float, default=0.001)
    parser.add_argument("--profile", choices=("full", "c1-ttfp"), default="full")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    concurrencies = (1,) if args.profile == "c1-ttfp" else CONCURRENCIES
    points = load_points(
        args.results_root,
        args.seed,
        args.expected_first_chunk_ms,
        args.first_chunk_tolerance_ms,
        concurrencies,
    )
    if args.profile == "c1-ttfp":
        write_c1_csv(points, args.output_dir / "c1-ttfp.csv")
        write_c1_readme(points, args.output_dir / "README.md", args.results_root)
        plot_c1_ttfp(points, args.output_dir / "c1-ttfp-p50-p95")
        return
    write_csv(points, args.output_dir / "ablation-data.csv")
    write_readme(points, args.output_dir / "README.md", args.results_root)
    plot_latency(points, args.output_dir / "ablation-ttfp-p50", "p50")
    plot_latency(points, args.output_dir / "ablation-ttfp-p95", "p95")
    plot_xrt(points, args.output_dir / "ablation-audio-xrt")


if __name__ == "__main__":
    main()
