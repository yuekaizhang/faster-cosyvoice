#!/usr/bin/env python3
"""Plot Nari-style fixed-concurrency CosyVoice3 latency and capacity curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402


@dataclass(frozen=True)
class SeriesSpec:
    label: str
    run_prefix: str
    concurrencies: tuple[int, ...]
    color: str
    marker: str
    linestyle: str
    backend: str
    mps: bool
    tensorrt: bool | None


@dataclass(frozen=True)
class Point:
    fixed_concurrency: int
    actual_rps: float
    ttfp_p50_ms: float
    ttfp_p95_ms: float
    audible_ttfa_p95_ms: float
    e2e_p95_ms: float
    started_requests: int
    pcm_complete_requests: int
    audible_requests: int
    successful_requests: int
    underrun_requests: int
    underrun_request_fraction: float
    received_audio_xrt: float
    first_chunk_p50_ms: float
    first_chunk_p95_ms: float
    run_name: str

    @property
    def clean(self) -> bool:
        return (
            self.pcm_complete_requests == self.started_requests
            and self.audible_requests == self.started_requests
            and self.successful_requests == self.started_requests
            and self.underrun_requests == 0
        )


COMMON_10 = (1, 2, 3, 4, 6, 8, 10)
COMMON_16 = (*COMMON_10, 12, 14, 16)

SERIES = (
    SeriesSpec(
        label="faster-cosyvoice (MPS ON)",
        run_prefix="fixed-faster-cosyvoice-mps-concurrency",
        concurrencies=COMMON_16,
        color="#159a78",
        marker="o",
        linestyle="-",
        backend="faster-cosyvoice",
        mps=True,
        tensorrt=None,
    ),
    SeriesSpec(
        label="faster-cosyvoice (MPS OFF)",
        run_prefix="fixed-faster-cosyvoice-mps-off-concurrency",
        concurrencies=COMMON_16,
        color="#159a78",
        marker="o",
        linestyle="--",
        backend="faster-cosyvoice",
        mps=False,
        tensorrt=None,
    ),
    SeriesSpec(
        label="Triton + TRT-LLM (760 ms, MPS ON)",
        run_prefix="fixed-triton-trtllm-padding760-mps-on-concurrency",
        concurrencies=COMMON_16,
        color="#348bd2",
        marker="s",
        linestyle="-",
        backend="triton-trtllm",
        mps=True,
        tensorrt=True,
    ),
    SeriesSpec(
        label="Triton + TRT-LLM (760 ms, MPS OFF)",
        run_prefix="fixed-triton-trtllm-padding760-mps-off-concurrency",
        concurrencies=COMMON_16,
        color="#348bd2",
        marker="s",
        linestyle="--",
        backend="triton-trtllm",
        mps=False,
        tensorrt=True,
    ),
    SeriesSpec(
        label="SGLang-Omni (MPS ON)†",
        run_prefix="fixed-sglang-omni-main-7bbdac6-cu130-mps-on-concurrency",
        concurrencies=COMMON_10,
        color="#d98516",
        marker="P",
        linestyle="-",
        backend="sglang-omni",
        mps=True,
        tensorrt=False,
    ),
    SeriesSpec(
        label="SGLang-Omni (MPS OFF)†",
        run_prefix="fixed-sglang-omni-main-7bbdac6-cu130-concurrency",
        concurrencies=COMMON_10,
        color="#d98516",
        marker="P",
        linestyle="--",
        backend="sglang-omni",
        mps=False,
        tensorrt=False,
    ),
    SeriesSpec(
        label="vLLM-Omni (TRT ON, MPS ON)*",
        run_prefix=(
            "fixed-vllm-omni-main-e2b83-schemafix-cu130-trt-on-mps-on-concurrency"
        ),
        concurrencies=(*COMMON_10, 12, 14),
        color="#756bd6",
        marker="^",
        linestyle="-",
        backend="vllm-omni",
        mps=True,
        tensorrt=True,
    ),
    SeriesSpec(
        label="vLLM-Omni (TRT ON, MPS OFF)*",
        run_prefix=(
            "fixed-vllm-omni-main-e2b83-schemafix-cu130-trt-on-mps-off-concurrency"
        ),
        concurrencies=COMMON_16,
        color="#756bd6",
        marker="^",
        linestyle="--",
        backend="vllm-omni",
        mps=False,
        tensorrt=True,
    ),
    SeriesSpec(
        label="vLLM-Omni (TRT OFF, MPS OFF)",
        run_prefix="fixed-vllm-omni-main-e2b83-cu130-trt-off-mps-off-concurrency",
        concurrencies=COMMON_10,
        color="#8b8a84",
        marker="D",
        linestyle=":",
        backend="vllm-omni",
        mps=False,
        tensorrt=False,
    ),
)

OLD_TRITON_SERIES = (
    SeriesSpec(
        label="Triton old 440 ms (MPS ON)",
        run_prefix="fixed-triton-trtllm-mps-on-concurrency",
        concurrencies=COMMON_10,
        color="#bf6958",
        marker="X",
        linestyle="-",
        backend="triton-trtllm-old-440ms",
        mps=True,
        tensorrt=True,
    ),
    SeriesSpec(
        label="Triton old 440 ms (MPS OFF)",
        run_prefix="fixed-triton-trtllm-mps-off-concurrency",
        concurrencies=COMMON_10,
        color="#bf6958",
        marker="X",
        linestyle="--",
        backend="triton-trtllm-old-440ms",
        mps=False,
        tensorrt=True,
    ),
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo_root / "benchmarks" / "results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "docs" / "assets_new",
    )
    return parser.parse_args()


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


def load_point(results_root: Path, spec: SeriesSpec, concurrency: int) -> Point:
    run_name = f"{spec.run_prefix}-{concurrency}-seed-0"
    run_dir = results_root / run_name
    summary_path = run_dir / "summary.json"
    status_path = run_dir / "status.json"
    if not summary_path.is_file() or not status_path.is_file():
        raise FileNotFoundError(f"missing fixed-concurrency result: {run_dir}")

    status = json.loads(status_path.read_text())
    if status.get("state") != "complete":
        raise ValueError(f"incomplete fixed-concurrency result: {run_dir}")

    summary = json.loads(summary_path.read_text())
    if summary.get("load_model") != "closed_loop_fixed_concurrency":
        raise ValueError(f"unexpected load model in {summary_path}")
    if int(summary["fixed_concurrency"]) != concurrency:
        raise ValueError(f"concurrency mismatch in {summary_path}")

    first_playable = summary["first_playable_ms"]
    first_chunks = first_chunk_durations_ms(run_dir / "events.jsonl")
    if not first_chunks:
        raise ValueError(f"no measurement-phase audio chunks in {run_dir}")
    return Point(
        fixed_concurrency=concurrency,
        actual_rps=float(summary["actual_rps"]),
        ttfp_p50_ms=float(first_playable["p50"]),
        ttfp_p95_ms=float(first_playable["p95"]),
        audible_ttfa_p95_ms=float(summary["audible_ttfa_ms"]["p95"]),
        e2e_p95_ms=float(summary["end_to_end_ms"]["p95"]),
        started_requests=int(summary["started_requests"]),
        pcm_complete_requests=int(summary["pcm_complete_requests"]),
        audible_requests=int(summary["audible_requests"]),
        successful_requests=int(summary["successful_requests"]),
        underrun_requests=int(summary["underrun_requests"]),
        underrun_request_fraction=float(summary["underrun_request_fraction"]),
        received_audio_xrt=float(summary["received_audio_xrt"]),
        first_chunk_p50_ms=percentile(first_chunks, 0.50),
        first_chunk_p95_ms=percentile(first_chunks, 0.95),
        run_name=run_name,
    )


def latency_tick(value: float, _position: int) -> str:
    return f"{value:,.0f}"


def plot_metric(
    loaded: list[tuple[SeriesSpec, list[Point]]],
    *,
    metric: str,
    percentile: str,
    output_stem: Path,
) -> None:
    figure_bg = "#faf9f5"
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=figure_bg)
    ax.set_facecolor(figure_bg)

    # Nari-style latency zones. They are visual guides, not pass/fail gates.
    ax.axhspan(50, 200, color="#e9f3df", alpha=0.88, zorder=0)
    ax.axhspan(200, 1_000, color="#f8f7f1", alpha=0.92, zorder=0)
    ax.axhspan(1_000, 10_000, color="#f8e9e6", alpha=0.72, zorder=0)
    ax.axhline(200, color="#71a94c", linestyle="--", linewidth=1.3, zorder=1)
    ax.axhline(1_000, color="#e06458", linestyle="--", linewidth=1.3, zorder=1)

    for index, (spec, points) in enumerate(loaded):
        x = [point.fixed_concurrency for point in points]
        y = [getattr(point, metric) for point in points]
        ax.plot(
            x,
            y,
            label=spec.label,
            color=spec.color,
            marker=spec.marker,
            linestyle=spec.linestyle,
            linewidth=2.7 if spec.mps else 2.35,
            markersize=8.6,
            markeredgewidth=1.1,
            markeredgecolor=figure_bg,
            zorder=4 + index * 0.02,
        )

        degraded = [point for point in points if not point.clean]
        if degraded:
            ax.scatter(
                [point.fixed_concurrency for point in degraded],
                [getattr(point, metric) for point in degraded],
                marker=spec.marker,
                s=98,
                facecolors=figure_bg,
                edgecolors=spec.color,
                linewidths=2.2,
                zorder=6,
            )

    ax.set_yscale("log")
    ax.set_ylim(10_000, 50)
    ax.set_xlim(0.55, 16.45)
    ax.set_xticks(COMMON_16)
    ax.yaxis.set_major_locator(
        FixedLocator((50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000))
    )
    ax.yaxis.set_major_formatter(FuncFormatter(latency_tick))
    ax.grid(
        True,
        which="major",
        color="#d7d4cc",
        linewidth=0.9,
        linestyle=(0, (1.5, 4.5)),
        alpha=0.9,
    )
    ax.grid(False, which="minor")

    ax.set_xlabel(
        "Fixed concurrency (outstanding requests)\nHigher concurrency →",
        fontsize=14.5,
        labelpad=17,
    )
    ax.set_ylabel(
        f"First-playable {percentile} TTFP (ms, log scale)\nLower is better →",
        fontsize=14.5,
        labelpad=18,
    )
    ax.tick_params(axis="both", which="major", labelsize=12.3, colors="#66645f")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8c8982")

    fig.suptitle(
        f"CosyVoice3 {percentile} TTFP at fixed concurrency",
        x=0.11,
        y=0.972,
        ha="left",
        fontsize=25,
        color="#262522",
    )
    fig.text(
        0.11,
        0.927,
        "Single NVIDIA H100 80GB · closed-loop workers · 15 s warmup + 60 s measurement · seed 0",
        ha="left",
        fontsize=13.2,
        color="#66645f",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.105, 0.902),
        ncol=3,
        frameon=False,
        fontsize=11.2,
        handlelength=3.0,
        columnspacing=1.45,
        labelspacing=0.75,
    )
    fig.text(
        0.11,
        0.018,
        "Hollow marker = incomplete/audibility/success mismatch or any simulated playback underrun.  "
        "*vLLM TRT-ON uses the one-field schema workaround.\n"
        "†SGLang-Omni currently buffers the full waveform, so TTFP ≈ E2E.  "
        "vLLM TRT-OFF ends at C10 because C12 crashed before producing valid TTFP.",
        ha="left",
        fontsize=9.7,
        color="#6f6c66",
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.73, bottom=0.16)

    for suffix in ("png", "svg"):
        output_path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(output_path, dpi=128, facecolor=figure_bg)
        print(output_path)
    plt.close(fig)


def plot_operational_metric(
    loaded: list[tuple[SeriesSpec, list[Point]]],
    *,
    metric: str,
    scale: float,
    title: str,
    ylabel: str,
    output_stem: Path,
    reference_760: bool = False,
) -> None:
    figure_bg = "#faf9f5"
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=figure_bg)
    ax.set_facecolor(figure_bg)

    for index, (spec, points) in enumerate(loaded):
        x = [point.fixed_concurrency for point in points]
        y = [getattr(point, metric) * scale for point in points]
        ax.plot(
            x,
            y,
            label=spec.label,
            color=spec.color,
            marker=spec.marker,
            linestyle=spec.linestyle,
            linewidth=2.7 if spec.mps else 2.35,
            markersize=8.6,
            markeredgewidth=1.1,
            markeredgecolor=figure_bg,
            zorder=4 + index * 0.02,
        )
        degraded = [point for point in points if not point.clean]
        if degraded:
            ax.scatter(
                [point.fixed_concurrency for point in degraded],
                [getattr(point, metric) * scale for point in degraded],
                marker=spec.marker,
                s=98,
                facecolors=figure_bg,
                edgecolors=spec.color,
                linewidths=2.2,
                zorder=6,
            )

    if reference_760:
        ax.axhline(760, color="#484641", linestyle=(0, (5, 4)), linewidth=1.5)
        ax.text(
            16.25,
            790,
            "streaming target: 760 ms",
            ha="right",
            va="bottom",
            fontsize=10.5,
            color="#5f5c56",
        )

    ax.set_xlim(0.55, 16.45)
    ax.set_xticks(COMMON_16)
    ax.set_ylim(bottom=0)
    ax.grid(
        True,
        which="major",
        color="#d7d4cc",
        linewidth=0.9,
        linestyle=(0, (1.5, 4.5)),
        alpha=0.9,
    )
    ax.set_xlabel(
        "Fixed concurrency (outstanding requests)\nHigher concurrency →",
        fontsize=14.5,
        labelpad=17,
    )
    ax.set_ylabel(ylabel, fontsize=14.5, labelpad=18)
    ax.tick_params(axis="both", which="major", labelsize=12.3, colors="#66645f")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8c8982")

    fig.suptitle(title, x=0.11, y=0.972, ha="left", fontsize=25, color="#262522")
    fig.text(
        0.11,
        0.927,
        "Single NVIDIA H100 80GB · closed-loop workers · 15 s warmup + 60 s measurement · seed 0",
        ha="left",
        fontsize=13.2,
        color="#66645f",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.105, 0.902),
        ncol=3,
        frameon=False,
        fontsize=11.2,
        handlelength=3.0,
        columnspacing=1.45,
        labelspacing=0.75,
    )
    fig.text(
        0.11,
        0.018,
        "Hollow marker = incomplete/audibility/success mismatch or any simulated playback underrun.  "
        "SGLang-Omni returns a buffered full waveform; its first chunk is therefore the whole output.",
        ha="left",
        fontsize=9.7,
        color="#6f6c66",
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.73, bottom=0.16)

    for suffix in ("png", "svg"):
        output_path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(output_path, dpi=128, facecolor=figure_bg)
        print(output_path)
    plt.close(fig)


def plot_latency_throughput_pareto(
    loaded: list[tuple[SeriesSpec, list[Point]]], output_stem: Path
) -> None:
    figure_bg = "#faf9f5"
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=figure_bg)
    ax.set_facecolor(figure_bg)

    for index, (spec, points) in enumerate(loaded):
        ax.plot(
            [point.actual_rps for point in points],
            [point.ttfp_p95_ms for point in points],
            label=spec.label,
            color=spec.color,
            marker=spec.marker,
            linestyle=spec.linestyle,
            linewidth=2.7 if spec.mps else 2.35,
            markersize=8.6,
            markeredgewidth=1.1,
            markeredgecolor=figure_bg,
            zorder=4 + index * 0.02,
        )
        degraded = [point for point in points if not point.clean]
        if degraded:
            ax.scatter(
                [point.actual_rps for point in degraded],
                [point.ttfp_p95_ms for point in degraded],
                marker=spec.marker,
                s=98,
                facecolors=figure_bg,
                edgecolors=spec.color,
                linewidths=2.2,
                zorder=6,
            )

    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.yaxis.set_major_formatter(FuncFormatter(latency_tick))
    ax.grid(
        True,
        which="major",
        color="#d7d4cc",
        linewidth=0.9,
        linestyle=(0, (1.5, 4.5)),
        alpha=0.9,
    )
    ax.grid(False, which="minor")
    ax.set_xlabel("Achieved requests/s\nHigher is better →", fontsize=14.5, labelpad=17)
    ax.set_ylabel(
        "First-playable p95 TTFP (ms, log scale)\nLower is better →",
        fontsize=14.5,
        labelpad=18,
    )
    ax.tick_params(axis="both", which="major", labelsize=12.3, colors="#66645f")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8c8982")

    fig.suptitle(
        "CosyVoice3 latency-throughput frontier",
        x=0.11,
        y=0.972,
        ha="left",
        fontsize=25,
        color="#262522",
    )
    fig.text(
        0.11,
        0.927,
        "Each line walks from C1 toward higher fixed concurrency · best operating region is upper-right",
        ha="left",
        fontsize=13.2,
        color="#66645f",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.105, 0.902),
        ncol=3,
        frameon=False,
        fontsize=11.2,
        handlelength=3.0,
        columnspacing=1.45,
        labelspacing=0.75,
    )
    fig.text(
        0.11,
        0.018,
        "Hollow marker = incomplete/audibility/success mismatch or any simulated playback underrun.",
        ha="left",
        fontsize=9.7,
        color="#6f6c66",
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.73, bottom=0.16)
    for suffix in ("png", "svg"):
        output_path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(output_path, dpi=128, facecolor=figure_bg)
        print(output_path)
    plt.close(fig)


def write_csv(loaded: list[tuple[SeriesSpec, list[Point]]], path: Path) -> None:
    fields = (
        "backend_label",
        "backend",
        "mps",
        "tensorrt",
        "fixed_concurrency",
        "actual_rps",
        "ttfp_p50_ms",
        "ttfp_p95_ms",
        "audible_ttfa_p95_ms",
        "e2e_p95_ms",
        "started_requests",
        "pcm_complete_requests",
        "audible_requests",
        "successful_requests",
        "underrun_requests",
        "underrun_request_fraction",
        "received_audio_xrt",
        "first_chunk_p50_ms",
        "first_chunk_p95_ms",
        "clean",
        "run_name",
    )
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for spec, points in loaded:
            for point in points:
                writer.writerow(
                    {
                        "backend_label": spec.label,
                        "backend": spec.backend,
                        "mps": spec.mps,
                        "tensorrt": spec.tensorrt,
                        **point.__dict__,
                        "clean": point.clean,
                    }
                )
    print(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = [
        (
            spec,
            [load_point(args.results_root, spec, value) for value in spec.concurrencies],
        )
        for spec in SERIES
    ]
    plot_metric(
        loaded,
        metric="ttfp_p50_ms",
        percentile="p50",
        output_stem=args.output_dir / "cosyvoice3-fixed-concurrency-p50-ttfp",
    )
    plot_metric(
        loaded,
        metric="ttfp_p95_ms",
        percentile="p95",
        output_stem=args.output_dir / "cosyvoice3-fixed-concurrency-p95-ttfp",
    )
    plot_operational_metric(
        loaded,
        metric="actual_rps",
        scale=1.0,
        title="CosyVoice3 achieved request throughput",
        ylabel="Completed requests/s\nHigher is better →",
        output_stem=args.output_dir / "cosyvoice3-fixed-concurrency-actual-rps",
    )
    plot_operational_metric(
        loaded,
        metric="received_audio_xrt",
        scale=1.0,
        title="CosyVoice3 generated-audio throughput",
        ylabel="Received audio × realtime\nHigher is better →",
        output_stem=args.output_dir / "cosyvoice3-fixed-concurrency-audio-xrt",
    )
    plot_operational_metric(
        loaded,
        metric="underrun_request_fraction",
        scale=100.0,
        title="CosyVoice3 simulated-playback underrun rate",
        ylabel="Requests with ≥1 playback underrun (%)\nLower is better →",
        output_stem=args.output_dir / "cosyvoice3-fixed-concurrency-underrun-rate",
    )
    plot_operational_metric(
        loaded,
        metric="first_chunk_p50_ms",
        scale=1.0,
        title="CosyVoice3 first audio-chunk buffer",
        ylabel="First response chunk p50 audio duration (ms)",
        output_stem=args.output_dir / "cosyvoice3-fixed-concurrency-first-chunk",
        reference_760=True,
    )
    plot_latency_throughput_pareto(
        loaded,
        args.output_dir / "cosyvoice3-fixed-concurrency-p95-ttfp-vs-rps",
    )
    old_triton_loaded = [
        (
            spec,
            [load_point(args.results_root, spec, value) for value in spec.concurrencies],
        )
        for spec in OLD_TRITON_SERIES
    ]
    triton_padding_comparison = [
        item
        for item in loaded
        if item[0].run_prefix.startswith("fixed-triton-trtllm-padding760-")
    ] + old_triton_loaded
    plot_operational_metric(
        triton_padding_comparison,
        metric="underrun_request_fraction",
        scale=100.0,
        title="Triton prompt-padding fix: playback continuity",
        ylabel="Requests with ≥1 playback underrun (%)\nLower is better →",
        output_stem=args.output_dir / "triton-padding-before-after-underrun-rate",
    )
    plot_operational_metric(
        triton_padding_comparison,
        metric="actual_rps",
        scale=1.0,
        title="Triton prompt-padding fix: request throughput",
        ylabel="Completed requests/s\nHigher is better →",
        output_stem=args.output_dir / "triton-padding-before-after-actual-rps",
    )
    plot_operational_metric(
        triton_padding_comparison,
        metric="first_chunk_p50_ms",
        scale=1.0,
        title="Triton prompt-padding fix: first audio buffer",
        ylabel="First response chunk p50 audio duration (ms)",
        output_stem=args.output_dir / "triton-padding-before-after-first-chunk",
        reference_760=True,
    )
    write_csv(loaded, args.output_dir / "cosyvoice3-fixed-concurrency-data.csv")
    write_csv(
        triton_padding_comparison,
        args.output_dir / "triton-padding-before-after-data.csv",
    )


if __name__ == "__main__":
    main()
