#!/usr/bin/env python3
"""Plot Nari-style CosyVoice3 latency-under-load curves from raw summaries."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402


@dataclass(frozen=True)
class SeriesSpec:
    label: str
    run_names: tuple[str, ...]
    color: str
    marker: str
    linestyle: str = "-"
    annotate_all: bool = False


@dataclass(frozen=True)
class Point:
    requested_rps: float
    actual_rps: float
    ttfp_p50_ms: float
    ttfp_p95_ms: float
    audible_ttfa_p50_ms: float
    audible_ttfa_p95_ms: float
    successful_requests: int
    scheduled_requests: int
    underrun_requests: int
    run_name: str

    @property
    def clean(self) -> bool:
        return (
            self.successful_requests == self.scheduled_requests
            and self.underrun_requests == 0
        )


SERIES = (
    SeriesSpec(
        label="faster-cosyvoice",
        run_names=tuple(
            f"scout-deadline-rps-{rps}-seed-0" for rps in (1, 2, 4, 6, 8, 10)
        ),
        color="#159a78",
        marker="o",
        annotate_all=True,
    ),
    SeriesSpec(
        label="Triton + TRT-LLM (MPS ON)",
        run_names=tuple(
            f"triton-trtllm-mps-scout-rps-{rps}-seed-0"
            for rps in (1, 2, 3, 4, 6)
        ),
        color="#348bd2",
        marker="s",
        annotate_all=True,
    ),
    SeriesSpec(
        label="Triton + TRT-LLM (MPS OFF)",
        run_names=tuple(
            f"triton-trtllm-readme-scout-rps-{rps}-seed-0"
            for rps in (1, 2, 3, 6)
        ),
        color="#8b8a84",
        marker="D",
        linestyle="--",
    ),
    SeriesSpec(
        label="vLLM-Omni*",
        run_names=(
            "vllm-omni-main-be335a86-rps-1-seed-0",
            "vllm-omni-main-be335a86-rps-2-seed-0",
            "vllm-omni-main-be335a86-resumablefix-rps-3-seed-0",
            "vllm-omni-main-be335a86-resumablefix-rps-4-seed-0",
            "vllm-omni-main-be335a86-resumablefix-rps-6-seed-0",
        ),
        color="#756bd6",
        marker="^",
    ),
    SeriesSpec(
        label="SGLang-Omni†",
        run_names=tuple(
            f"sglang-omni-cosyvoice3-main-7bbdac6-scout-rps-{rps}-seed-0"
            for rps in (1, 2, 3, 4, 6)
        ),
        color="#d98516",
        marker="P",
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
        default=repo_root / "docs" / "assets",
    )
    return parser.parse_args()


def load_point(results_root: Path, run_name: str) -> Point:
    summary_path = results_root / run_name / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing benchmark summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    return Point(
        requested_rps=float(summary["requested_rps"]),
        actual_rps=float(summary["actual_rps"]),
        ttfp_p50_ms=float(summary["first_playable_ms"]["p50"]),
        ttfp_p95_ms=float(summary["first_playable_ms"]["p95"]),
        audible_ttfa_p50_ms=float(summary["audible_ttfa_ms"]["p50"]),
        audible_ttfa_p95_ms=float(summary["audible_ttfa_ms"]["p95"]),
        successful_requests=int(summary["successful_requests"]),
        scheduled_requests=int(summary["scheduled_requests"]),
        underrun_requests=int(summary["underrun_requests"]),
        run_name=run_name,
    )


def format_latency(value: float) -> str:
    if value < 1_000:
        return f"{value:.0f} ms"
    if value < 10_000:
        return f"{value / 1_000:.1f} s"
    return f"{value / 1_000:.0f} s"


def latency_tick(value: float, _position: int) -> str:
    return f"{value:,.0f}"


def plot_metric(
    loaded: list[tuple[SeriesSpec, list[Point]]],
    metric: str,
    title: str,
    ylabel: str,
    output_stem: Path,
) -> None:
    figure_bg = "#faf9f5"
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=figure_bg)
    ax.set_facecolor(figure_bg)

    # Nari-style latency zones. These are visual guides, not pass/fail gates.
    ax.axhspan(50, 200, color="#e9f3df", alpha=0.88, zorder=0)
    ax.axhspan(200, 1_000, color="#f8f7f1", alpha=0.92, zorder=0)
    ax.axhspan(1_000, 200_000, color="#f8e9e6", alpha=0.72, zorder=0)
    ax.axhline(200, color="#71a94c", linestyle="--", linewidth=1.3, zorder=1)
    ax.axhline(1_000, color="#e06458", linestyle="--", linewidth=1.3, zorder=1)

    annotation_offsets = {
        "faster-cosyvoice": (0, 12),
        "Triton + TRT-LLM (MPS ON)": (0, -22),
    }
    for spec, points in loaded:
        x = [point.requested_rps for point in points]
        y = [getattr(point, metric) for point in points]
        ax.plot(
            x,
            y,
            label=spec.label,
            color=spec.color,
            marker=spec.marker,
            linestyle=spec.linestyle,
            linewidth=2.8,
            markersize=9.5,
            markeredgewidth=1.2,
            markeredgecolor=figure_bg,
            zorder=3,
        )

        degraded = [point for point in points if not point.clean]
        if degraded:
            ax.scatter(
                [point.requested_rps for point in degraded],
                [getattr(point, metric) for point in degraded],
                marker=spec.marker,
                s=112,
                facecolors=figure_bg,
                edgecolors=spec.color,
                linewidths=2.4,
                zorder=4,
            )

        if spec.annotate_all:
            offset = annotation_offsets[spec.label]
            for point in points:
                value = getattr(point, metric)
                ax.annotate(
                    format_latency(value),
                    (point.requested_rps, value),
                    xytext=offset,
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if offset[1] > 0 else "top",
                    color=spec.color,
                    fontsize=11.5,
                    bbox={
                        "boxstyle": "square,pad=0.12",
                        "facecolor": figure_bg,
                        "edgecolor": "none",
                        "alpha": 0.86,
                    },
                    zorder=5,
                )

    ax.set_yscale("log")
    ax.set_ylim(200_000, 50)
    ax.set_xlim(0.75, 10.35)
    ax.set_xticks((1, 2, 3, 4, 6, 8, 10))
    ax.yaxis.set_major_locator(
        FixedLocator((50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000,
                      50_000, 100_000, 200_000))
    )
    ax.yaxis.set_major_formatter(FuncFormatter(latency_tick))
    ax.grid(True, which="major", color="#d7d4cc", linewidth=0.9,
            linestyle=(0, (1.5, 4.5)), alpha=0.9)
    ax.grid(False, which="minor")

    ax.set_xlabel("Request rate (RPS)\nHigher is better →", fontsize=15, labelpad=18)
    ax.set_ylabel(f"{ylabel} (ms, log scale)\nLower is better →", fontsize=15, labelpad=18)
    ax.tick_params(axis="both", which="major", labelsize=12.5, colors="#66645f")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#8c8982")

    fig.suptitle(title, x=0.11, y=0.965, ha="left", fontsize=26, color="#262522")
    fig.text(
        0.11,
        0.914,
        "Single NVIDIA H100 80GB · Poisson arrivals · 60 s · seed 0",
        ha="left",
        fontsize=13.5,
        color="#66645f",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.105, 0.89),
        ncol=3,
        frameon=False,
        fontsize=12.5,
        handlelength=3.0,
        columnspacing=1.8,
    )
    fig.text(
        0.11,
        0.018,
        "Hollow marker = <100% successful requests or any underrun.  "
        "*vLLM-Omni uses the schema-only compatibility fix at RPS ≥3.\n"
        "†SGLang-Omni's current CosyVoice3 vocoder buffers the full waveform, so "
        "TTFP ≈ E2E.  Latency zones are visual guides only.",
        ha="left",
        fontsize=10.2,
        color="#6f6c66",
    )
    fig.subplots_adjust(left=0.11, right=0.975, top=0.79, bottom=0.16)

    for suffix in ("png", "svg"):
        path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=128, facecolor=figure_bg)
        print(path)
    plt.close(fig)


def write_csv(loaded: list[tuple[SeriesSpec, list[Point]]], path: Path) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "backend",
                "requested_rps",
                "actual_rps",
                "ttfp_p50_ms",
                "ttfp_p95_ms",
                "audible_ttfa_p50_ms",
                "audible_ttfa_p95_ms",
                "successful_requests",
                "scheduled_requests",
                "underrun_requests",
                "run_name",
            ),
        )
        writer.writeheader()
        for spec, points in loaded:
            for point in points:
                writer.writerow({"backend": spec.label, **point.__dict__})
    print(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = [
        (spec, [load_point(args.results_root, name) for name in spec.run_names])
        for spec in SERIES
    ]
    plot_metric(
        loaded,
        metric="audible_ttfa_p50_ms",
        title="CosyVoice3 p50 audible TTFA under load",
        ylabel="Audible p50 TTFA",
        output_stem=args.output_dir / "cosyvoice3-p50-audible-ttfa-under-load",
    )
    plot_metric(
        loaded,
        metric="ttfp_p50_ms",
        title="CosyVoice3 p50 TTFP under load",
        ylabel="First-playable p50 TTFP",
        output_stem=args.output_dir / "cosyvoice3-p50-ttfp-under-load",
    )
    plot_metric(
        loaded,
        metric="audible_ttfa_p95_ms",
        title="CosyVoice3 p95 audible TTFA under load",
        ylabel="Audible p95 TTFA",
        output_stem=args.output_dir / "cosyvoice3-p95-audible-ttfa-under-load",
    )
    plot_metric(
        loaded,
        metric="ttfp_p95_ms",
        title="CosyVoice3 p95 TTFP under load",
        ylabel="First-playable p95 TTFP",
        output_stem=args.output_dir / "cosyvoice3-p95-ttfp-under-load",
    )
    write_csv(loaded, args.output_dir / "cosyvoice3-load-curve-data.csv")


if __name__ == "__main__":
    main()
