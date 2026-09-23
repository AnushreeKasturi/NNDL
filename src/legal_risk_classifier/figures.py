"""Figures for the dataset report.

Each chart answers one design question, so each is a single-series magnitude
plot rather than a decorative multi-colour one. Identity lives on the axis;
colour carries magnitude or split membership only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .analysis import BERT_TOKEN_LIMIT, TOKENS_PER_WORD  # noqa: E402
from .labels import LABEL_NAMES  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

# Validated categorical slots 1-3 (all-pairs clean in both modes).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", ["#eaf1fb", "#2a78d6", "#123a68"])


def _style(ax) -> None:
    """Recessive axes and grid; text in ink tokens, never a series colour."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def document_lengths(report: dict, out_dir: Path) -> Path:
    """Where documents sit relative to the 512-token wall. Answers Problem 2."""
    stats = report["document_lengths"]
    words = stats["_raw_words"]
    limit_words = BERT_TOKEN_LIMIT / TOKENS_PER_WORD

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    ax.hist(words, bins=60, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(limit_words, color="#e34948", linewidth=2, linestyle="--")
    top = ax.get_ylim()[1]
    ax.annotate(
        f"BERT 512-token limit (~{limit_words:.0f} words)",
        xy=(limit_words, top * 0.92),
        xytext=(max(words) * 0.22, top * 0.92),
        color=INK_SECONDARY,
        fontsize=9,
        va="center",
        arrowprops={"arrowstyle": "-", "color": "#e34948", "linewidth": 1, "shrinkA": 2},
    )
    over = stats["exceeding_bert_512"]
    ax.set_title(
        f"{over} of {stats['documents']} contracts exceed BERT's context window",
        fontsize=12,
        pad=12,
        loc="left",
    )
    ax.set_xlabel("words per contract")
    ax.set_ylabel("contracts")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    return _save(fig, out_dir / "document_lengths.png")


def chunk_positive_rates(report: dict, out_dir: Path) -> Path:
    """The imbalance the loss actually sees. Direct labels supply contrast relief."""
    per_label = report["chunk_labels"]["per_label"]
    ordered = sorted(LABEL_NAMES, key=lambda n: per_label[n]["positive_rate"])
    rates = [per_label[n]["positive_rate"] * 100 for n in ordered]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    ax.barh(ordered, rates, height=0.62, color=SERIES[0])
    for name, rate in zip(ordered, rates):
        ax.text(
            rate + max(rates) * 0.015,
            name,
            f"{rate:.2f}%  ({per_label[name]['positive_chunks']:,} chunks)",
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )
    ax.set_xlim(0, max(rates) * 1.45)
    ax.set_title(
        f"Positive chunks per label — {report['chunk_labels']['unlabelled_rate'] * 100:.0f}% of "
        f"{report['chunk_labels']['chunks']:,} chunks carry no label",
        fontsize=12,
        pad=12,
        loc="left",
    )
    ax.set_xlabel("share of all chunks (%)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    return _save(fig, out_dir / "chunk_positive_rates.png")


def cooccurrence(report: dict, out_dir: Path) -> Path:
    """Off-diagonal mass is what makes the task multi-label."""
    matrix = report["label_cooccurrence"]
    grid = [[matrix[a][b] for b in LABEL_NAMES] for a in LABEL_NAMES]

    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor=SURFACE)
    image = ax.imshow(grid, cmap=SEQUENTIAL)
    ax.set_xticks(range(len(LABEL_NAMES)), LABEL_NAMES, rotation=40, ha="right")
    ax.set_yticks(range(len(LABEL_NAMES)), LABEL_NAMES)
    peak = max(max(row) for row in grid)
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=9,
                color="#ffffff" if value > peak * 0.55 else INK,
            )
    ax.set_title("Contracts containing both labels", fontsize=12, pad=12, loc="left")
    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=INK_SECONDARY, length=0, labelsize=9)
    _style(ax)
    ax.grid(False)
    return _save(fig, out_dir / "label_cooccurrence.png")


def split_balance(report: dict, out_dir: Path) -> Path:
    """Confirms no split is skewed — a held-out set must be representative."""
    balance = report["split_balance"]
    splits = ("train", "val", "test")
    width = 0.26

    fig, ax = plt.subplots(figsize=(8.6, 4.2), facecolor=SURFACE)
    for offset, (split, colour) in enumerate(zip(splits, SERIES)):
        rates = [balance[split]["label_rates"][n] * 100 for n in LABEL_NAMES]
        positions = [i + (offset - 1) * width for i in range(len(LABEL_NAMES))]
        ax.bar(
            positions,
            rates,
            width=width * 0.92,
            color=colour,
            label=f"{split} (n={balance[split]['documents']})",
        )
        for x, rate in zip(positions, rates):
            ax.text(x, rate + 1.2, f"{rate:.0f}", ha="center", fontsize=7.5, color=INK_SECONDARY)

    ax.set_xticks(range(len(LABEL_NAMES)), LABEL_NAMES, rotation=20, ha="right")
    # Headroom so the legend clears the tallest bar's direct label.
    ax.set_ylim(0, max(ax.get_ylim()[1], 1) * 1.3)
    ax.set_title("Documents containing each label, by split (%)", fontsize=12, pad=12, loc="left")
    ax.set_ylabel("documents containing the label (%)")
    legend = ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    return _save(fig, out_dir / "split_balance.png")


def render_all(report: dict, out_dir: Path) -> list[Path]:
    figures = [document_lengths, chunk_positive_rates, cooccurrence]
    if "split_balance" in report:
        figures.append(split_balance)
    return [make(report, out_dir) for make in figures]
