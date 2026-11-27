"""Summarise eval/results.jsonl into eval/results.json, eval/summary.md and two charts.

Run: python -m eval.summarize
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parent
RESULTS_JSONL = HERE / "results.jsonl"
RESULTS_JSON = HERE / "results.json"
SUMMARY_MD = HERE / "summary.md"
DOCS = ROOT / "docs"

TYPES = ["power", "hardware", "link", "multi"]
TYPE_LABEL = {
    "power": "Power loss",
    "hardware": "Hardware fault",
    "link": "Link cut",
    "multi": "Multi-fault",
    "all": "All scenarios",
}

# Palette: categorical slots 1-2 of the validated default palette (light surface).
SYSTEM_COLOR, BASELINE_COLOR = "#2a78d6", "#eb6834"
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load() -> list[dict]:
    if not RESULTS_JSONL.exists():
        return []
    return [json.loads(x) for x in RESULTS_JSONL.read_text(encoding="utf-8").splitlines() if x]


def pct(values) -> float:
    values = list(values)
    return 100 * sum(1 for v in values if v) / len(values) if values else 0.0


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(st.mean(values), 2) if values else None


def summarise(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["type"]].append(r)
        groups["all"].append(r)
    out = {}
    for key in TYPES + ["all"]:
        g = groups.get(key, [])
        if not g:
            continue
        sys_ = [r["system"] for r in g]
        base = [r["baseline"] for r in g]
        planned = [(r["system"], r["optimal_len"]) for r in g if r["system"].get("plan_len")]
        bplanned = [(r["baseline"], r["optimal_len"]) for r in g if r["baseline"].get("plan_len")]
        injected = [
            r for r in g if r["system"].get("replans") is not None and r["system"]["replans"] > 0
        ]
        out[key] = {
            "runs": len(g),
            "scenarios": len({r["scenario"] for r in g}),
            "valid_plan_rate": {
                "baseline": pct(b["valid"] for b in base),
                "system": pct(s["valid"] for s in sys_),
            },
            "goal_reached_rate": {
                "baseline": pct(b["goal_reached"] for b in base),
                "system": pct(s["goal_reached"] for s in sys_),
            },
            "unsafe_or_invalid_attempted": {
                "baseline": sum(b.get("attempted_invalid", 0) for b in base),
                "system": sum(s.get("attempted_invalid", 0) for s in sys_),
            },
            "runs_with_unsafe_or_invalid": {
                "baseline": pct(b.get("attempted_invalid", 0) > 0 for b in base),
                "system": pct(s.get("attempted_invalid", 0) > 0 for s in sys_),
            },
            "plan_len_over_optimal": {
                "baseline": mean(b["plan_len"] / o for b, o in bplanned),
                "system": mean(s["plan_len"] / o for s, o in planned),
            },
            "triage_exact_rate": pct(r["triage"]["exact"] for r in g),
            "triage_fault_acc": mean(r["triage"]["fault_acc"] for r in g),
            "triage_sector_acc": mean(r["triage"]["sector_acc"] for r in g),
            "avg_llm_rounds": mean(s.get("rounds") for s in sys_),
            "fallback_rate": pct(s.get("source") == "planner_fallback" for s in sys_),
            "llm_fixed_after_feedback": pct(
                s.get("source") == "llm_verified" and (s.get("rounds") or 0) > 1 for s in sys_
            ),
            "replan_recovery_rate": (
                pct(r["system"]["goal_reached"] for r in injected) if injected else None
            ),
            "replan_runs": len(injected),
            "latency_s": {
                "baseline": mean(b.get("latency_s") for b in base),
                "system": mean(s.get("latency_s") for s in sys_),
            },
        }
    return out


def _style(ax, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT2, length=0, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=TEXT2, fontsize=10)


def grouped_bars(
    summary: dict, metric: str, title: str, ylabel: str, path: Path, percent: bool
) -> None:
    keys = [k for k in TYPES + ["all"] if k in summary]
    base = [summary[k][metric]["baseline"] for k in keys]
    syst = [summary[k][metric]["system"] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    width, gap = 0.36, 0.03
    xs = range(len(keys))
    for offset, vals, color, label in [
        (-(width + gap) / 2, base, BASELINE_COLOR, "LLM-only baseline"),
        ((width + gap) / 2, syst, SYSTEM_COLOR, "Full system (verified)"),
    ]:
        bars = ax.bar([x + offset for x in xs], vals, width, color=color, label=label)
        for b, v in zip(bars, vals, strict=True):
            ax.annotate(
                f"{v:.0f}%" if percent else f"{v:g}",
                (b.get_x() + b.get_width() / 2, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=TEXT,
            )
    ax.set_xticks(list(xs), [TYPE_LABEL[k] for k in keys])
    if percent:
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100])
    _style(ax, ylabel)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, pad=14)
    leg = ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0, 1.02),
        ncol=2,
        fontsize=9,
        labelcolor=TEXT2,
    )
    leg.set_in_layout(True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write_markdown(summary: dict, rows: list[dict]) -> str:
    a = summary.get("all", {})
    if not a:
        return "# Evaluation summary\n\nNo results yet.\n"
    model = rows[0]["model"] if rows else "?"
    lines = [
        "# Evaluation summary",
        "",
        f"{a['runs']} runs over {a['scenarios']} scenarios; model `{model}`; "
        "goal reached is checked on the true network after execution.",
        "",
        f"**Headline:** valid plans {a['valid_plan_rate']['system']:.0f}% (system) vs "
        f"{a['valid_plan_rate']['baseline']:.0f}% (LLM-only baseline); unsafe or invalid "
        f"actions attempted: {a['unsafe_or_invalid_attempted']['system']} vs "
        f"{a['unsafe_or_invalid_attempted']['baseline']}.",
        "",
        "| Metric | Baseline | System |",
        "| --- | --- | --- |",
        f"| Valid-plan rate | {a['valid_plan_rate']['baseline']:.0f}% | "
        f"{a['valid_plan_rate']['system']:.0f}% |",
        f"| Goal reached after execution | {a['goal_reached_rate']['baseline']:.0f}% | "
        f"{a['goal_reached_rate']['system']:.0f}% |",
        f"| Unsafe or invalid actions attempted (total) | "
        f"{a['unsafe_or_invalid_attempted']['baseline']} | "
        f"{a['unsafe_or_invalid_attempted']['system']} |",
        f"| Runs with any unsafe or invalid action | "
        f"{a['runs_with_unsafe_or_invalid']['baseline']:.0f}% | "
        f"{a['runs_with_unsafe_or_invalid']['system']:.0f}% |",
        f"| Plan length / optimal | {a['plan_len_over_optimal']['baseline']} | "
        f"{a['plan_len_over_optimal']['system']} |",
        f"| Latency per incident (s) | {a['latency_s']['baseline']} | {a['latency_s']['system']} |",
        "",
        "| System-only metric | Value |",
        "| --- | --- |",
        f"| Triage exact match | {a['triage_exact_rate']:.0f}% |",
        f"| Triage fault accuracy / sector accuracy | {a['triage_fault_acc']} / "
        f"{a['triage_sector_acc']} |",
        f"| Average LLM plan rounds | {a['avg_llm_rounds']} |",
        f"| LLM plan fixed after verifier feedback | {a['llm_fixed_after_feedback']:.0f}% |",
        f"| Planner fallback rate | {a['fallback_rate']:.0f}% |",
        "| Replan recovery rate | "
        + (
            f"{a['replan_recovery_rate']:.0f}% of {a['replan_runs']} runs |"
            if a["replan_recovery_rate"] is not None
            else "n/a |"
        ),
        "",
        "## By scenario type",
        "",
        "| Type | Runs | Valid plans (base / sys) | Goal reached (base / sys) | "
        "Unsafe attempts (base / sys) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for k in TYPES:
        if k in summary:
            s = summary[k]
            lines.append(
                f"| {TYPE_LABEL[k]} | {s['runs']} | {s['valid_plan_rate']['baseline']:.0f}% / "
                f"{s['valid_plan_rate']['system']:.0f}% | {s['goal_reached_rate']['baseline']:.0f}%"
                f" / {s['goal_reached_rate']['system']:.0f}% | "
                f"{s['unsafe_or_invalid_attempted']['baseline']} / "
                f"{s['unsafe_or_invalid_attempted']['system']} |"
            )
    lines += [
        "",
        "![Valid plans](../docs/valid_plan_rate.png)",
        "![Unsafe actions](../docs/unsafe_actions.png)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    rows = load()
    summary = summarise(rows)
    RESULTS_JSON.write_text(
        json.dumps({"summary": summary, "runs": rows}, indent=1), encoding="utf-8"
    )
    SUMMARY_MD.write_text(write_markdown(summary, rows), encoding="utf-8")
    if summary:
        DOCS.mkdir(exist_ok=True)
        grouped_bars(
            summary,
            "valid_plan_rate",
            "Valid plans: LLM-only vs verified system",
            "Runs with a valid plan (%)",
            DOCS / "valid_plan_rate.png",
            True,
        )
        grouped_bars(
            summary,
            "unsafe_or_invalid_attempted",
            "Unsafe or invalid actions attempted on the network",
            "Actions attempted (total)",
            DOCS / "unsafe_actions.png",
            False,
        )
    print(SUMMARY_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
