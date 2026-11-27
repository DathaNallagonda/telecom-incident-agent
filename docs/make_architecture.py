"""Draw docs/architecture.png (run: python docs/make_architecture.py)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

SURFACE, TEXT, TEXT2, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
AGENT_FILL, AGENT_EDGE = "#dbe9fb", "#2a78d6"  # LLM agents: judgment
CODE_FILL, CODE_EDGE = "#e5f4e4", "#008300"  # deterministic code: guarantees
HUMAN_FILL, HUMAN_EDGE = "#f9eeda", "#c98500"
NEUTRAL_FILL, NEUTRAL_EDGE = "#f0efec", "#a8a7a0"

fig, ax = plt.subplots(figsize=(16, 7), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(0, 16)
ax.set_ylim(-0.6, 6.4)
ax.axis("off")

boxes = {}


def box(key, x, y, w, h, title, sub, fill, edge):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.6,
        )
    )
    if title:
        ax.text(
            x + w / 2,
            y + h * 0.65,
            title,
            ha="center",
            va="center",
            fontsize=11,
            color=TEXT,
            fontweight="bold",
        )
    if sub:
        ax.text(
            x + w / 2,
            y + h * 0.3,
            sub,
            ha="center",
            va="center",
            fontsize=8.5,
            color=TEXT2,
            linespacing=1.3,
        )
    boxes[key] = (x, y, w, h)


def anchor(key, side):
    x, y, w, h = boxes[key]
    return {
        "l": (x, y + h / 2),
        "r": (x + w, y + h / 2),
        "t": (x + w / 2, y + h),
        "b": (x + w / 2, y),
    }[side]


def arrow(a, b, label="", dashed=False, lx=0.0, ly=0.08):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color=LINE,
            linestyle=(0, (4, 3)) if dashed else "-",
            shrinkA=2,
            shrinkB=2,
        )
    )
    if label:
        mx, my = (a[0] + b[0]) / 2 + lx, (a[1] + b[1]) / 2 + ly
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=8.5, color=TEXT2, linespacing=1.2)


# --- boxes -------------------------------------------------------------------
Y, H = 3.0, 1.2
box(
    "text",
    0.2,
    Y + 0.1,
    1.6,
    1.0,
    "Incident text",
    "customer or\nNOC report",
    NEUTRAL_FILL,
    NEUTRAL_EDGE,
)
box(
    "triage",
    2.4,
    Y,
    2.0,
    H,
    "Triage Agent",
    "LLM + 5 network tools\nfaults, sectors, P1-P3",
    AGENT_FILL,
    AGENT_EDGE,
)
box("planner", 5.0, Y, 2.0, H, "Planner Agent", "LLM proposes a\nJSON plan", AGENT_FILL, AGENT_EDGE)
box("verifier", 8.4, Y, 2.0, H, "Verifier", "unified-planning\nPlanValidator", CODE_FILL, CODE_EDGE)
box(
    "operator",
    11.2,
    Y,
    2.0,
    H,
    "Operator Agent",
    "step by step,\nre-checks live state",
    AGENT_FILL,
    AGENT_EDGE,
)
box(
    "comms",
    13.8,
    Y,
    2.0,
    H,
    "Comms Agent",
    "customer SMS +\nincident report",
    AGENT_FILL,
    AGENT_EDGE,
)
box("human", 11.3, 5.0, 1.8, 0.85, "Human", "approves risky steps", HUMAN_FILL, HUMAN_EDGE)
box("pyper", 8.4, 1.35, 2.0, 1.0, "pyperplan", "classical planner", CODE_FILL, CODE_EDGE)
box("network", 2.4, 0.15, 10.8, 0.6, "", "", NEUTRAL_FILL, NEUTRAL_EDGE)
ax.text(
    7.8,
    0.45,
    "Network (simulated towers, links, sectors)",
    ha="center",
    va="center",
    fontsize=10,
    color=TEXT,
    fontweight="bold",
)

# --- arrows ------------------------------------------------------------------
mid = Y + H / 2
arrow(anchor("text", "r"), anchor("triage", "l"))
arrow(anchor("triage", "r"), anchor("planner", "l"), "triage")
arrow((7.0, mid + 0.25), (8.4, mid + 0.25), "plan")
arrow((8.4, mid - 0.25), (7.0, mid - 0.25), "failing step +\ncondition (<= 3 rounds)", ly=-0.72)
arrow(anchor("verifier", "r"), anchor("operator", "l"), "valid")
arrow(anchor("operator", "r"), anchor("comms", "l"), "log")
arrow(anchor("human", "b"), anchor("operator", "t"), "approve / reject", lx=0.8, ly=-0.12)
arrow(
    anchor("verifier", "b"), anchor("pyper", "t"), "still invalid", dashed=True, lx=0.52, ly=-0.08
)
arrow((10.4, 2.15), (11.5, Y), "new plan", dashed=True, lx=-0.05, ly=0.12)
arrow((11.9, Y), (10.4, 1.6), "step failed:\nreplan", dashed=True, lx=0.6, ly=-0.6)
arrow(anchor("triage", "b"), (3.4, 0.75), "investigates (tools)", lx=0.95, ly=0.3)
arrow((12.7, Y), (12.7, 0.75), "applies actions", lx=0.8, ly=0.3)

# --- legend and title --------------------------------------------------------
for i, (fill, edge, label) in enumerate(
    [
        (AGENT_FILL, AGENT_EDGE, "LLM agent (judgment)"),
        (CODE_FILL, CODE_EDGE, "Deterministic code (guarantees)"),
        (HUMAN_FILL, HUMAN_EDGE, "Human in the loop"),
    ]
):
    x = 2.4 + i * 3.6
    ax.add_patch(
        FancyBboxPatch(
            (x, -0.45),
            0.32,
            0.22,
            boxstyle="round,pad=0.01",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.3,
        )
    )
    ax.text(x + 0.45, -0.34, label, va="center", fontsize=9, color=TEXT2)

ax.text(0.2, 5.95, "Telecom Incident Agent", fontsize=15, color=TEXT, fontweight="bold")
ax.text(
    0.2,
    5.5,
    "LLM-Modulo: the LLM proposes, a symbolic verifier checks every plan "
    "before it touches the network",
    fontsize=10,
    color=TEXT2,
)

out = Path(__file__).with_name("architecture.png")
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print(f"wrote {out}")
