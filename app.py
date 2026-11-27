"""Telecom Incident Agent - Streamlit UI.

Incident text -> Triage Agent -> verified plan -> Operator (approvals, replanning)
-> Comms Agent, plus a Benchmark tab that reads saved results (no live LLM calls).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agents.graph import build_graph, config_for, initial_state, pending_interrupt  # noqa: E402
from core import llm as llm_mod  # noqa: E402
from core.simulator import CORE_ID, from_dict, inject_failure, sample_network  # noqa: E402
from core.spec import RISKY  # noqa: E402

ROOT = Path(__file__).parent
SCENARIOS = json.loads((ROOT / "eval" / "scenarios.json").read_text(encoding="utf-8"))
SUMMARY_MD = ROOT / "eval" / "summary.md"

st.set_page_config(page_title="Telecom Incident Agent", page_icon="📡", layout="wide")
# Wide Markdown tables scroll inside their own box instead of widening the page (phones).
st.markdown(
    "<style>.stMarkdown table{display:block;overflow-x:auto;max-width:100%;}</style>",
    unsafe_allow_html=True,
)

# --- colours (validated default palette, light surface) -----------------------
OK, FAULT, REROUTE, MUTED, TEXT = "#2a78d6", "#e34948", "#1baf7a", "#a8a7a0", "#0b0b0b"
POS = {
    "C0": (0, 0),
    "T1": (-1.6, 1.2),
    "T2": (0, 1.8),
    "T3": (1.6, 1.2),
    "T4": (1.6, -1.2),
    "T5": (0, -1.8),
    "T6": (-1.6, -1.2),
}


QUOTA_MARKERS = ("429", "rate limit", "Rate limit", "quota", "RESOURCE_EXHAUSTED", "RateLimit")


def show_error(message: str) -> None:
    """Plain-language error; quota errors get their own explanation, details folded away."""
    if any(m in message for m in QUOTA_MARKERS):
        st.error(
            "**The free LLM quota is used up right now.** Every model in the chain "
            "(Groq gpt-oss-120b, gpt-oss-20b, Gemini) hit its free-tier limit. Groq limits "
            "reset within a few minutes to hours; Gemini's reset daily. The Benchmark tab "
            "still works because it reads saved results."
        )
    else:
        st.error(message[:300])
    with st.expander("Error details"):
        st.code(message, language="text")


def has_llm_key() -> bool:
    try:
        llm_mod.chat_models()
        return True
    except llm_mod.LLMUnavailableError:
        return False


def scenario_network(sc: dict):
    net = sample_network()
    for f in sc["failures"]:
        inject_failure(net, f["kind"], f["node"])
    return net


# --- network drawing ---------------------------------------------------------
def draw_network(net, reroutes: dict[str, str]):
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=130)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for lk in net.links:
        (x1, y1), (x2, y2) = POS[lk.a], POS[lk.b]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=MUTED if lk.up else FAULT,
            lw=2,
            ls="-" if lk.up else (0, (3, 3)),
            zorder=1,
        )
    for s in net.sectors.values():
        tx, ty = POS[s.served_by]
        sx, sy = tx * 1.45, ty * 1.3
        via = reroutes.get(s.id)
        if via:
            bx, by = POS[via]
            ax.annotate(
                "",
                (sx, sy),
                (bx, by),
                zorder=2,
                arrowprops={"arrowstyle": "->", "color": REROUTE, "lw": 2},
            )
        color = OK if s.served else FAULT
        ax.scatter([sx], [sy], s=260, marker="s", color=color, edgecolor="#fcfcfb", lw=2, zorder=3)
        ax.text(sx, sy - 0.32, s.id, ha="center", va="top", fontsize=8, color=TEXT)
    for n in net.nodes.values():
        x, y = POS[n.id]
        color = FAULT if (n.alarm or not n.up) else OK
        ax.scatter(
            [x],
            [y],
            s=900 if n.id == CORE_ID else 620,
            color=color,
            edgecolor="#fcfcfb",
            lw=2,
            zorder=4,
        )
        ax.text(
            x,
            y,
            n.id,
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
            zorder=5,
        )
    ax.set_xlim(-2.9, 2.9)
    ax.set_ylim(-3.0, 3.0)
    ax.axis("off")
    fig.tight_layout()
    return fig


def reroutes_from(log: list[dict]) -> dict[str, str]:
    return {
        e["args"][0]: e["args"][1]
        for e in log
        if e["action"] == "reroute_traffic" and e["result"] == "ok"
    }


# --- graph session -----------------------------------------------------------
def start_run(text: str, net, goal_mode: str, inject: bool) -> None:
    st.session_state.graph = build_graph()
    st.session_state.thread = str(uuid.uuid4())
    st.session_state.result = None
    drive(initial_state(text, net, goal_mode, inject_failure=inject, auto_approve=False))


def drive(payload) -> None:
    """Stream the graph until it ends or pauses for an approval."""
    graph, thread = st.session_state.graph, st.session_state.thread
    status = st.empty()
    labels = {
        "triage": "Triage Agent investigating...",
        "plan": "Planner Agent verifying...",
        "execute": "Operator executing...",
        "communicate": "Comms Agent writing...",
    }
    with st.spinner("Agents working..."):
        for values in graph.stream(payload, config_for(thread), stream_mode="values"):
            st.session_state.result = values
            last = values.get("execution_log") or []
            if last:
                status.caption(
                    f"Last step: {last[-1]['action']}({', '.join(last[-1]['args'])})"
                    f" -> {last[-1]['result']}"
                )
            elif values.get("triage"):
                status.caption(labels["plan"])
            else:
                status.caption(labels["triage"])
    status.empty()
    st.session_state.pending = pending_interrupt(graph, thread)


def resume_run(approved: bool) -> None:
    drive(Command(resume={"approved": approved, "by": "operator (UI)"}))


# --- sidebar -----------------------------------------------------------------
st.sidebar.title("Incident")
labels = {s["id"]: f"{s['id']} - {s['incident_text'][:48]}..." for s in SCENARIOS}
choice = st.sidebar.selectbox("Scenario", list(labels), format_func=labels.get)
sc = next(s for s in SCENARIOS if s["id"] == choice)
custom = st.sidebar.text_area(
    "...or type your own incident",
    placeholder=sc["incident_text"],
    help="Your text is investigated on the selected scenario's network.",
)
goal_mode = st.sidebar.radio(
    "Goal",
    ["default", "full_repair"],
    index=0 if sc["goal_mode"] == "default" else 1,
    format_func={"default": "Restore service", "full_repair": "Restore service + repair nodes"}.get,
)
inject = st.sidebar.toggle(
    "Inject a restart failure",
    value=sc["inject_failure_during_execution"],
    help="The first restart fails, forcing the Operator to replan.",
)
run_clicked = st.sidebar.button("Run agents", type="primary", width="stretch")
st.sidebar.caption("Risky steps (" + ", ".join(sorted(RISKY)) + ") pause for your approval.")

# --- header ------------------------------------------------------------------
st.title("📡 Telecom Incident Agent")
st.caption(
    "LLM agents diagnose, act and communicate; a symbolic verifier checks every plan "
    "before it touches the network."
)

tab_run, tab_bench, tab_how = st.tabs(["Run an incident", "Benchmark", "How it works"])

with tab_run:
    llm_ok = has_llm_key()
    if not llm_ok:
        st.warning(
            "No LLM API key is configured, so the agents can't run here. Add "
            "GEMINI_API_KEY and/or GROQ_API_KEY to `.env` locally or to the app's "
            "Secrets on Streamlit Cloud. The Benchmark tab still works."
        )
    if run_clicked and llm_ok:
        try:
            start_run(
                custom.strip() or sc["incident_text"], scenario_network(sc), goal_mode, inject
            )
        except Exception as e:  # noqa: BLE001
            show_error(f"Run failed: {type(e).__name__}: {e}")

    res = st.session_state.get("result")
    net = from_dict(res["network"]) if res else scenario_network(sc)
    log = (res or {}).get("execution_log", [])

    col_net, col_triage = st.columns([1, 1])
    with col_net:
        st.subheader("1 · Network")
        st.pyplot(draw_network(net, reroutes_from(log)), width="stretch")
        st.caption(
            "Red = alarm, down or no service · dashed red = cut link · "
            "green arrow = traffic rerouted to a backup tower · squares are sectors"
        )

    with col_triage:
        st.subheader("2 · Triage")
        if not res:
            st.info(
                f"Incident: _{custom.strip() or sc['incident_text']}_\n\n"
                "Press **Run agents** in the sidebar."
            )
        elif res.get("triage"):
            t = res["triage"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Priority", t["priority"])
            c2.metric("Confidence", f"{t['confidence']:.0%}")
            c3.metric("Sectors down", len(t["affected_sectors"]))
            st.write(
                "**Faults:** "
                + ", ".join(f"{f['node_id']} ({f['fault_type']})" for f in t["fault_types"])
            )
            st.write("**Affected sectors:** " + (", ".join(t["affected_sectors"]) or "none"))
            st.caption(t["reasoning"])
            with st.expander(f"Tool calls ({len(res.get('triage_calls', []))})"):
                for c in res.get("triage_calls", []):
                    st.code(
                        f"{c['tool']}({json.dumps(c['args'])})\n-> {c.get('result', '')}",
                        language="text",
                    )
        for err in (res or {}).get("errors", []):
            show_error(err)

    if res and res.get("first_plan"):
        st.subheader("3 · Plan")
        fp = res["first_plan"]
        badge = {
            "llm_verified": ("LLM plan, verified", "green"),
            "planner_fallback": ("Planner fallback (pyperplan)", "orange"),
        }.get(fp["source"], (fp["source"], "gray"))
        st.badge(badge[0], color=badge[1])
        st.caption(f"{fp['message']} · LLM rounds: {fp['rounds_used']}")
        cur = res["plan"]
        rows = []
        for i, s in enumerate(cur["steps"], 1):
            done = next(
                (
                    e["result"]
                    for e in log
                    if e["index"] == i and e.get("plan_version") == res.get("plan_version", 0)
                ),
                "",
            )
            rows.append(
                {
                    "#": i,
                    "step": f"{s['action']}({', '.join(s['args'])})",
                    "risky": "⚠ approval" if s["action"] in RISKY else "",
                    "result": done,
                }
            )
        if res.get("plan_version"):
            st.caption(f"Showing plan version {res['plan_version']} (after replanning)")
        st.dataframe(rows, hide_index=True, width="stretch")
        with st.expander("Verify-and-fix log"):
            for r in fp["validation_log"]:
                icon = "✅" if r["valid"] else "❌"
                st.write(
                    f"{icon} Round {r['round']}: "
                    + (r["reason"] or "valid")
                    + f" ({len(r['steps'])} steps)"
                )

    pending = st.session_state.get("pending")
    if res and pending:
        st.warning(f"**Approval needed:** {pending['question']}")
        a, b, _ = st.columns([1, 1, 4])
        if a.button("Approve", type="primary"):
            resume_run(True)
            st.rerun()
        if b.button("Reject"):
            resume_run(False)
            st.rerun()

    if res and log:
        st.subheader("4 · Execution log")
        st.dataframe(
            [
                {
                    "plan": e.get("plan_version", 0),
                    "#": e["index"],
                    "step": f"{e['action']}({', '.join(e['args'])})",
                    "result": e["result"],
                    "approved by": e.get("approved_by") or "",
                    "note": e["message"],
                    "time": e["timestamp"][11:],
                }
                for e in log
            ],
            hide_index=True,
            width="stretch",
        )
        for r in res.get("replans", []):
            st.info(
                f"🔁 Replanned: {r['reason']} → "
                + " → ".join(f"{s['action']}({', '.join(s['args'])})" for s in r["new_plan"])
            )

    if res and res.get("comms"):
        st.subheader("5 · Customer SMS and incident report")
        status = res.get("status")
        (st.success if status == "completed" else st.error)(f"Incident status: **{status}**")
        if res["comms"].get("source") == "template":
            st.caption(
                "The LLM was unavailable (likely a rate limit), so these messages use "
                "the built-in templates."
            )
        for m in res["comms"]["sms"]:
            st.text_area(
                f"SMS · {m['area']} ({len(m['text'])} chars)",
                m["text"],
                height=70,
                disabled=True,
                key=f"sms-{m['sector_id']}",
            )
        with st.expander("Incident report", expanded=True):
            st.markdown(res["comms"]["report"])

with tab_bench:
    st.subheader("LLM-only baseline vs the verified system")
    if SUMMARY_MD.exists():
        text = SUMMARY_MD.read_text(encoding="utf-8")
        body = "\n".join(
            line
            for line in text.splitlines()
            if not line.startswith("![") and not line.startswith("# ")
        )
        goal_png = ROOT / "docs" / "goal_reached_rate.png"
        if goal_png.exists():
            st.image(str(goal_png), width="stretch")
        st.markdown(body)
        c1, c2 = st.columns(2)
        for col, name in [(c1, "valid_plan_rate.png"), (c2, "unsafe_actions.png")]:
            p = ROOT / "docs" / name
            if p.exists():
                col.image(str(p), width="stretch")
        st.caption("Results are loaded from eval/results.json; this tab makes no LLM calls.")
    else:
        st.info(
            "No benchmark results yet. Run `python -m eval.run_eval` then "
            "`python -m eval.summarize`."
        )

with tab_how:
    st.subheader("How it works")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box style="rounded,filled" fillcolor="#f0efec" fontname=Helvetica
  fontsize=11 color="#a8a7a0"]; edge [color="#52514e" fontname=Helvetica fontsize=9]
  text [label="Incident text" shape=note fillcolor="#fcfcfb"]
  triage [label="Triage Agent\\n(LLM + network tools)" fillcolor="#dbe9fb"]
  planner [label="Planner Agent\\n(LLM proposes)" fillcolor="#dbe9fb"]
  verifier [label="Verifier\\n(PlanValidator)" fillcolor="#e5f4e4"]
  pyper [label="pyperplan\\nfallback" fillcolor="#e5f4e4"]
  operator [label="Operator Agent\\n(human approval)" fillcolor="#dbe9fb"]
  comms [label="Comms Agent\\n(SMS + report)" fillcolor="#dbe9fb"]
  text -> triage -> planner -> verifier
  verifier -> planner [label="failing step\\n(<=3 rounds)"]
  verifier -> operator [label="valid"]
  planner -> pyper [label="still invalid" style=dashed]
  pyper -> operator
  operator -> pyper [label="step failed:\\nreplan" style=dashed]
  operator -> comms
}""")
    st.markdown(
        "1. **Triage Agent** (LLM with tools) reads the messy report and investigates the "
        "network.\n"
        "2. **Planner Agent** (LLM) proposes a JSON plan; a **symbolic verifier** checks every "
        "step and sends back the exact failing condition, for up to 3 fix rounds, with "
        "**pyperplan** as the fallback.\n"
        "3. **Operator Agent** re-checks each step on the live network, pauses for **human "
        "approval** on risky steps, and **replans** when a step fails.\n"
        "4. **Comms Agent** writes the customer SMS and the incident report.\n"
        "5. Rule: an LLM wherever judgment is needed; deterministic code wherever a guarantee "
        "is needed (LLM-Modulo)."
    )
