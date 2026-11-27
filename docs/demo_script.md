# Demo video script (2 minutes)

Record at 1080p with OBS or the Windows Game Bar (Win + Alt + R). Upload to YouTube as **unlisted** and put the link at the top of the README.

**Before recording:** open the live app 10 minutes early so it is awake, and check a run works (free-tier quotas). If the live run is slow or rate limited, record locally with `streamlit run app.py`. Close other tabs; zoom the browser to 110%.

## 0:00–0:15 · The problem (15 s)

> "When a phone tower breaks, an AI agent can read the complaints and suggest a fix. But a plausible plan isn't a safe plan: restart a tower before its power is back and you make the outage worse. LLMs don't check preconditions, and they don't recover when a step fails."

Show: the app's title and the network panel with the red tower.

## 0:15–1:15 · Live demo (60 s)

1. Sidebar: pick **P1** ("customers near the mall say no signal..."). Point at **Inject a restart failure** (on). Click **Run agents**.
   > "This is a real customer complaint, not engineer language."
2. **Triage card:** open **Tool calls**.
   > "The Triage Agent investigated with tools: alarms, diagnostics, sectors. T3 lost power; the mall sector is down, priority P1."
3. **Plan:** point at the green **LLM plan, verified** badge, then open **Verify-and-fix log**.
   > "The LLM proposes a plan, and a symbolic verifier checks every step's preconditions before anything runs. If a step is wrong, it sends back the exact failing condition and the LLM fixes it."
4. **Approval:** the yellow box appears for `restart_node(T3)`. Click **Approve**.
   > "Risky actions wait for a human."
5. **Replan:** the restart fails; point at the blue **Replanned** line.
   > "The restart failed: hardware fault. The Operator replans from the live network state: repair the hardware, then restart." Click **Approve** again.
6. **SMS and report:** scroll to section 5.
   > "The Comms Agent writes the customer SMS, no internal ids, and the incident report."

## 1:15–1:45 · Benchmark (30 s)

Open the **Benchmark** tab.
> "I benchmarked 20 scenarios, 3 runs each, against an LLM-only baseline: the same model's first plan, no verifier, no replanning. [Read the headline numbers from the README.] And with a smaller model, the verifier catches more invalid plans, which is the point: verification lets you run a cheaper model safely."

## 1:45–2:00 · Close (15 s)

Open the **How it works** tab.
> "Agents where judgment is needed, deterministic code where guarantees are needed. Built with LangGraph, unified-planning and pyperplan, on free tiers. The code and results are on GitHub."
