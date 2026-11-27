# Resume entry and LinkedIn post (drafts)

Fill the bracketed numbers from the README's Results section (`python -m eval.summarize` keeps it current) once all 60 runs per model are done. Use numbers exactly as reported; don't round up.

## Resume (first project entry)

**Telecom Incident Agent** · Python, LangGraph, unified-planning, Streamlit · [live demo](https://telecom-incident-agent.streamlit.app/) · [GitHub](https://github.com/DathaNallagonda/telecom-incident-agent)

- Built a multi-agent telecom incident-resolution system (LangGraph): a tool-calling Triage Agent, an LLM planner, an Operator with human-in-the-loop approval and automatic replanning, and a Comms Agent for customer SMS and incident reports.
- Added symbolic plan verification (LLM-Modulo): every LLM plan is checked step by step with a formal planning domain; failures go back as precise feedback, with a classical planner (pyperplan) as fallback.
- Benchmarked 20 scenarios x 3 runs against an LLM-only baseline: [X]% vs [Y]% of incidents fully resolved, [A] vs [B] unsafe or invalid actions attempted; [summarise the smaller-model result in one clause].

One-line version: *Multi-agent telecom incident system (LangGraph) combining LLM agents with symbolic plan verification; [X]% vs [Y]% incidents resolved vs an LLM-only baseline over 20 scenarios; human-in-the-loop approvals and automatic replanning; live on Streamlit Cloud.*

## LinkedIn post

> LLM agents can read a messy outage report and suggest a fix. But in network operations, a plausible plan isn't a safe plan: restart a tower before its power is back and you make things worse.
>
> So I built a Telecom Incident Agent where LLM agents do the judgment and a symbolic verifier provides the guarantees:
>
> 🔎 A Triage Agent investigates the network with tools
> 🧠 A Planner Agent proposes a repair plan
> ✅ A verifier checks every step's preconditions and sends back the exact failure, so the LLM can fix it (LLM-Modulo)
> 🙋 Risky actions wait for human approval, and the system replans when a step fails
> 📱 A Comms Agent writes the customer SMS and the incident report
>
> Over 20 scenarios against an LLM-only baseline: [headline numbers].
>
> Built with LangGraph, unified-planning and pyperplan, all on free tiers.
> Live demo: https://telecom-incident-agent.streamlit.app/
> Code: https://github.com/DathaNallagonda/telecom-incident-agent
>
> #AgenticAI #LLM #Telecom #LangGraph #AIPlanning

Attach a 20–30 second clip of the approval and replan moment (crop from the demo video).
