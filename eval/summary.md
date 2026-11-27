# Evaluation summary

27 runs over 20 scenarios; model `openai/gpt-oss-120b`; goal reached is checked on the true network after execution.

**Headline:** incidents fully resolved 100% (system) vs 81% (LLM-only baseline); valid plans 100% vs 96%; unsafe or invalid actions attempted: 0 vs 1.

| Metric | Baseline | System |
| --- | --- | --- |
| Valid-plan rate | 96% | 100% |
| Goal reached after execution | 81% | 100% |
| ...without injected failures (22 runs) | 100% | 100% |
| ...with an injected restart failure (5 runs) | 0% | 100% |
| Unsafe or invalid actions attempted (total) | 1 | 0 |
| Runs with any unsafe or invalid action | 4% | 0% |
| Plan length / optimal | 1.26 | 1.28 |
| Latency per incident (s) | 25.46 | 23.74 |

| System-only metric | Value |
| --- | --- |
| Triage exact match | 96% |
| Triage fault accuracy / sector accuracy | 1.0 / 0.98 |
| Average LLM plan rounds | 1.04 |
| LLM plan fixed after verifier feedback | 4% |
| Planner fallback rate | 0% |
| Replan recovery rate | 100% of 5 runs |

## By scenario type

| Type | Runs | Valid plans (base / sys) | Goal reached (base / sys) | Unsafe attempts (base / sys) |
| --- | --- | --- | --- | --- |
| Power loss | 10 | 100% / 100% | 80% / 100% | 0 / 0 |
| Hardware fault | 7 | 100% / 100% | 86% / 100% | 0 / 0 |
| Link cut | 5 | 100% / 100% | 100% / 100% | 0 / 0 |
| Multi-fault | 5 | 80% / 100% | 60% / 100% | 1 / 0 |

## How to read this

- The baseline has no verifier and no replanning, by design. In the 4 scenarios with an injected restart failure it cannot recover, so most of the goal-reached gap measures replanning, not plan quality.
- The verifier's direct effect shows in valid plans and unsafe actions: an LLM plan that breaks a precondition is fixed before it runs (system) or runs as-is (baseline).
- The system plans from what triage believes. When triage misses an affected sector, the first plan misses it too; only a replan from the live network recovers it.
- One model (Groq gpt-oss-120b) for both arms; temperature 0; cache off so each run is a fresh sample.

![Goal reached](../docs/goal_reached_rate.png)
![Valid plans](../docs/valid_plan_rate.png)
![Unsafe actions](../docs/unsafe_actions.png)
