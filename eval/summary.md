# Evaluation summary

12 runs over 12 scenarios; model `openai/gpt-oss-20b`; goal reached is checked on the true network after execution.

**Headline:** incidents fully resolved 92% (system) vs 75% (LLM-only baseline); valid plans 92% vs 83%; unsafe or invalid actions attempted: 0 vs 1.

| Metric | Baseline | System |
| --- | --- | --- |
| Valid-plan rate | 83% | 92% |
| Goal reached after execution | 75% | 92% |
| ...without injected failures (10 runs) | 90% | 90% |
| ...with an injected restart failure (2 runs) | 0% | 100% |
| Unsafe or invalid actions attempted (total) | 1 | 0 |
| Runs with any unsafe or invalid action | 8% | 0% |
| Plan length / optimal | 1.15 | 1.14 |
| Latency per incident (s) | None | 18.03 |

| System-only metric | Value |
| --- | --- |
| Triage exact match | 92% |
| Triage fault accuracy / sector accuracy | 0.92 / 0.92 |
| Average LLM plan rounds | 1.09 |
| LLM plan fixed after verifier feedback | 8% |
| Planner fallback rate | 0% |
| Replan recovery rate | 100% of 2 runs |

## By scenario type

| Type | Runs | Valid plans (base / sys) | Goal reached (base / sys) | Unsafe attempts (base / sys) |
| --- | --- | --- | --- | --- |
| Power loss | 5 | 80% / 80% | 60% / 80% | 0 / 0 |
| Hardware fault | 5 | 80% / 100% | 80% / 100% | 1 / 0 |
| Link cut | 2 | 100% / 100% | 100% / 100% | 0 / 0 |

## How to read this

- The baseline has no verifier and no replanning, by design. In the 4 scenarios with an injected restart failure it cannot recover, so most of the goal-reached gap measures replanning, not plan quality.
- The verifier's direct effect shows in valid plans and unsafe actions: an LLM plan that breaks a precondition is fixed before it runs (system) or runs as-is (baseline).
- The system plans from what triage believes. When triage misses an affected sector, the first plan misses it too; only a replan from the live network recovers it.
- One model (Groq gpt-oss-120b) for both arms; temperature 0; cache off so each run is a fresh sample.

![Goal reached](../docs/goal_reached_rate.png)
![Valid plans](../docs/valid_plan_rate.png)
![Unsafe actions](../docs/unsafe_actions.png)
