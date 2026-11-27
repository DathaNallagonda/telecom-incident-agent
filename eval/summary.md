# Evaluation summary

33 runs over 20 scenarios; model `openai/gpt-oss-20b`; goal reached is checked on the true network after execution.

**Headline:** incidents fully resolved 94% (system) vs 67% (LLM-only baseline); valid plans 94% vs 76%; unsafe or invalid actions attempted: 0 vs 5.

| Metric | Baseline | System |
| --- | --- | --- |
| Valid-plan rate | 76% | 94% |
| Goal reached after execution | 67% | 94% |
| ...without injected failures (27 runs) | 81% | 93% |
| ...with an injected restart failure (6 runs) | 0% | 100% |
| Unsafe or invalid actions attempted (total) | 5 | 0 |
| Runs with any unsafe or invalid action | 9% | 0% |
| Plan length / optimal | 1.3 | 1.28 |
| Latency per incident (s) | — (shares the first call) | 25.24 |

| System-only metric | Value |
| --- | --- |
| Triage exact match | 82% |
| Triage fault accuracy / sector accuracy | 0.94 / 0.86 |
| Average LLM plan rounds | 1.32 |
| LLM plan fixed after verifier feedback | 15% |
| Planner fallback rate | 3% |
| Replan recovery rate | 100% of 6 runs |

## By scenario type

| Type | Runs | Valid plans (base / sys) | Goal reached (base / sys) | Unsafe attempts (base / sys) |
| --- | --- | --- | --- | --- |
| Power loss | 10 | 80% / 80% | 60% / 80% | 0 / 0 |
| Hardware fault | 10 | 90% / 100% | 80% / 100% | 1 / 0 |
| Link cut | 8 | 75% / 100% | 75% / 100% | 4 / 0 |
| Multi-fault | 5 | 40% / 100% | 40% / 100% | 0 / 0 |

## How to read this

- The baseline has no verifier and no replanning, by design. In the 4 scenarios with an injected restart failure it cannot recover, so most of the goal-reached gap measures replanning, not plan quality.
- The verifier's direct effect shows in valid plans and unsafe actions: an LLM plan that breaks a precondition is fixed before it runs (system) or runs as-is (baseline).
- The system plans from what triage believes. When triage misses an affected sector, the first plan misses it too; only a replan from the live network recovers it.
- One model (Groq gpt-oss-120b) for both arms; temperature 0; cache off so each run is a fresh sample.

![Goal reached](../docs/goal_reached_rate.png)
![Valid plans](../docs/valid_plan_rate.png)
![Unsafe actions](../docs/unsafe_actions.png)
