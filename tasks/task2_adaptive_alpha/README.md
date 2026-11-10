# Task 2: Instance-Adaptive Stability–Suitability Balancing

## Overview
This directory contains the implementation, test harness, and analysis scripts for **Task 2: Instance-Adaptive Stability–Suitability Balancing** (Person 2).

Standard SS-TPT blends suitability and prediction stability using a fixed global parameter $\alpha = 0.40$. Task 2 replaces this with a sample-dependent coefficient $\alpha(x) \in [0, 1]$ calculated from instance score distributions:
$$\alpha(x) = \frac{r_{suit}}{r_{suit} + r_{stab} + \epsilon}$$

## Implemented Reliability Modes (`ALPHA_MODES`)
1. **`variance` (Best Performing):** Computes score variance across views $\text{Var}(S)$. Higher variance indicates a more decisive and reliable quality signal.
2. **`margin`:** Difference between top-1 and top-2 scores ($s_{(1)} - s_{(2)}$).
3. **`entropy`:** Normalized entropy over softmaxed quality scores ($1 - H(p)/\log N$).
4. **`topk`:** Probability mass concentrated within top-$k$ views.
5. **`separation`:** Mean difference between the upper and lower halves of view scores.
6. **`agreement`:** Spearman rank correlation $\rho(S^{suit}, S^{stab})$ interpolating between fixed prior $\alpha_0$ and margin-based $\alpha$.
7. **`view0`:** Compares stability and suitability confidence on the unaugmented base view.
8. **`fixed`:** Static prior $\alpha(x) = \alpha_0 = 0.40$ (reproduces original SS-TPT baseline).

## Key Files
- `instance_alpha.py`: Core module computing $\alpha(x)$ and reliabilities $r_{suit}, r_{stab}$.
- `test_instance_alpha.py`: Comprehensive unit test suite (17 tests covering all modes, degenerate cases, boundaries, and fallbacks).
- `alpha_pilot.py`: Paired pilot test comparing instance-adaptive $\alpha(x)$ vs fixed $\alpha=0.40$ on identical views.
- `alpha_vote.py`: Multi-alpha voting ensemble testing.
- `../../output_results/task2/`: Full evaluation logs and `alpha_trace.jsonl` files for all 7 modes across Clean and Adversarial PGD regimes.
