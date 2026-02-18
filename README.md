# Spring Design Optimization with Metaheuristics

Monte Carlo benchmark of single-solution metaheuristics on a constrained, non-convex engineering problem (spring weight minimization).

## What this project does

- Implements the constrained spring design objective and constraints.
- Compares 6 optimization variants:
	- Random Search (baseline)
	- Hill Climbing `(1+1)`
	- Generalized Hill Climbing `(1, lambda)`
	- Simulated Annealing (exponential, linear, logarithmic cooling)
- Runs statistically meaningful Monte Carlo campaigns.
- Produces convergence profiles and final performance statistics.

## Problem summary

The optimization target is a classic constrained spring design benchmark with 3 variables and 4 inequality constraints. The objective is to minimize spring weight while satisfying feasibility constraints and variable bounds.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --runs 50 --max-iter 2000 --output-dir outputs_main
```

## Key CLI parameters

- `--runs`: Monte Carlo runs per algorithm
- `--max-iter`: maximum iterations
- `--stagnation-iter-limit`: secondary stopping criterion (no significant improvement)
- `--epsilon`: minimum significant improvement
- `--seed`: base random seed
- `--step-scale`: neighborhood scale for Gaussian perturbations
- `--lambda-neighbors`: number of neighbors for `(1,lambda)` and SA
- `--penalty-coeff`: feasibility penalty coefficient
- `--sa-t0`, `--sa-alpha`, `--sa-min-temperature`, `--sa-reheat-factor`: SA controls

## Outputs

- `outputs_main/runs_history.csv`
- `outputs_main/runs_history.pkl`
- `outputs_main/final_statistics.csv`
- `outputs_main/convergence_profile.csv`
- `outputs_main/convergence_*.png`
- `outputs_main/convergence_comparative.png`

## Main result snapshot

Across 50 runs and 2000 iterations, Simulated Annealing variants provide the best median performance and robustness (lowest dispersion), followed by Generalized Hill Climbing.

