from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms import (
    hill_climbing_1_lambda,
    hill_climbing_1p1,
    random_search,
    simulated_annealing,
)


@dataclass
class ExperimentConfig:
    runs: int = 50
    max_iter: int = 2000
    stagnation_iter_limit: int = 100
    epsilon: float = 1e-3
    seed: int = 42
    step_scale: float = 0.05
    lambda_neighbors: int = 10
    penalty_coeff: float = 1e6
    sa_t0: float = 1.0
    sa_alpha: float = 0.995
    sa_min_temperature: float = 1e-6
    sa_reheat_factor: float = 1.5


def run_monte_carlo(config: ExperimentConfig) -> pd.DataFrame:
    rows: list[dict] = []

    for run_id in range(config.runs):
        rng = np.random.default_rng(config.seed + run_id)

        results = [
            random_search(
                rng=rng,
                max_iter=config.max_iter,
                penalty_coeff=config.penalty_coeff,
            ),
            hill_climbing_1p1(
                rng=rng,
                max_iter=config.max_iter,
                stagnation_iter_limit=config.stagnation_iter_limit,
                epsilon=config.epsilon,
                step_scale=config.step_scale,
                penalty_coeff=config.penalty_coeff,
            ),
            hill_climbing_1_lambda(
                rng=rng,
                max_iter=config.max_iter,
                stagnation_iter_limit=config.stagnation_iter_limit,
                epsilon=config.epsilon,
                step_scale=config.step_scale,
                lambda_neighbors=config.lambda_neighbors,
                penalty_coeff=config.penalty_coeff,
            ),
            simulated_annealing(
                rng=rng,
                schedule="exponential",
                max_iter=config.max_iter,
                stagnation_iter_limit=config.stagnation_iter_limit,
                epsilon=config.epsilon,
                step_scale=config.step_scale,
                lambda_neighbors=config.lambda_neighbors,
                penalty_coeff=config.penalty_coeff,
                t0=config.sa_t0,
                alpha=config.sa_alpha,
                min_temperature=config.sa_min_temperature,
                reheat_factor=config.sa_reheat_factor,
            ),
            simulated_annealing(
                rng=rng,
                schedule="linear",
                max_iter=config.max_iter,
                stagnation_iter_limit=config.stagnation_iter_limit,
                epsilon=config.epsilon,
                step_scale=config.step_scale,
                lambda_neighbors=config.lambda_neighbors,
                penalty_coeff=config.penalty_coeff,
                t0=config.sa_t0,
                alpha=config.sa_alpha,
                min_temperature=config.sa_min_temperature,
                reheat_factor=config.sa_reheat_factor,
            ),
            simulated_annealing(
                rng=rng,
                schedule="logarithmic",
                max_iter=config.max_iter,
                stagnation_iter_limit=config.stagnation_iter_limit,
                epsilon=config.epsilon,
                step_scale=config.step_scale,
                lambda_neighbors=config.lambda_neighbors,
                penalty_coeff=config.penalty_coeff,
                t0=config.sa_t0,
                alpha=config.sa_alpha,
                min_temperature=config.sa_min_temperature,
                reheat_factor=config.sa_reheat_factor,
            ),
        ]

        for result in results:
            for iteration, (cost, best_cost) in enumerate(zip(result.history_cost, result.history_best_cost)):
                rows.append(
                    {
                        "algo": result.name,
                        "run": run_id,
                        "iteration": iteration,
                        "cost": cost,
                        "best_cost": best_cost,
                        "final_best_cost": result.best_eval.penalized_cost,
                        "final_objective": result.best_eval.objective,
                        "feasible": result.best_eval.feasible,
                    }
                )

    return pd.DataFrame(rows)


def save_experiment_outputs(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "runs_history.csv"
    pkl_path = output_dir / "runs_history.pkl"
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    return csv_path, pkl_path
