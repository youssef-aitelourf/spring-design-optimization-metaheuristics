from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from problem import BOUNDS, Evaluation, evaluate_with_penalty, random_solution


@dataclass
class AlgorithmResult:
    name: str
    best_x: np.ndarray
    best_eval: Evaluation
    history_cost: list[float]
    history_best_cost: list[float]


def _make_neighbor(
    x: np.ndarray,
    rng: np.random.Generator,
    step_scale: float,
) -> np.ndarray:
    span = BOUNDS[:, 1] - BOUNDS[:, 0]
    step = rng.normal(loc=0.0, scale=step_scale * span, size=x.shape[0])
    return x + step


def random_search(
    rng: np.random.Generator,
    max_iter: int,
    penalty_coeff: float,
) -> AlgorithmResult:
    current = evaluate_with_penalty(random_solution(rng), penalty_coeff=penalty_coeff)
    best = current
    history_cost: list[float] = []
    history_best_cost: list[float] = []

    for _ in range(max_iter):
        candidate = evaluate_with_penalty(random_solution(rng), penalty_coeff=penalty_coeff)
        current = candidate
        if candidate.penalized_cost < best.penalized_cost:
            best = candidate

        history_cost.append(current.penalized_cost)
        history_best_cost.append(best.penalized_cost)

    return AlgorithmResult(
        name="random_search",
        best_x=best.x,
        best_eval=best,
        history_cost=history_cost,
        history_best_cost=history_best_cost,
    )


def hill_climbing_1p1(
    rng: np.random.Generator,
    max_iter: int,
    stagnation_iter_limit: int,
    epsilon: float,
    step_scale: float,
    penalty_coeff: float,
) -> AlgorithmResult:
    current = evaluate_with_penalty(random_solution(rng), penalty_coeff=penalty_coeff)
    best = current
    history_cost: list[float] = []
    history_best_cost: list[float] = []
    no_significant_improve = 0

    for _ in range(max_iter):
        candidate_x = _make_neighbor(current.x, rng, step_scale=step_scale)
        candidate = evaluate_with_penalty(candidate_x, penalty_coeff=penalty_coeff)

        if candidate.penalized_cost < current.penalized_cost:
            current = candidate

        improvement = best.penalized_cost - current.penalized_cost
        if current.penalized_cost < best.penalized_cost:
            best = current

        if improvement >= epsilon:
            no_significant_improve = 0
        else:
            no_significant_improve += 1

        history_cost.append(current.penalized_cost)
        history_best_cost.append(best.penalized_cost)

        if no_significant_improve >= stagnation_iter_limit:
            break

    return AlgorithmResult(
        name="hill_climbing_1p1",
        best_x=best.x,
        best_eval=best,
        history_cost=history_cost,
        history_best_cost=history_best_cost,
    )


def hill_climbing_1_lambda(
    rng: np.random.Generator,
    max_iter: int,
    stagnation_iter_limit: int,
    epsilon: float,
    step_scale: float,
    lambda_neighbors: int,
    penalty_coeff: float,
) -> AlgorithmResult:
    current = evaluate_with_penalty(random_solution(rng), penalty_coeff=penalty_coeff)
    best = current
    history_cost: list[float] = []
    history_best_cost: list[float] = []
    no_significant_improve = 0

    for _ in range(max_iter):
        candidates = [
            evaluate_with_penalty(_make_neighbor(current.x, rng, step_scale=step_scale), penalty_coeff=penalty_coeff)
            for _ in range(lambda_neighbors)
        ]
        candidate_best = min(candidates, key=lambda item: item.penalized_cost)

        if candidate_best.penalized_cost < current.penalized_cost:
            current = candidate_best

        improvement = best.penalized_cost - current.penalized_cost
        if current.penalized_cost < best.penalized_cost:
            best = current

        if improvement >= epsilon:
            no_significant_improve = 0
        else:
            no_significant_improve += 1

        history_cost.append(current.penalized_cost)
        history_best_cost.append(best.penalized_cost)

        if no_significant_improve >= stagnation_iter_limit:
            break

    return AlgorithmResult(
        name="hill_climbing_1_lambda",
        best_x=best.x,
        best_eval=best,
        history_cost=history_cost,
        history_best_cost=history_best_cost,
    )


def _cooling_temperature(
    schedule: Literal["exponential", "linear", "logarithmic"],
    t0: float,
    alpha: float,
    iteration: int,
) -> float:
    if schedule == "exponential":
        return t0 * (alpha**iteration)
    if schedule == "linear":
        return max(1e-12, t0 - alpha * iteration)
    if schedule == "logarithmic":
        return t0 / (1.0 + alpha * np.log(1.0 + iteration))
    raise ValueError(f"Unknown schedule: {schedule}")


def simulated_annealing(
    rng: np.random.Generator,
    schedule: Literal["exponential", "linear", "logarithmic"],
    max_iter: int,
    stagnation_iter_limit: int,
    epsilon: float,
    step_scale: float,
    lambda_neighbors: int,
    penalty_coeff: float,
    t0: float,
    alpha: float,
    min_temperature: float,
    reheat_factor: float,
) -> AlgorithmResult:
    current = evaluate_with_penalty(random_solution(rng), penalty_coeff=penalty_coeff)
    best = current
    last_best = best
    history_cost: list[float] = []
    history_best_cost: list[float] = []
    no_significant_improve = 0
    temperature = t0

    for i in range(max_iter):
        temperature = max(min_temperature, _cooling_temperature(schedule, t0, alpha, i))

        candidates = [
            evaluate_with_penalty(_make_neighbor(current.x, rng, step_scale=step_scale), penalty_coeff=penalty_coeff)
            for _ in range(lambda_neighbors)
        ]
        candidate = min(candidates, key=lambda item: item.penalized_cost)

        delta = candidate.penalized_cost - current.penalized_cost
        if delta <= 0:
            current = candidate
        else:
            accept_prob = float(np.exp(-delta / max(temperature, 1e-12)))
            if rng.random() < accept_prob:
                current = candidate

        improvement = best.penalized_cost - current.penalized_cost
        if current.penalized_cost < best.penalized_cost:
            best = current
            last_best = best

        if improvement >= epsilon:
            no_significant_improve = 0
        else:
            no_significant_improve += 1

        if no_significant_improve >= stagnation_iter_limit:
            current = last_best
            no_significant_improve = 0
            t0 = max(t0, temperature * reheat_factor)

        history_cost.append(current.penalized_cost)
        history_best_cost.append(best.penalized_cost)

    return AlgorithmResult(
        name=f"simulated_annealing_{schedule}",
        best_x=best.x,
        best_eval=best,
        history_cost=history_cost,
        history_best_cost=history_best_cost,
    )
