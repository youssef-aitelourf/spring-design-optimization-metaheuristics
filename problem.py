from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BOUNDS = np.array(
    [
        [0.05, 2.00],
        [0.25, 1.30],
        [2.00, 15.0],
    ],
    dtype=float,
)


def objective(x: np.ndarray) -> float:
    x1, x2, x3 = x
    return float((x1**2) * x2 * (2.0 + x3))


def constraints(x: np.ndarray) -> np.ndarray:
    x1, x2, x3 = x

    g1 = 1.0 - (x2**3 * x3) / (71785.0 * x1**4)
    g2 = (4.0 * x2**2 - x1 * x2) / (12566.0 * (x2 * x1**3 - x1**4)) + 1.0 / (5108.0 * x1**2) - 1.0
    g3 = 1.0 - (140.45 * x1) / (x2**2 * x3)
    g4 = (x1 + x2) / 1.5 - 1.0

    return np.array([g1, g2, g3, g4], dtype=float)


def bounds_violation(x: np.ndarray) -> float:
    lower = BOUNDS[:, 0]
    upper = BOUNDS[:, 1]
    below = np.maximum(0.0, lower - x)
    above = np.maximum(0.0, x - upper)
    return float(np.sum(below + above))


def constraints_violation(x: np.ndarray) -> float:
    g = constraints(x)
    return float(np.sum(np.maximum(0.0, g)))


@dataclass
class Evaluation:
    x: np.ndarray
    objective: float
    penalty: float
    penalized_cost: float
    feasible: bool


def evaluate_with_penalty(x: np.ndarray, penalty_coeff: float = 1e6) -> Evaluation:
    f = objective(x)
    v_bounds = bounds_violation(x)
    v_constraints = constraints_violation(x)
    total_violation = v_bounds + v_constraints
    penalty = penalty_coeff * total_violation
    feasible = bool(total_violation <= 0.0)
    return Evaluation(
        x=np.array(x, dtype=float),
        objective=f,
        penalty=penalty,
        penalized_cost=f + penalty,
        feasible=feasible,
    )


def random_solution(rng: np.random.Generator) -> np.ndarray:
    lower = BOUNDS[:, 0]
    upper = BOUNDS[:, 1]
    return rng.uniform(lower, upper)
