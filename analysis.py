from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def final_statistics_table(df: pd.DataFrame) -> pd.DataFrame:
    final_rows = (
        df.sort_values(["algo", "run", "iteration"]).groupby(["algo", "run"], as_index=False).tail(1)
    )

    stats = (
        final_rows.groupby("algo")["final_best_cost"]
        .agg(best="min", median="median", std="std", q1=lambda x: np.quantile(x, 0.25), q3=lambda x: np.quantile(x, 0.75))
        .reset_index()
    )
    stats["iqr"] = stats["q3"] - stats["q1"]
    return stats


def convergence_profile(df: pd.DataFrame) -> pd.DataFrame:
    profile = (
        df.groupby(["algo", "iteration"], as_index=False)
        .agg(
            median_cost=("best_cost", "median"),
            q1_cost=("best_cost", lambda x: np.quantile(x, 0.25)),
            q3_cost=("best_cost", lambda x: np.quantile(x, 0.75)),
            min_cost=("best_cost", "min"),
            max_cost=("best_cost", "max"),
        )
        .sort_values(["algo", "iteration"])
    )
    return profile


def plot_convergence_by_algo(profile_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for algo, data in profile_df.groupby("algo"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(data["iteration"], data["median_cost"], label=f"{algo} - médiane")
        ax.fill_between(data["iteration"], data["q1_cost"], data["q3_cost"], alpha=0.2, label="IQR")
        ax.set_yscale("log")
        ax.set_xlabel("Itération")
        ax.set_ylabel("Coût (pénalisé)")
        ax.set_title(f"Profil de convergence: {algo}")
        ax.legend(loc="best")
        fig.tight_layout()

        file_path = output_dir / f"convergence_{algo}.png"
        fig.savefig(file_path, dpi=200)
        plt.close(fig)
        paths.append(file_path)

    return paths


def plot_comparative_convergence(profile_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, data in profile_df.groupby("algo"):
        ax.plot(data["iteration"], data["median_cost"], label=algo)

    ax.set_yscale("log")
    ax.set_xlabel("Itération")
    ax.set_ylabel("Médiane du coût pénalisé")
    ax.set_title("Comparaison interalgorithmes des profils de convergence")
    ax.legend(loc="best")
    fig.tight_layout()

    file_path = output_dir / "convergence_comparative.png"
    fig.savefig(file_path, dpi=200)
    plt.close(fig)
    return file_path
