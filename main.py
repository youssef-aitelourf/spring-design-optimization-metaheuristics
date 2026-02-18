from __future__ import annotations

import argparse
from pathlib import Path

from analysis import (
    convergence_profile,
    final_statistics_table,
    plot_comparative_convergence,
    plot_convergence_by_algo,
)
from experiments import ExperimentConfig, run_monte_carlo, save_experiment_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TP1 - 8INF852 - Métaheuristiques")
    parser.add_argument("--runs", type=int, default=50, help="Nombre d'exécutions Monte-Carlo par algorithme")
    parser.add_argument("--max-iter", type=int, default=2000, help="Nombre maximal d'itérations")
    parser.add_argument(
        "--stagnation-iter-limit",
        type=int,
        default=100,
        help="Arrêt secondaire après N itérations sans amélioration significative",
    )
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Seuil d'amélioration significative")
    parser.add_argument("--seed", type=int, default=42, help="Seed de base")
    parser.add_argument("--step-scale", type=float, default=0.05, help="Échelle du voisinage (normalisé)")
    parser.add_argument("--lambda-neighbors", type=int, default=10, help="Nombre de voisins pour (1,lambda) et SA")
    parser.add_argument("--penalty-coeff", type=float, default=1e6, help="Coefficient de pénalisation")
    parser.add_argument("--sa-t0", type=float, default=1.0, help="Température initiale du SA")
    parser.add_argument("--sa-alpha", type=float, default=0.995, help="Paramètre de refroidissement SA")
    parser.add_argument("--sa-min-temperature", type=float, default=1e-6, help="Température minimale SA")
    parser.add_argument("--sa-reheat-factor", type=float, default=1.5, help="Facteur de réchauffement SA")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Dossier des résultats")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    config = ExperimentConfig(
        runs=args.runs,
        max_iter=args.max_iter,
        stagnation_iter_limit=args.stagnation_iter_limit,
        epsilon=args.epsilon,
        seed=args.seed,
        step_scale=args.step_scale,
        lambda_neighbors=args.lambda_neighbors,
        penalty_coeff=args.penalty_coeff,
        sa_t0=args.sa_t0,
        sa_alpha=args.sa_alpha,
        sa_min_temperature=args.sa_min_temperature,
        sa_reheat_factor=args.sa_reheat_factor,
    )

    print("[1/4] Lancement des simulations Monte-Carlo...")
    df = run_monte_carlo(config)

    print("[2/4] Sauvegarde des historiques...")
    csv_path, pkl_path = save_experiment_outputs(df, output_dir)

    print("[3/4] Calcul des statistiques finales...")
    stats = final_statistics_table(df)
    stats_path = output_dir / "final_statistics.csv"
    stats.to_csv(stats_path, index=False)

    print("[4/4] Génération des profils de convergence...")
    profile_df = convergence_profile(df)
    profile_path = output_dir / "convergence_profile.csv"
    profile_df.to_csv(profile_path, index=False)
    plot_convergence_by_algo(profile_df, output_dir)
    comparative_path = plot_comparative_convergence(profile_df, output_dir)

    print("Terminé.")
    print(f"- Historique CSV: {csv_path}")
    print(f"- Historique PKL: {pkl_path}")
    print(f"- Statistiques: {stats_path}")
    print(f"- Profil convergence: {profile_path}")
    print(f"- Comparatif: {comparative_path}")


if __name__ == "__main__":
    main()
