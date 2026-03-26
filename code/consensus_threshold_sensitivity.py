from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from VSCOUT import VSCOUT


@dataclass(frozen=True)
class ScenarioConfig:
    scenario: str
    shift_type: str
    dist: str
    n_inlier: int
    n_features: int
    gamma: float
    contamination: float


DEFAULT_SCENARIOS = [
    # Cleaner/light benchmark case.
    ScenarioConfig(
        scenario="Transient Normal",
        shift_type="Transient",
        dist="normal",
        n_inlier=500,
        n_features=150,
        gamma=2.0,
        contamination=0.10,
    ),
    # Sustained clean shift to preserve the notebook's transient/sustained framing.
    ScenarioConfig(
        scenario="Sustained Normal",
        shift_type="Sustained",
        dist="normal",
        n_inlier=500,
        n_features=150,
        gamma=2.0,
        contamination=0.10,
    ),
    # Heavier-tailed/harder case.
    ScenarioConfig(
        scenario="Sustained t5",
        shift_type="Sustained",
        dist="t5",
        n_inlier=500,
        n_features=150,
        gamma=2.0,
        contamination=0.10,
    ),
    # More heterogeneous case.
    ScenarioConfig(
        scenario="Sustained Mixed",
        shift_type="Sustained",
        dist="mixed",
        n_inlier=500,
        n_features=250,
        gamma=2.0,
        contamination=0.10,
    ),
]

THRESHOLD_RULES = {
    1: "1-of-4",
    2: "2-of-4",
    3: "3-of-4",
}


def simulate_transient_outliers(
    n_inlier: int,
    n_outlier: int,
    n_features: int,
    gamma: float,
    dist: str = "normal",
    random_state: int | None = None,
):
    rng = np.random.RandomState(random_state)

    if dist == "normal":
        X_in = rng.normal(0, 1, (n_inlier, n_features))
        X_out = rng.normal(gamma, 1, (n_outlier, n_features))
    elif dist == "t5":
        X_in = rng.standard_t(df=5, size=(n_inlier, n_features))
        X_out = rng.standard_t(df=5, size=(n_outlier, n_features)) + gamma
    elif dist == "lognormal":
        X_in = rng.lognormal(mean=0, sigma=1, size=(n_inlier, n_features))
        X_out = rng.lognormal(mean=gamma, sigma=1, size=(n_outlier, n_features))
    elif dist == "mixed":
        n_half_in = n_inlier // 2
        n_half_out = n_outlier // 2
        X_norm_in = rng.normal(0, 1, (n_half_in, n_features))
        X_t_in = rng.standard_t(df=5, size=(n_inlier - n_half_in, n_features))
        X_in = np.vstack([X_norm_in, X_t_in])
        rng.shuffle(X_in)

        X_norm_out = rng.normal(gamma, 1, (n_half_out, n_features))
        X_t_out = rng.standard_t(df=5, size=(n_outlier - n_half_out, n_features)) + gamma
        X_out = np.vstack([X_norm_out, X_t_out])
        rng.shuffle(X_out)
    elif dist == "multimodal":
        n_half = n_inlier // 2
        X_in1 = rng.normal(loc=-5, scale=1.0, size=(n_half, n_features))
        X_in2 = rng.normal(loc=+5, scale=1.0, size=(n_inlier - n_half, n_features))
        X_in = np.vstack([X_in1, X_in2])
        rng.shuffle(X_in)
        X_out = rng.normal(loc=gamma, scale=1.0, size=(n_outlier, n_features))
    else:
        raise ValueError(f"Unknown dist: {dist}")

    y_in = np.zeros(n_inlier, dtype=int)
    y_out = np.ones(n_outlier, dtype=int)
    X_all = np.vstack([X_in, X_out])
    y_all = np.concatenate([y_in, y_out])
    idx = np.arange(len(X_all))
    rng.shuffle(idx)
    return X_all[idx], y_all[idx]


def simulate_sustained_outliers(
    n_inlier: int,
    n_outlier: int,
    n_features: int,
    gamma: float,
    dist: str = "normal",
    random_state: int | None = None,
):
    rng = np.random.RandomState(random_state)

    if dist == "normal":
        X_in = rng.normal(0, 1, (n_inlier, n_features))
        X_out = rng.normal(gamma, 1, (n_outlier, n_features))
    elif dist == "t5":
        X_in = rng.standard_t(df=5, size=(n_inlier, n_features))
        X_out = rng.standard_t(df=5, size=(n_outlier, n_features)) + gamma
    elif dist == "lognormal":
        X_in = rng.lognormal(mean=0, sigma=1, size=(n_inlier, n_features))
        X_out = rng.lognormal(mean=gamma, sigma=1, size=(n_outlier, n_features))
    elif dist == "mixed":
        n_half_in = n_inlier // 2
        n_half_out = n_outlier // 2
        X_norm_in = rng.normal(0, 1, (n_half_in, n_features))
        X_t_in = rng.standard_t(df=5, size=(n_inlier - n_half_in, n_features))
        X_in = np.vstack([X_norm_in, X_t_in])

        X_norm_out = rng.normal(gamma, 1, (n_half_out, n_features))
        X_t_out = rng.standard_t(df=5, size=(n_outlier - n_half_out, n_features)) + gamma
        X_out = np.vstack([X_norm_out, X_t_out])
    elif dist == "multimodal":
        n_half = n_inlier // 2
        X_in1 = rng.normal(loc=-5, scale=1.0, size=(n_half, n_features))
        X_in2 = rng.normal(loc=+5, scale=1.0, size=(n_inlier - n_half, n_features))
        X_in = np.vstack([X_in1, X_in2])
        X_out = rng.normal(loc=gamma, scale=1.0, size=(n_outlier, n_features))
    else:
        raise ValueError(f"Unknown dist: {dist}")

    y_in = np.zeros(n_inlier, dtype=int)
    y_out = np.ones(n_outlier, dtype=int)
    X_all = np.vstack([X_in, X_out])
    y_all = np.concatenate([y_in, y_out])
    return X_all, y_all


def _compute_metrics(y_true: np.ndarray, pred_outlier_mask: np.ndarray) -> dict[str, float]:
    y = y_true.astype(int)
    p = pred_outlier_mask.astype(bool)

    tp = np.sum((y == 1) & p)
    fp = np.sum((y == 0) & p)
    fn = np.sum((y == 1) & (~p))
    tn = np.sum((y == 0) & (~p))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
    }


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _simulate_scenario(config: ScenarioConfig, seed: int):
    n_outlier = int(config.n_inlier * config.contamination / (1 - config.contamination))
    generator = (
        simulate_transient_outliers
        if config.shift_type.lower() == "transient"
        else simulate_sustained_outliers
    )
    return generator(
        n_inlier=config.n_inlier,
        n_outlier=n_outlier,
        n_features=config.n_features,
        gamma=config.gamma,
        dist=config.dist,
        random_state=seed,
    )


def _run_single_replication(
    config: ScenarioConfig,
    replication: int,
    alpha: float,
    epochs: int,
    batch_size: int,
):
    tf.keras.backend.clear_session()
    _set_all_seeds(replication)

    X, y = _simulate_scenario(config, seed=replication)
    X_scaled = StandardScaler().fit_transform(X)

    model = VSCOUT(alpha=alpha)
    model.fit(X_scaled, verbose=0, epochs=epochs, batch_size=batch_size)
    _, cp_mask, ensemble_mask, t2_mask, recon_mask, *_ = model.is_outlier(
        X_scaled, batch_size=batch_size
    )

    votes = np.column_stack(
        [
            cp_mask.astype(int),
            ensemble_mask.astype(int),
            t2_mask.astype(int),
            recon_mask.astype(int),
        ]
    )
    vote_sum = votes.sum(axis=1)

    rows = []
    scenario_meta = asdict(config)
    scenario_meta["Scenario"] = scenario_meta.pop("scenario")
    scenario_meta["ShiftType"] = scenario_meta.pop("shift_type")
    scenario_meta["Contam (%)"] = 100 * scenario_meta.pop("contamination")
    scenario_meta["Distribution"] = scenario_meta.pop("dist")
    scenario_meta["p"] = scenario_meta.pop("n_features")
    scenario_meta["N(clean)"] = scenario_meta.pop("n_inlier")

    for k, rule_label in THRESHOLD_RULES.items():
        y_hat = vote_sum >= k
        metrics = _compute_metrics(y, y_hat)
        rows.append(
            {
                **scenario_meta,
                "ThresholdK": k,
                "ThresholdRule": rule_label,
                "Replication": replication,
                **metrics,
            }
        )

    return rows


def build_summary_tables(results_df: pd.DataFrame):
    summary_df = (
        results_df.groupby(
            ["Scenario", "ShiftType", "Distribution", "p", "gamma", "Contam (%)", "ThresholdRule"]
        )[["Precision", "Recall", "F1", "FPR"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df.columns = [
        "_".join(str(part) for part in col if part).rstrip("_") for col in summary_df.columns
    ]
    summary_df = summary_df.rename(
        columns={
            "Scenario_": "Scenario",
            "ShiftType_": "ShiftType",
            "Distribution_": "Distribution",
            "p_": "p",
            "gamma_": "gamma",
            "Contam (%)_": "Contam (%)",
            "ThresholdRule_": "ThresholdRule",
            "Precision_std": "Precision_sd",
            "Recall_std": "Recall_sd",
            "F1_std": "F1_sd",
            "FPR_std": "FPR_sd",
        }
    )

    overall_df = (
        summary_df.groupby("ThresholdRule", as_index=False)[
            ["Precision_mean", "Recall_mean", "F1_mean", "FPR_mean"]
        ]
        .mean()
        .rename(
            columns={
                "Precision_mean": "Precision",
                "Recall_mean": "Recall",
                "F1_mean": "F1",
                "FPR_mean": "FPR",
            }
        )
    )
    overall_df["ThresholdRule"] = pd.Categorical(
        overall_df["ThresholdRule"],
        categories=[THRESHOLD_RULES[k] for k in sorted(THRESHOLD_RULES)],
        ordered=True,
    )
    overall_df = overall_df.sort_values("ThresholdRule").reset_index(drop=True)

    return summary_df, overall_df


def write_output_tables(
    results_df: pd.DataFrame,
    output_dir: Path | str,
    prefix: str = "consensus_threshold_sensitivity",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df, overall_df = build_summary_tables(results_df)
    results_df.to_csv(output_dir / f"{prefix}_results.csv", index=False)
    summary_df.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    overall_df.to_csv(output_dir / f"{prefix}_overall.csv", index=False)
    return summary_df, overall_df


def run_consensus_threshold_sensitivity(
    scenarios: list[ScenarioConfig] | None = None,
    n_repeats: int = 25,
    alpha: float = 0.05,
    epochs: int = 100,
    batch_size: int = 32,
    checkpoint_dir: Path | str | None = None,
    checkpoint_every: int = 1,
):
    scenarios = DEFAULT_SCENARIOS if scenarios is None else scenarios
    rows = []
    completed_reps = 0

    for config in scenarios:
        print(f"=== Scenario: {config.scenario} ({config.shift_type}, {config.dist}) ===")
        for replication in range(n_repeats):
            print(f"Replication {replication + 1}/{n_repeats}")
            rows.extend(
                _run_single_replication(
                    config=config,
                    replication=replication,
                    alpha=alpha,
                    epochs=epochs,
                    batch_size=batch_size,
                )
            )
            completed_reps += 1
            if checkpoint_dir is not None and completed_reps % checkpoint_every == 0:
                write_output_tables(
                    results_df=pd.DataFrame(rows),
                    output_dir=checkpoint_dir,
                    prefix="consensus_threshold_sensitivity_checkpoint",
                )

    results_df = pd.DataFrame(rows)
    if checkpoint_dir is not None:
        write_output_tables(
            results_df=results_df,
            output_dir=checkpoint_dir,
            prefix="consensus_threshold_sensitivity_checkpoint",
        )
    summary_df, overall_df = build_summary_tables(results_df)
    return results_df, summary_df, overall_df


def build_latex_table(overall_df: pd.DataFrame) -> str:
    latex_df = overall_df.copy()
    latex_df = latex_df.rename(columns={"ThresholdRule": "Threshold Rule"})
    for metric in ["Precision", "Recall", "F1", "FPR"]:
        latex_df[metric] = latex_df[metric].map(lambda x: f"{x:.3f}")
    return latex_df.to_latex(index=False, escape=False)


def build_interpretation(overall_df: pd.DataFrame) -> str:
    summary = overall_df.set_index("ThresholdRule")
    recall_1 = summary.loc["1-of-4", "Recall"]
    fpr_1 = summary.loc["1-of-4", "FPR"]
    recall_2 = summary.loc["2-of-4", "Recall"]
    f1_2 = summary.loc["2-of-4", "F1"]
    fpr_2 = summary.loc["2-of-4", "FPR"]
    recall_3 = summary.loc["3-of-4", "Recall"]
    fpr_3 = summary.loc["3-of-4", "FPR"]
    best_f1_rule = overall_df.loc[overall_df["F1"].idxmax(), "ThresholdRule"]

    base = (
        "Across the selected shifted cases, the 1-of-4 rule was the most sensitive, "
        f"with higher average recall ({recall_1:.3f}) but also the highest false positive rate ({fpr_1:.3f}). "
        f"The 3-of-4 rule was the most conservative, lowering false positives ({fpr_3:.3f}) at the cost of recall ({recall_3:.3f}). "
    )
    if best_f1_rule == "2-of-4":
        return (
            base
            + f"The 2-of-4 rule provided the best overall balance, with mean recall {recall_2:.3f}, "
            f"mean F1 {f1_2:.3f}, and a materially lower FPR ({fpr_2:.3f}) than 1-of-4."
        )
    return (
        base
        + f"In this run, {best_f1_rule} attained the highest mean F1, but 2-of-4 still served as the middle-ground rule "
        f"with recall {recall_2:.3f}, F1 {f1_2:.3f}, and FPR {fpr_2:.3f}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sensitivity study for the final VSCOUT consensus threshold."
    )
    parser.add_argument("--n-repeats", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df, summary_df, overall_df = run_consensus_threshold_sensitivity(
        n_repeats=args.n_repeats,
        alpha=args.alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=output_dir,
        checkpoint_every=args.checkpoint_every,
    )
    summary_df, overall_df = write_output_tables(
        results_df=results_df,
        output_dir=output_dir,
        prefix="consensus_threshold_sensitivity",
    )

    print("\nOverall summary:")
    print(overall_df.to_string(index=False))

    print("\nLaTeX table:")
    print(build_latex_table(overall_df))

    print("\nInterpretation:")
    print(build_interpretation(overall_df))


if __name__ == "__main__":
    main()
