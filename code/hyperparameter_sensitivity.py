from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from VSCOUT import VSCOUT
from ablation_framework import (
    DEFAULT_BASE_SEED,
    _compute_metrics,
    _get_reconstruction_errors,
    set_global_seed,
)
from consensus_threshold_sensitivity import simulate_sustained_outliers


DEFAULT_REPEATS = 25
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True)
class ScenarioConfig:
    label: str
    dist: str
    contamination: float
    n_inlier: int
    n_features: int
    gamma: float
    seed_offset: int


@dataclass(frozen=True)
class HyperparameterSpec:
    key: str
    title: str
    values: tuple[Any, ...]
    default: Any
    formatter: Callable[[Any], str]


SCENARIOS = [
    ScenarioConfig(
        label="Multivariate Normal",
        dist="normal",
        contamination=0.10,
        n_inlier=500,
        n_features=150,
        gamma=2.0,
        seed_offset=1,
    ),
    ScenarioConfig(
        label="Mixed (Normal and t-Distribution)",
        dist="mixed",
        contamination=0.15,
        n_inlier=500,
        n_features=250,
        gamma=2.0,
        seed_offset=2,
    ),
]


HYPERPARAMETERS = [
    HyperparameterSpec(
        key="latent_dim",
        title="Latent size",
        values=(16, 32, 64),
        default=32,
        formatter=lambda x: f"{int(x)}",
    ),
    HyperparameterSpec(
        key="kl_threshold",
        title="KL threshold",
        values=(0.5, 1.0, 2.0),
        default=1.0,
        formatter=lambda x: f"{float(x):.1f}",
    ),
    HyperparameterSpec(
        key="penalty",
        title="PELT penalty",
        values=(20, 40, 60),
        default=40,
        formatter=lambda x: f"{int(x)}",
    ),
    HyperparameterSpec(
        key="alpha",
        title="Significance level",
        values=(0.01, 0.05, 0.10),
        default=0.05,
        formatter=lambda x: f"{float(x):.2f}",
    ),
    HyperparameterSpec(
        key="flag_rule",
        title="Ensemble rule",
        values=("any", "majority", "all"),
        default="any",
        formatter=str,
    ),
    HyperparameterSpec(
        key="hidden_width",
        title="Hidden-layer width",
        values=(32, 64, 128),
        default=64,
        formatter=lambda x: f"{int(x)}",
    ),
]


DEFAULTS = {
    "latent_dim": 32,
    "kl_threshold": 1.0,
    "penalty": 40,
    "alpha": 0.05,
    "flag_rule": "any",
    "hidden_width": 64,
}


class VSCOUTSensitivity(VSCOUT):
    def _ensemble_predict(self, z_mean_relevant, rule="majority"):
        base_preds = np.array([clf.predict(z_mean_relevant) for clf in self.base_detectors])
        votes = np.sum(base_preds, axis=0)
        n_detectors = base_preds.shape[0]
        if rule == "majority":
            return votes >= (n_detectors // 2 + 1)
        if rule == "any":
            return votes >= 1
        if rule == "all":
            return votes == n_detectors
        raise ValueError("flag_rule must be 'any', 'majority', or 'all'")


def _build_vscout_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    width = int(params["hidden_width"])
    hidden_layers = (width,)
    return {
        "encoder_neurons": hidden_layers,
        "decoder_neurons": hidden_layers,
        "latent_dim": int(params["latent_dim"]),
        "kl_threshold": float(params["kl_threshold"]),
        "penalty": int(params["penalty"]),
        "alpha": float(params["alpha"]),
        "flag_rule": str(params["flag_rule"]),
    }


def _setting_value_label(spec: HyperparameterSpec, value: Any) -> str:
    return spec.formatter(value)


def _simulate_scenario(config: ScenarioConfig, seed: int):
    n_outlier = int(config.n_inlier * config.contamination / (1.0 - config.contamination))
    X, y_true = simulate_sustained_outliers(
        n_inlier=config.n_inlier,
        n_outlier=n_outlier,
        n_features=config.n_features,
        gamma=config.gamma,
        dist=config.dist,
        random_state=seed,
    )
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled.astype(np.float32), y_true.astype(int)


def _make_seed(scenario: ScenarioConfig, hyper_idx: int, value_idx: int, replication: int) -> int:
    return (
        DEFAULT_BASE_SEED
        + scenario.seed_offset * 100000
        + hyper_idx * 10000
        + value_idx * 1000
        + replication
    )


def _run_single_setting(
    scenario: ScenarioConfig,
    spec: HyperparameterSpec,
    value: Any,
    hyper_idx: int,
    value_idx: int,
    replication: int,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    params = dict(DEFAULTS)
    params[spec.key] = value
    seed = _make_seed(
        scenario=scenario,
        hyper_idx=hyper_idx,
        value_idx=value_idx,
        replication=replication,
    )

    set_global_seed(seed)
    X, y_true = _simulate_scenario(config=scenario, seed=seed)
    model_kwargs = _build_vscout_kwargs(params)

    start = time.perf_counter()
    model = VSCOUTSensitivity(**model_kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X, epochs=epochs, batch_size=batch_size, verbose=0)
        (
            y_pred,
            cp_mask,
            ensemble_mask,
            t2_mask,
            recon_mask,
            mahal_sq,
            mahal_threshold,
            recon_threshold,
        ) = model.is_outlier(X, batch_size=batch_size)
    recon_errors = _get_reconstruction_errors(
        model=model,
        X=X,
        batch_size=batch_size,
        use_sampled_z=True,
    )
    runtime = time.perf_counter() - start

    scores = (
        cp_mask.astype(float)
        + ensemble_mask.astype(float)
        + (mahal_sq / max(float(mahal_threshold), 1e-12))
        + (recon_errors / max(float(recon_threshold), 1e-12))
    )
    metrics = _compute_metrics(
        y_true=y_true,
        y_pred=np.asarray(y_pred, dtype=int),
        scores=scores,
        runtime=runtime,
    )

    return {
        "Scenario": scenario.label,
        "Distribution": scenario.dist,
        "Contam (%)": 100.0 * scenario.contamination,
        "N(clean)": scenario.n_inlier,
        "p": scenario.n_features,
        "Shift": scenario.gamma,
        "Hyperparameter": spec.title,
        "HyperparameterKey": spec.key,
        "Value": _setting_value_label(spec, value),
        "ValueRaw": value,
        "Replication": replication,
        "Seed": seed,
        **metrics,
    }


def run_hyperparameter_sensitivity(
    scenarios: list[ScenarioConfig] | None = None,
    hyperparameters: list[HyperparameterSpec] | None = None,
    n_repeats: int = DEFAULT_REPEATS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    checkpoint_dir: Path | str | None = None,
    checkpoint_every: int = 1,
) -> pd.DataFrame:
    scenarios = SCENARIOS if scenarios is None else scenarios
    hyperparameters = HYPERPARAMETERS if hyperparameters is None else hyperparameters

    rows = []
    completed_runs = 0
    for hyper_idx, spec in enumerate(hyperparameters, start=1):
        print(f"=== {spec.title} ===")
        for value_idx, value in enumerate(spec.values, start=1):
            print(f"Value: {_setting_value_label(spec, value)}")
            for scenario in scenarios:
                print(f"  Scenario: {scenario.label}")
                for replication in range(1, n_repeats + 1):
                    rows.append(
                        _run_single_setting(
                            scenario=scenario,
                            spec=spec,
                            value=value,
                            hyper_idx=hyper_idx,
                            value_idx=value_idx,
                            replication=replication,
                            epochs=epochs,
                            batch_size=batch_size,
                        )
                    )
                    completed_runs += 1
                    if checkpoint_dir is not None and completed_runs % checkpoint_every == 0:
                        write_output_tables(
                            results_df=pd.DataFrame(rows),
                            output_dir=checkpoint_dir,
                            prefix="hyperparameter_sensitivity_checkpoint",
                        )

    results_df = pd.DataFrame(rows)
    if checkpoint_dir is not None:
        write_output_tables(
            results_df=results_df,
            output_dir=checkpoint_dir,
            prefix="hyperparameter_sensitivity_checkpoint",
        )
    return results_df


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["F1", "Recall", "FPR", "Precision", "Runtime"]
    summary_df = (
        results_df.groupby(["Scenario", "Hyperparameter", "HyperparameterKey", "Value"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df.columns = [
        "_".join(str(part) for part in col if part).rstrip("_") for col in summary_df.columns
    ]
    return summary_df.rename(
        columns={
            "F1_std": "F1_sd",
            "Recall_std": "Recall_sd",
            "FPR_std": "FPR_sd",
            "Precision_std": "Precision_sd",
            "Runtime_std": "Runtime_sd",
        }
    )


def write_output_tables(
    results_df: pd.DataFrame,
    output_dir: Path | str,
    prefix: str = "hyperparameter_sensitivity",
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = summarize_results(results_df)
    results_df.to_csv(output_dir / f"{prefix}_results.csv", index=False)
    summary_df.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    return summary_df


def plot_sensitivity(summary_df: pd.DataFrame, output_path: Path | str) -> None:
    output_path = Path(output_path)
    scenario_colors = {
        "Multivariate Normal": "#1f77b4",
        "Mixed (Normal and t-Distribution)": "#d95f02",
    }
    scenario_markers = {
        "Multivariate Normal": "o",
        "Mixed (Normal and t-Distribution)": "s",
    }

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharey=True)
    axes = axes.ravel()

    legend_handles = None
    legend_labels = None

    for ax, spec in zip(axes, HYPERPARAMETERS):
        spec_df = summary_df[summary_df["HyperparameterKey"] == spec.key].copy()
        ordered_labels = [_setting_value_label(spec, value) for value in spec.values]
        x_positions = np.arange(len(ordered_labels))

        for scenario in [cfg.label for cfg in SCENARIOS]:
            scenario_df = spec_df[spec_df["Scenario"] == scenario].copy()
            scenario_df["Value"] = pd.Categorical(
                scenario_df["Value"],
                categories=ordered_labels,
                ordered=True,
            )
            scenario_df = scenario_df.sort_values("Value")
            line = ax.errorbar(
                x_positions,
                scenario_df["F1_mean"],
                yerr=scenario_df["F1_sd"],
                marker=scenario_markers[scenario],
                color=scenario_colors[scenario],
                linewidth=1.8,
                markersize=5.5,
                capsize=3,
                label=scenario,
            )
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        ax.set_title(spec.title)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(ordered_labels)
        ax.set_ylabel("Mean F1")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.set_axisbelow(True)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_interpretation(summary_df: pd.DataFrame) -> str:
    overall = (
        summary_df.groupby(["HyperparameterKey", "Value"])[["F1_mean", "Recall_mean", "FPR_mean"]]
        .mean()
        .reset_index()
    )

    def _subset(key: str) -> pd.DataFrame:
        spec = next(item for item in HYPERPARAMETERS if item.key == key)
        ordered_labels = [_setting_value_label(spec, value) for value in spec.values]
        df = overall[overall["HyperparameterKey"] == key].copy()
        df["Value"] = pd.Categorical(df["Value"], categories=ordered_labels, ordered=True)
        return df.sort_values("Value")

    latent_df = _subset("latent_dim")
    width_df = _subset("hidden_width")
    alpha_df = _subset("alpha")
    penalty_df = _subset("penalty")
    ensemble_df = _subset("flag_rule")

    latent_span = latent_df["F1_mean"].max() - latent_df["F1_mean"].min()
    width_span = width_df["F1_mean"].max() - width_df["F1_mean"].min()

    alpha_low = alpha_df.iloc[0]
    alpha_high = alpha_df.iloc[-1]
    penalty_best = penalty_df.loc[penalty_df["F1_mean"].idxmax(), "Value"]
    ensemble_best = ensemble_df.loc[ensemble_df["F1_mean"].idxmax(), "Value"]

    return (
        f"Performance is {'stable' if latent_span < 0.05 and width_span < 0.05 else 'moderately sensitive'} "
        f"across latent size and hidden-layer width, with mean F1 ranges of {latent_span:.3f} and {width_span:.3f}, respectively. "
        f"Moving from alpha={alpha_low['Value']} to alpha={alpha_high['Value']} changes mean recall from "
        f"{alpha_low['Recall_mean']:.3f} to {alpha_high['Recall_mean']:.3f} while FPR shifts from "
        f"{alpha_low['FPR_mean']:.3f} to {alpha_high['FPR_mean']:.3f}, consistent with more aggressive significance levels increasing sensitivity at some false-positive cost. "
        f"The best average F1 for the PELT sweep occurs at penalty={penalty_best}, suggesting larger penalties can reduce responsiveness when they become too conservative. "
        f"Across ensemble rules, the strongest mean F1 occurs for '{ensemble_best}', highlighting that the aggregation rule meaningfully affects overall aggressiveness."
    )


def main() -> None:
    output_dir = Path.cwd()
    results_df = run_hyperparameter_sensitivity(
        checkpoint_dir=output_dir,
        checkpoint_every=1,
    )
    summary_df = write_output_tables(
        results_df=results_df,
        output_dir=output_dir,
        prefix="hyperparameter_sensitivity",
    )
    plot_sensitivity(
        summary_df=summary_df,
        output_path=output_dir / "hyperparameter_sensitivity_2x3.png",
    )

    print("\nSummary:")
    print(summary_df.head(18).to_string(index=False))

    print("\nInterpretation:")
    print(build_interpretation(summary_df))


if __name__ == "__main__":
    main()
