from __future__ import annotations

import os
import time
import random
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
import tensorflow as tf
from scipy.stats import chi2
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from VSCOUT import VSCOUT


DEFAULT_N = 500
DEFAULT_REPLICATIONS = 30
DEFAULT_SHIFT = 2.0
DEFAULT_BASE_SEED = 20260312

VARIANT_LABELS = {
    1: "VAE_AllLatents",
    2: "ARDVAE_Mahalanobis",
    3: "ARDVAE_Mahalanobis_CP",
    4: "ARDVAE_Mahalanobis_Ensemble",
    5: "ARDVAE_CP_Ensemble_NoRefine",
    6: "Full_VSCOUT",
}

DGP_LABELS = {
    1: "DGP1_MVN",
    2: "DGP2_HeavyTailed",
    3: "DGP3_Lognormal",
    4: "DGP4_MixedNormalT",
    5: "DGP5_PersistentBlockShift",
}


@dataclass(frozen=True)
class DGPConfig:
    dgp_id: int
    name: str
    p: int
    gamma: float
    delta: float
    df: Optional[int] = None
    block_feature_fraction: float = 0.30


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def get_dgp_config(dgp_id: int, delta: float = DEFAULT_SHIFT) -> DGPConfig:
    configs = {
        1: DGPConfig(1, DGP_LABELS[1], p=150, gamma=0.03, delta=delta, df=None),
        2: DGPConfig(2, DGP_LABELS[2], p=150, gamma=0.10, delta=delta, df=5),
        3: DGPConfig(3, DGP_LABELS[3], p=250, gamma=0.10, delta=delta, df=None),
        4: DGPConfig(4, DGP_LABELS[4], p=250, gamma=0.15, delta=delta, df=5),
        5: DGPConfig(5, DGP_LABELS[5], p=250, gamma=0.15, delta=delta, df=None),
    }
    if dgp_id not in configs:
        raise ValueError(f"Unsupported DGP id: {dgp_id}")
    return configs[dgp_id]


def inject_contamination(
    X: np.ndarray,
    rng: np.random.Generator,
    config: DGPConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    X_cont = np.array(X, copy=True)
    n, p = X_cont.shape
    labels = np.zeros(n, dtype=int)
    meta: Dict[str, int] = {}

    if config.dgp_id == 1:
        n_cont = max(1, int(round(config.gamma * n)))
        idx = rng.choice(n, size=n_cont, replace=False)
        X_cont[idx] = rng.normal(loc=config.delta, scale=1.0, size=(n_cont, p))
        labels[idx] = 1
        meta["tau"] = -1
    elif config.dgp_id == 2:
        n_cont = max(1, int(round(config.gamma * n)))
        idx = rng.choice(n, size=n_cont, replace=False)
        X_cont[idx] = rng.standard_t(df=config.df, size=(n_cont, p)) + config.delta
        labels[idx] = 1
        meta["tau"] = -1
    elif config.dgp_id == 3:
        n_cont = max(1, int(round(config.gamma * n)))
        idx = rng.choice(n, size=n_cont, replace=False)
        shifted_z = rng.normal(loc=config.delta, scale=1.0, size=(n_cont, p))
        X_cont[idx] = np.exp(shifted_z)
        labels[idx] = 1
        meta["tau"] = -1
    elif config.dgp_id == 4:
        n_cont = max(1, int(round(config.gamma * n)))
        idx = rng.choice(n, size=n_cont, replace=False)
        comp = rng.integers(0, 2, size=n_cont)
        shifted = np.empty((n_cont, p))
        normal_mask = comp == 0
        t_mask = ~normal_mask
        if normal_mask.any():
            shifted[normal_mask] = rng.normal(
                loc=config.delta,
                scale=1.0,
                size=(normal_mask.sum(), p),
            )
        if t_mask.any():
            shifted[t_mask] = (
                rng.standard_t(df=config.df, size=(t_mask.sum(), p)) + config.delta
            )
        X_cont[idx] = shifted
        labels[idx] = 1
        meta["tau"] = -1
    elif config.dgp_id == 5:
        block_len = max(1, int(round(config.gamma * n)))
        tau_min = int(np.floor(0.6 * n))
        tau_max = min(int(np.floor(0.8 * n)), n - block_len)
        tau = int(rng.integers(tau_min, tau_max + 1))
        feature_count = max(1, int(round(config.block_feature_fraction * p)))
        feature_idx = rng.choice(p, size=feature_count, replace=False)
        block_slice = slice(tau, tau + block_len)
        X_cont[block_slice, feature_idx] += config.delta
        labels[block_slice] = 1
        meta["tau"] = tau
    else:
        raise ValueError(f"Unsupported DGP id for contamination: {config.dgp_id}")

    return X_cont, labels, meta


def simulate_dgp(
    dgp_id: int,
    n: int = DEFAULT_N,
    delta: float = DEFAULT_SHIFT,
    seed: int = DEFAULT_BASE_SEED,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], DGPConfig]:
    config = get_dgp_config(dgp_id=dgp_id, delta=delta)
    rng = np.random.default_rng(seed)

    if dgp_id == 1:
        X = rng.normal(loc=0.0, scale=1.0, size=(n, config.p))
    elif dgp_id == 2:
        X = rng.standard_t(df=config.df, size=(n, config.p))
    elif dgp_id == 3:
        X = np.exp(rng.normal(loc=0.0, scale=1.0, size=(n, config.p)))
    elif dgp_id == 4:
        half = n // 2
        X = np.empty((n, config.p))
        X[:half] = rng.normal(loc=0.0, scale=1.0, size=(half, config.p))
        X[half:] = rng.standard_t(df=config.df, size=(n - half, config.p))
        rng.shuffle(X, axis=0)
    elif dgp_id == 5:
        X = rng.normal(loc=0.0, scale=1.0, size=(n, config.p))
    else:
        raise ValueError(f"Unsupported DGP id: {dgp_id}")

    X_cont, y_true, meta = inject_contamination(X=X, rng=rng, config=config)
    return X_cont.astype(np.float32), y_true.astype(int), meta, config


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return np.nan


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    runtime: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    scores = np.asarray(scores, dtype=float)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    negatives = np.sum(y_true == 0)
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    fpr = false_positives / negatives if negatives > 0 else np.nan

    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "FPR": float(fpr),
        "AUROC": _safe_auc(y_true, scores),
        "Runtime": float(runtime),
    }


def _fit_base_vae(
    X: np.ndarray,
    alpha: float,
    seed: int,
    vscout_kwargs: Optional[Dict] = None,
    epochs: int = 30,
    batch_size: int = 32,
    verbose: int = 0,
) -> VSCOUT:
    set_global_seed(seed)
    kwargs = dict(vscout_kwargs or {})
    model = VSCOUT(alpha=alpha, **kwargs)
    model._build_model(X)
    model.vae.compile(optimizer=tf.keras.optimizers.Adam(model.learning_rate))
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        patience=10,
        restore_best_weights=True,
    )
    model.vae.fit(
        X,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[early_stop],
    )
    return model


def _get_reconstruction_errors(
    model: VSCOUT,
    X: np.ndarray,
    batch_size: int = 32,
    use_sampled_z: bool = True,
) -> np.ndarray:
    z_mean, _, z_sample = model.encoder.predict(X, batch_size=batch_size, verbose=0)
    latent_input = z_sample if use_sampled_z else z_mean
    X_recon = model.decoder.predict(latent_input, batch_size=batch_size, verbose=0)
    return np.sum((X - X_recon) ** 2, axis=1)


def _extract_ard_state(
    X: np.ndarray,
    alpha: float,
    seed: int,
    vscout_kwargs: Optional[Dict] = None,
    epochs: int = 30,
    batch_size: int = 32,
    verbose: int = 0,
) -> Dict[str, np.ndarray]:
    model = _fit_base_vae(
        X=X,
        alpha=alpha,
        seed=seed,
        vscout_kwargs=vscout_kwargs,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )
    z_mean, z_log_var, _ = model.encoder.predict(X, batch_size=batch_size, verbose=0)
    kl_divs = np.mean(
        0.5 * (-1.0 - z_log_var + np.square(z_mean) + np.exp(z_log_var)),
        axis=0,
    )
    relevant_latents = np.where(kl_divs > model.kl_threshold)[0]
    if relevant_latents.size == 0:
        relevant_latents = np.array([0], dtype=int)

    z_relevant = z_mean[:, relevant_latents]
    center = np.mean(z_relevant, axis=0)
    cov = np.cov(z_relevant, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[cov]])
    inv_cov = np.linalg.pinv(cov)
    diff = z_relevant - center
    mahal_sq = np.sum(diff @ inv_cov * diff, axis=1)
    mahal_threshold = chi2.ppf(1.0 - model.alpha, df=z_relevant.shape[1])
    mahal_mask = mahal_sq > mahal_threshold

    l2_norm = np.linalg.norm(z_relevant, axis=1)
    algo = rpt.Pelt(model="rbf").fit(l2_norm.reshape(-1, 1))
    change_points = algo.predict(pen=model.penalty)
    cp_mask = np.zeros(X.shape[0], dtype=bool)
    if change_points:
        first_cp = change_points[0]
        if first_cp < X.shape[0]:
            cp_mask[first_cp:] = True

    model._fit_ensemble(z_relevant)
    ensemble_mask = np.asarray(
        model._ensemble_predict(z_relevant, rule=model.flag_rule),
        dtype=bool,
    )
    if hasattr(model, "base_detectors") and model.base_detectors:
        base_preds = np.array([clf.predict(z_relevant) for clf in model.base_detectors])
        ensemble_votes = base_preds.sum(axis=0).astype(float)
    else:
        ensemble_votes = ensemble_mask.astype(float)

    return {
        "model": model,
        "z_relevant": z_relevant,
        "relevant_latents": relevant_latents,
        "mahal_sq": mahal_sq,
        "mahal_threshold": np.full(X.shape[0], mahal_threshold, dtype=float),
        "mahal_mask": mahal_mask,
        "cp_mask": cp_mask,
        "ensemble_mask": ensemble_mask,
        "ensemble_votes": ensemble_votes,
        "change_points": np.asarray(change_points, dtype=int),
    }


def run_variant(
    X: np.ndarray,
    y_true: np.ndarray,
    variant_id: int,
    seed: int,
    alpha: float = 0.05,
    vscout_kwargs: Optional[Dict] = None,
    epochs: int = 30,
    batch_size: int = 32,
    verbose: int = 0,
) -> Dict[str, float]:
    start = time.perf_counter()
    set_global_seed(seed)

    if variant_id == 1:
        model = _fit_base_vae(
            X=X,
            alpha=alpha,
            seed=seed,
            vscout_kwargs=vscout_kwargs,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        recon_errors = _get_reconstruction_errors(
            model=model,
            X=X,
            batch_size=batch_size,
            use_sampled_z=True,
        )
        threshold = np.percentile(recon_errors, 100.0 * (1.0 - model.alpha))
        y_pred = recon_errors > threshold
        scores = recon_errors
    elif variant_id in {2, 3, 4, 5}:
        state = _extract_ard_state(
            X=X,
            alpha=alpha,
            seed=seed,
            vscout_kwargs=vscout_kwargs,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        mahal_mask = state["mahal_mask"]
        cp_mask = state["cp_mask"]
        ensemble_mask = state["ensemble_mask"]

        if variant_id == 2:
            y_pred = mahal_mask
            scores = state["mahal_sq"] / np.maximum(state["mahal_threshold"], 1e-12)
        elif variant_id == 3:
            y_pred = np.logical_or(mahal_mask, cp_mask)
            scores = (
                state["mahal_sq"] / np.maximum(state["mahal_threshold"], 1e-12)
                + cp_mask.astype(float)
            )
        elif variant_id == 4:
            y_pred = np.logical_or(mahal_mask, ensemble_mask)
            scores = (
                state["mahal_sq"] / np.maximum(state["mahal_threshold"], 1e-12)
                + state["ensemble_votes"] / max(1.0, state["ensemble_votes"].max())
            )
        else:
            votes = np.stack([mahal_mask, cp_mask, ensemble_mask], axis=1)
            y_pred = votes.sum(axis=1) >= 2
            scores = (
                state["mahal_sq"] / np.maximum(state["mahal_threshold"], 1e-12)
                + cp_mask.astype(float)
                + state["ensemble_votes"] / max(1.0, state["ensemble_votes"].max())
            )
    elif variant_id == 6:
        model = VSCOUT(alpha=alpha, **(vscout_kwargs or {}))
        model.fit(X, epochs=epochs, batch_size=batch_size, verbose=verbose)
        (
            y_pred,
            cp_mask,
            suod_mask,
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
        scores = (
            cp_mask.astype(float)
            + suod_mask.astype(float)
            + (mahal_sq / max(float(mahal_threshold), 1e-12))
            + (recon_errors / max(float(recon_threshold), 1e-12))
        )
    else:
        raise ValueError(f"Unsupported variant id: {variant_id}")

    runtime = time.perf_counter() - start
    metrics = _compute_metrics(y_true=y_true, y_pred=y_pred, scores=scores, runtime=runtime)
    return metrics


def _single_run(task: Tuple[int, int, int, int, float, Dict, int, int]) -> Dict[str, float]:
    dgp_id, variant_id, replication, n, delta, vscout_kwargs, epochs, batch_size = task
    seed = DEFAULT_BASE_SEED + dgp_id * 10000 + variant_id * 1000 + replication
    X, y_true, meta, config = simulate_dgp(dgp_id=dgp_id, n=n, delta=delta, seed=seed)
    metrics = run_variant(
        X=X,
        y_true=y_true,
        variant_id=variant_id,
        seed=seed,
        alpha=0.05,
        vscout_kwargs=vscout_kwargs,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )
    metrics.update(
        {
            "DGP": config.name,
            "Variant": VARIANT_LABELS[variant_id],
            "Replication": replication,
            "Seed": seed,
            "n": n,
            "p": config.p,
            "gamma": config.gamma,
            "delta": config.delta,
            "tau": meta.get("tau", -1),
        }
    )
    return metrics


def run_ablation(
    dgps: Sequence[int] = (1, 2, 3, 4, 5),
    variants: Sequence[int] = (1, 2, 3, 4, 5, 6),
    n: int = DEFAULT_N,
    replications: int = DEFAULT_REPLICATIONS,
    delta: float = DEFAULT_SHIFT,
    vscout_kwargs: Optional[Dict] = None,
    epochs: int = 30,
    batch_size: int = 32,
    max_workers: int = 1,
    output_dir: str = "ablation_results",
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    tasks = [
        (dgp_id, variant_id, replication, n, delta, dict(vscout_kwargs or {}), epochs, batch_size)
        for dgp_id in dgps
        for variant_id in variants
        for replication in range(1, replications + 1)
    ]

    results: List[Dict[str, float]] = []
    if max_workers and max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_single_run, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for task in tasks:
            results.append(_single_run(task))

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["DGP", "Variant", "Replication"]).reset_index(drop=True)
    results_path = os.path.join(output_dir, "ablation_results.csv")
    results_df.to_csv(results_path, index=False)
    return results_df


def summarize_results(
    results_df: pd.DataFrame,
    output_dir: str = "ablation_results",
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    metrics = ["Precision", "Recall", "F1", "FPR", "AUROC", "Runtime"]

    summary = (
        results_df.groupby(["DGP", "Variant"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns.to_flat_index()
    ]

    summary_path = os.path.join(output_dir, "ablation_summary.csv")
    summary.to_csv(summary_path, index=False)

    for metric in ["F1", "FPR", "Runtime"]:
        _plot_metric_bars(
            results_df=results_df,
            metric=metric,
            output_path=os.path.join(output_dir, f"{metric.lower()}_comparison.png"),
        )

    return summary


def _plot_metric_bars(results_df: pd.DataFrame, metric: str, output_path: str) -> None:
    agg = (
        results_df.groupby(["DGP", "Variant"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    dgp_order = [DGP_LABELS[i] for i in sorted(DGP_LABELS)]
    variant_order = [VARIANT_LABELS[i] for i in sorted(VARIANT_LABELS)]

    pivot_mean = (
        agg.pivot(index="DGP", columns="Variant", values="mean")
        .reindex(index=dgp_order, columns=variant_order)
    )
    pivot_std = (
        agg.pivot(index="DGP", columns="Variant", values="std")
        .reindex(index=dgp_order, columns=variant_order)
    )

    ax = pivot_mean.plot(
        kind="bar",
        figsize=(14, 6),
        yerr=pivot_std,
        capsize=3,
        rot=20,
    )
    ax.set_ylabel(metric)
    ax.set_xlabel("DGP")
    ax.set_title(f"{metric} Comparison Across DGPs")
    ax.legend(title="Variant", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    results = run_ablation()
    summarize_results(results)


if __name__ == "__main__":
    main()
