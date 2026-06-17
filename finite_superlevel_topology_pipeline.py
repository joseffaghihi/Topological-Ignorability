"""Finite superlevel-set topological diagnostics replication pipeline.

This script reproduces four analyses:

1. A synthetic structural-equation benchmark with known potential outcomes.
2. A Wisconsin breast-cancer semi-synthetic benchmark with known potential outcomes.
3. A randomized JOBS II application using the public Rdatasets extract by default.
4. An optional observational eICU analysis when the required CSV.GZ files are supplied.

Scientific scope
----------------
The synthetic and semi-synthetic benchmarks are causal benchmarks because the
potential outcomes are generated and therefore known. The JOBS II analysis is a
randomized empirical application without unit-level oracle potential outcomes.
The eICU analysis is observational and is reported as a descriptive sensitivity
analysis; its diagnostics do not establish causal identification or conditional
exchangeability.

Installation
------------
Create an environment with Python 3.10 or newer, then install:

    python -m pip install numpy pandas matplotlib scipy scikit-learn jupyter

Default replication
-------------------
Run the synthetic benchmark, the breast-cancer semi-synthetic benchmark, and the
JOBS II randomized application:

    python finite_superlevel_topology_pipeline.py \
        --outdir superlevel_outputs \
        --bootstrap 30 \
        --diagnostic-permutations 99 \
        --skip-eicu

Optional eICU analysis
----------------------
The optional eICU branch expects the files ``apachePredVar.csv.gz`` and
``apachePatientResult.csv.gz`` in the directory passed with ``--eicu-dir``:

    python finite_superlevel_topology_pipeline.py \
        --outdir superlevel_outputs_with_eicu \
        --eicu-dir data/eicu

Outputs
-------
Each dataset is written under ``--outdir/<dataset-name>/`` as CSV diagnostics and
PNG figures. The root output directory contains ``superlevel_results_summary.tex``
and ``output_manifest.csv``. A local JOBS II CSV can be supplied with
``--jobs-csv`` to avoid dependence on the remote Rdatasets URL.
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, label
from scipy.special import expit, logit
from scipy.spatial.distance import cdist, pdist
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)
np.set_printoptions(suppress=True, linewidth=140)


DEFAULT_RANDOM_SEED = 13
DEFAULT_TOPOLOGY_BOUNDS = (-3.0, 3.0, -1.5, 1.5)
FOREGROUND_CONNECTIVITY = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
BACKGROUND_CONNECTIVITY = np.ones((3, 3), dtype=int)
DEFAULT_EICU_DIRECTORY = Path("data") / "eicu"
JOBS_II_RDATASETS_CSV_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/mediation/jobs.csv"
JOBS_II_ARCHIVE_URL = "https://www.icpsr.umich.edu/web/ICPSR/studies/2739"
JOBS_II_ARCHIVE_DOI = "https://doi.org/10.3886/ICPSR02739.v1"


DatasetBundle = dict[str, Any]
TopologyParameters = dict[str, Any]


def print_progress(message: str, current: int | None = None, total: int | None = None, *, enabled: bool = True) -> None:
    if not enabled:
        return
    stamp = time.strftime("%H:%M:%S")
    if current is not None and total is not None:
        prefix = f"[{current:02d}/{total:02d}]"
    else:
        prefix = "[--/--]"
    print(f"{stamp} {prefix} {message}", flush=True)


def compute_weighted_mean(x, w=None):
    x = np.asarray(x, float)
    if w is None:
        return float(np.mean(x))
    w = np.asarray(w, float)
    return float(np.sum(w * x) / np.sum(w))


def compute_weighted_variance(x, w=None):
    x = np.asarray(x, float)
    if w is None:
        return float(np.var(x, ddof=0))
    w = np.asarray(w, float)
    m = compute_weighted_mean(x, w)
    return float(np.sum(w * (x - m) ** 2) / np.sum(w))


def compute_weighted_mean_difference(y, treatment, w=None, coord=0):
    y = np.asarray(y, float)
    treatment = np.asarray(treatment, int)
    if w is None:
        return float(y[treatment == 1, coord].mean() - y[treatment == 0, coord].mean())
    w = np.asarray(w, float)
    return compute_weighted_mean(y[treatment == 1, coord], w[treatment == 1]) - compute_weighted_mean(y[treatment == 0, coord], w[treatment == 0])


def compute_effective_sample_size(w):
    w = np.asarray(w, float)
    denom = np.sum(w**2)
    if denom <= 0:
        return 0.0
    return float((np.sum(w) ** 2) / denom)


def estimate_sampled_energy_distance(x, y, max_n=500, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) > max_n:
        x = x[rng.choice(len(x), size=max_n, replace=False)]
    if len(y) > max_n:
        y = y[rng.choice(len(y), size=max_n, replace=False)]
    dxy = cdist(x, y).mean()
    dxx = pdist(x).mean() if len(x) > 1 else 0.0
    dyy = pdist(y).mean() if len(y) > 1 else 0.0
    return float(2 * dxy - dxx - dyy)


def build_covariate_design_matrix(covariates: pd.DataFrame):
    X = pd.get_dummies(covariates.copy(), drop_first=False)
    X = X.apply(pd.to_numeric, errors="coerce")
    imp = SimpleImputer(strategy="median")
    imputed_covariate_design = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)
    return imputed_covariate_design


def estimate_propensity_scores(covariates: pd.DataFrame, treatment: np.ndarray, clip=0.02, C=1.0, max_iter=5000):
    X = build_covariate_design_matrix(covariates)
    scaler = StandardScaler()
    standardized_covariates = scaler.fit_transform(X)
    model = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=max_iter)
    model.fit(standardized_covariates, treatment)
    propensity_score = np.clip(model.predict_proba(standardized_covariates)[:, 1], clip, 1 - clip)
    return {
        "propensity": propensity_score,
        "model": model,
        "X_design": X,
        "X_scaled": standardized_covariates,
        "scaler": scaler,
    }


def compute_inverse_probability_weights(treatment, propensity_score, stabilized=True):
    treatment = np.asarray(treatment, int)
    propensity_score = np.asarray(propensity_score, float)
    pi = treatment.mean()
    if stabilized:
        return np.where(treatment == 1, pi / propensity_score, (1 - pi) / (1 - propensity_score))
    return np.where(treatment == 1, 1 / propensity_score, 1 / (1 - propensity_score))


def compute_overlap_weights(treatment, propensity_score):
    treatment = np.asarray(treatment, int)
    propensity_score = np.asarray(propensity_score, float)
    return np.where(treatment == 1, 1 - propensity_score, propensity_score)


def compute_standardized_mean_differences(numeric_covariates: pd.DataFrame, treatment: np.ndarray, w=None):
    rows = []
    treatment = np.asarray(treatment, int)
    weight_array = None if w is None else np.asarray(w, float)
    for col in numeric_covariates.columns:
        x = numeric_covariates[col].to_numpy(dtype=float)
        mt = compute_weighted_mean(x[treatment == 1], None if weight_array is None else weight_array[treatment == 1])
        mc = compute_weighted_mean(x[treatment == 0], None if weight_array is None else weight_array[treatment == 0])
        vt = compute_weighted_variance(x[treatment == 1], None if weight_array is None else weight_array[treatment == 1])
        vc = compute_weighted_variance(x[treatment == 0], None if weight_array is None else weight_array[treatment == 0])
        pooled = math.sqrt(max((vt + vc) / 2.0, 1e-12))
        rows.append({"feature": col, "smd": (mt - mc) / pooled})
    out = pd.DataFrame(rows)
    out["abs_smd"] = out["smd"].abs()
    return out.sort_values("abs_smd", ascending=False).reset_index(drop=True)


def summarize_covariate_balance(numeric_covariates, treatment, w):
    raw = compute_standardized_mean_differences(numeric_covariates, treatment, None)
    wt = compute_standardized_mean_differences(numeric_covariates, treatment, w)
    return (
        pd.DataFrame(
            [
                {
                    "n_features": int(numeric_covariates.shape[1]),
                    "max_abs_smd_raw": float(raw["abs_smd"].max()),
                    "mean_abs_smd_raw": float(raw["abs_smd"].mean()),
                    "max_abs_smd_weighted": float(wt["abs_smd"].max()),
                    "mean_abs_smd_weighted": float(wt["abs_smd"].mean()),
                }
            ]
        ),
        raw,
        wt,
    )


def compute_weighted_ks_statistic(x_t, x_c, w_t=None, w_c=None):
    x_t = np.asarray(x_t, float)
    x_c = np.asarray(x_c, float)
    if len(x_t) == 0 or len(x_c) == 0:
        return np.nan
    grid = np.unique(np.r_[x_t, x_c])
    if len(grid) > 600:
        grid = np.unique(np.quantile(np.r_[x_t, x_c], np.linspace(0, 1, 600)))
    if w_t is None:
        ft = np.searchsorted(np.sort(x_t), grid, side="right") / len(x_t)
    else:
        order = np.argsort(x_t)
        xs = x_t[order]
        ws = np.maximum(np.asarray(w_t, float)[order], 0.0)
        cs = np.cumsum(ws)
        denom = cs[-1] if len(cs) and cs[-1] > 0 else 1.0
        ft = cs[np.searchsorted(xs, grid, side="right") - 1] / denom
        ft[np.searchsorted(xs, grid, side="right") == 0] = 0.0
    if w_c is None:
        fc = np.searchsorted(np.sort(x_c), grid, side="right") / len(x_c)
    else:
        order = np.argsort(x_c)
        xs = x_c[order]
        ws = np.maximum(np.asarray(w_c, float)[order], 0.0)
        cs = np.cumsum(ws)
        denom = cs[-1] if len(cs) and cs[-1] > 0 else 1.0
        fc = cs[np.searchsorted(xs, grid, side="right") - 1] / denom
        fc[np.searchsorted(xs, grid, side="right") == 0] = 0.0
    return float(np.max(np.abs(ft - fc)))


def compute_symmetric_variance_ratio(x, treatment, w=None):
    treatment = np.asarray(treatment, int)
    x = np.asarray(x, float)
    weight_array = None if w is None else np.asarray(w, float)
    vt = compute_weighted_variance(x[treatment == 1], None if weight_array is None else weight_array[treatment == 1])
    vc = compute_weighted_variance(x[treatment == 0], None if weight_array is None else weight_array[treatment == 0])
    if vc <= 1e-12 or vt <= 1e-12:
        return np.nan
    ratio = vt / vc
    return float(max(ratio, 1.0 / ratio))


def compute_covariate_balance_diagnostics(numeric_covariates: pd.DataFrame, treatment: np.ndarray, w=None, n_permutations=199, seed=DEFAULT_RANDOM_SEED):
    treatment = np.asarray(treatment, int)
    X = numeric_covariates.apply(pd.to_numeric, errors="coerce").fillna(numeric_covariates.median(numeric_only=True)).to_numpy(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    weight_array = None if w is None else np.asarray(w, float)
    rows = []
    for j, col in enumerate(numeric_covariates.columns):
        x = X[:, j]
        raw_smd_table = compute_standardized_mean_differences(pd.DataFrame({col: x}), treatment, None)["smd"].iloc[0]
        weighted_smd_table = compute_standardized_mean_differences(pd.DataFrame({col: x}), treatment, weight_array)["smd"].iloc[0] if weight_array is not None else np.nan
        rows.append(
            {
                "feature": col,
                "raw_smd": float(raw_smd_table),
                "abs_raw_smd": float(abs(raw_smd_table)),
                "weighted_smd": float(weighted_smd_table) if not pd.isna(weighted_smd_table) else np.nan,
                "abs_weighted_smd": float(abs(weighted_smd_table)) if not pd.isna(weighted_smd_table) else np.nan,
                "raw_variance_ratio_or_inverse": compute_symmetric_variance_ratio(x, treatment, None),
                "weighted_variance_ratio_or_inverse": compute_symmetric_variance_ratio(x, treatment, weight_array) if weight_array is not None else np.nan,
                "raw_ks_statistic": compute_weighted_ks_statistic(x[treatment == 1], x[treatment == 0]),
                "weighted_ks_statistic": compute_weighted_ks_statistic(
                    x[treatment == 1],
                    x[treatment == 0],
                    None if weight_array is None else weight_array[treatment == 1],
                    None if weight_array is None else weight_array[treatment == 0],
                )
                if weight_array is not None
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_raw_smd", ascending=False).reset_index(drop=True)


def _standardize_nonconstant_columns(X):
    X = np.asarray(X, float)
    sd = X.std(axis=0)
    keep = sd > 1e-10
    if keep.sum() == 0:
        return np.zeros((X.shape[0], 1))
    return (X[:, keep] - X[:, keep].mean(axis=0)) / sd[keep]


def compute_omnibus_balance_statistic(X, treatment, w=None):
    treatment = np.asarray(treatment, int)
    standardized_covariates = _standardize_nonconstant_columns(X)
    weight_array = None if w is None else np.asarray(w, float)
    if weight_array is None:
        d = standardized_covariates[treatment == 1].mean(axis=0) - standardized_covariates[treatment == 0].mean(axis=0)
    else:
        d = np.asarray(
            [
                compute_weighted_mean(standardized_covariates[treatment == 1, j], weight_array[treatment == 1]) - compute_weighted_mean(standardized_covariates[treatment == 0, j], weight_array[treatment == 0])
                for j in range(standardized_covariates.shape[1])
            ]
        )
    return float(np.mean(d**2))


def run_omnibus_balance_permutation_test(numeric_covariates, treatment, w=None, n_permutations=199, seed=DEFAULT_RANDOM_SEED):
    rng = np.random.default_rng(seed)
    X = numeric_covariates.apply(pd.to_numeric, errors="coerce").fillna(numeric_covariates.median(numeric_only=True)).to_numpy(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    treatment = np.asarray(treatment, int)
    obs = compute_omnibus_balance_statistic(X, treatment, w)
    vals = []
    n_permutations = int(n_permutations)
    for _ in range(n_permutations):
        ap = rng.permutation(treatment)
        vals.append(compute_omnibus_balance_statistic(X, ap, w))
    vals = np.asarray(vals, float)
    p = float((1 + np.sum(vals >= obs)) / (1 + len(vals))) if len(vals) else np.nan
    return pd.DataFrame(
        [
            {
                "test": "omnibus_standardized_mean_balance",
                "statistic": obs,
                "permutation_p_value": p,
                "n_permutations": int(n_permutations),
                "weighted": bool(w is not None),
                "interpretation": "large p-value indicates no detectable aggregate imbalance in observed covariates",
            }
        ]
    )


def compute_propensity_overlap_diagnostics(propensity_score, treatment, ate_weights, overlap_weight_values=None):
    propensity_score = np.asarray(propensity_score, float)
    treatment = np.asarray(treatment, int)
    ate_weights = np.asarray(ate_weights, float)
    rows = []
    for label_name, w in [("ATE", ate_weights), ("overlap", overlap_weight_values)]:
        if w is None:
            continue
        norm_w = w / max(w.sum(), 1e-12)
        rows.append(
            {
                "weight_scheme": label_name,
                "propensity_min": float(propensity_score.min()),
                "propensity_max": float(propensity_score.max()),
                "fraction_outside_0p05_0p95": float(np.mean((propensity_score < 0.05) | (propensity_score > 0.95))),
                "fraction_outside_0p10_0p90": float(np.mean((propensity_score < 0.10) | (propensity_score > 0.90))),
                "ESS_treated": compute_effective_sample_size(w[treatment == 1]),
                "ESS_control": compute_effective_sample_size(w[treatment == 0]),
                "ESS_fraction_total": compute_effective_sample_size(w) / len(w),
                "weight_min": float(np.min(w)),
                "weight_p50": float(np.quantile(w, 0.50)),
                "weight_p95": float(np.quantile(w, 0.95)),
                "weight_p99": float(np.quantile(w, 0.99)),
                "weight_max": float(np.max(w)),
                "weight_cv": float(np.std(w) / max(np.mean(w), 1e-12)),
                "max_normalized_weight": float(np.max(norm_w)),
            }
        )
    return pd.DataFrame(rows)


def estimate_sampled_rbf_mmd2(x, y, max_n=500, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) > max_n:
        x = x[rng.choice(len(x), size=max_n, replace=False)]
    if len(y) > max_n:
        y = y[rng.choice(len(y), size=max_n, replace=False)]
    z = np.vstack([x, y])
    d = cdist(z, z, metric="sqeuclidean")
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    gamma = 1.0 / max(med, 1e-8)
    kxx = np.exp(-gamma * cdist(x, x, metric="sqeuclidean"))
    kyy = np.exp(-gamma * cdist(y, y, metric="sqeuclidean"))
    kxy = np.exp(-gamma * cdist(x, y, metric="sqeuclidean"))
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean())


def run_outcome_distribution_permutation_tests(observed_outcomes, treatment, n_permutations=199, max_n=450, seed=DEFAULT_RANDOM_SEED):
    rng = np.random.default_rng(seed)
    observed_outcomes = np.asarray(observed_outcomes, float)
    treatment = np.asarray(treatment, int)
    n_permutations = int(n_permutations)
    stats = {
        "mean_norm": float(np.linalg.norm(observed_outcomes[treatment == 1].mean(axis=0) - observed_outcomes[treatment == 0].mean(axis=0))),
        "energy_distance": estimate_sampled_energy_distance(observed_outcomes[treatment == 1], observed_outcomes[treatment == 0], max_n=max_n, seed=seed + 1),
        "rbf_mmd2": estimate_sampled_rbf_mmd2(observed_outcomes[treatment == 1], observed_outcomes[treatment == 0], max_n=max_n, seed=seed + 2),
    }
    perm_values = {key: [] for key in stats}
    for _ in range(n_permutations):
        ap = rng.permutation(treatment)
        perm_values["mean_norm"].append(float(np.linalg.norm(observed_outcomes[ap == 1].mean(axis=0) - observed_outcomes[ap == 0].mean(axis=0))))
        perm_values["energy_distance"].append(estimate_sampled_energy_distance(observed_outcomes[ap == 1], observed_outcomes[ap == 0], max_n=max_n, seed=int(rng.integers(1, 1_000_000))))
        perm_values["rbf_mmd2"].append(estimate_sampled_rbf_mmd2(observed_outcomes[ap == 1], observed_outcomes[ap == 0], max_n=max_n, seed=int(rng.integers(1, 1_000_000))))
    rows = []
    for key, obs in stats.items():
        vals = np.asarray(perm_values[key], float)
        rows.append(
            {
                "test": key,
                "statistic": obs,
                "permutation_p_value": float((1 + np.sum(vals >= obs)) / (1 + len(vals))) if len(vals) else np.nan,
                "null_q95": float(np.quantile(vals, 0.95)) if len(vals) else np.nan,
                "n_permutations": int(n_permutations),
            }
        )
    out = pd.DataFrame(rows)
    out["bh_q_value"] = compute_benjamini_hochberg_q_values(out["permutation_p_value"].to_numpy(float))
    return out


def run_topological_signature_permutation_tests(observed_outcomes, treatment, topology_parameters, n_permutations=99, seed=DEFAULT_RANDOM_SEED):
    rng = np.random.default_rng(seed)
    observed_outcomes = np.asarray(observed_outcomes, float)
    treatment = np.asarray(treatment, int)
    obs_row, _ = compute_superlevel_topology_contrast(observed_outcomes[treatment == 0], observed_outcomes[treatment == 1], "unadjusted_observed", **topology_parameters)
    keys = ["betti0_effect_ref", "betti1_effect_ref", "euler_effect_ref", "betti0_l1", "euler_l1", "ect_l2"]
    obs = {key: float(abs(obs_row[key])) if key.endswith("_effect_ref") else float(obs_row[key]) for key in keys}
    perm_values = {key: [] for key in keys}
    n_permutations = int(n_permutations)
    for _ in range(n_permutations):
        ap = rng.permutation(treatment)
        row, _ = compute_superlevel_topology_contrast(observed_outcomes[ap == 0], observed_outcomes[ap == 1], "permuted", **topology_parameters)
        for key in keys:
            perm_values[key].append(float(abs(row[key])) if key.endswith("_effect_ref") else float(row[key]))
    rows = []
    for key in keys:
        vals = np.asarray(perm_values[key], float)
        rows.append(
            {
                "test": key,
                "statistic": obs[key],
                "permutation_p_value": float((1 + np.sum(vals >= obs[key])) / (1 + len(vals))) if len(vals) else np.nan,
                "null_q95": float(np.quantile(vals, 0.95)) if len(vals) else np.nan,
                "n_permutations": int(n_permutations),
            }
        )
    out = pd.DataFrame(rows)
    out["bh_q_value"] = compute_benjamini_hochberg_q_values(out["permutation_p_value"].to_numpy(float))
    return out


def compute_benjamini_hochberg_q_values(p_values):
    p = np.asarray(p_values, float)
    out = np.full_like(p, np.nan, dtype=float)
    good = np.isfinite(p)
    if good.sum() == 0:
        return out
    idx = np.argsort(p[good])
    sorted_p = p[good][idx]
    m = len(sorted_p)
    q = sorted_p * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty_like(q)
    tmp[idx] = q
    out[np.where(good)[0]] = tmp
    return out


def estimate_cross_fitted_aipw_mean_contrasts(covariates, treatment, observed_outcomes, n_splits=5, seed=DEFAULT_RANDOM_SEED):
    X = build_covariate_design_matrix(covariates)
    standardized_covariates = StandardScaler().fit_transform(X)
    treatment = np.asarray(treatment, int)
    y = np.asarray(observed_outcomes, float)
    kf = KFold(n_splits=min(int(n_splits), max(2, len(treatment) // 50)), shuffle=True, random_state=seed)
    propensity_score = np.zeros(len(treatment), float)
    mu0 = np.zeros_like(y, float)
    mu1 = np.zeros_like(y, float)
    for train, test in kf.split(standardized_covariates):
        ps = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=3000)
        ps.fit(standardized_covariates[train], treatment[train])
        propensity_score[test] = np.clip(ps.predict_proba(standardized_covariates[test])[:, 1], 0.02, 0.98)
        for coord in range(y.shape[1]):
            m0 = Ridge(alpha=1.0)
            m1 = Ridge(alpha=1.0)
            if np.sum(treatment[train] == 0) < 3 or np.sum(treatment[train] == 1) < 3:
                mu0[test, coord] = y[train, coord].mean()
                mu1[test, coord] = y[train, coord].mean()
            else:
                m0.fit(standardized_covariates[train][treatment[train] == 0], y[train, coord][treatment[train] == 0])
                m1.fit(standardized_covariates[train][treatment[train] == 1], y[train, coord][treatment[train] == 1])
                mu0[test, coord] = m0.predict(standardized_covariates[test])
                mu1[test, coord] = m1.predict(standardized_covariates[test])
    rows = []
    for coord in range(y.shape[1]):
        pseudo = mu1[:, coord] - mu0[:, coord] + treatment * (y[:, coord] - mu1[:, coord]) / propensity_score - (1 - treatment) * (y[:, coord] - mu0[:, coord]) / (1 - propensity_score)
        se = float(np.std(pseudo, ddof=1) / math.sqrt(len(pseudo)))
        est = float(np.mean(pseudo))
        rows.append(
            {
                "coordinate": int(coord),
                "estimand": f"AIPW_crossfit_mean_diff_coord_{coord}",
                "estimate": est,
                "standard_error": se,
                "ci95_low": est - 1.96 * se,
                "ci95_high": est + 1.96 * se,
            }
        )
    return pd.DataFrame(rows)


def compute_e_value_for_risk_ratio(rr):
    rr = float(rr)
    if not np.isfinite(rr) or rr <= 0:
        return np.nan
    rr = rr if rr >= 1 else 1.0 / rr
    return float(rr + math.sqrt(rr * max(rr - 1.0, 0.0)))


def compute_binary_outcome_sensitivity(binary_outcome, treatment, ate_weights=None, overlap_weight_values=None, label="binary_outcome"):
    y = np.asarray(binary_outcome, float)
    treatment = np.asarray(treatment, int)
    rows = []
    for scheme, w in [("raw", None), ("IPW", ate_weights), ("overlap", overlap_weight_values)]:
        wt = None if w is None else np.asarray(w, float)
        p1 = compute_weighted_mean(y[treatment == 1], None if wt is None else wt[treatment == 1])
        p0 = compute_weighted_mean(y[treatment == 0], None if wt is None else wt[treatment == 0])
        rr = (p1 + 1e-9) / (p0 + 1e-9)
        rd = p1 - p0
        rows.append(
            {
                "outcome": label,
                "analysis": scheme,
                "risk_treated": float(p1),
                "risk_control": float(p0),
                "risk_difference": float(rd),
                "risk_ratio": float(rr),
                "e_value_for_risk_ratio": compute_e_value_for_risk_ratio(rr),
            }
        )
    return pd.DataFrame(rows)


def compute_overlap_trim_sensitivity(observed_outcomes, treatment, propensity_score, ate_weights, overlap_weight_values, topology_parameters, trims=(0.02, 0.05, 0.10)):
    rows = []
    propensity_score = np.asarray(propensity_score, float)
    treatment = np.asarray(treatment, int)
    for lo in trims:
        mask = (propensity_score >= lo) & (propensity_score <= 1 - lo)
        if mask.sum() == 0 or np.sum(mask & (treatment == 1)) < 10 or np.sum(mask & (treatment == 0)) < 10:
            continue
        for analysis, w in [("raw", None), ("IPW", ate_weights), ("overlap", overlap_weight_values)]:
            wt = None if w is None else np.asarray(w, float)[mask]
            row, _ = compute_superlevel_topology_contrast(
                observed_outcomes[mask & (treatment == 0)],
                observed_outcomes[mask & (treatment == 1)],
                analysis,
                None if wt is None else wt[treatment[mask] == 0],
                None if wt is None else wt[treatment[mask] == 1],
                **topology_parameters,
            )
            rows.append(
                {
                    "trim_rule": f"{lo:.2f} <= e <= {1-lo:.2f}",
                    "analysis": analysis,
                    "n_retained": int(mask.sum()),
                    "fraction_retained": float(mask.mean()),
                    "n_treated": int(np.sum(mask & (treatment == 1))),
                    "n_control": int(np.sum(mask & (treatment == 0))),
                    "mean_diff_coord_0": compute_weighted_mean_difference(observed_outcomes[mask], treatment[mask], wt, coord=0),
                    "mean_diff_coord_1": compute_weighted_mean_difference(observed_outcomes[mask], treatment[mask], wt, coord=1),
                    "betti0_effect_ref": row["betti0_effect_ref"],
                    "euler_effect_ref": row["euler_effect_ref"],
                    "betti0_l1": row["betti0_l1"],
                    "euler_l1": row["euler_l1"],
                    "ect_l2": row["ect_l2"],
                }
            )
    return pd.DataFrame(rows)


def default_diagnostic_topology_parameters(bounds=None):
    kwargs = {
        "bins": 48,
        "sigma": 3.0,
        "level_fracs": default_superlevel_fractions(n_levels=9),
        "n_dirs": 4,
        "n_slices": 10,
        "min_pixels": 3,
    }
    if bounds is not None:
        kwargs["bounds"] = bounds
    return kwargs


def compute_design_and_sensitivity_diagnostics(
    covariates,
    covariate_design,
    treatment,
    observed_outcomes,
    propensity_score,
    ate_weights,
    overlap_weight_values,
    *,
    bounds=None,
    n_permutations=99,
    seed=DEFAULT_RANDOM_SEED,
    include_binary_sensitivity=False,
    binary_outcome=None,
    binary_label="binary_outcome",
):
    """Compute balance, overlap, outcome-distribution, topology, and sensitivity diagnostics."""
    n_permutations = int(max(0, n_permutations))
    topology_permutations = n_permutations
    distribution_perm = n_permutations
    balance_diag = compute_covariate_balance_diagnostics(covariate_design, treatment, overlap_weight_values, seed=seed)
    balance_omnibus = pd.concat(
        [
            run_omnibus_balance_permutation_test(covariate_design, treatment, None, n_permutations=n_permutations, seed=seed + 1),
            run_omnibus_balance_permutation_test(covariate_design, treatment, overlap_weight_values, n_permutations=n_permutations, seed=seed + 2),
        ],
        ignore_index=True,
    )
    overlap_diagnostic_table = compute_propensity_overlap_diagnostics(propensity_score, treatment, ate_weights, overlap_weight_values)
    distribution_test_table = run_outcome_distribution_permutation_tests(observed_outcomes, treatment, n_permutations=distribution_perm, max_n=450, seed=seed + 3)
    topology_parameters = default_diagnostic_topology_parameters(bounds)
    topology_test_table = run_topological_signature_permutation_tests(observed_outcomes, treatment, topology_parameters, n_permutations=topology_permutations, seed=seed + 4)
    aipw_table = estimate_cross_fitted_aipw_mean_contrasts(covariates, treatment, observed_outcomes, n_splits=5, seed=seed + 5)
    trim_sensitivity_table = compute_overlap_trim_sensitivity(observed_outcomes, treatment, propensity_score, ate_weights, overlap_weight_values, topology_parameters)
    binary_sensitivity_table = (
        compute_binary_outcome_sensitivity(binary_outcome, treatment, ate_weights, overlap_weight_values, label=binary_label)
        if include_binary_sensitivity and binary_outcome is not None
        else pd.DataFrame()
    )
    return {
        "covariate_balance_diagnostics": balance_diag,
        "design_balance_omnibus": balance_omnibus,
        "design_overlap": overlap_diagnostic_table,
        "design_distribution_tests": distribution_test_table,
        "design_topology_tests": topology_test_table,
        "design_aipw": aipw_table,
        "design_trim": trim_sensitivity_table,
        "design_binary_sensitivity": binary_sensitivity_table,
    }


def assign_quantile_bins(values, n_bins=6):
    s = pd.Series(np.asarray(values, float))
    try:
        bins = pd.qcut(s, q=n_bins, duplicates="drop")
    except ValueError:
        bins = pd.cut(s, bins=n_bins, duplicates="drop")
    codes = bins.cat.codes.to_numpy()
    labels = [str(x) for x in bins.cat.categories]
    return codes, labels


def count_foreground_components(binary):
    _, n = label(binary.astype(bool), structure=FOREGROUND_CONNECTIVITY)
    return int(n)


def count_enclosed_background_components(binary):
    bg = ~binary.astype(bool)
    lab, n = label(bg, structure=BACKGROUND_CONNECTIVITY)
    border_ids = set(np.unique(np.r_[lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
    holes = sum((idx not in border_ids) for idx in range(1, n + 1))
    return int(holes)


def compute_euler_characteristic(binary):
    return count_foreground_components(binary) - count_enclosed_background_components(binary)


def remove_small_foreground_components(binary, min_pixels=3):
    if min_pixels <= 1:
        return binary.astype(bool)
    lab, n = label(binary.astype(bool), structure=FOREGROUND_CONNECTIVITY)
    out = np.zeros_like(binary, dtype=bool)
    for idx in range(1, n + 1):
        comp = lab == idx
        if int(comp.sum()) >= min_pixels:
            out |= comp
    return out


def estimate_smoothed_density_grid(points, weights=None, bounds=DEFAULT_TOPOLOGY_BOUNDS, bins=96, sigma=1.2):
    """Estimate a normalized two-dimensional Gaussian-smoothed histogram density."""
    pts = np.asarray(points, float)
    if len(pts) == 0:
        raise ValueError("Cannot estimate a density from an empty point cloud.")
    xmin, xmax, ymin, ymax = bounds
    w = None if weights is None else np.asarray(weights, float)
    H, xedges, yedges = np.histogram2d(
        pts[:, 0],
        pts[:, 1],
        bins=[bins, bins],
        range=[[xmin, xmax], [ymin, ymax]],
        weights=w,
    )
    H = H.T
    Hs = gaussian_filter(H.astype(float), sigma=sigma)
    dx = (xmax - xmin) / bins
    dy = (ymax - ymin) / bins
    total = float(Hs.sum() * dx * dy)
    if total > 0:
        Hs = Hs / total
    xcent = 0.5 * (xedges[:-1] + xedges[1:])
    ycent = 0.5 * (yedges[:-1] + yedges[1:])
    return {"density": Hs, "xcent": xcent, "ycent": ycent, "dx": dx, "dy": dy}


def default_superlevel_fractions(n_levels=19, high=0.62, low=0.08):
    return np.linspace(high, low, int(n_levels))


def compute_superlevel_betti_curves(density, levels, min_pixels=3):
    b0 = []
    b1 = []
    chi = []
    for level_value in levels:
        binary = remove_small_foreground_components(density >= level_value, min_pixels=min_pixels)
        b0.append(count_foreground_components(binary))
        b1.append(count_enclosed_background_components(binary))
        chi.append(b0[-1] - b1[-1])
    return {
        "betti0": np.asarray(b0, float),
        "betti1": np.asarray(b1, float),
        "euler": np.asarray(chi, float),
    }


def compute_sliced_euler_signature_from_density(density, xcent, ycent, levels, n_dirs=8, n_slices=24, min_pixels=3):
    yy, xx = np.meshgrid(ycent, xcent, indexing="ij")
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    signature = []
    thetas = np.linspace(0.0, np.pi, num=int(n_dirs), endpoint=False)
    projection_masks = []
    for th in thetas:
        v = np.array([math.cos(th), math.sin(th)])
        proj = coords @ v
        cuts = np.linspace(float(proj.min()), float(proj.max()), num=int(n_slices))
        projection_masks.append([proj <= cut for cut in cuts])
    for level_value in levels:
        superlevel_mask = remove_small_foreground_components(density >= level_value, min_pixels=min_pixels)
        flat_superlevel_mask = superlevel_mask.ravel()
        for masks in projection_masks:
            for mask in masks:
                sliced = (flat_superlevel_mask & mask).reshape(superlevel_mask.shape)
                signature.append(compute_euler_characteristic(sliced))
    return np.asarray(signature, float)


def compute_superlevel_topological_signature(
    density_obj,
    levels,
    n_dirs=8,
    n_slices=24,
    min_pixels=3,
):
    density = density_obj["density"]
    curves = compute_superlevel_betti_curves(density, levels, min_pixels=min_pixels)
    ect = compute_sliced_euler_signature_from_density(
        density,
        density_obj["xcent"],
        density_obj["ycent"],
        levels,
        n_dirs=n_dirs,
        n_slices=n_slices,
        min_pixels=min_pixels,
    )
    return {**curves, "ect": ect}


def mean_absolute_signature_difference(left_signature, right_signature):
    left_signature = np.asarray(left_signature, float)
    right_signature = np.asarray(right_signature, float)
    return float(np.mean(np.abs(left_signature - right_signature)))


def root_mean_square_signature_difference(left_signature, right_signature):
    left_signature = np.asarray(left_signature, float)
    right_signature = np.asarray(right_signature, float)
    return float(np.linalg.norm(left_signature - right_signature) / math.sqrt(max(len(left_signature), 1)))


def compute_level_grid_margin(density, levels):
    values = np.asarray(density, float).ravel()
    levels = np.asarray(levels, float)
    if len(values) == 0 or len(levels) == 0:
        return 0.0
    return float(np.min(np.abs(values[:, None] - levels[None, :])))


def sliced_euler_effect_from_details(det):
    return np.asarray(det["treatment"]["ect"], float) - np.asarray(det["control"]["ect"], float)


def decode_sliced_euler_coordinate(idx, n_dirs, n_slices):
    per_level = int(n_dirs) * int(n_slices)
    level_idx = int(idx // per_level)
    rem = int(idx % per_level)
    direction_idx = int(rem // int(n_slices))
    slice_idx = int(rem % int(n_slices))
    return level_idx, direction_idx, slice_idx


def compare_sliced_euler_effect_to_reference(oracle_det, current_det, atol=1e-9):
    if oracle_det is None or current_det is None:
        return {
            "same_slice_wise_ect_effect_as_oracle": False,
            "fraction_slices_same_ect_effect": np.nan,
            "n_ect_slices_checked": 0,
            "max_abs_ect_slice_effect_delta": np.nan,
            "ect_slice_effect_l2_delta_vs_oracle": np.nan,
        }
    oracle_effect = sliced_euler_effect_from_details(oracle_det)
    current_effect = sliced_euler_effect_from_details(current_det)
    if len(oracle_effect) != len(current_effect) or len(oracle_effect) == 0:
        return {
            "same_slice_wise_ect_effect_as_oracle": False,
            "fraction_slices_same_ect_effect": 0.0,
            "n_ect_slices_checked": int(min(len(oracle_effect), len(current_effect))),
            "max_abs_ect_slice_effect_delta": np.nan,
            "ect_slice_effect_l2_delta_vs_oracle": np.nan,
        }
    delta = current_effect - oracle_effect
    same = np.isclose(delta, 0.0, atol=atol, rtol=0.0)
    return {
        "same_slice_wise_ect_effect_as_oracle": bool(np.all(same)),
        "fraction_slices_same_ect_effect": float(np.mean(same)),
        "n_ect_slices_checked": int(len(same)),
        "max_abs_ect_slice_effect_delta": float(np.max(np.abs(delta))),
        "ect_slice_effect_l2_delta_vs_oracle": root_mean_square_signature_difference(current_effect, oracle_effect),
    }


def build_sliced_euler_coordinate_check_table(details_by_analysis, *, bin_code=None, score_bin=None, atol=1e-9):
    if not details_by_analysis or "oracle_do" not in details_by_analysis:
        return pd.DataFrame()
    oracle_det = details_by_analysis["oracle_do"]
    oracle_effect = sliced_euler_effect_from_details(oracle_det)
    n_dirs = int(oracle_det.get("n_dirs", 1))
    n_slices = int(oracle_det.get("n_slices", max(1, len(oracle_effect))))
    levels = np.asarray(oracle_det.get("levels", []), float)
    rows = []
    for analysis, det in details_by_analysis.items():
        current_effect = sliced_euler_effect_from_details(det)
        n = min(len(oracle_effect), len(current_effect))
        for idx in range(n):
            level_idx, direction_idx, slice_idx = decode_sliced_euler_coordinate(idx, n_dirs, n_slices)
            delta = float(current_effect[idx] - oracle_effect[idx])
            row = {
                "analysis": analysis,
                "ect_coordinate": int(idx),
                "level_index": int(level_idx),
                "direction_index": int(direction_idx),
                "slice_index": int(slice_idx),
                "level_value": float(levels[level_idx]) if level_idx < len(levels) else np.nan,
                "oracle_ect_effect": float(oracle_effect[idx]),
                "analysis_ect_effect": float(current_effect[idx]),
                "ect_effect_delta_from_oracle": delta,
                "same_slice_ect_effect_as_oracle": bool(abs(delta) <= atol),
            }
            if bin_code is not None:
                row["bin_code"] = int(bin_code)
                row["score_bin"] = score_bin
            rows.append(row)
    return pd.DataFrame(rows)


def build_binned_sliced_euler_coordinate_check_table(score_details, score_topology_df, atol=1e-9):
    if not score_details or score_topology_df is None or score_topology_df.empty:
        return pd.DataFrame()
    tables = []
    for code, g in score_topology_df.groupby("bin_code"):
        details_by_analysis = {
            analysis: score_details[(int(code), analysis)]
            for analysis in sorted(set(g["analysis"]))
            if (int(code), analysis) in score_details
        }
        if "oracle_do" not in details_by_analysis:
            continue
        score_bin = g["score_bin"].iloc[0]
        table = build_sliced_euler_coordinate_check_table(
            details_by_analysis,
            bin_code=int(code),
            score_bin=score_bin,
            atol=atol,
        )
        if not table.empty:
            tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def compute_superlevel_topology_contrast(
    ctrl_points,
    trt_points,
    label_name,
    ctrl_weights=None,
    trt_weights=None,
    bounds=DEFAULT_TOPOLOGY_BOUNDS,
    bins=96,
    sigma=1.2,
    level_fracs=None,
    n_dirs=8,
    n_slices=24,
    min_pixels=3,
):
    """Compute finite superlevel-set topological contrasts between two outcome point clouds."""
    if level_fracs is None:
        level_fracs = default_superlevel_fractions()
    d0 = estimate_smoothed_density_grid(ctrl_points, ctrl_weights, bounds=bounds, bins=bins, sigma=sigma)
    d1 = estimate_smoothed_density_grid(trt_points, trt_weights, bounds=bounds, bins=bins, sigma=sigma)
    pooled_max = max(float(d0["density"].max()), float(d1["density"].max()), 1e-12)
    levels = np.asarray(level_fracs, float) * pooled_max
    s0 = compute_superlevel_topological_signature(d0, levels, n_dirs=n_dirs, n_slices=n_slices, min_pixels=min_pixels)
    s1 = compute_superlevel_topological_signature(d1, levels, n_dirs=n_dirs, n_slices=n_slices, min_pixels=min_pixels)
    ref_idx = len(levels) // 2
    row = {
        "analysis": label_name,
        "control_n": int(len(ctrl_points)),
        "treatment_n": int(len(trt_points)),
        "n_levels": int(len(levels)),
        "reference_level_fraction": float(np.asarray(level_fracs)[ref_idx]),
        "control_betti0_ref": float(s0["betti0"][ref_idx]),
        "treatment_betti0_ref": float(s1["betti0"][ref_idx]),
        "betti0_effect_ref": float(s1["betti0"][ref_idx] - s0["betti0"][ref_idx]),
        "control_betti1_ref": float(s0["betti1"][ref_idx]),
        "treatment_betti1_ref": float(s1["betti1"][ref_idx]),
        "betti1_effect_ref": float(s1["betti1"][ref_idx] - s0["betti1"][ref_idx]),
        "control_euler_ref": float(s0["euler"][ref_idx]),
        "treatment_euler_ref": float(s1["euler"][ref_idx]),
        "euler_effect_ref": float(s1["euler"][ref_idx] - s0["euler"][ref_idx]),
        "betti0_l1": mean_absolute_signature_difference(s1["betti0"], s0["betti0"]),
        "betti0_l2": root_mean_square_signature_difference(s1["betti0"], s0["betti0"]),
        "betti1_l1": mean_absolute_signature_difference(s1["betti1"], s0["betti1"]),
        "betti1_l2": root_mean_square_signature_difference(s1["betti1"], s0["betti1"]),
        "euler_l1": mean_absolute_signature_difference(s1["euler"], s0["euler"]),
        "euler_l2": root_mean_square_signature_difference(s1["euler"], s0["euler"]),
        "ect_l2": root_mean_square_signature_difference(s1["ect"], s0["ect"]),
        "control_betti0_auc": float(np.mean(s0["betti0"])),
        "treatment_betti0_auc": float(np.mean(s1["betti0"])),
        "control_betti1_auc": float(np.mean(s0["betti1"])),
        "treatment_betti1_auc": float(np.mean(s1["betti1"])),
        "control_level_margin": compute_level_grid_margin(d0["density"], levels),
        "treatment_level_margin": compute_level_grid_margin(d1["density"], levels),
    }
    row["min_level_margin"] = float(min(row["control_level_margin"], row["treatment_level_margin"]))
    row["min_level_margin_relative_to_max_density"] = float(row["min_level_margin"] / pooled_max)
    return row, {
        "control": s0,
        "treatment": s1,
        "density0": d0,
        "density1": d1,
        "levels": levels,
        "level_fracs": np.asarray(level_fracs, float),
        "n_dirs": int(n_dirs),
        "n_slices": int(n_slices),
    }


def generate_single_component_outcomes(unobserved_group, structural_score, rng):
    unobserved_group = np.asarray(unobserved_group, int)
    structural_score = np.asarray(structural_score, float)
    latent_shift = 0.38 * (2 * unobserved_group - 1) + 0.16 * np.tanh(structural_score)
    x = rng.normal(latent_shift, 0.82, size=len(unobserved_group))
    x = np.clip(x, -2.15, 2.15)
    y = rng.normal(0.0, 0.21, size=len(unobserved_group))
    y = np.clip(y, -0.58, 0.58)
    return np.column_stack([x, y])


def generate_two_component_outcomes(unobserved_group, structural_score, rng):
    unobserved_group = np.asarray(unobserved_group, int)
    structural_score = np.asarray(structural_score, float)
    latent_shift = 0.38 * (2 * unobserved_group - 1) + 0.16 * np.tanh(structural_score)
    side = np.where(rng.random(len(unobserved_group)) < 0.5, -1.0, 1.0)
    x = side * 1.45 + latent_shift + rng.normal(0.0, 0.18, size=len(unobserved_group))
    y = rng.normal(0.0, 0.20, size=len(unobserved_group))
    return np.column_stack([x, y])


def assign_structural_benchmark_treatment(structural_score, unobserved_group, rng):
    structural_score = np.asarray(structural_score, float)
    unobserved_group = np.asarray(unobserved_group, int)
    true_propensity_score = expit(-0.20 + 0.95 * structural_score + 1.35 * unobserved_group)
    treatment = (rng.random(len(unobserved_group)) < true_propensity_score).astype(int)
    return treatment, true_propensity_score


def build_synthetic_structural_benchmark(n: int = 3600, seed: int = DEFAULT_RANDOM_SEED) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, size=n)
    structural_score = 0.90 * z
    unobserved_group = rng.binomial(1, 0.5, size=n)
    control_potential_outcomes = generate_single_component_outcomes(unobserved_group, structural_score, rng)
    treated_potential_outcomes = generate_two_component_outcomes(unobserved_group, structural_score, rng)
    treatment, true_propensity_score = assign_structural_benchmark_treatment(structural_score, unobserved_group, rng)
    observed_outcomes = np.where(treatment[:, None] == 1, treated_potential_outcomes, control_potential_outcomes)
    covariates = pd.DataFrame({"z": z})
    return {
        "name": "synthetic_exact_superlevel",
        "X": covariates,
        "eta_true": structural_score,
        "u": unobserved_group,
        "a": treatment,
        "e_true": true_propensity_score,
        "y0": control_potential_outcomes,
        "y1": treated_potential_outcomes,
        "y_obs": observed_outcomes,
    }


def build_breast_cancer_semisynthetic_benchmark(seed: int = DEFAULT_RANDOM_SEED) -> DatasetBundle:
    rng = np.random.default_rng(seed + 1)
    ds = load_breast_cancer(as_frame=True)
    covariates = ds.data.copy()
    standardized_covariates = StandardScaler().fit_transform(covariates)
    beta = np.zeros(standardized_covariates.shape[1])
    beta[:8] = np.array([0.85, -0.75, 0.65, -0.55, 0.40, -0.30, 0.25, -0.20])
    raw_structural_score = standardized_covariates @ beta
    structural_score = (raw_structural_score - raw_structural_score.mean()) / raw_structural_score.std()
    unobserved_group = rng.binomial(1, 0.5, size=len(covariates))
    control_potential_outcomes = generate_single_component_outcomes(unobserved_group, structural_score, rng)
    treated_potential_outcomes = generate_two_component_outcomes(unobserved_group, structural_score, rng)
    treatment, true_propensity_score = assign_structural_benchmark_treatment(structural_score, unobserved_group, rng)
    observed_outcomes = np.where(treatment[:, None] == 1, treated_potential_outcomes, control_potential_outcomes)
    return {
        "name": "real_based_semisynthetic_breast_cancer_superlevel",
        "X": covariates,
        "eta_true": structural_score,
        "u": unobserved_group,
        "a": treatment,
        "e_true": true_propensity_score,
        "y0": control_potential_outcomes,
        "y1": treated_potential_outcomes,
        "y_obs": observed_outcomes,
    }


def build_jobs_ii_randomized_application(jobs_csv: Path | None = None, jobs_url: str = JOBS_II_RDATASETS_CSV_URL) -> DatasetBundle:
    if jobs_csv is not None and Path(jobs_csv).exists():
        source = str(Path(jobs_csv))
        jobs = pd.read_csv(jobs_csv)
    else:
        source = str(jobs_url)
        jobs = pd.read_csv(jobs_url)
    jobs = jobs.drop(columns=["rownames"], errors="ignore")
    required = [
        "treat",
        "job_seek",
        "depress2",
        "econ_hard",
        "depress1",
        "sex",
        "age",
        "occp",
        "marital",
        "nonwhite",
        "educ",
        "income",
    ]
    missing = [col for col in required if col not in jobs.columns]
    if missing:
        raise ValueError("JOBS II data are missing required columns: " + ", ".join(missing))
    jobs = jobs.dropna(subset=required).copy()
    treatment = jobs["treat"].astype(int).to_numpy()
    if set(np.unique(treatment)) != {0, 1}:
        raise ValueError("JOBS II treatment column must be binary with values 0/1.")

    job_seek = pd.to_numeric(jobs["job_seek"], errors="coerce")
    neg_depress2 = -pd.to_numeric(jobs["depress2"], errors="coerce")
    keep = job_seek.notna() & neg_depress2.notna()
    jobs = jobs.loc[keep].copy()
    treatment = jobs["treat"].astype(int).to_numpy()
    job_seek = job_seek.loc[keep]
    neg_depress2 = neg_depress2.loc[keep]
    job_seek_z = ((job_seek - job_seek.mean()) / job_seek.std(ddof=1)).to_numpy(float)
    neg_depress2_z = ((neg_depress2 - neg_depress2.mean()) / neg_depress2.std(ddof=1)).to_numpy(float)
    observed_outcomes = np.column_stack([job_seek_z, neg_depress2_z])
    raw_outcomes = np.column_stack([job_seek.to_numpy(float), neg_depress2.to_numpy(float)])
    bounds = (
        float(min(-4.5, observed_outcomes[:, 0].min() - 0.2)),
        float(max(2.0, observed_outcomes[:, 0].max() + 0.2)),
        float(min(-5.25, observed_outcomes[:, 1].min() - 0.2)),
        float(max(2.0, observed_outcomes[:, 1].max() + 0.2)),
    )
    z_cols = ["econ_hard", "depress1", "sex", "age", "occp", "marital", "nonwhite", "educ", "income"]
    covariates = jobs[z_cols].copy()
    info = pd.DataFrame(
        [
            {
                "source_dataset": "Rdatasets mediation/jobs.csv",
                "source_used_by_code": source,
                "formal_archive": "ICPSR 2739: Jobs II Preventive Intervention for Unemployed Job Seekers, 1991-1993: Southeast Michigan",
                "formal_archive_url": JOBS_II_ARCHIVE_URL,
                "formal_archive_doi": JOBS_II_ARCHIVE_DOI,
                "analysis_rows": int(len(jobs)),
                "treated_jobs_ii": int(treatment.sum()),
                "control_jobs_ii": int((1 - treatment).sum()),
                "treated_fraction": float(treatment.mean()),
                "treatment_definition": "JOBS II intervention assignment, treat == 1",
                "outcome_1": "standardized job_seek score; higher means stronger job-search self-efficacy",
                "outcome_2": "standardized negative depress2 score; higher means less depression at follow-up",
                "scientific_scope": (
                    "randomized empirical application; no unit-level oracle potential outcomes, "
                    "not a structural unobserved-confounding benchmark"
                ),
            }
        ]
    )
    return {
        "name": "jobs_ii_randomized_job_search_depression_superlevel",
        "mode": "randomized",
        "X": covariates,
        "a": treatment,
        "y_obs": observed_outcomes,
        "y_raw": raw_outcomes,
        "jobs_info": info,
        "cohort": jobs,
        "topology_bounds": bounds,
        "treatment_label": "JOBS II intervention",
        "control_label": "control",
        "outcome_axis_labels": ["standardized job_seek", "standardized -depress2"],
        "raw_outcome_labels": ["job_seek", "-depress2"],
    }


def build_eicu_observational_application(input_data_dir: Path = DEFAULT_EICU_DIRECTORY) -> DatasetBundle:
    input_data_dir = Path(input_data_dir)
    prediction_file = input_data_dir / "apachePredVar.csv.gz"
    result_file = input_data_dir / "apachePatientResult.csv.gz"
    required = [prediction_file, result_file]
    if not all(p.exists() for p in required):
        missing = [str(p) for p in required if not p.exists()]
        raise FileNotFoundError("Missing required eICU files: " + "; ".join(missing))

    prediction_columns = [
        "patientunitstayid",
        "ventday1",
        "gender",
        "age",
        "teachtype",
        "region",
        "bedcount",
        "admitsource",
        "admitdiagnosis",
        "diedinhospital",
        "aids",
        "hepaticfailure",
        "lymphoma",
        "metastaticcancer",
        "leukemia",
        "immunosuppression",
        "cirrhosis",
        "diabetes",
        "readmit",
        "meds",
        "verbal",
        "motor",
        "eyes",
        "creatinine",
        "pao2",
        "fio2",
    ]
    result_columns = [
        "patientunitstayid",
        "apacheversion",
        "acutephysiologyscore",
        "apachescore",
        "predictedicumortality",
        "actualicumortality",
        "predictediculos",
        "actualiculos",
        "predictedhospitalmortality",
        "actualhospitalmortality",
        "predictedhospitallos",
        "actualhospitallos",
    ]
    prediction_table = pd.read_csv(prediction_file, compression="gzip", usecols=prediction_columns, low_memory=False)
    results = pd.read_csv(result_file, compression="gzip", usecols=result_columns, low_memory=False)
    results = results[results["apacheversion"].astype(str).str.lower().eq("iva")].drop_duplicates("patientunitstayid")
    raw_n = int(len(results))
    df = prediction_table.drop_duplicates("patientunitstayid").merge(results, on="patientunitstayid", how="inner")
    merged_n = int(len(df))
    numeric_cols = [
        "age",
        "ventday1",
        "actualiculos",
        "actualhospitallos",
        "predictedhospitalmortality",
        "predictedicumortality",
        "acutephysiologyscore",
        "apachescore",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = (
        df["age"].between(18, 89)
        & df["predictedhospitalmortality"].between(0, 1)
        & df["predictedicumortality"].between(0, 1)
        & (df["actualiculos"] > 0)
        & (df["actualhospitallos"] > 0)
        & df["ventday1"].isin([0, 1])
    )
    df = df.loc[keep].copy()
    diagnosis_counts = df["admitdiagnosis"].astype(str).value_counts()
    common_diagnoses = set(diagnosis_counts[diagnosis_counts >= 250].index)
    df["admitdiagnosis_grouped"] = df["admitdiagnosis"].astype(str).where(
        df["admitdiagnosis"].astype(str).isin(common_diagnoses),
        "OTHER",
    )
    x_cols = [
        "age",
        "gender",
        "teachtype",
        "region",
        "bedcount",
        "admitsource",
        "admitdiagnosis_grouped",
        "aids",
        "hepaticfailure",
        "lymphoma",
        "metastaticcancer",
        "leukemia",
        "immunosuppression",
        "cirrhosis",
        "diabetes",
        "readmit",
        "meds",
        "verbal",
        "motor",
        "eyes",
        "creatinine",
        "pao2",
        "fio2",
        "acutephysiologyscore",
        "apachescore",
    ]
    covariates = df[x_cols].copy()
    treatment = df["ventday1"].astype(int).to_numpy()
    raw_outcomes = np.column_stack(
        [
            np.log1p(df["actualiculos"].to_numpy(float)),
            np.log1p(df["actualhospitallos"].to_numpy(float)),
        ]
    )
    y_center = raw_outcomes.mean(axis=0)
    y_scale = raw_outcomes.std(axis=0)
    y_scale = np.where(y_scale <= 0, 1.0, y_scale)
    observed_outcomes = (raw_outcomes - y_center) / y_scale
    mortality = df["actualhospitalmortality"].astype(str).str.upper().eq("EXPIRED").astype(float).to_numpy()
    info = pd.DataFrame(
        [
            {
                "raw_apache_iva_rows": raw_n,
                "merged_rows": merged_n,
                "analysis_rows": int(len(df)),
                "treated_ventday1": int(treatment.sum()),
                "control_no_ventday1": int((1 - treatment).sum()),
                "treated_fraction": float(treatment.mean()),
                "n_common_admitdiagnosis_levels": int(len(common_diagnoses)),
                "treatment_definition": "APACHE ventday1 == 1 (ventilated during ICU day 1)",
                "outcome_1": "standardized log1p(actual ICU LOS)",
                "outcome_2": "standardized log1p(actual hospital LOS)",
                "scientific_scope": "observational application; no oracle potential outcomes available",
            }
        ]
    )
    return {
        "name": "eicu_observational_day1_ventilation_los_superlevel",
        "mode": "observational",
        "X": covariates,
        "a": treatment,
        "y_obs": observed_outcomes,
        "y_raw": raw_outcomes,
        "mortality": mortality,
        "patientunitstayid": df["patientunitstayid"].to_numpy(),
        "cohort": df,
        "eicu_info": info,
        "outcome_center": y_center,
        "outcome_scale": y_scale,
        "topology_bounds": (-3.5, 3.5, -3.5, 3.5),
        "treatment_label": "day-1 ventilation",
        "control_label": "no day-1 ventilation",
    }


def compute_latent_variable_diagnostics(structural_score, unobserved_group, treatment, control_potential_outcomes, treated_potential_outcomes, n_bins=6):
    codes, labels = assign_quantile_bins(structural_score, n_bins=n_bins)
    rows = []
    for code, label_text in enumerate(labels):
        mask = codes == code
        if mask.sum() == 0 or treatment[mask].min() == treatment[mask].max():
            continue
        mt = mask & (treatment == 1)
        mc = mask & (treatment == 0)
        rows.append(
            {
                "eta_bin": label_text,
                "n": int(mask.sum()),
                "n_treated": int(mt.sum()),
                "n_control": int(mc.sum()),
                "p_u1_treated": float(unobserved_group[mt].mean()),
                "p_u1_control": float(unobserved_group[mc].mean()),
                "delta_p_u1": float(unobserved_group[mt].mean() - unobserved_group[mc].mean()),
                "delta_mean_x_y0": float(control_potential_outcomes[mt, 0].mean() - control_potential_outcomes[mc, 0].mean()),
                "delta_mean_x_y1": float(treated_potential_outcomes[mt, 0].mean() - treated_potential_outcomes[mc, 0].mean()),
                "energy_y0_treated_vs_control": estimate_sampled_energy_distance(control_potential_outcomes[mt], control_potential_outcomes[mc], seed=301 + code),
                "energy_y1_treated_vs_control": estimate_sampled_energy_distance(treated_potential_outcomes[mt], treated_potential_outcomes[mc], seed=401 + code),
            }
        )
    return pd.DataFrame(rows)


def summarize_structural_benchmark_mean_effects(control_potential_outcomes, treated_potential_outcomes, observed_outcomes, treatment, ate_weights, overlap_weight_values):
    true_ate_x = float(np.mean(treated_potential_outcomes[:, 0] - control_potential_outcomes[:, 0]))
    true_ate_y = float(np.mean(treated_potential_outcomes[:, 1] - control_potential_outcomes[:, 1]))
    naive_x = float(observed_outcomes[treatment == 1, 0].mean() - observed_outcomes[treatment == 0, 0].mean())
    naive_y = float(observed_outcomes[treatment == 1, 1].mean() - observed_outcomes[treatment == 0, 1].mean())
    ipw_x = float(compute_weighted_mean_difference(observed_outcomes, treatment, ate_weights, coord=0))
    ipw_y = float(compute_weighted_mean_difference(observed_outcomes, treatment, ate_weights, coord=1))
    ov_x = float(compute_weighted_mean_difference(observed_outcomes, treatment, overlap_weight_values, coord=0))
    ov_y = float(compute_weighted_mean_difference(observed_outcomes, treatment, overlap_weight_values, coord=1))
    return pd.DataFrame(
        [
            {"estimand": "true_ATE_mean_x_oracle", "value": true_ate_x},
            {"estimand": "true_ATE_mean_y_oracle", "value": true_ate_y},
            {"estimand": "naive_observational_diff_mean_x", "value": naive_x},
            {"estimand": "ipw_observed_covariate_diff_mean_x", "value": ipw_x},
            {"estimand": "overlap_weighted_observed_covariate_diff_mean_x", "value": ov_x},
            {"estimand": "naive_minus_true_bias_x", "value": naive_x - true_ate_x},
            {"estimand": "ipw_minus_true_bias_x", "value": ipw_x - true_ate_x},
            {"estimand": "overlap_minus_true_bias_x", "value": ov_x - true_ate_x},
            {"estimand": "naive_observational_diff_mean_y", "value": naive_y},
            {"estimand": "ipw_observed_covariate_diff_mean_y", "value": ipw_y},
            {"estimand": "overlap_weighted_observed_covariate_diff_mean_y", "value": ov_y},
        ]
    )


def summarize_structural_benchmark_score_bin_effects(score, control_potential_outcomes, treated_potential_outcomes, observed_outcomes, treatment, propensity_score, n_bins=6):
    codes, labels = assign_quantile_bins(score, n_bins=n_bins)
    rows = []
    for code, label_text in enumerate(labels):
        mask = codes == code
        if mask.sum() == 0 or treatment[mask].min() == treatment[mask].max():
            continue
        mb = mask
        mt = mask & (treatment == 1)
        mc = mask & (treatment == 0)
        wb = compute_inverse_probability_weights(treatment[mb], propensity_score[mb], stabilized=True)
        rows.append(
            {
                "score_bin": label_text,
                "n": int(mb.sum()),
                "n_treated": int(mt.sum()),
                "n_control": int(mc.sum()),
                "bin_weight": float(mb.mean()),
                "true_CATE_x_oracle": float(np.mean(treated_potential_outcomes[mb, 0] - control_potential_outcomes[mb, 0])),
                "true_CATE_y_oracle": float(np.mean(treated_potential_outcomes[mb, 1] - control_potential_outcomes[mb, 1])),
                "naive_diff_x": float(observed_outcomes[mt, 0].mean() - observed_outcomes[mc, 0].mean()),
                "IPW_diff_x": compute_weighted_mean_difference(observed_outcomes[mb], treatment[mb], wb, coord=0),
                "naive_bias_x": float(observed_outcomes[mt, 0].mean() - observed_outcomes[mc, 0].mean() - np.mean(treated_potential_outcomes[mb, 0] - control_potential_outcomes[mb, 0])),
                "IPW_bias_x": compute_weighted_mean_difference(observed_outcomes[mb], treatment[mb], wb, coord=0) - float(np.mean(treated_potential_outcomes[mb, 0] - control_potential_outcomes[mb, 0])),
            }
        )
    return pd.DataFrame(rows)


def compute_structural_benchmark_topology_contrasts(control_potential_outcomes, treated_potential_outcomes, observed_outcomes, treatment, ate_weights, overlap_weight_values, topology_parameters):
    rows = []
    topology_details = {}
    configs = [
        ("oracle_do", control_potential_outcomes, treated_potential_outcomes, None, None),
        ("unadjusted_observed", observed_outcomes[treatment == 0], observed_outcomes[treatment == 1], None, None),
        (
            "ipw_adjusted_observed_density",
            observed_outcomes[treatment == 0],
            observed_outcomes[treatment == 1],
            ate_weights[treatment == 0],
            ate_weights[treatment == 1],
        ),
        (
            "overlap_weighted_observed_density",
            observed_outcomes[treatment == 0],
            observed_outcomes[treatment == 1],
            overlap_weight_values[treatment == 0],
            overlap_weight_values[treatment == 1],
        ),
    ]
    for label_name, p0, p1, ww0, ww1 in configs:
        row, det = compute_superlevel_topology_contrast(p0, p1, label_name, ww0, ww1, **topology_parameters)
        rows.append(row)
        topology_details[label_name] = det
    return pd.DataFrame(rows), topology_details


def compute_structural_benchmark_binned_topology_contrasts(
    control_potential_outcomes,
    treated_potential_outcomes,
    observed_outcomes,
    treatment,
    score,
    ate_weights,
    overlap_weight_values,
    n_bins=5,
    topology_parameters=None,
    min_arm=20,
    return_details=False,
):
    if topology_parameters is None:
        topology_parameters = {}
    codes, labels = assign_quantile_bins(score, n_bins=n_bins)
    rows = []
    topology_details = {}
    for code, label_text in enumerate(labels):
        mask = codes == code
        if mask.sum() == 0:
            continue
        mt = mask & (treatment == 1)
        mc = mask & (treatment == 0)
        if mt.sum() < min_arm or mc.sum() < min_arm:
            continue
        bin_weight = float(mask.mean())
        configs = [
            ("oracle_do", control_potential_outcomes[mask], treated_potential_outcomes[mask], None, None),
            ("unadjusted_observed", observed_outcomes[mc], observed_outcomes[mt], None, None),
            ("ipw_adjusted_observed_density", observed_outcomes[mc], observed_outcomes[mt], ate_weights[mc], ate_weights[mt]),
            ("overlap_weighted_observed_density", observed_outcomes[mc], observed_outcomes[mt], overlap_weight_values[mc], overlap_weight_values[mt]),
        ]
        for label_name, p0, p1, ww0, ww1 in configs:
            row, det = compute_superlevel_topology_contrast(p0, p1, label_name, ww0, ww1, **topology_parameters)
            row.update({"score_bin": label_text, "bin_code": int(code), "bin_weight": bin_weight})
            rows.append(row)
            if return_details:
                topology_details[(int(code), label_name)] = det
    bin_df = pd.DataFrame(rows)
    if bin_df.empty:
        if return_details:
            return bin_df, pd.DataFrame(), topology_details
        return bin_df, pd.DataFrame()
    metric_cols = [
        "betti0_l1",
        "betti0_l2",
        "betti1_l1",
        "betti1_l2",
        "euler_l1",
        "euler_l2",
        "ect_l2",
        "betti0_effect_ref",
        "betti1_effect_ref",
        "euler_effect_ref",
    ]
    agg_rows = []
    for label_name, g in bin_df.groupby("analysis"):
        weights = g["bin_weight"].to_numpy(float)
        weights = weights / weights.sum()
        row = {"analysis": label_name, "n_bins_used": int(len(g)), "total_bin_weight": float(g["bin_weight"].sum())}
        for col in metric_cols:
            row[f"standardized_{col}"] = float(np.sum(weights * g[col].to_numpy(float)))
        agg_rows.append(row)
    if return_details:
        return bin_df, pd.DataFrame(agg_rows), topology_details
    return bin_df, pd.DataFrame(agg_rows)


def summarize_observational_mean_differences(raw_outcomes, y_std, mortality, treatment, ate_weights, overlap_weight_values):
    rows = []
    for prefix, y in [("raw_log_los", raw_outcomes), ("standardized_log_los", y_std)]:
        for coord, name in [(0, "icu_los"), (1, "hospital_los")]:
            rows.extend(
                [
                    {"estimand": f"naive_diff_{prefix}_{name}", "value": compute_weighted_mean_difference(y, treatment, None, coord=coord)},
                    {"estimand": f"IPW_diff_{prefix}_{name}", "value": compute_weighted_mean_difference(y, treatment, ate_weights, coord=coord)},
                    {"estimand": f"overlap_diff_{prefix}_{name}", "value": compute_weighted_mean_difference(y, treatment, overlap_weight_values, coord=coord)},
                ]
            )
    mortality = np.asarray(mortality, float)
    rows.extend(
        [
            {"estimand": "naive_diff_hospital_mortality", "value": compute_weighted_mean(mortality[treatment == 1]) - compute_weighted_mean(mortality[treatment == 0])},
            {"estimand": "IPW_diff_hospital_mortality", "value": compute_weighted_mean(mortality[treatment == 1], ate_weights[treatment == 1]) - compute_weighted_mean(mortality[treatment == 0], ate_weights[treatment == 0])},
            {"estimand": "overlap_diff_hospital_mortality", "value": compute_weighted_mean(mortality[treatment == 1], overlap_weight_values[treatment == 1]) - compute_weighted_mean(mortality[treatment == 0], overlap_weight_values[treatment == 0])},
        ]
    )
    return pd.DataFrame(rows)


def summarize_observational_score_bin_differences(score, raw_outcomes, y_std, mortality, treatment, propensity_score, n_bins=4):
    codes, labels = assign_quantile_bins(score, n_bins=n_bins)
    rows = []
    mortality = np.asarray(mortality, float)
    for code, label_text in enumerate(labels):
        mask = codes == code
        if mask.sum() == 0 or treatment[mask].min() == treatment[mask].max():
            continue
        mt = mask & (treatment == 1)
        mc = mask & (treatment == 0)
        wb = compute_inverse_probability_weights(treatment[mask], propensity_score[mask], stabilized=True)
        rows.append(
            {
                "score_bin": label_text,
                "n": int(mask.sum()),
                "n_treated": int(mt.sum()),
                "n_control": int(mc.sum()),
                "bin_weight": float(mask.mean()),
                "naive_diff_std_log_icu_los": compute_weighted_mean_difference(y_std[mask], treatment[mask], None, coord=0),
                "IPW_diff_std_log_icu_los": compute_weighted_mean_difference(y_std[mask], treatment[mask], wb, coord=0),
                "naive_diff_std_log_hospital_los": compute_weighted_mean_difference(y_std[mask], treatment[mask], None, coord=1),
                "IPW_diff_std_log_hospital_los": compute_weighted_mean_difference(y_std[mask], treatment[mask], wb, coord=1),
                "naive_diff_hospital_mortality": compute_weighted_mean(mortality[mt]) - compute_weighted_mean(mortality[mc]),
                "IPW_diff_hospital_mortality": compute_weighted_mean(mortality[mt], wb[treatment[mask] == 1]) - compute_weighted_mean(mortality[mc], wb[treatment[mask] == 0]),
            }
        )
    return pd.DataFrame(rows)


def summarize_randomized_mean_differences(raw_outcomes, y_std, treatment, ate_weights, overlap_weight_values):
    rows = []
    for prefix, y in [("raw", raw_outcomes), ("standardized", y_std)]:
        for coord, name in [(0, "outcome_1"), (1, "outcome_2")]:
            rows.extend(
                [
                    {"estimand": f"naive_diff_{prefix}_{name}", "value": compute_weighted_mean_difference(y, treatment, None, coord=coord)},
                    {"estimand": f"IPW_diff_{prefix}_{name}", "value": compute_weighted_mean_difference(y, treatment, ate_weights, coord=coord)},
                    {"estimand": f"overlap_diff_{prefix}_{name}", "value": compute_weighted_mean_difference(y, treatment, overlap_weight_values, coord=coord)},
                ]
            )
    return pd.DataFrame(rows)


def summarize_randomized_score_bin_differences(score, raw_outcomes, y_std, treatment, propensity_score, n_bins=4):
    codes, labels = assign_quantile_bins(score, n_bins=n_bins)
    rows = []
    for code, label_text in enumerate(labels):
        mask = codes == code
        if mask.sum() == 0 or treatment[mask].min() == treatment[mask].max():
            continue
        mt = mask & (treatment == 1)
        mc = mask & (treatment == 0)
        wb = compute_inverse_probability_weights(treatment[mask], propensity_score[mask], stabilized=True)
        rows.append(
            {
                "score_bin": label_text,
                "n": int(mask.sum()),
                "n_treated": int(mt.sum()),
                "n_control": int(mc.sum()),
                "bin_weight": float(mask.mean()),
                "naive_diff_std_outcome_1": compute_weighted_mean_difference(y_std[mask], treatment[mask], None, coord=0),
                "IPW_diff_std_outcome_1": compute_weighted_mean_difference(y_std[mask], treatment[mask], wb, coord=0),
                "naive_diff_std_outcome_2": compute_weighted_mean_difference(y_std[mask], treatment[mask], None, coord=1),
                "IPW_diff_std_outcome_2": compute_weighted_mean_difference(y_std[mask], treatment[mask], wb, coord=1),
                "naive_diff_raw_outcome_1": compute_weighted_mean_difference(raw_outcomes[mask], treatment[mask], None, coord=0),
                "IPW_diff_raw_outcome_1": compute_weighted_mean_difference(raw_outcomes[mask], treatment[mask], wb, coord=0),
                "naive_diff_raw_outcome_2": compute_weighted_mean_difference(raw_outcomes[mask], treatment[mask], None, coord=1),
                "IPW_diff_raw_outcome_2": compute_weighted_mean_difference(raw_outcomes[mask], treatment[mask], wb, coord=1),
            }
        )
    return pd.DataFrame(rows)


def compute_empirical_topology_contrasts(observed_outcomes, treatment, ate_weights, overlap_weight_values, topology_parameters):
    rows = []
    topology_details = {}
    configs = [
        ("unadjusted_observed", observed_outcomes[treatment == 0], observed_outcomes[treatment == 1], None, None),
        ("ipw_adjusted_observed_density", observed_outcomes[treatment == 0], observed_outcomes[treatment == 1], ate_weights[treatment == 0], ate_weights[treatment == 1]),
        ("overlap_weighted_observed_density", observed_outcomes[treatment == 0], observed_outcomes[treatment == 1], overlap_weight_values[treatment == 0], overlap_weight_values[treatment == 1]),
    ]
    for label_name, p0, p1, ww0, ww1 in configs:
        row, det = compute_superlevel_topology_contrast(p0, p1, label_name, ww0, ww1, **topology_parameters)
        rows.append(row)
        topology_details[label_name] = det
    return pd.DataFrame(rows), topology_details


def compute_empirical_binned_topology_contrasts(
    observed_outcomes,
    treatment,
    score,
    ate_weights,
    overlap_weight_values,
    n_bins=4,
    topology_parameters=None,
    min_arm=200,
    return_details=False,
):
    if topology_parameters is None:
        topology_parameters = {}
    codes, labels = assign_quantile_bins(score, n_bins=n_bins)
    rows = []
    topology_details = {}
    for code, label_text in enumerate(labels):
        mask = codes == code
        if mask.sum() == 0:
            continue
        mt = mask & (treatment == 1)
        mc = mask & (treatment == 0)
        if mt.sum() < min_arm or mc.sum() < min_arm:
            continue
        bin_weight = float(mask.mean())
        configs = [
            ("unadjusted_observed", observed_outcomes[mc], observed_outcomes[mt], None, None),
            ("ipw_adjusted_observed_density", observed_outcomes[mc], observed_outcomes[mt], ate_weights[mc], ate_weights[mt]),
            ("overlap_weighted_observed_density", observed_outcomes[mc], observed_outcomes[mt], overlap_weight_values[mc], overlap_weight_values[mt]),
        ]
        for label_name, p0, p1, ww0, ww1 in configs:
            row, det = compute_superlevel_topology_contrast(p0, p1, label_name, ww0, ww1, **topology_parameters)
            row.update({"score_bin": label_text, "bin_code": int(code), "bin_weight": bin_weight})
            rows.append(row)
            if return_details:
                topology_details[(int(code), label_name)] = det
    bin_df = pd.DataFrame(rows)
    if bin_df.empty:
        if return_details:
            return bin_df, pd.DataFrame(), topology_details
        return bin_df, pd.DataFrame()
    metric_cols = [
        "betti0_l1",
        "betti0_l2",
        "betti1_l1",
        "betti1_l2",
        "euler_l1",
        "euler_l2",
        "ect_l2",
        "betti0_effect_ref",
        "betti1_effect_ref",
        "euler_effect_ref",
    ]
    agg_rows = []
    for label_name, g in bin_df.groupby("analysis"):
        weights = g["bin_weight"].to_numpy(float)
        weights = weights / weights.sum()
        row = {"analysis": label_name, "n_bins_used": int(len(g)), "total_bin_weight": float(g["bin_weight"].sum())}
        for col in metric_cols:
            row[f"standardized_{col}"] = float(np.sum(weights * g[col].to_numpy(float)))
        agg_rows.append(row)
    if return_details:
        return bin_df, pd.DataFrame(agg_rows), topology_details
    return bin_df, pd.DataFrame(agg_rows)


def compare_topology_contrasts_to_reference(topology_contrast_table, topology_details=None, reference_analysis="ipw_adjusted_observed_density"):
    if topology_contrast_table.empty or reference_analysis not in set(topology_contrast_table["analysis"]):
        return pd.DataFrame()
    ref = topology_contrast_table[topology_contrast_table["analysis"] == reference_analysis].iloc[0]
    ref_det = topology_details.get(reference_analysis) if topology_details else None
    rows = []
    for _, row in topology_contrast_table.iterrows():
        out = {
            "analysis": row["analysis"],
            "reference_analysis": reference_analysis,
            "same_reference_betti0_as_reference": bool(row["betti0_effect_ref"] == ref["betti0_effect_ref"]),
            "same_reference_betti1_as_reference": bool(row["betti1_effect_ref"] == ref["betti1_effect_ref"]),
            "same_reference_euler_as_reference": bool(row["euler_effect_ref"] == ref["euler_effect_ref"]),
            "betti0_l1_difference_from_reference": float(abs(row["betti0_l1"] - ref["betti0_l1"])),
            "betti1_l1_difference_from_reference": float(abs(row["betti1_l1"] - ref["betti1_l1"])),
            "euler_l1_difference_from_reference": float(abs(row["euler_l1"] - ref["euler_l1"])),
            "ect_l2_difference_from_reference": float(abs(row["ect_l2"] - ref["ect_l2"])),
            "min_level_margin": float(row.get("min_level_margin", np.nan)),
        }
        if topology_details:
            out.update(compare_sliced_euler_effect_to_reference(ref_det, topology_details.get(row["analysis"])))
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_propensity_overlap(propensity_score, treatment, ate_weights, n_bins=6):
    codes, labels = assign_quantile_bins(propensity_score, n_bins=n_bins)
    min_t = np.inf
    min_c = np.inf
    for code, _ in enumerate(labels):
        mask = codes == code
        min_t = min(min_t, int(np.sum(mask & (treatment == 1))))
        min_c = min(min_c, int(np.sum(mask & (treatment == 0))))
    wt = ate_weights[treatment == 1]
    wc = ate_weights[treatment == 0]
    return pd.DataFrame(
        [
            {
                "propensity_min": float(np.min(propensity_score)),
                "propensity_max": float(np.max(propensity_score)),
                "propensity_mean": float(np.mean(propensity_score)),
                "fraction_outside_0p1_0p9": float(np.mean((propensity_score < 0.1) | (propensity_score > 0.9))),
                "min_treated_count_by_score_bin": int(min_t if np.isfinite(min_t) else 0),
                "min_control_count_by_score_bin": int(min_c if np.isfinite(min_c) else 0),
                "ESS_treated_ATE_weights": compute_effective_sample_size(wt),
                "ESS_control_ATE_weights": compute_effective_sample_size(wc),
                "max_ATE_weight": float(np.max(ate_weights)),
                "p95_ATE_weight": float(np.quantile(ate_weights, 0.95)),
            }
        ]
    )


def apply_odds_ratio_propensity_tilt(propensity_score, gamma, sign):
    propensity_score = np.clip(np.asarray(propensity_score, float), 1e-5, 1 - 1e-5)
    shifted = logit(propensity_score) + float(sign) * math.log(float(gamma))
    return np.clip(expit(shifted), 0.02, 0.98)


def run_structural_odds_tilt_sensitivity(control_potential_outcomes, treated_potential_outcomes, observed_outcomes, treatment, propensity_score, gammas, topology_parameters, progress_enabled=False, prefix=""):
    rows = []
    true_ate_x = float(np.mean(treated_potential_outcomes[:, 0] - control_potential_outcomes[:, 0]))
    for gamma_i, gamma in enumerate(gammas, 1):
        print_progress(f"{prefix}odds-tilt sensitivity gamma={gamma:g}", gamma_i, len(gammas), enabled=progress_enabled)
        signs = [0] if abs(float(gamma) - 1.0) < 1e-12 else [-1, 1]
        for sign in signs:
            et = apply_odds_ratio_propensity_tilt(propensity_score, gamma, sign)
            wt = compute_inverse_probability_weights(treatment, et, stabilized=True)
            top_row, _ = compute_superlevel_topology_contrast(
                observed_outcomes[treatment == 0],
                observed_outcomes[treatment == 1],
                f"odds_tilt_gamma_{gamma:g}_sign_{sign:+d}",
                wt[treatment == 0],
                wt[treatment == 1],
                **topology_parameters,
            )
            ipw_x = compute_weighted_mean_difference(observed_outcomes, treatment, wt, coord=0)
            rows.append(
                {
                    "gamma": float(gamma),
                    "tilt_sign": int(sign),
                    "IPW_ATE_x": ipw_x,
                    "IPW_bias_x_vs_oracle": ipw_x - true_ate_x,
                    "ESS_treated": compute_effective_sample_size(wt[treatment == 1]),
                    "ESS_control": compute_effective_sample_size(wt[treatment == 0]),
                    "max_weight": float(np.max(wt)),
                    "betti0_l1": top_row["betti0_l1"],
                    "betti1_l1": top_row["betti1_l1"],
                    "euler_l1": top_row["euler_l1"],
                    "ect_l2": top_row["ect_l2"],
                    "betti0_effect_ref": top_row["betti0_effect_ref"],
                    "betti1_effect_ref": top_row["betti1_effect_ref"],
                    "euler_effect_ref": top_row["euler_effect_ref"],
                }
            )
    return pd.DataFrame(rows)


def run_structural_bootstrap_sensitivity(
    dataset_bundle,
    n_bootstrap=30,
    seed=900,
    topology_parameters=None,
    progress_enabled=False,
    prefix="",
    n_bins=4,
    min_arm=8,
):
    if topology_parameters is None:
        topology_parameters = {}
    rng = np.random.default_rng(seed)
    X = dataset_bundle["X"].reset_index(drop=True)
    treatment = np.asarray(dataset_bundle["a"], int)
    control_potential_outcomes = np.asarray(dataset_bundle["y0"], float)
    treated_potential_outcomes = np.asarray(dataset_bundle["y1"], float)
    observed_outcomes = np.asarray(dataset_bundle["y_obs"], float)
    rows = []
    n = len(treatment)
    n_bootstrap = int(n_bootstrap)
    for b in range(n_bootstrap):
        if b == 0 or (b + 1) == n_bootstrap or (b + 1) % max(1, n_bootstrap // 5) == 0:
            print_progress(f"{prefix}bootstrap replicate {b + 1}/{n_bootstrap}", b + 1, n_bootstrap, enabled=progress_enabled)
        idx = rng.choice(n, size=n, replace=True)
        try:
            fit = estimate_propensity_scores(X.iloc[idx].reset_index(drop=True), treatment[idx])
            propensity_score = fit["propensity"]
            wt = compute_inverse_probability_weights(treatment[idx], propensity_score, stabilized=True)
            wov = compute_overlap_weights(treatment[idx], propensity_score)
            ipw_x = compute_weighted_mean_difference(observed_outcomes[idx], treatment[idx], wt, coord=0)
            true_ate_x = float(np.mean(treated_potential_outcomes[idx, 0] - control_potential_outcomes[idx, 0]))

            global_df, _ = compute_structural_benchmark_topology_contrasts(control_potential_outcomes[idx], treated_potential_outcomes[idx], observed_outcomes[idx], treatment[idx], wt, wov, topology_parameters)
            for _, top_row in global_df.iterrows():
                rows.append(
                    {
                        "bootstrap_id": b,
                        "scope": "global",
                        "analysis": top_row["analysis"],
                        "true_ATE_x_oracle": true_ate_x,
                        "IPW_ATE_x": ipw_x,
                        "IPW_bias_x_vs_oracle": ipw_x - true_ate_x,
                        "betti0_l1": top_row["betti0_l1"],
                        "betti1_l1": top_row["betti1_l1"],
                        "euler_l1": top_row["euler_l1"],
                        "ect_l2": top_row["ect_l2"],
                        "betti0_effect_ref": top_row["betti0_effect_ref"],
                        "betti1_effect_ref": top_row["betti1_effect_ref"],
                        "euler_effect_ref": top_row["euler_effect_ref"],
                        "min_level_margin": top_row["min_level_margin"],
                    }
                )

            _, std_df = compute_structural_benchmark_binned_topology_contrasts(
                control_potential_outcomes[idx],
                treated_potential_outcomes[idx],
                observed_outcomes[idx],
                treatment[idx],
                propensity_score,
                wt,
                wov,
                n_bins=n_bins,
                topology_parameters=topology_parameters,
                min_arm=min_arm,
            )
            for _, std_row in std_df.iterrows():
                rows.append(
                    {
                        "bootstrap_id": b,
                        "scope": "balancing_bin",
                        "analysis": std_row["analysis"],
                        "true_ATE_x_oracle": true_ate_x,
                        "IPW_ATE_x": ipw_x,
                        "IPW_bias_x_vs_oracle": ipw_x - true_ate_x,
                        "betti0_l1": std_row["standardized_betti0_l1"],
                        "betti1_l1": std_row["standardized_betti1_l1"],
                        "euler_l1": std_row["standardized_euler_l1"],
                        "ect_l2": std_row["standardized_ect_l2"],
                        "betti0_effect_ref": std_row["standardized_betti0_effect_ref"],
                        "betti1_effect_ref": std_row["standardized_betti1_effect_ref"],
                        "euler_effect_ref": std_row["standardized_euler_effect_ref"],
                        "min_level_margin": np.nan,
                    }
                )
        except Exception as exc:
            rows.append({"bootstrap_id": b, "scope": "error", "analysis": "error", "error": repr(exc)})
    return pd.DataFrame(rows)


def run_structural_signature_tuning_sensitivity(control_potential_outcomes, treated_potential_outcomes, observed_outcomes, treatment, signature_parameter_grid, progress_enabled=False, prefix=""):
    rows = []
    for setting_id, kwargs in enumerate(signature_parameter_grid):
        print_progress(
            f"{prefix}finite-signature grid {setting_id + 1}/{len(signature_parameter_grid)} "
            f"(bins={kwargs.get('bins')}, sigma={kwargs.get('sigma')})",
            setting_id + 1,
            len(signature_parameter_grid),
            enabled=progress_enabled,
        )
        for label_name, p0, p1 in [
            ("oracle_do", control_potential_outcomes, treated_potential_outcomes),
            ("unadjusted_observed", observed_outcomes[treatment == 0], observed_outcomes[treatment == 1]),
        ]:
            row, _ = compute_superlevel_topology_contrast(p0, p1, label_name, **kwargs)
            keep = {
                "setting_id": int(setting_id),
                "analysis": label_name,
                "bins": int(kwargs.get("bins", 96)),
                "sigma": float(kwargs.get("sigma", 1.2)),
                "n_levels": int(len(kwargs.get("level_fracs", default_superlevel_fractions()))),
                "level_fraction_high": float(np.max(kwargs.get("level_fracs", default_superlevel_fractions()))),
                "level_fraction_low": float(np.min(kwargs.get("level_fracs", default_superlevel_fractions()))),
                "n_dirs": int(kwargs.get("n_dirs", 8)),
                "n_slices": int(kwargs.get("n_slices", 24)),
                "min_pixels": int(kwargs.get("min_pixels", 3)),
                "betti0_l1": row["betti0_l1"],
                "betti1_l1": row["betti1_l1"],
                "euler_l1": row["euler_l1"],
                "ect_l2": row["ect_l2"],
                "betti0_effect_ref": row["betti0_effect_ref"],
                "betti1_effect_ref": row["betti1_effect_ref"],
                "euler_effect_ref": row["euler_effect_ref"],
                "min_level_margin": row["min_level_margin"],
            }
            rows.append(keep)
    return pd.DataFrame(rows)


def run_observational_odds_tilt_sensitivity(observed_outcomes, raw_outcomes, mortality, treatment, propensity_score, gammas, topology_parameters, progress_enabled=False, prefix=""):
    rows = []
    mortality = np.asarray(mortality, float)
    for gamma_i, gamma in enumerate(gammas, 1):
        print_progress(f"{prefix}observational odds-tilt sensitivity gamma={gamma:g}", gamma_i, len(gammas), enabled=progress_enabled)
        signs = [0] if abs(float(gamma) - 1.0) < 1e-12 else [-1, 1]
        for sign in signs:
            et = apply_odds_ratio_propensity_tilt(propensity_score, gamma, sign)
            wt = compute_inverse_probability_weights(treatment, et, stabilized=True)
            top_row, _ = compute_superlevel_topology_contrast(
                observed_outcomes[treatment == 0],
                observed_outcomes[treatment == 1],
                f"odds_tilt_gamma_{gamma:g}_sign_{sign:+d}",
                wt[treatment == 0],
                wt[treatment == 1],
                **topology_parameters,
            )
            mortality_diff = compute_weighted_mean(mortality[treatment == 1], wt[treatment == 1]) - compute_weighted_mean(mortality[treatment == 0], wt[treatment == 0])
            rows.append(
                {
                    "gamma": float(gamma),
                    "tilt_sign": int(sign),
                    "IPW_diff_std_log_icu_los": compute_weighted_mean_difference(observed_outcomes, treatment, wt, coord=0),
                    "IPW_diff_std_log_hospital_los": compute_weighted_mean_difference(observed_outcomes, treatment, wt, coord=1),
                    "IPW_diff_raw_log_icu_los": compute_weighted_mean_difference(raw_outcomes, treatment, wt, coord=0),
                    "IPW_diff_raw_log_hospital_los": compute_weighted_mean_difference(raw_outcomes, treatment, wt, coord=1),
                    "IPW_diff_hospital_mortality": mortality_diff,
                    "ESS_treated": compute_effective_sample_size(wt[treatment == 1]),
                    "ESS_control": compute_effective_sample_size(wt[treatment == 0]),
                    "max_weight": float(np.max(wt)),
                    "betti0_l1": top_row["betti0_l1"],
                    "betti1_l1": top_row["betti1_l1"],
                    "euler_l1": top_row["euler_l1"],
                    "ect_l2": top_row["ect_l2"],
                    "betti0_effect_ref": top_row["betti0_effect_ref"],
                    "betti1_effect_ref": top_row["betti1_effect_ref"],
                    "euler_effect_ref": top_row["euler_effect_ref"],
                }
            )
    return pd.DataFrame(rows)


def run_randomized_odds_tilt_sensitivity(observed_outcomes, raw_outcomes, treatment, propensity_score, gammas, topology_parameters, progress_enabled=False, prefix=""):
    rows = []
    for gamma_i, gamma in enumerate(gammas, 1):
        print_progress(f"{prefix}randomized/empirical odds-tilt sensitivity gamma={gamma:g}", gamma_i, len(gammas), enabled=progress_enabled)
        signs = [0] if abs(float(gamma) - 1.0) < 1e-12 else [-1, 1]
        for sign in signs:
            et = apply_odds_ratio_propensity_tilt(propensity_score, gamma, sign)
            wt = compute_inverse_probability_weights(treatment, et, stabilized=True)
            top_row, _ = compute_superlevel_topology_contrast(
                observed_outcomes[treatment == 0],
                observed_outcomes[treatment == 1],
                f"odds_tilt_gamma_{gamma:g}_sign_{sign:+d}",
                wt[treatment == 0],
                wt[treatment == 1],
                **topology_parameters,
            )
            rows.append(
                {
                    "gamma": float(gamma),
                    "tilt_sign": int(sign),
                    "IPW_diff_std_outcome_1": compute_weighted_mean_difference(observed_outcomes, treatment, wt, coord=0),
                    "IPW_diff_std_outcome_2": compute_weighted_mean_difference(observed_outcomes, treatment, wt, coord=1),
                    "IPW_diff_raw_outcome_1": compute_weighted_mean_difference(raw_outcomes, treatment, wt, coord=0),
                    "IPW_diff_raw_outcome_2": compute_weighted_mean_difference(raw_outcomes, treatment, wt, coord=1),
                    "ESS_treated": compute_effective_sample_size(wt[treatment == 1]),
                    "ESS_control": compute_effective_sample_size(wt[treatment == 0]),
                    "max_weight": float(np.max(wt)),
                    "betti0_l1": top_row["betti0_l1"],
                    "betti1_l1": top_row["betti1_l1"],
                    "euler_l1": top_row["euler_l1"],
                    "ect_l2": top_row["ect_l2"],
                    "betti0_effect_ref": top_row["betti0_effect_ref"],
                    "betti1_effect_ref": top_row["betti1_effect_ref"],
                    "euler_effect_ref": top_row["euler_effect_ref"],
                }
            )
    return pd.DataFrame(rows)


def run_observational_bootstrap_sensitivity(
    dataset_bundle,
    n_bootstrap=10,
    seed=1900,
    topology_parameters=None,
    progress_enabled=False,
    prefix="",
    n_bins=4,
    min_arm=200,
):
    if topology_parameters is None:
        topology_parameters = {}
    rng = np.random.default_rng(seed)
    X = dataset_bundle["X"].reset_index(drop=True)
    treatment = np.asarray(dataset_bundle["a"], int)
    observed_outcomes = np.asarray(dataset_bundle["y_obs"], float)
    raw_outcomes = np.asarray(dataset_bundle["y_raw"], float)
    mortality = np.asarray(dataset_bundle["mortality"], float)
    rows = []
    n = len(treatment)
    n_bootstrap = int(n_bootstrap)
    for b in range(n_bootstrap):
        if b == 0 or (b + 1) == n_bootstrap or (b + 1) % max(1, n_bootstrap // 5) == 0:
            print_progress(f"{prefix}observational bootstrap replicate {b + 1}/{n_bootstrap}", b + 1, n_bootstrap, enabled=progress_enabled)
        idx = rng.choice(n, size=n, replace=True)
        try:
            fit = estimate_propensity_scores(X.iloc[idx].reset_index(drop=True), treatment[idx], C=0.2, max_iter=1000)
            propensity_score = fit["propensity"]
            wt = compute_inverse_probability_weights(treatment[idx], propensity_score, stabilized=True)
            wov = compute_overlap_weights(treatment[idx], propensity_score)
            ipw_icu = compute_weighted_mean_difference(observed_outcomes[idx], treatment[idx], wt, coord=0)
            ipw_hosp = compute_weighted_mean_difference(observed_outcomes[idx], treatment[idx], wt, coord=1)
            mort_diff = compute_weighted_mean(mortality[idx][treatment[idx] == 1], wt[treatment[idx] == 1]) - compute_weighted_mean(
                mortality[idx][treatment[idx] == 0],
                wt[treatment[idx] == 0],
            )
            global_df, _ = compute_empirical_topology_contrasts(observed_outcomes[idx], treatment[idx], wt, wov, topology_parameters)
            for _, top_row in global_df.iterrows():
                rows.append(
                    {
                        "bootstrap_id": b,
                        "scope": "global",
                        "analysis": top_row["analysis"],
                        "IPW_diff_std_log_icu_los": ipw_icu,
                        "IPW_diff_std_log_hospital_los": ipw_hosp,
                        "IPW_diff_hospital_mortality": mort_diff,
                        "betti0_l1": top_row["betti0_l1"],
                        "betti1_l1": top_row["betti1_l1"],
                        "euler_l1": top_row["euler_l1"],
                        "ect_l2": top_row["ect_l2"],
                        "betti0_effect_ref": top_row["betti0_effect_ref"],
                        "betti1_effect_ref": top_row["betti1_effect_ref"],
                        "euler_effect_ref": top_row["euler_effect_ref"],
                        "min_level_margin": top_row["min_level_margin"],
                    }
                )
            _, std_df = compute_empirical_binned_topology_contrasts(
                observed_outcomes[idx],
                treatment[idx],
                propensity_score,
                wt,
                wov,
                n_bins=n_bins,
                topology_parameters=topology_parameters,
                min_arm=min_arm,
            )
            for _, std_row in std_df.iterrows():
                rows.append(
                    {
                        "bootstrap_id": b,
                        "scope": "balancing_bin",
                        "analysis": std_row["analysis"],
                        "IPW_diff_std_log_icu_los": ipw_icu,
                        "IPW_diff_std_log_hospital_los": ipw_hosp,
                        "IPW_diff_hospital_mortality": mort_diff,
                        "betti0_l1": std_row["standardized_betti0_l1"],
                        "betti1_l1": std_row["standardized_betti1_l1"],
                        "euler_l1": std_row["standardized_euler_l1"],
                        "ect_l2": std_row["standardized_ect_l2"],
                        "betti0_effect_ref": std_row["standardized_betti0_effect_ref"],
                        "betti1_effect_ref": std_row["standardized_betti1_effect_ref"],
                        "euler_effect_ref": std_row["standardized_euler_effect_ref"],
                        "min_level_margin": np.nan,
                    }
                )
        except Exception as exc:
            rows.append({"bootstrap_id": b, "scope": "error", "analysis": "error", "error": repr(exc)})
    return pd.DataFrame(rows)


def run_randomized_bootstrap_sensitivity(
    dataset_bundle,
    n_bootstrap=10,
    seed=2900,
    topology_parameters=None,
    progress_enabled=False,
    prefix="",
    n_bins=4,
    min_arm=25,
):
    if topology_parameters is None:
        topology_parameters = {}
    rng = np.random.default_rng(seed)
    X = dataset_bundle["X"].reset_index(drop=True)
    treatment = np.asarray(dataset_bundle["a"], int)
    observed_outcomes = np.asarray(dataset_bundle["y_obs"], float)
    raw_outcomes = np.asarray(dataset_bundle["y_raw"], float)
    rows = []
    n = len(treatment)
    n_bootstrap = int(n_bootstrap)
    for b in range(n_bootstrap):
        if b == 0 or (b + 1) == n_bootstrap or (b + 1) % max(1, n_bootstrap // 5) == 0:
            print_progress(f"{prefix}randomized/empirical bootstrap replicate {b + 1}/{n_bootstrap}", b + 1, n_bootstrap, enabled=progress_enabled)
        idx = rng.choice(n, size=n, replace=True)
        try:
            fit = estimate_propensity_scores(X.iloc[idx].reset_index(drop=True), treatment[idx], C=1.0, max_iter=3000)
            propensity_score = fit["propensity"]
            wt = compute_inverse_probability_weights(treatment[idx], propensity_score, stabilized=True)
            wov = compute_overlap_weights(treatment[idx], propensity_score)
            ipw_1 = compute_weighted_mean_difference(observed_outcomes[idx], treatment[idx], wt, coord=0)
            ipw_2 = compute_weighted_mean_difference(observed_outcomes[idx], treatment[idx], wt, coord=1)
            global_df, _ = compute_empirical_topology_contrasts(observed_outcomes[idx], treatment[idx], wt, wov, topology_parameters)
            for _, top_row in global_df.iterrows():
                rows.append(
                    {
                        "bootstrap_id": b,
                        "scope": "global",
                        "analysis": top_row["analysis"],
                        "IPW_diff_std_outcome_1": ipw_1,
                        "IPW_diff_std_outcome_2": ipw_2,
                        "betti0_l1": top_row["betti0_l1"],
                        "betti1_l1": top_row["betti1_l1"],
                        "euler_l1": top_row["euler_l1"],
                        "ect_l2": top_row["ect_l2"],
                        "betti0_effect_ref": top_row["betti0_effect_ref"],
                        "betti1_effect_ref": top_row["betti1_effect_ref"],
                        "euler_effect_ref": top_row["euler_effect_ref"],
                        "min_level_margin": top_row["min_level_margin"],
                    }
                )
            _, std_df = compute_empirical_binned_topology_contrasts(
                observed_outcomes[idx],
                treatment[idx],
                propensity_score,
                wt,
                wov,
                n_bins=n_bins,
                topology_parameters=topology_parameters,
                min_arm=min_arm,
            )
            for _, std_row in std_df.iterrows():
                rows.append(
                    {
                        "bootstrap_id": b,
                        "scope": "balancing_bin",
                        "analysis": std_row["analysis"],
                        "IPW_diff_std_outcome_1": ipw_1,
                        "IPW_diff_std_outcome_2": ipw_2,
                        "betti0_l1": std_row["standardized_betti0_l1"],
                        "betti1_l1": std_row["standardized_betti1_l1"],
                        "euler_l1": std_row["standardized_euler_l1"],
                        "ect_l2": std_row["standardized_ect_l2"],
                        "betti0_effect_ref": std_row["standardized_betti0_effect_ref"],
                        "betti1_effect_ref": std_row["standardized_betti1_effect_ref"],
                        "euler_effect_ref": std_row["standardized_euler_effect_ref"],
                        "min_level_margin": np.nan,
                    }
                )
        except Exception as exc:
            rows.append({"bootstrap_id": b, "scope": "error", "analysis": "error", "error": repr(exc)})
    return pd.DataFrame(rows)


def run_empirical_signature_tuning_sensitivity(observed_outcomes, treatment, signature_parameter_grid, progress_enabled=False, prefix=""):
    rows = []
    for setting_id, kwargs in enumerate(signature_parameter_grid):
        print_progress(
            f"{prefix}empirical finite-signature grid {setting_id + 1}/{len(signature_parameter_grid)} "
            f"(bins={kwargs.get('bins')}, sigma={kwargs.get('sigma')})",
            setting_id + 1,
            len(signature_parameter_grid),
            enabled=progress_enabled,
        )
        row, _ = compute_superlevel_topology_contrast(observed_outcomes[treatment == 0], observed_outcomes[treatment == 1], "unadjusted_observed", **kwargs)
        keep = {
            "setting_id": int(setting_id),
            "analysis": "unadjusted_observed",
            "bins": int(kwargs.get("bins", 96)),
            "sigma": float(kwargs.get("sigma", 1.2)),
            "n_levels": int(len(kwargs.get("level_fracs", default_superlevel_fractions()))),
            "level_fraction_high": float(np.max(kwargs.get("level_fracs", default_superlevel_fractions()))),
            "level_fraction_low": float(np.min(kwargs.get("level_fracs", default_superlevel_fractions()))),
            "n_dirs": int(kwargs.get("n_dirs", 8)),
            "n_slices": int(kwargs.get("n_slices", 24)),
            "min_pixels": int(kwargs.get("min_pixels", 3)),
            "betti0_l1": row["betti0_l1"],
            "betti1_l1": row["betti1_l1"],
            "euler_l1": row["euler_l1"],
            "ect_l2": row["ect_l2"],
            "betti0_effect_ref": row["betti0_effect_ref"],
            "betti1_effect_ref": row["betti1_effect_ref"],
            "euler_effect_ref": row["euler_effect_ref"],
            "min_level_margin": row["min_level_margin"],
        }
        rows.append(keep)
    return pd.DataFrame(rows)


def compute_structural_signature_chamber_checks(topology_contrast_table, topology_details=None, score_topology_df=None, score_details=None):
    oracle = topology_contrast_table[topology_contrast_table["analysis"] == "oracle_do"].iloc[0]
    oracle_det = topology_details.get("oracle_do") if topology_details else None
    rows = []
    for _, row in topology_contrast_table.iterrows():
        out_row = {
            "analysis": row["analysis"],
            "same_reference_betti0_as_oracle": bool(row["betti0_effect_ref"] == oracle["betti0_effect_ref"]),
            "same_reference_betti1_as_oracle": bool(row["betti1_effect_ref"] == oracle["betti1_effect_ref"]),
            "same_reference_euler_as_oracle": bool(row["euler_effect_ref"] == oracle["euler_effect_ref"]),
            "betti0_l1_difference_from_oracle": float(abs(row["betti0_l1"] - oracle["betti0_l1"])),
            "betti1_l1_difference_from_oracle": float(abs(row["betti1_l1"] - oracle["betti1_l1"])),
            "euler_l1_difference_from_oracle": float(abs(row["euler_l1"] - oracle["euler_l1"])),
            "ect_l2_difference_from_oracle": float(abs(row["ect_l2"] - oracle["ect_l2"])),
            "min_level_margin": float(row.get("min_level_margin", np.nan)),
            "min_level_margin_relative_to_max_density": float(row.get("min_level_margin_relative_to_max_density", np.nan)),
        }
        if topology_details:
            out_row.update(
                compare_sliced_euler_effect_to_reference(
                    oracle_det,
                    topology_details.get(row["analysis"]),
                )
            )
        rows.append(out_row)
    out = pd.DataFrame(rows)
    if score_topology_df is not None and not score_topology_df.empty:
        flags = []
        for code, g in score_topology_df.groupby("bin_code"):
            if "oracle_do" not in set(g["analysis"]):
                continue
            o = g[g["analysis"] == "oracle_do"].iloc[0]
            oracle_bin_det = score_details.get((int(code), "oracle_do")) if score_details else None
            for _, row in g.iterrows():
                flag = {
                    "bin_code": int(code),
                    "score_bin": row["score_bin"],
                    "analysis": row["analysis"],
                    "same_bin_betti0_effect_as_oracle": bool(row["betti0_effect_ref"] == o["betti0_effect_ref"]),
                    "same_bin_betti1_effect_as_oracle": bool(row["betti1_effect_ref"] == o["betti1_effect_ref"]),
                    "same_bin_euler_effect_as_oracle": bool(row["euler_effect_ref"] == o["euler_effect_ref"]),
                }
                if score_details:
                    flag.update(
                        compare_sliced_euler_effect_to_reference(
                            oracle_bin_det,
                            score_details.get((int(code), row["analysis"])),
                        )
                    )
                flags.append(flag)
        if flags:
            flag_df = pd.DataFrame(flags)
            mean_cols = [
                "same_bin_betti0_effect_as_oracle",
                "same_bin_betti1_effect_as_oracle",
                "same_bin_euler_effect_as_oracle",
            ]
            if "same_slice_wise_ect_effect_as_oracle" in flag_df.columns:
                mean_cols.extend(
                    [
                        "same_slice_wise_ect_effect_as_oracle",
                        "fraction_slices_same_ect_effect",
                        "ect_slice_effect_l2_delta_vs_oracle",
                    ]
                )
            agg = flag_df.groupby("analysis")[mean_cols].mean().reset_index()
            agg = agg.rename(
                columns={
                    "same_bin_betti0_effect_as_oracle": "fraction_bins_same_betti0_effect",
                    "same_bin_betti1_effect_as_oracle": "fraction_bins_same_betti1_effect",
                    "same_bin_euler_effect_as_oracle": "fraction_bins_same_euler_effect",
                    "same_slice_wise_ect_effect_as_oracle": "fraction_bins_same_slice_wise_ect_effect",
                    "fraction_slices_same_ect_effect": "mean_fraction_slices_same_ect_effect",
                    "ect_slice_effect_l2_delta_vs_oracle": "mean_ect_slice_effect_l2_delta_vs_oracle",
                }
            )
            if "max_abs_ect_slice_effect_delta" in flag_df.columns:
                max_delta = (
                    flag_df.groupby("analysis")["max_abs_ect_slice_effect_delta"]
                    .max()
                    .reset_index()
                    .rename(columns={"max_abs_ect_slice_effect_delta": "max_abs_ect_slice_effect_delta"})
                )
                agg = agg.merge(max_delta, on="analysis", how="left")
            return out, flag_df, agg
    return out, pd.DataFrame(), pd.DataFrame()


def save_structural_summary_figure(name, control_potential_outcomes, treated_potential_outcomes, observed_outcomes, treatment, propensity_score, raw_smd_table, weighted_smd_table, topology_contrast_table, topology_details, dataset_output_dir):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    ax = axes[0, 0]
    ax.scatter(observed_outcomes[treatment == 0, 0], observed_outcomes[treatment == 0, 1], s=6, alpha=0.30, label="observed control")
    ax.scatter(observed_outcomes[treatment == 1, 0], observed_outcomes[treatment == 1, 1], s=6, alpha=0.30, label="observed treatment")
    ax.set_title(f"{name}: observed clouds")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.scatter(control_potential_outcomes[:, 0], control_potential_outcomes[:, 1], s=6, alpha=0.22, label="known control potential outcome")
    ax.scatter(treated_potential_outcomes[:, 0], treated_potential_outcomes[:, 1], s=6, alpha=0.22, label="known treatment potential outcome")
    ax.set_title("Known interventional outcome clouds")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.hist(propensity_score[treatment == 0], bins=25, alpha=0.6, label="control")
    ax.hist(propensity_score[treatment == 1], bins=25, alpha=0.6, label="treatment")
    ax.set_title("Estimated propensity overlap")
    ax.legend(fontsize=8)

    ax = axes[0, 3]
    top10 = raw_smd_table.head(10).copy()
    merged = top10[["feature", "abs_smd"]].rename(columns={"abs_smd": "raw"})
    merged = merged.merge(weighted_smd_table[["feature", "abs_smd"]].rename(columns={"abs_smd": "weighted"}), on="feature", how="left")
    idx = np.arange(len(merged))
    width = 0.38
    ax.bar(idx - width / 2, merged["raw"], width=width, label="raw")
    ax.bar(idx + width / 2, merged["weighted"], width=width, label="weighted")
    ax.set_xticks(idx)
    ax.set_xticklabels(merged["feature"], rotation=75, ha="right", fontsize=8)
    ax.set_title("Top absolute SMDs")
    ax.legend(fontsize=8)

    oracle_det = topology_details["oracle_do"]
    level_fracs = np.linspace(0.62, 0.08, len(oracle_det["levels"]))
    ax = axes[1, 0]
    ax.plot(level_fracs, oracle_det["control"]["betti0"], label="control")
    ax.plot(level_fracs, oracle_det["treatment"]["betti0"], label="treatment")
    ax.invert_xaxis()
    ax.set_title("Known-potential-outcome superlevel Betti-0")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(level_fracs, oracle_det["control"]["betti1"], label="control")
    ax.plot(level_fracs, oracle_det["treatment"]["betti1"], label="treatment")
    ax.invert_xaxis()
    ax.set_title("Known-potential-outcome superlevel Betti-1")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.bar(topology_contrast_table["analysis"], topology_contrast_table["euler_l1"])
    ax.set_title("Superlevel Euler curve contrast")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1, 3]
    ax.bar(topology_contrast_table["analysis"], topology_contrast_table["ect_l2"])
    ax.set_title("Finite sliced Euler signature contrast")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(dataset_output_dir / f"{name}_superlevel_summary.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_empirical_summary_figure(
    name,
    observed_outcomes,
    treatment,
    propensity_score,
    raw_smd_table,
    weighted_smd_table,
    topology_contrast_table,
    standardized_binned_topology_table,
    dataset_output_dir,
    x_label="standardized log ICU LOS",
    y_label="standardized log hospital LOS",
    figure_suffix="observational_superlevel_summary",
):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ax = axes[0, 0]
    ax.scatter(observed_outcomes[treatment == 0, 0], observed_outcomes[treatment == 0, 1], s=3, alpha=0.08, label="control")
    ax.scatter(observed_outcomes[treatment == 1, 0], observed_outcomes[treatment == 1, 1], s=3, alpha=0.08, label="treated")
    ax.set_title(f"{name}: observed outcome cloud")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(propensity_score[treatment == 0], bins=30, alpha=0.6, label="control")
    ax.hist(propensity_score[treatment == 1], bins=30, alpha=0.6, label="treated")
    ax.set_title("Estimated propensity overlap")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    top10 = raw_smd_table.head(10).copy()
    merged = top10[["feature", "abs_smd"]].rename(columns={"abs_smd": "raw"})
    merged = merged.merge(weighted_smd_table[["feature", "abs_smd"]].rename(columns={"abs_smd": "weighted"}), on="feature", how="left")
    idx = np.arange(len(merged))
    width = 0.38
    ax.bar(idx - width / 2, merged["raw"], width=width, label="raw")
    ax.bar(idx + width / 2, merged["weighted"], width=width, label="weighted")
    ax.set_xticks(idx)
    ax.set_xticklabels(merged["feature"], rotation=75, ha="right", fontsize=8)
    ax.set_title("Top absolute SMDs")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.bar(topology_contrast_table["analysis"], topology_contrast_table["betti0_l1"])
    ax.set_title("Superlevel Betti-0 curve contrast")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1, 1]
    ax.bar(topology_contrast_table["analysis"], topology_contrast_table["euler_l1"])
    ax.set_title("Superlevel Euler curve contrast")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1, 2]
    if standardized_binned_topology_table is not None and not standardized_binned_topology_table.empty:
        ax.bar(standardized_binned_topology_table["analysis"], standardized_binned_topology_table["standardized_ect_l2"])
        ax.set_title("Balancing-bin standardized sliced Euler signature")
    else:
        ax.bar(topology_contrast_table["analysis"], topology_contrast_table["ect_l2"])
        ax.set_title("Finite sliced Euler signature contrast")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(dataset_output_dir / f"{name}_{figure_suffix}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_design_diagnostics_figure(name, balance_diag, overlap_diagnostic_table, distribution_test_table, topology_test_table, trim_sensitivity_table, dataset_output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax = axes[0, 0]
    if balance_diag is not None and not balance_diag.empty:
        top = balance_diag.head(20).copy()
        y = np.arange(len(top))
        ax.barh(y - 0.18, top["abs_raw_smd"], height=0.35, label="raw")
        if "abs_weighted_smd" in top:
            ax.barh(y + 0.18, top["abs_weighted_smd"], height=0.35, label="weighted")
        ax.axvline(0.10, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(top["feature"], fontsize=7)
        ax.invert_yaxis()
        ax.set_title("Top covariate SMDs")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No balance diagnostics", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[0, 1]
    if balance_diag is not None and not balance_diag.empty and "weighted_ks_statistic" in balance_diag:
        vals = [
            balance_diag["raw_ks_statistic"].dropna().to_numpy(float),
            balance_diag["weighted_ks_statistic"].dropna().to_numpy(float),
        ]
        ax.boxplot(vals, tick_labels=["raw", "weighted"], showfliers=False)
        ax.set_title("Covariate distribution KS stats")
        ax.set_ylabel("KS")
    else:
        ax.text(0.5, 0.5, "No KS diagnostics", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[0, 2]
    if overlap_diagnostic_table is not None and not overlap_diagnostic_table.empty:
        labels = overlap_diagnostic_table["weight_scheme"].astype(str)
        ax.bar(labels, overlap_diagnostic_table["ESS_fraction_total"])
        ax.set_ylim(0, 1)
        ax.set_title("Effective sample size fraction")
    else:
        ax.text(0.5, 0.5, "No overlap diagnostics", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1, 0]
    if distribution_test_table is not None and not distribution_test_table.empty:
        ax.bar(distribution_test_table["test"], distribution_test_table["permutation_p_value"])
        ax.axhline(0.05, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_title("Distributional permutation p-values")
        ax.tick_params(axis="x", rotation=30)
    else:
        ax.text(0.5, 0.5, "No distribution tests", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1, 1]
    if topology_test_table is not None and not topology_test_table.empty:
        ax.bar(topology_test_table["test"], topology_test_table["permutation_p_value"])
        ax.axhline(0.05, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_title("Topological-signature permutation p-values")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "No topology tests", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1, 2]
    if trim_sensitivity_table is not None and not trim_sensitivity_table.empty:
        plot_df = trim_sensitivity_table[trim_sensitivity_table["analysis"].isin(["raw", "IPW", "overlap"])].copy()
        for analysis, g in plot_df.groupby("analysis"):
            ax.plot(g["trim_rule"], g["ect_l2"], marker="o", label=analysis)
        ax.set_title("Overlap-trim sliced Euler L2 sensitivity")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No trim sensitivity", ha="center", va="center")
        ax.set_axis_off()
    fig.suptitle(f"{name}: design diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(dataset_output_dir / f"{name}_design_diagnostics.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def summarize_bootstrap_results(df):
    if df.empty or "error" in df.columns and df["error"].notna().all():
        return pd.DataFrame()
    cols = [
        "IPW_bias_x_vs_oracle",
        "IPW_diff_std_log_icu_los",
        "IPW_diff_std_log_hospital_los",
        "IPW_diff_hospital_mortality",
        "IPW_diff_std_outcome_1",
        "IPW_diff_std_outcome_2",
        "betti0_l1",
        "betti1_l1",
        "euler_l1",
        "ect_l2",
        "betti0_effect_ref",
        "betti1_effect_ref",
        "euler_effect_ref",
        "min_level_margin",
    ]
    rows = []
    if "scope" in df.columns:
        good_df = df[df["scope"] != "error"].copy()
    else:
        good_df = df.copy()
    group_cols = [c for c in ["scope", "analysis"] if c in good_df.columns]
    if not group_cols:
        good_df["scope"] = "legacy"
        good_df["analysis"] = "legacy"
        group_cols = ["scope", "analysis"]
    for keys, group in good_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        good = group[[c for c in cols if c in group.columns]].apply(pd.to_numeric, errors="coerce")
        for col in good.columns:
            vals = good[col].dropna().to_numpy(float)
            if len(vals) == 0:
                continue
            row = {
                **key_map,
                "quantity": col,
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "q025": float(np.quantile(vals, 0.025)),
                "q500": float(np.quantile(vals, 0.500)),
                "q975": float(np.quantile(vals, 0.975)),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_structural_benchmark(
    dataset_bundle,
    output_dir: Path,
    n_bootstrap=30,
    *,
    progress_enabled=True,
    dataset_index=1,
    n_datasets=1,
    diagnostic_permutations=99,
):
    """Run the full analysis for a benchmark with known generated potential outcomes."""
    name = dataset_bundle["name"]
    prefix = f"{name}: "
    print_progress(f"Starting dataset {dataset_index}/{n_datasets}: {name}", 1, 13, enabled=progress_enabled)
    covariates = dataset_bundle["X"]
    structural_score = dataset_bundle["eta_true"]
    unobserved_group = dataset_bundle["u"]
    treatment = dataset_bundle["a"]
    control_potential_outcomes = dataset_bundle["y0"]
    treated_potential_outcomes = dataset_bundle["y1"]
    observed_outcomes = dataset_bundle["y_obs"]

    print_progress(f"{prefix}fitting observed-covariate propensity model", 2, 13, enabled=progress_enabled)
    fit = estimate_propensity_scores(covariates, treatment)
    propensity_score = fit["propensity"]
    ate_weights = compute_inverse_probability_weights(treatment, propensity_score, stabilized=True)
    overlap_weight_values = compute_overlap_weights(treatment, propensity_score)
    covariate_design = fit["X_design"]

    print_progress(f"{prefix}computing balance, ATE, CATE, and latent-confounding diagnostics", 3, 13, enabled=progress_enabled)
    balance_summary_table, raw_smd_table, weighted_smd_table = summarize_covariate_balance(covariate_design, treatment, overlap_weight_values)
    latent_confounding_df = compute_latent_variable_diagnostics(structural_score, unobserved_group, treatment, control_potential_outcomes, treated_potential_outcomes)
    mean_effect_table = summarize_structural_benchmark_mean_effects(control_potential_outcomes, treated_potential_outcomes, observed_outcomes, treatment, ate_weights, overlap_weight_values)
    score_bin_effect_table = summarize_structural_benchmark_score_bin_effects(propensity_score, control_potential_outcomes, treated_potential_outcomes, observed_outcomes, treatment, propensity_score, n_bins=4)
    overlap_summary_table = summarize_propensity_overlap(propensity_score, treatment, ate_weights)

    primary_topology_parameters = {
        "bins": 96,
        "sigma": 3.0,
        "level_fracs": default_superlevel_fractions(n_levels=19),
        "n_dirs": 8,
        "n_slices": 20,
        "min_pixels": 3,
    }
    sensitivity_topology_parameters = {
        "bins": 72,
        "sigma": 3.0,
        "level_fracs": default_superlevel_fractions(n_levels=13),
        "n_dirs": 6,
        "n_slices": 14,
        "min_pixels": 3,
    }
    print_progress(f"{prefix}running design and sensitivity diagnostics for the structural benchmark", None, None, enabled=progress_enabled)
    design_diagnostics = compute_design_and_sensitivity_diagnostics(
        covariates,
        covariate_design,
        treatment,
        observed_outcomes,
        propensity_score,
        ate_weights,
        overlap_weight_values,
        n_permutations=diagnostic_permutations,
        seed=DEFAULT_RANDOM_SEED + dataset_index * 100,
    )
    print_progress(f"{prefix}computing global finite level-grid superlevel signatures", 4, 13, enabled=progress_enabled)
    topology_contrast_table, topology_details = compute_structural_benchmark_topology_contrasts(control_potential_outcomes, treated_potential_outcomes, observed_outcomes, treatment, ate_weights, overlap_weight_values, primary_topology_parameters)
    print_progress(f"{prefix}computing balancing-bin superlevel signatures", 5, 13, enabled=progress_enabled)
    binned_topology_table, standardized_binned_topology_table, binned_topology_details = compute_structural_benchmark_binned_topology_contrasts(
        control_potential_outcomes,
        treated_potential_outcomes,
        observed_outcomes,
        treatment,
        propensity_score,
        ate_weights,
        overlap_weight_values,
        n_bins=4,
        topology_parameters=sensitivity_topology_parameters,
        min_arm=18,
        return_details=True,
    )
    print_progress(f"{prefix}computing energy-distance checks", 6, 13, enabled=progress_enabled)
    energy_distance_table = pd.DataFrame(
        [
            {"comparison": "treated_obs_vs_do_treated", "energy_distance": estimate_sampled_energy_distance(observed_outcomes[treatment == 1], treated_potential_outcomes, seed=11)},
            {"comparison": "control_obs_vs_do_control", "energy_distance": estimate_sampled_energy_distance(observed_outcomes[treatment == 0], control_potential_outcomes, seed=22)},
        ]
    )
    print_progress(f"{prefix}running odds-tilt unmeasured-confounding sensitivity", 7, 13, enabled=progress_enabled)
    odds_tilt_table = run_structural_odds_tilt_sensitivity(
        control_potential_outcomes,
        treated_potential_outcomes,
        observed_outcomes,
        treatment,
        propensity_score,
        gammas=[1, 1.25, 1.5, 2, 3, 5],
        topology_parameters=sensitivity_topology_parameters,
        progress_enabled=progress_enabled,
        prefix=prefix,
    )
    print_progress(f"{prefix}running nonparametric bootstrap sensitivity", 8, 13, enabled=progress_enabled)
    bootstrap_table = run_structural_bootstrap_sensitivity(
        dataset_bundle,
        n_bootstrap=n_bootstrap,
        seed=777,
        topology_parameters=sensitivity_topology_parameters,
        progress_enabled=progress_enabled,
        prefix=prefix,
    )
    bootstrap_summary_table = summarize_bootstrap_results(bootstrap_table)
    signature_parameter_grid = [
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 14, "min_pixels": 3},
        {"bins": 96, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 14, "min_pixels": 3},
        {"bins": 72, "sigma": 3.6, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 14, "min_pixels": 3},
        {"bins": 72, "sigma": 4.2, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 14, "min_pixels": 3},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=9), "n_dirs": 6, "n_slices": 14, "min_pixels": 3},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=17), "n_dirs": 6, "n_slices": 14, "min_pixels": 3},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13, high=0.56, low=0.06), "n_dirs": 6, "n_slices": 14, "min_pixels": 3},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13, high=0.68, low=0.10), "n_dirs": 6, "n_slices": 14, "min_pixels": 3},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 4, "n_slices": 14, "min_pixels": 3},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 8, "n_slices": 14, "min_pixels": 3},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 10, "min_pixels": 3},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 18, "min_pixels": 3},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 14, "min_pixels": 1},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 14, "min_pixels": 5},
    ]
    print_progress(f"{prefix}running finite-signature tuning sensitivity", 9, 13, enabled=progress_enabled)
    signature_tuning_table = run_structural_signature_tuning_sensitivity(
        control_potential_outcomes,
        treated_potential_outcomes,
        observed_outcomes,
        treatment,
        signature_parameter_grid,
        progress_enabled=progress_enabled,
        prefix=prefix,
    )
    print_progress(f"{prefix}checking level-grid chamber agreement", 10, 13, enabled=progress_enabled)
    signature_chamber_table, binned_signature_chamber_table, binned_signature_chamber_summary_table = compute_structural_signature_chamber_checks(
        topology_contrast_table,
        topology_details=topology_details,
        score_topology_df=binned_topology_table,
        score_details=binned_topology_details,
    )
    sliced_euler_coordinate_table = build_sliced_euler_coordinate_check_table(topology_details)
    binned_sliced_euler_coordinate_table = build_binned_sliced_euler_coordinate_check_table(binned_topology_details, binned_topology_table)

    print_progress(f"{prefix}saving CSV tables", 11, 13, enabled=progress_enabled)
    dataset_output_dir = output_dir / name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    balance_summary_table.to_csv(dataset_output_dir / "balance_summary.csv", index=False)
    raw_smd_table.to_csv(dataset_output_dir / "balance_raw_smd.csv", index=False)
    weighted_smd_table.to_csv(dataset_output_dir / "balance_weighted_smd.csv", index=False)
    latent_confounding_df.to_csv(dataset_output_dir / "latent_confounding_by_score_bins.csv", index=False)
    mean_effect_table.to_csv(dataset_output_dir / "causal_benchmark_metric_targets.csv", index=False)
    score_bin_effect_table.to_csv(dataset_output_dir / "causal_benchmark_score_bin_effects.csv", index=False)
    overlap_summary_table.to_csv(dataset_output_dir / "overlap_diagnostics.csv", index=False)
    topology_contrast_table.to_csv(dataset_output_dir / "topology_superlevel_contrasts.csv", index=False)
    binned_topology_table.to_csv(dataset_output_dir / "topology_balancing_bin_contrasts.csv", index=False)
    standardized_binned_topology_table.to_csv(dataset_output_dir / "topology_balancing_bin_standardized.csv", index=False)
    energy_distance_table.to_csv(dataset_output_dir / "energy_distance_checks.csv", index=False)
    odds_tilt_table.to_csv(dataset_output_dir / "odds_tilt_sensitivity.csv", index=False)
    bootstrap_table.to_csv(dataset_output_dir / "bootstrap_sensitivity.csv", index=False)
    bootstrap_summary_table.to_csv(dataset_output_dir / "bootstrap_sensitivity_summary.csv", index=False)
    signature_tuning_table.to_csv(dataset_output_dir / "finite_signature_tuning_sensitivity.csv", index=False)
    signature_chamber_table.to_csv(dataset_output_dir / "finite_signature_chamber_checks.csv", index=False)
    binned_signature_chamber_table.to_csv(dataset_output_dir / "balancing_bin_chamber_checks.csv", index=False)
    binned_signature_chamber_summary_table.to_csv(dataset_output_dir / "balancing_bin_chamber_check_summary.csv", index=False)
    sliced_euler_coordinate_table.to_csv(dataset_output_dir / "ect_slice_chamber_checks.csv", index=False)
    binned_sliced_euler_coordinate_table.to_csv(dataset_output_dir / "balancing_bin_ect_slice_chamber_checks.csv", index=False)
    design_diagnostics["covariate_balance_diagnostics"].to_csv(dataset_output_dir / "covariate_balance_diagnostics.csv", index=False)
    design_diagnostics["design_balance_omnibus"].to_csv(dataset_output_dir / "design_balance_omnibus_tests.csv", index=False)
    design_diagnostics["design_overlap"].to_csv(dataset_output_dir / "design_overlap_weight_diagnostics.csv", index=False)
    design_diagnostics["design_distribution_tests"].to_csv(dataset_output_dir / "design_distribution_permutation_tests.csv", index=False)
    design_diagnostics["design_topology_tests"].to_csv(dataset_output_dir / "design_topology_permutation_tests.csv", index=False)
    design_diagnostics["design_aipw"].to_csv(dataset_output_dir / "design_crossfit_aipw_mean_checks.csv", index=False)
    design_diagnostics["design_trim"].to_csv(dataset_output_dir / "design_overlap_trim_sensitivity.csv", index=False)

    print_progress(f"{prefix}saving summary figure", 12, 13, enabled=progress_enabled)
    save_structural_summary_figure(name, control_potential_outcomes, treated_potential_outcomes, observed_outcomes, treatment, propensity_score, raw_smd_table, weighted_smd_table, topology_contrast_table, topology_details, dataset_output_dir)
    save_design_diagnostics_figure(
        name,
        design_diagnostics["covariate_balance_diagnostics"],
        design_diagnostics["design_overlap"],
        design_diagnostics["design_distribution_tests"],
        design_diagnostics["design_topology_tests"],
        design_diagnostics["design_trim"],
        dataset_output_dir,
    )
    print_progress(f"Finished dataset {dataset_index}/{n_datasets}: {name}", 13, 13, enabled=progress_enabled)

    return {
        "name": name,
        "base": dataset_output_dir,
        "balance_summary": balance_summary_table,
        "raw_smd": raw_smd_table,
        "weighted_smd": weighted_smd_table,
        "latent_confounding": latent_confounding_df,
        "metric_bias": mean_effect_table,
        "cate": score_bin_effect_table,
        "overlap": overlap_summary_table,
        "topology": topology_contrast_table,
        "score_topology": binned_topology_table,
        "score_topology_standardized": standardized_binned_topology_table,
        "energy": energy_distance_table,
        "odds_tilt": odds_tilt_table,
        "bootstrap": bootstrap_table,
        "bootstrap_summary": bootstrap_summary_table,
        "tuning": signature_tuning_table,
        "chamber": signature_chamber_table,
        "score_chamber_summary": binned_signature_chamber_summary_table,
        "ect_slice_chamber": sliced_euler_coordinate_table,
        "balancing_bin_ect_slice_chamber": binned_sliced_euler_coordinate_table,
        **design_diagnostics,
    }


def analyze_randomized_application(
    dataset_bundle,
    output_dir: Path,
    n_bootstrap=10,
    *,
    progress_enabled=True,
    dataset_index=1,
    n_datasets=1,
    diagnostic_permutations=99,
):
    """Run the empirical randomized JOBS II analysis without oracle potential outcomes."""
    name = dataset_bundle["name"]
    prefix = f"{name}: "
    print_progress(f"Starting dataset {dataset_index}/{n_datasets}: {name}", 1, 13, enabled=progress_enabled)
    covariates = dataset_bundle["X"]
    treatment = dataset_bundle["a"]
    observed_outcomes = dataset_bundle["y_obs"]
    raw_outcomes = dataset_bundle["y_raw"]
    bounds = dataset_bundle.get("topology_bounds", (-3.5, 3.5, -3.5, 3.5))
    x_label, y_label = dataset_bundle.get("outcome_axis_labels", ["standardized outcome 1", "standardized outcome 2"])

    print_progress(f"{prefix}fitting covariate-adjustment propensity model", 2, 13, enabled=progress_enabled)
    fit = estimate_propensity_scores(covariates, treatment, clip=0.02, C=1.0, max_iter=3000)
    propensity_score = fit["propensity"]
    ate_weights = compute_inverse_probability_weights(treatment, propensity_score, stabilized=True)
    overlap_weight_values = compute_overlap_weights(treatment, propensity_score)
    covariate_design = fit["X_design"]

    print_progress(f"{prefix}computing randomization balance, outcome contrasts, and score-bin contrasts", 3, 13, enabled=progress_enabled)
    balance_summary_table, raw_smd_table, weighted_smd_table = summarize_covariate_balance(covariate_design, treatment, overlap_weight_values)
    mean_effect_table = summarize_randomized_mean_differences(raw_outcomes, observed_outcomes, treatment, ate_weights, overlap_weight_values)
    score_bin_effect_table = summarize_randomized_score_bin_differences(propensity_score, raw_outcomes, observed_outcomes, treatment, propensity_score, n_bins=4)
    overlap_summary_table = summarize_propensity_overlap(propensity_score, treatment, ate_weights)

    primary_topology_parameters = {
        "bins": 96,
        "sigma": 3.0,
        "level_fracs": default_superlevel_fractions(n_levels=19),
        "n_dirs": 8,
        "n_slices": 20,
        "min_pixels": 3,
        "bounds": bounds,
    }
    sensitivity_topology_parameters = {
        "bins": 72,
        "sigma": 3.0,
        "level_fracs": default_superlevel_fractions(n_levels=13),
        "n_dirs": 6,
        "n_slices": 14,
        "min_pixels": 3,
        "bounds": bounds,
    }
    print_progress(f"{prefix}running design and sensitivity diagnostics for the randomized application", None, None, enabled=progress_enabled)
    design_diagnostics = compute_design_and_sensitivity_diagnostics(
        covariates,
        covariate_design,
        treatment,
        observed_outcomes,
        propensity_score,
        ate_weights,
        overlap_weight_values,
        bounds=bounds,
        n_permutations=diagnostic_permutations,
        seed=DEFAULT_RANDOM_SEED + dataset_index * 100,
    )
    print_progress(f"{prefix}computing global finite level-grid superlevel signatures", 4, 13, enabled=progress_enabled)
    topology_contrast_table, topology_details = compute_empirical_topology_contrasts(observed_outcomes, treatment, ate_weights, overlap_weight_values, primary_topology_parameters)
    print_progress(f"{prefix}computing balancing-bin superlevel signatures", 5, 13, enabled=progress_enabled)
    binned_topology_table, standardized_binned_topology_table, binned_topology_details = compute_empirical_binned_topology_contrasts(
        observed_outcomes,
        treatment,
        propensity_score,
        ate_weights,
        overlap_weight_values,
        n_bins=4,
        topology_parameters=sensitivity_topology_parameters,
        min_arm=40,
        return_details=True,
    )
    print_progress(f"{prefix}computing energy-distance checks", 6, 13, enabled=progress_enabled)
    energy_distance_table = pd.DataFrame(
        [
            {
                "comparison": "randomized_treated_vs_control_observed_outcomes",
                "energy_distance": estimate_sampled_energy_distance(observed_outcomes[treatment == 1], observed_outcomes[treatment == 0], seed=707),
            }
        ]
    )
    print_progress(f"{prefix}running odds-tilt sensitivity", 7, 13, enabled=progress_enabled)
    odds_tilt_table = run_randomized_odds_tilt_sensitivity(
        observed_outcomes,
        raw_outcomes,
        treatment,
        propensity_score,
        gammas=[1, 1.25, 1.5, 2, 3, 5],
        topology_parameters=sensitivity_topology_parameters,
        progress_enabled=progress_enabled,
        prefix=prefix,
    )
    print_progress(f"{prefix}running nonparametric bootstrap sensitivity", 8, 13, enabled=progress_enabled)
    bootstrap_table = run_randomized_bootstrap_sensitivity(
        dataset_bundle,
        n_bootstrap=n_bootstrap,
        seed=2777,
        topology_parameters=sensitivity_topology_parameters,
        progress_enabled=progress_enabled,
        prefix=prefix,
        min_arm=25,
    )
    bootstrap_summary_table = summarize_bootstrap_results(bootstrap_table)
    signature_parameter_grid = [
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 96, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.6, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 4.2, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=9), "n_dirs": 6, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=17), "n_dirs": 6, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13, high=0.56, low=0.06), "n_dirs": 6, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13, high=0.68, low=0.10), "n_dirs": 6, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 4, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 8, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 10, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 18, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 14, "min_pixels": 1, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 14, "min_pixels": 5, "bounds": bounds},
    ]
    print_progress(f"{prefix}running finite-signature tuning sensitivity", 9, 13, enabled=progress_enabled)
    signature_tuning_table = run_empirical_signature_tuning_sensitivity(
        observed_outcomes,
        treatment,
        signature_parameter_grid,
        progress_enabled=progress_enabled,
        prefix=prefix,
    )
    print_progress(f"{prefix}checking topology stability and slice-wise Euler coordinates", 10, 13, enabled=progress_enabled)
    stability_df = compare_topology_contrasts_to_reference(topology_contrast_table, topology_details)
    reference_details = {"oracle_do": topology_details["ipw_adjusted_observed_density"], **topology_details}
    sliced_euler_coordinate_table = build_sliced_euler_coordinate_check_table(reference_details)
    binned_sliced_euler_coordinate_table = pd.DataFrame()

    print_progress(f"{prefix}saving CSV tables", 11, 13, enabled=progress_enabled)
    dataset_output_dir = output_dir / name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    dataset_bundle["jobs_info"].to_csv(dataset_output_dir / "randomized_application_metadata.csv", index=False)
    balance_summary_table.to_csv(dataset_output_dir / "balance_summary.csv", index=False)
    raw_smd_table.to_csv(dataset_output_dir / "balance_raw_smd.csv", index=False)
    weighted_smd_table.to_csv(dataset_output_dir / "balance_weighted_smd.csv", index=False)
    mean_effect_table.to_csv(dataset_output_dir / "randomized_metric_diffs.csv", index=False)
    score_bin_effect_table.to_csv(dataset_output_dir / "randomized_score_bin_effects.csv", index=False)
    overlap_summary_table.to_csv(dataset_output_dir / "overlap_diagnostics.csv", index=False)
    topology_contrast_table.to_csv(dataset_output_dir / "topology_superlevel_contrasts.csv", index=False)
    binned_topology_table.to_csv(dataset_output_dir / "topology_balancing_bin_contrasts.csv", index=False)
    standardized_binned_topology_table.to_csv(dataset_output_dir / "topology_balancing_bin_standardized.csv", index=False)
    energy_distance_table.to_csv(dataset_output_dir / "energy_distance_checks.csv", index=False)
    odds_tilt_table.to_csv(dataset_output_dir / "odds_tilt_sensitivity.csv", index=False)
    bootstrap_table.to_csv(dataset_output_dir / "bootstrap_sensitivity.csv", index=False)
    bootstrap_summary_table.to_csv(dataset_output_dir / "bootstrap_sensitivity_summary.csv", index=False)
    signature_tuning_table.to_csv(dataset_output_dir / "finite_signature_tuning_sensitivity.csv", index=False)
    stability_df.to_csv(dataset_output_dir / "randomized_topology_stability_checks.csv", index=False)
    sliced_euler_coordinate_table.to_csv(dataset_output_dir / "ect_slice_stability_checks.csv", index=False)
    binned_sliced_euler_coordinate_table.to_csv(dataset_output_dir / "balancing_bin_ect_slice_stability_checks.csv", index=False)
    design_diagnostics["covariate_balance_diagnostics"].to_csv(dataset_output_dir / "covariate_balance_diagnostics.csv", index=False)
    design_diagnostics["design_balance_omnibus"].to_csv(dataset_output_dir / "design_balance_omnibus_tests.csv", index=False)
    design_diagnostics["design_overlap"].to_csv(dataset_output_dir / "design_overlap_weight_diagnostics.csv", index=False)
    design_diagnostics["design_distribution_tests"].to_csv(dataset_output_dir / "design_distribution_permutation_tests.csv", index=False)
    design_diagnostics["design_topology_tests"].to_csv(dataset_output_dir / "design_topology_permutation_tests.csv", index=False)
    design_diagnostics["design_aipw"].to_csv(dataset_output_dir / "design_crossfit_aipw_mean_checks.csv", index=False)
    design_diagnostics["design_trim"].to_csv(dataset_output_dir / "design_overlap_trim_sensitivity.csv", index=False)

    print_progress(f"{prefix}saving summary figure", 12, 13, enabled=progress_enabled)
    save_empirical_summary_figure(
        name,
        observed_outcomes,
        treatment,
        propensity_score,
        raw_smd_table,
        weighted_smd_table,
        topology_contrast_table,
        standardized_binned_topology_table,
        dataset_output_dir,
        x_label=x_label,
        y_label=y_label,
        figure_suffix="randomized_superlevel_summary",
    )
    save_design_diagnostics_figure(
        name,
        design_diagnostics["covariate_balance_diagnostics"],
        design_diagnostics["design_overlap"],
        design_diagnostics["design_distribution_tests"],
        design_diagnostics["design_topology_tests"],
        design_diagnostics["design_trim"],
        dataset_output_dir,
    )
    print_progress(f"Finished dataset {dataset_index}/{n_datasets}: {name}", 13, 13, enabled=progress_enabled)

    return {
        "name": name,
        "mode": "randomized",
        "base": dataset_output_dir,
        "jobs_info": dataset_bundle["jobs_info"],
        "balance_summary": balance_summary_table,
        "raw_smd": raw_smd_table,
        "weighted_smd": weighted_smd_table,
        "metric_bias": mean_effect_table,
        "cate": score_bin_effect_table,
        "overlap": overlap_summary_table,
        "topology": topology_contrast_table,
        "score_topology": binned_topology_table,
        "score_topology_standardized": standardized_binned_topology_table,
        "energy": energy_distance_table,
        "odds_tilt": odds_tilt_table,
        "bootstrap": bootstrap_table,
        "bootstrap_summary": bootstrap_summary_table,
        "tuning": signature_tuning_table,
        "stability": stability_df,
        "ect_slice_stability": sliced_euler_coordinate_table,
        **design_diagnostics,
    }


def analyze_observational_application(
    dataset_bundle,
    output_dir: Path,
    n_bootstrap=10,
    *,
    progress_enabled=True,
    dataset_index=1,
    n_datasets=1,
    diagnostic_permutations=99,
):
    """Run the optional observational eICU analysis as a descriptive sensitivity study."""
    name = dataset_bundle["name"]
    prefix = f"{name}: "
    print_progress(f"Starting dataset {dataset_index}/{n_datasets}: {name}", 1, 13, enabled=progress_enabled)
    covariates = dataset_bundle["X"]
    treatment = dataset_bundle["a"]
    observed_outcomes = dataset_bundle["y_obs"]
    raw_outcomes = dataset_bundle["y_raw"]
    mortality = dataset_bundle["mortality"]
    bounds = dataset_bundle.get("topology_bounds", (-3.5, 3.5, -3.5, 3.5))

    print_progress(f"{prefix}fitting observed-covariate propensity model", 2, 13, enabled=progress_enabled)
    fit = estimate_propensity_scores(covariates, treatment, clip=0.02, C=0.2, max_iter=1000)
    propensity_score = fit["propensity"]
    ate_weights = compute_inverse_probability_weights(treatment, propensity_score, stabilized=True)
    overlap_weight_values = compute_overlap_weights(treatment, propensity_score)
    covariate_design = fit["X_design"]

    print_progress(f"{prefix}computing balance, observational effects, and score-bin contrasts", 3, 13, enabled=progress_enabled)
    balance_summary_table, raw_smd_table, weighted_smd_table = summarize_covariate_balance(covariate_design, treatment, overlap_weight_values)
    mean_effect_table = summarize_observational_mean_differences(raw_outcomes, observed_outcomes, mortality, treatment, ate_weights, overlap_weight_values)
    score_bin_effect_table = summarize_observational_score_bin_differences(propensity_score, raw_outcomes, observed_outcomes, mortality, treatment, propensity_score, n_bins=4)
    overlap_summary_table = summarize_propensity_overlap(propensity_score, treatment, ate_weights)

    primary_topology_parameters = {
        "bins": 96,
        "sigma": 3.0,
        "level_fracs": default_superlevel_fractions(n_levels=19),
        "n_dirs": 8,
        "n_slices": 20,
        "min_pixels": 3,
        "bounds": bounds,
    }
    sensitivity_topology_parameters = {
        "bins": 72,
        "sigma": 3.0,
        "level_fracs": default_superlevel_fractions(n_levels=13),
        "n_dirs": 6,
        "n_slices": 14,
        "min_pixels": 3,
        "bounds": bounds,
    }
    print_progress(f"{prefix}running design and sensitivity diagnostics for the observational analysis", None, None, enabled=progress_enabled)
    design_diagnostics = compute_design_and_sensitivity_diagnostics(
        covariates,
        covariate_design,
        treatment,
        observed_outcomes,
        propensity_score,
        ate_weights,
        overlap_weight_values,
        bounds=bounds,
        n_permutations=diagnostic_permutations,
        seed=DEFAULT_RANDOM_SEED + dataset_index * 100,
        include_binary_sensitivity=True,
        binary_outcome=mortality,
        binary_label="hospital_mortality",
    )
    print_progress(f"{prefix}computing global finite level-grid superlevel signatures", 4, 13, enabled=progress_enabled)
    topology_contrast_table, topology_details = compute_empirical_topology_contrasts(observed_outcomes, treatment, ate_weights, overlap_weight_values, primary_topology_parameters)
    print_progress(f"{prefix}computing balancing-bin superlevel signatures", 5, 13, enabled=progress_enabled)
    binned_topology_table, standardized_binned_topology_table, binned_topology_details = compute_empirical_binned_topology_contrasts(
        observed_outcomes,
        treatment,
        propensity_score,
        ate_weights,
        overlap_weight_values,
        n_bins=4,
        topology_parameters=sensitivity_topology_parameters,
        min_arm=200,
        return_details=True,
    )
    print_progress(f"{prefix}computing energy-distance checks", 6, 13, enabled=progress_enabled)
    energy_distance_table = pd.DataFrame(
        [
            {
                "comparison": "ventday1_treated_vs_control_observed_outcomes",
                "energy_distance": estimate_sampled_energy_distance(observed_outcomes[treatment == 1], observed_outcomes[treatment == 0], seed=505),
            }
        ]
    )
    print_progress(f"{prefix}running odds-tilt unmeasured-confounding sensitivity", 7, 13, enabled=progress_enabled)
    odds_tilt_table = run_observational_odds_tilt_sensitivity(
        observed_outcomes,
        raw_outcomes,
        mortality,
        treatment,
        propensity_score,
        gammas=[1, 1.25, 1.5, 2, 3, 5],
        topology_parameters=sensitivity_topology_parameters,
        progress_enabled=progress_enabled,
        prefix=prefix,
    )
    print_progress(f"{prefix}running nonparametric bootstrap sensitivity", 8, 13, enabled=progress_enabled)
    bootstrap_table = run_observational_bootstrap_sensitivity(
        dataset_bundle,
        n_bootstrap=n_bootstrap,
        seed=1777,
        topology_parameters=sensitivity_topology_parameters,
        progress_enabled=progress_enabled,
        prefix=prefix,
        min_arm=200,
    )
    bootstrap_summary_table = summarize_bootstrap_results(bootstrap_table)
    signature_parameter_grid = [
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 96, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.6, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 4.2, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=9), "n_dirs": 6, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=17), "n_dirs": 6, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13, high=0.56, low=0.06), "n_dirs": 6, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13, high=0.68, low=0.10), "n_dirs": 6, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 4, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 8, "n_slices": 14, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 10, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 18, "min_pixels": 3, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 14, "min_pixels": 1, "bounds": bounds},
        {"bins": 72, "sigma": 3.0, "level_fracs": default_superlevel_fractions(n_levels=13), "n_dirs": 6, "n_slices": 14, "min_pixels": 5, "bounds": bounds},
    ]
    print_progress(f"{prefix}running finite-signature tuning sensitivity", 9, 13, enabled=progress_enabled)
    signature_tuning_table = run_empirical_signature_tuning_sensitivity(
        observed_outcomes,
        treatment,
        signature_parameter_grid,
        progress_enabled=progress_enabled,
        prefix=prefix,
    )
    print_progress(f"{prefix}checking observational topology stability", 10, 13, enabled=progress_enabled)
    stability_df = compare_topology_contrasts_to_reference(topology_contrast_table, topology_details)
    reference_details = {"oracle_do": topology_details["ipw_adjusted_observed_density"], **topology_details}
    sliced_euler_coordinate_table = build_sliced_euler_coordinate_check_table(reference_details)
    binned_sliced_euler_coordinate_table = pd.DataFrame()

    print_progress(f"{prefix}saving CSV tables", 11, 13, enabled=progress_enabled)
    dataset_output_dir = output_dir / name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    dataset_bundle["eicu_info"].to_csv(dataset_output_dir / "observational_cohort_metadata.csv", index=False)
    balance_summary_table.to_csv(dataset_output_dir / "balance_summary.csv", index=False)
    raw_smd_table.to_csv(dataset_output_dir / "balance_raw_smd.csv", index=False)
    weighted_smd_table.to_csv(dataset_output_dir / "balance_weighted_smd.csv", index=False)
    mean_effect_table.to_csv(dataset_output_dir / "observational_metric_diffs.csv", index=False)
    score_bin_effect_table.to_csv(dataset_output_dir / "observational_score_bin_effects.csv", index=False)
    overlap_summary_table.to_csv(dataset_output_dir / "overlap_diagnostics.csv", index=False)
    topology_contrast_table.to_csv(dataset_output_dir / "topology_superlevel_contrasts.csv", index=False)
    binned_topology_table.to_csv(dataset_output_dir / "topology_balancing_bin_contrasts.csv", index=False)
    standardized_binned_topology_table.to_csv(dataset_output_dir / "topology_balancing_bin_standardized.csv", index=False)
    energy_distance_table.to_csv(dataset_output_dir / "energy_distance_checks.csv", index=False)
    odds_tilt_table.to_csv(dataset_output_dir / "odds_tilt_sensitivity.csv", index=False)
    bootstrap_table.to_csv(dataset_output_dir / "bootstrap_sensitivity.csv", index=False)
    bootstrap_summary_table.to_csv(dataset_output_dir / "bootstrap_sensitivity_summary.csv", index=False)
    signature_tuning_table.to_csv(dataset_output_dir / "finite_signature_tuning_sensitivity.csv", index=False)
    stability_df.to_csv(dataset_output_dir / "observational_topology_stability_checks.csv", index=False)
    sliced_euler_coordinate_table.to_csv(dataset_output_dir / "ect_slice_stability_checks.csv", index=False)
    binned_sliced_euler_coordinate_table.to_csv(dataset_output_dir / "balancing_bin_ect_slice_stability_checks.csv", index=False)
    design_diagnostics["covariate_balance_diagnostics"].to_csv(dataset_output_dir / "covariate_balance_diagnostics.csv", index=False)
    design_diagnostics["design_balance_omnibus"].to_csv(dataset_output_dir / "design_balance_omnibus_tests.csv", index=False)
    design_diagnostics["design_overlap"].to_csv(dataset_output_dir / "design_overlap_weight_diagnostics.csv", index=False)
    design_diagnostics["design_distribution_tests"].to_csv(dataset_output_dir / "design_distribution_permutation_tests.csv", index=False)
    design_diagnostics["design_topology_tests"].to_csv(dataset_output_dir / "design_topology_permutation_tests.csv", index=False)
    design_diagnostics["design_aipw"].to_csv(dataset_output_dir / "design_crossfit_aipw_mean_checks.csv", index=False)
    design_diagnostics["design_trim"].to_csv(dataset_output_dir / "design_overlap_trim_sensitivity.csv", index=False)
    design_diagnostics["design_binary_sensitivity"].to_csv(dataset_output_dir / "design_binary_evalue_sensitivity.csv", index=False)

    print_progress(f"{prefix}saving summary figure", 12, 13, enabled=progress_enabled)
    save_empirical_summary_figure(name, observed_outcomes, treatment, propensity_score, raw_smd_table, weighted_smd_table, topology_contrast_table, standardized_binned_topology_table, dataset_output_dir)
    save_design_diagnostics_figure(
        name,
        design_diagnostics["covariate_balance_diagnostics"],
        design_diagnostics["design_overlap"],
        design_diagnostics["design_distribution_tests"],
        design_diagnostics["design_topology_tests"],
        design_diagnostics["design_trim"],
        dataset_output_dir,
    )
    print_progress(f"Finished dataset {dataset_index}/{n_datasets}: {name}", 13, 13, enabled=progress_enabled)

    return {
        "name": name,
        "mode": "observational",
        "base": dataset_output_dir,
        "eicu_info": dataset_bundle["eicu_info"],
        "balance_summary": balance_summary_table,
        "raw_smd": raw_smd_table,
        "weighted_smd": weighted_smd_table,
        "metric_bias": mean_effect_table,
        "cate": score_bin_effect_table,
        "overlap": overlap_summary_table,
        "topology": topology_contrast_table,
        "score_topology": binned_topology_table,
        "score_topology_standardized": standardized_binned_topology_table,
        "energy": energy_distance_table,
        "odds_tilt": odds_tilt_table,
        "bootstrap": bootstrap_table,
        "bootstrap_summary": bootstrap_summary_table,
        "tuning": signature_tuning_table,
        "stability": stability_df,
        "ect_slice_stability": sliced_euler_coordinate_table,
        **design_diagnostics,
    }


def tex_escape(x):
    s = str(x)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def extract_metric_value(mean_effect_table, name):
    rows = mean_effect_table.loc[mean_effect_table["estimand"] == name, "value"]
    return float(rows.iloc[0]) if len(rows) else float("nan")


def format_compact_float(x, digits=3):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def format_bool_for_latex(x):
    if pd.isna(x):
        return "NA"
    return "yes" if bool(x) else "no"


def build_score_bin_topology_comparison_table(res, analysis="unadjusted_observed"):
    cate = res["cate"].copy()
    topology_tests = res["score_topology"].copy()
    if cate.empty or topology_tests.empty:
        return pd.DataFrame()
    topology_tests = topology_tests[topology_tests["analysis"] == analysis]
    if topology_tests.empty:
        return pd.DataFrame()
    keep = ["score_bin", "betti0_effect_ref", "betti1_effect_ref", "euler_effect_ref", "ect_l2"]
    return cate.merge(topology_tests[keep], on="score_bin", how="left")


def write_latex_summary(results: list[dict[str, Any]], output_dir: Path, method_description: str) -> Path:
    semi_results = [res for res in results if res.get("mode", "semisynthetic") not in {"observational", "randomized"}]
    randomized_results = [res for res in results if res.get("mode") == "randomized"]
    observational_results = [res for res in results if res.get("mode") == "observational"]
    lines = []
    lines.append(r"\section*{Finite Superlevel-Set Topological Diagnostics}")
    lines.append("")
    lines.append(
        "This report summarizes the replication pipeline. The code uses the density-based formulation "
        r"$\Psi(\mu)=\Phi(f_\mu)$: topology is computed from finite superlevel-set filtrations "
        r"$\{y:f_\mu(y)\ge \lambda\}$ rather than from ordinary supports."
    )
    lines.append(
        "The reported finite vector is a finite level-grid approximation to the full "
        r"filtration $\{y:f_\mu(y)\ge\lambda\}_\lambda$, not a single-level support calculation."
    )
    lines.append("")
    lines.append(r"\paragraph{Method specification.}")
    lines.append(tex_escape(method_description))
    lines.append(
        "The implemented balancing-bin estimator follows this order: estimate or condition on the observed balancing score, "
        "construct arm-specific factual densities inside each bin, apply the finite topological signature, and then "
        "average binwise contrasts over the empirical bin distribution. This estimates a binwise "
        "approximation to the covariate-standardized structural effect, with binning error controlled by stability "
        "of the selected signatures inside bins."
    )
    lines.append("")
    lines.append(r"\subsection*{Main Numerical Comparison}")
    lines.append(r"\begin{tabular}{lrrrrrrr}")
    lines.append(r"Dataset & raw SMD & weighted SMD & true ATE$_x$ & naive bias & IPW bias & Betti0 ref. & Euler ref. \\")
    lines.append(r"\hline")
    for res in semi_results:
        bal = res["balance_summary"].iloc[0]
        topology_tests = res["topology"]
        oracle = topology_tests[topology_tests["analysis"] == "oracle_do"].iloc[0]
        lines.append(
            f"{tex_escape(res['name'])} & "
            f"{format_compact_float(bal['max_abs_smd_raw'])} & "
            f"{format_compact_float(bal['max_abs_smd_weighted'])} & "
            f"{format_compact_float(extract_metric_value(res['metric_bias'], 'true_ATE_mean_x_oracle'))} & "
            f"{format_compact_float(extract_metric_value(res['metric_bias'], 'naive_minus_true_bias_x'))} & "
            f"{format_compact_float(extract_metric_value(res['metric_bias'], 'ipw_minus_true_bias_x'))} & "
            f"{format_compact_float(oracle['betti0_effect_ref'])} & "
            f"{format_compact_float(oracle['euler_effect_ref'])} \\\\"
        )
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(
        "The table above is a global structural-benchmark diagnostic. For the noninjective conditional "
        "finite topological target considered here, the finite target is the balancing-bin standardized "
        r"$\tau_\Psi^{(K)}$: condition or bin first, apply the finite superlevel signature second, "
        "and average binwise contrasts third. The corresponding rows are reported below and saved in "
        r"\texttt{topology\_balancing\_bin\_standardized.csv}."
    )
    lines.append("")
    lines.append(r"\subsection*{Balancing-Bin Standardized Topological Target}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"Dataset & bins & oracle B0 & observed B0 & observed Euler & observed sliced Euler & IPW B0 \\")
    lines.append(r"\hline")
    for res in semi_results:
        std = res["score_topology_standardized"]

        def std_value(analysis, col):
            if std.empty or analysis not in set(std["analysis"]):
                return "NA"
            return format_compact_float(std[std["analysis"] == analysis].iloc[0][col])

        n_bins = 0
        if not std.empty and "n_bins_used" in std.columns:
            n_bins = int(std["n_bins_used"].max())
        lines.append(
            f"{tex_escape(res['name'])} & "
            f"{n_bins} & "
            f"{std_value('oracle_do', 'standardized_betti0_effect_ref')} & "
            f"{std_value('unadjusted_observed', 'standardized_betti0_effect_ref')} & "
            f"{std_value('unadjusted_observed', 'standardized_euler_effect_ref')} & "
            f"{std_value('unadjusted_observed', 'standardized_ect_l2')} & "
            f"{std_value('ipw_adjusted_observed_density', 'standardized_betti0_effect_ref')} \\\\"
        )
    lines.append(r"\end{tabular}")
    lines.append(
        "In the structural benchmarks, known potential outcomes make it possible to compare factual estimators "
        "with known-potential-outcome finite Betti0, Betti1, and Euler superlevel signatures. "
        "The sliced Euler signature column is location-sensitive and is reported as an additional descriptive distributional "
        "signature unless the slice-wise chamber-invariance checks below pass exactly."
    )
    lines.append("")
    lines.append(r"\subsection*{Slice-Wise sliced Euler signature Chamber Invariance}")
    lines.append(
        "For every finite sliced Euler signature coordinate, the code compares the treated-minus-control sliced Euler "
        "effect against the corresponding known-potential-outcome effect. This is the slice-wise coordinate check."
    )
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"Dataset & global pass & global slice frac. & bin pass frac. & bin slice frac. & max delta \\")
    lines.append(r"\hline")
    ect_all_pass = True
    for res in semi_results:
        ch = res["chamber"]
        score_ch = res["score_chamber_summary"]
        global_pass = np.nan
        global_frac = np.nan
        max_delta = np.nan
        bin_pass_frac = np.nan
        bin_slice_frac = np.nan
        if not ch.empty and "unadjusted_observed" in set(ch["analysis"]):
            obs_ch = ch[ch["analysis"] == "unadjusted_observed"].iloc[0]
            global_pass = obs_ch.get("same_slice_wise_ect_effect_as_oracle", np.nan)
            global_frac = obs_ch.get("fraction_slices_same_ect_effect", np.nan)
            max_delta = obs_ch.get("max_abs_ect_slice_effect_delta", np.nan)
        if not score_ch.empty and "unadjusted_observed" in set(score_ch["analysis"]):
            obs_bin = score_ch[score_ch["analysis"] == "unadjusted_observed"].iloc[0]
            bin_pass_frac = obs_bin.get("fraction_bins_same_slice_wise_ect_effect", np.nan)
            bin_slice_frac = obs_bin.get("mean_fraction_slices_same_ect_effect", np.nan)
            if not pd.isna(obs_bin.get("max_abs_ect_slice_effect_delta", np.nan)):
                max_delta = max(float(max_delta), float(obs_bin["max_abs_ect_slice_effect_delta"])) if not pd.isna(max_delta) else float(obs_bin["max_abs_ect_slice_effect_delta"])
        if not (bool(global_pass) and not pd.isna(bin_pass_frac) and float(bin_pass_frac) == 1.0):
            ect_all_pass = False
        lines.append(
            f"{tex_escape(res['name'])} & "
            f"{format_bool_for_latex(global_pass)} & "
            f"{format_compact_float(global_frac)} & "
            f"{format_compact_float(bin_pass_frac)} & "
            f"{format_compact_float(bin_slice_frac)} & "
            f"{format_compact_float(max_delta)} \\\\"
        )
    lines.append(r"\end{tabular}")
    if ect_all_pass:
        lines.append(
            "The finite-grid sliced Euler signature slice-wise chamber-invariance check passes globally and in every "
            "balancing bin for the observed factual estimator, so the finite sliced Euler signature matches "
            "the known-potential-outcome finite target for this generated run."
        )
    else:
        lines.append(
            "The table reports where finite-grid sliced Euler signature slice-wise chamber invariance is exact and where "
            "it is only approximate. sliced Euler signature should be interpreted descriptively unless both global and "
            "balancing-bin slice-wise agreement are exact."
        )
    lines.append(
        r"Detailed per-coordinate checks are saved in \texttt{ect\_slice\_chamber\_checks.csv} and "
        r"\texttt{balancing\_bin\_ect\_slice\_chamber\_checks.csv}."
    )
    lines.append("")
    lines.append(r"\subsection*{CATE and Binwise Topology Comparison}")
    lines.append(
        "The following tables compare conventional score-bin CATE diagnostics with the observed factual "
        "finite-superlevel topology in the same balancing bins."
    )
    for res in semi_results:
        comp = build_score_bin_topology_comparison_table(res, analysis="unadjusted_observed")
        lines.append(r"\paragraph{" + tex_escape(res["name"]) + ".}")
        if comp.empty:
            lines.append("No bin had enough treated and control units for the CATE/topology comparison.")
            continue
        lines.append(r"\begin{tabular}{rrrrrrrr}")
        lines.append(r"Bin & true CATE$_x$ & naive bias & IPW bias & B0 ref. & B1 ref. & Euler ref. & sliced Euler L2 \\")
        lines.append(r"\hline")
        for j, row in comp.reset_index(drop=True).iterrows():
            lines.append(
                f"{j + 1} & "
                f"{format_compact_float(row['true_CATE_x_oracle'])} & "
                f"{format_compact_float(row['naive_bias_x'])} & "
                f"{format_compact_float(row['IPW_bias_x'])} & "
                f"{format_compact_float(row['betti0_effect_ref'])} & "
                f"{format_compact_float(row['betti1_effect_ref'])} & "
                f"{format_compact_float(row['euler_effect_ref'])} & "
                f"{format_compact_float(row['ect_l2'])} \\\\"
            )
        lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\section*{Design and Sensitivity Diagnostics}")
    lines.append(
        "This section collects diagnostics that standard causal-inference workflows often report before treating "
        "an empirical contrast as credible: covariate balance after weighting, distributional balance beyond "
        "means, positivity/overlap and effective sample size, doubly robust mean checks, permutation or "
        "randomization tests, false-discovery-rate adjusted $q$-values, propensity trimming, and explicit "
        "unmeasured-confounding sensitivity. These diagnostics do not establish identifying assumptions; "
        "they document whether the conventional empirical design is stable enough for comparison with the "
        "finite-superlevel target."
    )
    all_results = semi_results + randomized_results + observational_results
    lines.append(r"\subsection*{Balance, Overlap, and Positivity Audit}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"Dataset & max wt. SMD & max wt. KS & raw balance p & wt. balance p & ATE ESS frac. & overlap ESS frac. \\")
    lines.append(r"\hline")
    for res in all_results:
        balance_diagnostic_table = res.get("covariate_balance_diagnostics", pd.DataFrame())
        omnibus_balance_table = res.get("design_balance_omnibus", pd.DataFrame())
        overlap_diagnostic_table = res.get("design_overlap", pd.DataFrame())
        max_w_smd = balance_diagnostic_table["abs_weighted_smd"].max() if not balance_diagnostic_table.empty and "abs_weighted_smd" in balance_diagnostic_table else np.nan
        max_w_ks = balance_diagnostic_table["weighted_ks_statistic"].max() if not balance_diagnostic_table.empty and "weighted_ks_statistic" in balance_diagnostic_table else np.nan
        raw_p = np.nan
        wt_p = np.nan
        if not omnibus_balance_table.empty:
            raw_rows = omnibus_balance_table[omnibus_balance_table["weighted"] == False]  # noqa: E712
            wt_rows = omnibus_balance_table[omnibus_balance_table["weighted"] == True]  # noqa: E712
            raw_p = raw_rows["permutation_p_value"].iloc[0] if not raw_rows.empty else np.nan
            wt_p = wt_rows["permutation_p_value"].iloc[0] if not wt_rows.empty else np.nan
        ate_ess = np.nan
        ov_ess = np.nan
        if not overlap_diagnostic_table.empty:
            ate_rows = overlap_diagnostic_table[overlap_diagnostic_table["weight_scheme"] == "ATE"]
            ov_rows = overlap_diagnostic_table[overlap_diagnostic_table["weight_scheme"] == "overlap"]
            ate_ess = ate_rows["ESS_fraction_total"].iloc[0] if not ate_rows.empty else np.nan
            ov_ess = ov_rows["ESS_fraction_total"].iloc[0] if not ov_rows.empty else np.nan
        lines.append(
            f"{tex_escape(res['name'])} & "
            f"{format_compact_float(max_w_smd)} & "
            f"{format_compact_float(max_w_ks)} & "
            f"{format_compact_float(raw_p)} & "
            f"{format_compact_float(wt_p)} & "
            f"{format_compact_float(ate_ess)} & "
            f"{format_compact_float(ov_ess)} \\\\"
        )
    lines.append(r"\end{tabular}")
    lines.append(
        "The weighted SMD target is conventionally below 0.10, but the table also reports weighted "
        "Kolmogorov--Smirnov distances because good mean balance alone can miss distributional imbalance. "
        "The omnibus permutation balance test is a design diagnostic: after weighting, a large p-value is "
        "consistent with no detected aggregate covariate imbalance. ESS fractions flag unstable weights "
        "and practical positivity stress."
    )
    lines.append(r"\subsection*{Outcome Distribution and Topology Permutation Tests}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"Dataset & mean $p$ & energy $p$ & MMD $p$ & B0 $p$ & Euler $p$ & sliced Euler $p$ \\")
    lines.append(r"\hline")
    for res in all_results:
        dist = res.get("design_distribution_tests", pd.DataFrame())
        topo_tests = res.get("design_topology_tests", pd.DataFrame())

        def test_p(df, key):
            if df.empty or "test" not in df:
                return np.nan
            rows = df[df["test"] == key]
            return rows["permutation_p_value"].iloc[0] if not rows.empty else np.nan

        lines.append(
            f"{tex_escape(res['name'])} & "
            f"{format_compact_float(test_p(dist, 'mean_norm'))} & "
            f"{format_compact_float(test_p(dist, 'energy_distance'))} & "
            f"{format_compact_float(test_p(dist, 'rbf_mmd2'))} & "
            f"{format_compact_float(test_p(topo_tests, 'betti0_effect_ref'))} & "
            f"{format_compact_float(test_p(topo_tests, 'euler_effect_ref'))} & "
            f"{format_compact_float(test_p(topo_tests, 'ect_l2'))} \\\\"
        )
    lines.append(r"\end{tabular}")
    lines.append(
        "The distributional tests ask whether treated and control observed outcome clouds differ under label "
        "permutation, using mean norm, energy distance, and RBF-MMD. The topology tests apply the same finite "
        "superlevel estimator under permuted labels, so a small topology $p$-value means the observed topological "
        "contrast is larger than expected under random relabeling. The corresponding CSV files also store "
        "Benjamini--Hochberg $q$-values for multiplicity control."
    )
    lines.append(r"\subsection*{Doubly Robust Mean Checks and Trimming Sensitivity}")
    for res in all_results:
        aipw = res.get("design_aipw", pd.DataFrame())
        trim = res.get("design_trim", pd.DataFrame())
        lines.append(r"\paragraph{" + tex_escape(res["name"]) + ".}")
        if not aipw.empty:
            parts = []
            for _, row in aipw.iterrows():
                parts.append(
                    f"coord. {int(row['coordinate'])}: {format_compact_float(row['estimate'])} "
                    f"[{format_compact_float(row['ci95_low'])}, {format_compact_float(row['ci95_high'])}]"
                )
            lines.append(
                "Cross-fit AIPW mean contrasts are "
                + tex_escape("; ".join(parts))
                + ". They are included as a doubly robust conventional check against the IPW/overlap mean estimates."
            )
        else:
            lines.append("Cross-fit AIPW checks were not available for this dataset.")
        if not trim.empty:
            retained = trim.groupby("trim_rule")["fraction_retained"].max().reset_index()
            retain_text = "; ".join(
                f"{row['trim_rule']}: retained {format_compact_float(row['fraction_retained'])}" for _, row in retained.iterrows()
            )
            ect_range = trim.groupby("analysis")["ect_l2"].agg(["min", "max"]).reset_index()
            ect_text = "; ".join(
                f"{row['analysis']} sliced Euler [{format_compact_float(row['min'])}, {format_compact_float(row['max'])}]"
                for _, row in ect_range.iterrows()
            )
            lines.append(
                "Overlap trimming sensitivity retained "
                + tex_escape(retain_text)
                + ". Topological sliced Euler ranges under trimming were "
                + tex_escape(ect_text)
                + "."
            )
        else:
            lines.append("Overlap trimming did not retain enough treated and control observations for this dataset.")
        binary_sens = res.get("design_binary_sensitivity", pd.DataFrame())
        if binary_sens is not None and not binary_sens.empty:
            e_parts = []
            for _, row in binary_sens.iterrows():
                e_parts.append(
                    f"{row['analysis']}: RR {format_compact_float(row['risk_ratio'])}, "
                    f"E-value {format_compact_float(row['e_value_for_risk_ratio'])}"
                )
            lines.append(
                "For the binary mortality endpoint, E-values for risk-ratio sensitivity are "
                + tex_escape("; ".join(e_parts))
                + "."
            )
    lines.append("")
    lines.append(r"\subsection*{Interpretation}")
    for res in semi_results:
        bal = res["balance_summary"].iloc[0]
        latent_confounding = res["latent_confounding"]
        topology_tests = res["topology"]
        oracle = topology_tests[topology_tests["analysis"] == "oracle_do"].iloc[0]
        observed = topology_tests[topology_tests["analysis"] == "unadjusted_observed"].iloc[0]
        score_std = res["score_topology_standardized"]
        if not score_std.empty and "oracle_do" in set(score_std["analysis"]):
            std_oracle = score_std[score_std["analysis"] == "oracle_do"].iloc[0]
            std_text = (
                "The balancing-bin standardized known-potential-outcome persistent-homology diagnostic has "
                f"Betti0 L1 {format_compact_float(std_oracle['standardized_betti0_l1'])}, "
                f"Betti1 L1 {format_compact_float(std_oracle['standardized_betti1_l1'])}, "
                f"and Euler L1 {format_compact_float(std_oracle['standardized_euler_l1'])}. "
                f"The descriptive sliced Euler L2 value is {format_compact_float(std_oracle['standardized_ect_l2'])}."
            )
        else:
            std_text = "The balancing-bin standardized topological table did not have enough arm counts in every bin."
        lines.append(r"\paragraph{" + tex_escape(res["name"]) + ".}")
        lines.append(
            "Observed covariate balance diagnostics improve after weighting: the maximum absolute SMD changes from "
            f"{format_compact_float(bal['max_abs_smd_raw'])} to {format_compact_float(bal['max_abs_smd_weighted'])}. "
            "However, standard exchangeability is deliberately violated: within observed-score bins, treated and control "
            f"units differ in unobserved $U$ prevalence by {format_compact_float(latent_confounding['delta_p_u1'].min())} to "
            f"{format_compact_float(latent_confounding['delta_p_u1'].max())}. "
            "The coordinate-mean ATE remains biased after observed-covariate IPW, with IPW bias "
            f"{format_compact_float(extract_metric_value(res['metric_bias'], 'ipw_minus_true_bias_x'))}. "
            "For the generated structural benchmark, the superlevel topology is compared with the known structural contrast: at the reference "
            f"density level the known-potential-outcome Betti0 effect is {format_compact_float(oracle['betti0_effect_ref'])}, "
            f"the Betti1 effect is {format_compact_float(oracle['betti1_effect_ref'])}, and the Euler effect is "
            f"{format_compact_float(oracle['euler_effect_ref'])}. The raw observed reference effects are "
            f"{format_compact_float(observed['betti0_effect_ref'])}, {format_compact_float(observed['betti1_effect_ref'])}, "
            f"and {format_compact_float(observed['euler_effect_ref'])}, respectively. {std_text}"
        )
    lines.append("")
    lines.append(r"\subsection*{Sensitivity Analysis}")
    for res in semi_results:
        odds = res["odds_tilt"]
        boot = res["bootstrap_summary"]
        tuning = res["tuning"]
        overlap = res["overlap"].iloc[0]
        lines.append(r"\paragraph{" + tex_escape(res["name"]) + ".}")
        lines.append(
            "Overlap diagnostics are: propensity range "
            f"[{format_compact_float(overlap['propensity_min'])}, {format_compact_float(overlap['propensity_max'])}], "
            f"fraction outside [0.1,0.9] = {format_compact_float(overlap['fraction_outside_0p1_0p9'])}, "
            f"minimum treated/control bin counts = {int(overlap['min_treated_count_by_score_bin'])}/"
            f"{int(overlap['min_control_count_by_score_bin'])}, and ATE-weight ESS = "
            f"{format_compact_float(overlap['ESS_treated_ATE_weights'])}/"
            f"{format_compact_float(overlap['ESS_control_ATE_weights'])}."
        )
        if not odds.empty:
            worst = odds.loc[odds["gamma"] == odds["gamma"].max()]
            lines.append(
                "The odds-tilt unmeasured-confounding stress test stores the full Gamma curve and recomputes the "
                "weighted structural effect under "
                r"$\logit e_\Gamma^\pm(z)=\logit e(z)\pm\log\Gamma$. "
                f"At Gamma={format_compact_float(odds['gamma'].max(), 2)}, Betti0 reference effects range from "
                f"{format_compact_float(worst['betti0_effect_ref'].min())} to {format_compact_float(worst['betti0_effect_ref'].max())}, "
                f"Euler reference effects range from {format_compact_float(worst['euler_effect_ref'].min())} to "
                f"{format_compact_float(worst['euler_effect_ref'].max())}, while IPW mean-bias ranges from "
                f"{format_compact_float(worst['IPW_bias_x_vs_oracle'].min())} to "
                f"{format_compact_float(worst['IPW_bias_x_vs_oracle'].max())}."
            )
        if not boot.empty and {"scope", "analysis", "quantity"}.issubset(boot.columns):
            interval_parts = []
            for scope, label_name in [("global", "global observed"), ("balancing_bin", "balancing-bin observed")]:
                key = boot[
                    (boot["scope"] == scope)
                    & (boot["analysis"] == "unadjusted_observed")
                    & (boot["quantity"] == "betti0_effect_ref")
                ]
                if not key.empty:
                    key = key.iloc[0]
                    interval_parts.append(
                        f"{label_name}: [{format_compact_float(key['q025'])}, {format_compact_float(key['q975'])}], "
                        f"median {format_compact_float(key['q500'])}"
                    )
            if interval_parts:
                lines.append(
                    "The expanded nonparametric bootstrap reports intervals by scope and analysis for "
                    "mean bias, Betti curves, Euler curves, sliced Euler L2, and reference effects. Betti0 "
                    "reference-effect intervals are "
                    + tex_escape("; ".join(interval_parts))
                    + "."
                )
        if not tuning.empty:
            g = tuning.groupby("analysis")["betti0_effect_ref"].agg(["min", "max"]).reset_index()
            parts = []
            for _, row in g.iterrows():
                parts.append(f"{row['analysis']}: [{format_compact_float(row['min'])}, {format_compact_float(row['max'])}]")
            margin_text = ""
            if "min_level_margin" in tuning.columns and tuning["min_level_margin"].notna().any():
                min_margin = float(tuning["min_level_margin"].min())
                margin_text = (
                    " The smallest discrete level-margin proxy was "
                    f"{min_margin:.2e}."
                )
            lines.append(
                "Finite-signature tuning varied grid size, smoothing, level-grid size/range, sliced Euler directions, "
                "sliced Euler slices, and small-component thresholds. Betti0 reference-effect ranges were "
                + tex_escape("; ".join(parts))
                + "."
                + margin_text
            )
    lines.append("")
    if randomized_results:
        lines.append(r"\section*{Randomized JOBS II Application}")
        lines.append(
            "JOBS II is a real randomized application. It shows how finite superlevel topology can summarize "
            "distributional structure when conventional mean contrasts are modest, but it is not a structural "
            "benchmark because unit-level oracle potential outcomes are not observed."
        )
        lines.append(
            "The code loads the Rdatasets mediation/jobs extract and records the formal original archive as ICPSR "
            r"study 2739, \textit{Jobs II Preventive Intervention for Unemployed Job Seekers, 1991--1993: "
            r"[Southeast Michigan]}, DOI \texttt{10.3886/ICPSR02739.v1}."
        )
        lines.append("The generated tables below are the JOBS II results produced by this replication pipeline.")
        for res in randomized_results:
            info = res["jobs_info"].iloc[0]
            bal = res["balance_summary"].iloc[0]
            overlap = res["overlap"].iloc[0]
            metric = res["metric_bias"]
            topology_tests = res["topology"]
            std = res["score_topology_standardized"]
            tuning = res["tuning"]
            boot = res["bootstrap_summary"]
            odds = res["odds_tilt"]
            lines.append(r"\subsection*{" + tex_escape(res["name"]) + "}")
            lines.append(
                f"The analysis retained {int(info['analysis_rows'])} randomized participants: "
                f"{int(info['treated_jobs_ii'])} assigned to the JOBS II intervention and "
                f"{int(info['control_jobs_ii'])} assigned to control "
                f"(treated fraction {format_compact_float(info['treated_fraction'])}). "
                f"The two-dimensional outcome is {tex_escape(info['outcome_1'])} paired with "
                f"{tex_escape(info['outcome_2'])}."
            )
            lines.append(
                "Baseline balance is consistent with a randomized field experiment but still audited: max absolute "
                f"SMD is {format_compact_float(bal['max_abs_smd_raw'])} before adjustment and "
                f"{format_compact_float(bal['max_abs_smd_weighted'])} after overlap weighting. The fitted adjustment-score "
                f"range is [{format_compact_float(overlap['propensity_min'])}, {format_compact_float(overlap['propensity_max'])}], "
                f"with {format_compact_float(overlap['fraction_outside_0p1_0p9'])} outside [0.1,0.9]."
            )
            lines.append(r"\paragraph{Classical randomized outcome contrasts.}")
            lines.append(r"\begin{tabular}{lrrr}")
            lines.append(r"Outcome & randomized diff. & IPW adj. & overlap adj. \\")
            lines.append(r"\hline")
            for label, suffix in [
                ("job search self-efficacy z", "standardized_outcome_1"),
                ("negative depression z", "standardized_outcome_2"),
            ]:
                lines.append(
                    f"{label} & "
                    f"{format_compact_float(extract_metric_value(metric, 'naive_diff_' + suffix))} & "
                    f"{format_compact_float(extract_metric_value(metric, 'IPW_diff_' + suffix))} & "
                    f"{format_compact_float(extract_metric_value(metric, 'overlap_diff_' + suffix))} \\\\"
                )
            lines.append(r"\end{tabular}")
            lines.append(
                "These are valid randomized-trial contrasts under the usual trial assumptions. They are deliberately "
                "reported next to the topology so the reader can see what a mean-only analysis compresses away."
            )
            lines.append(r"\paragraph{Finite-superlevel topology.}")
            lines.append(r"\begin{tabular}{lrrrr}")
            lines.append(r"Analysis & B0 ref. & Euler ref. & B0 L1 & sliced Euler L2 \\")
            lines.append(r"\hline")
            for _, row in topology_tests.iterrows():
                lines.append(
                    f"{tex_escape(row['analysis'])} & "
                    f"{format_compact_float(row['betti0_effect_ref'])} & "
                    f"{format_compact_float(row['euler_effect_ref'])} & "
                    f"{format_compact_float(row['betti0_l1'])} & "
                    f"{format_compact_float(row['ect_l2'])} \\\\"
                )
            lines.append(r"\end{tabular}")
            if not std.empty:
                obs_std = std[std["analysis"] == "unadjusted_observed"].iloc[0] if "unadjusted_observed" in set(std["analysis"]) else std.iloc[0]
                ipw_std = std[std["analysis"] == "ipw_adjusted_observed_density"].iloc[0] if "ipw_adjusted_observed_density" in set(std["analysis"]) else obs_std
                lines.append(
                    "Balancing-bin standardization gives observed/IPW sliced Euler L2 values "
                    f"{format_compact_float(obs_std['standardized_ect_l2'])}/"
                    f"{format_compact_float(ipw_std['standardized_ect_l2'])}; observed Betti0/Euler reference effects are "
                    f"{format_compact_float(obs_std['standardized_betti0_effect_ref'])}/"
                    f"{format_compact_float(obs_std['standardized_euler_effect_ref'])}."
                )
            comp = res["cate"].merge(
                res["score_topology"][res["score_topology"]["analysis"] == "unadjusted_observed"][
                    ["score_bin", "betti0_effect_ref", "euler_effect_ref", "ect_l2"]
                ],
                on="score_bin",
                how="left",
            ) if not res["cate"].empty and not res["score_topology"].empty else pd.DataFrame()
            if not comp.empty:
                lines.append(r"\paragraph{Score-bin randomized contrasts versus topology.}")
                lines.append(r"\begin{tabular}{rrrrrrr}")
                lines.append(r"Bin & n$_1$ & n$_0$ & job IPW & dep. IPW & B0 ref. & sliced Euler L2 \\")
                lines.append(r"\hline")
                for j, row in comp.reset_index(drop=True).iterrows():
                    lines.append(
                        f"{j + 1} & "
                        f"{int(row['n_treated'])} & "
                        f"{int(row['n_control'])} & "
                        f"{format_compact_float(row['IPW_diff_std_outcome_1'])} & "
                        f"{format_compact_float(row['IPW_diff_std_outcome_2'])} & "
                        f"{format_compact_float(row['betti0_effect_ref'])} & "
                        f"{format_compact_float(row['ect_l2'])} \\\\"
                    )
                lines.append(r"\end{tabular}")
            lines.append(
                "Interpretation: JOBS II should be read as a randomized distributional application. Mean effects "
                "summarize average shifts in job-search self-efficacy and depression, whereas the finite superlevel "
                "signatures summarize the geometry of the joint outcome density. A nonzero sliced Euler signature contrast indicates "
                "a directional distributional shape/location difference even when Betti/Euler summaries show no "
                "large component or hole change."
            )
            if not odds.empty:
                worst = odds.loc[odds["gamma"] == odds["gamma"].max()]
                lines.append(
                    "Odds-tilt sensitivity is included as a diagnostic stress test, even though randomization is "
                    "the primary source of exchangeability. At Gamma="
                    f"{format_compact_float(odds['gamma'].max(), 2)}, adjusted job-search contrasts range from "
                    f"{format_compact_float(worst['IPW_diff_std_outcome_1'].min())} to "
                    f"{format_compact_float(worst['IPW_diff_std_outcome_1'].max())}, adjusted depression contrasts range from "
                    f"{format_compact_float(worst['IPW_diff_std_outcome_2'].min())} to "
                    f"{format_compact_float(worst['IPW_diff_std_outcome_2'].max())}, and sliced Euler L2 ranges from "
                    f"{format_compact_float(worst['ect_l2'].min())} to {format_compact_float(worst['ect_l2'].max())}."
                )
            if not boot.empty and {"scope", "analysis", "quantity"}.issubset(boot.columns):
                key = boot[
                    (boot["scope"] == "global")
                    & (boot["analysis"] == "unadjusted_observed")
                    & (boot["quantity"] == "ect_l2")
                ]
                if not key.empty:
                    key = key.iloc[0]
                    lines.append(
                        "Bootstrap sensitivity for global observed sliced Euler L2 gives interval "
                        f"[{format_compact_float(key['q025'])}, {format_compact_float(key['q975'])}], "
                        f"median {format_compact_float(key['q500'])}."
                    )
            if not tuning.empty:
                lines.append(
                    "Finite-signature tuning varied grid size, smoothing, level range/count, sliced Euler directions/slices, "
                    "and small-component thresholds. Observed sliced Euler L2 ranged from "
                    f"{format_compact_float(tuning['ect_l2'].min())} to {format_compact_float(tuning['ect_l2'].max())}; "
                    f"Betti0 reference effect ranged from {format_compact_float(tuning['betti0_effect_ref'].min())} to "
                    f"{format_compact_float(tuning['betti0_effect_ref'].max())}."
                )
        lines.append("")
    if observational_results:
        lines.append(r"\section*{Observational eICU Appendix Stress Test}")
        lines.append(
            "We include eICU as a large-scale observational stress test of the finite-superlevel estimator on "
            "clinically meaningful two-dimensional outcomes. Because no oracle potential outcomes are available "
            "and overlap is imperfect, this analysis is not interpreted as evidence for causal identification."
        )
        for res in observational_results:
            info = res["eicu_info"].iloc[0]
            bal = res["balance_summary"].iloc[0]
            overlap = res["overlap"].iloc[0]
            metric = res["metric_bias"]
            topology_tests = res["topology"]
            std = res["score_topology_standardized"]
            tuning = res["tuning"]
            boot = res["bootstrap_summary"]
            odds = res["odds_tilt"]
            lines.append(r"\subsection*{" + tex_escape(res["name"]) + "}")
            lines.append(
                "Cohort construction retained "
                f"{int(info['analysis_rows'])} adult ICU stays from {int(info['raw_apache_iva_rows'])} APACHE-IVa rows. "
                f"The exposure is {tex_escape(info['treatment_definition'])}; "
                f"{int(info['treated_ventday1'])} stays are treated and {int(info['control_no_ventday1'])} are controls "
                f"(treated fraction {format_compact_float(info['treated_fraction'])}). Outcomes are "
                f"{tex_escape(info['outcome_1'])} and {tex_escape(info['outcome_2'])}."
            )
            lines.append(
                "Observed confounding is substantial but the propensity/overlap weighting balances the recorded covariates: "
                f"max absolute SMD changes from {format_compact_float(bal['max_abs_smd_raw'])} to "
                f"{format_compact_float(bal['max_abs_smd_weighted'])}. Propensity range is "
                f"[{format_compact_float(overlap['propensity_min'])}, {format_compact_float(overlap['propensity_max'])}], "
                f"with {format_compact_float(overlap['fraction_outside_0p1_0p9'])} outside [0.1,0.9]."
            )
            lines.append(r"\paragraph{Classical outcome contrasts.}")
            lines.append(r"\begin{tabular}{lrrr}")
            lines.append(r"Outcome & naive & IPW & overlap \\")
            lines.append(r"\hline")
            for label, suffix in [
                ("std. log ICU LOS", "standardized_log_los_icu_los"),
                ("std. log hospital LOS", "standardized_log_los_hospital_los"),
                ("hospital mortality", "hospital_mortality"),
            ]:
                lines.append(
                    f"{label} & "
                    f"{format_compact_float(extract_metric_value(metric, 'naive_diff_' + suffix))} & "
                    f"{format_compact_float(extract_metric_value(metric, 'IPW_diff_' + suffix))} & "
                    f"{format_compact_float(extract_metric_value(metric, 'overlap_diff_' + suffix))} \\\\"
                )
            lines.append(r"\end{tabular}")
            lines.append(
                "These mean/proportion contrasts show longer LOS and higher mortality for day-1 ventilated stays "
                "after observed-covariate weighting, but they only summarize first moments or marginal risks."
            )
            lines.append(r"\paragraph{Finite-superlevel topology.}")
            lines.append(r"\begin{tabular}{lrrrr}")
            lines.append(r"Analysis & B0 ref. & Euler ref. & B0 L1 & sliced Euler L2 \\")
            lines.append(r"\hline")
            for _, row in topology_tests.iterrows():
                lines.append(
                    f"{tex_escape(row['analysis'])} & "
                    f"{format_compact_float(row['betti0_effect_ref'])} & "
                    f"{format_compact_float(row['euler_effect_ref'])} & "
                    f"{format_compact_float(row['betti0_l1'])} & "
                    f"{format_compact_float(row['ect_l2'])} \\\\"
                )
            lines.append(r"\end{tabular}")
            if not std.empty:
                obs_std = std[std["analysis"] == "unadjusted_observed"].iloc[0] if "unadjusted_observed" in set(std["analysis"]) else std.iloc[0]
                ipw_std = std[std["analysis"] == "ipw_adjusted_observed_density"].iloc[0] if "ipw_adjusted_observed_density" in set(std["analysis"]) else obs_std
                lines.append(
                    "Balancing-bin standardization gives observed/IPW sliced Euler L2 values "
                    f"{format_compact_float(obs_std['standardized_ect_l2'])}/"
                    f"{format_compact_float(ipw_std['standardized_ect_l2'])}, while Betti0 and Euler reference effects remain "
                    f"{format_compact_float(obs_std['standardized_betti0_effect_ref'])} and "
                    f"{format_compact_float(obs_std['standardized_euler_effect_ref'])}."
                )
            lines.append(
                "Interpretation: ordinary Betti/Euler superlevel summaries say the high-density LOS clouds remain "
                "one connected chamber with no gross component/hole change. The sliced Euler signature, however, remains "
                "large after weighting, detecting directional shape and location differences in the joint LOS distribution "
                "that mean/proportion models do not represent."
            )
            if not odds.empty:
                worst = odds.loc[odds["gamma"] == odds["gamma"].max()]
                lines.append(
                    "Odds-tilt sensitivity stores the full Gamma curve. At Gamma="
                    f"{format_compact_float(odds['gamma'].max(), 2)}, IPW standardized ICU-LOS contrasts range from "
                    f"{format_compact_float(worst['IPW_diff_std_log_icu_los'].min())} to "
                    f"{format_compact_float(worst['IPW_diff_std_log_icu_los'].max())}, mortality contrasts range from "
                    f"{format_compact_float(worst['IPW_diff_hospital_mortality'].min())} to "
                    f"{format_compact_float(worst['IPW_diff_hospital_mortality'].max())}, and sliced Euler L2 ranges from "
                    f"{format_compact_float(worst['ect_l2'].min())} to {format_compact_float(worst['ect_l2'].max())}."
                )
            if not boot.empty and {"scope", "analysis", "quantity"}.issubset(boot.columns):
                key = boot[
                    (boot["scope"] == "global")
                    & (boot["analysis"] == "unadjusted_observed")
                    & (boot["quantity"] == "ect_l2")
                ]
                if not key.empty:
                    key = key.iloc[0]
                    lines.append(
                        "Bootstrap sensitivity for global observed sliced Euler L2 gives interval "
                        f"[{format_compact_float(key['q025'])}, {format_compact_float(key['q975'])}], "
                        f"median {format_compact_float(key['q500'])}."
                    )
            if not tuning.empty:
                lines.append(
                    "Finite-signature tuning varied grid size, smoothing, level range/count, sliced Euler directions/slices, "
                    "and small-component thresholds. Observed sliced Euler L2 ranged from "
                    f"{format_compact_float(tuning['ect_l2'].min())} to {format_compact_float(tuning['ect_l2'].max())}; "
                    f"Betti0 reference effect ranged from {format_compact_float(tuning['betti0_effect_ref'].min())} to "
                    f"{format_compact_float(tuning['betti0_effect_ref'].max())}."
                )
            lines.append(
                "Scientific caution: because this is real observational eICU data, unmeasured confounding and timing "
                "ambiguity cannot be ruled out, and the overlap diagnostics show substantial stress. The result is "
                "scientifically useful as a secondary finite-superlevel distributional analysis of day-1 ventilation outcomes, "
                "not as a randomized causal effect estimate or evidence for causal identification."
            )
        lines.append("")
    lines.append(r"\subsection*{Persistent-Homology and sliced Euler signature Scope}")
    lines.append(
        "The persistent-homology summaries in this implementation are finite Betti-0 and Betti-1 "
        "curves over density superlevel sets, together with the induced Euler curve. They are finite "
        "level-grid summaries, not full persistence diagrams, barcodes, or landscapes. This is "
        "consistent with the finite-signature framework; a cubical-persistence backend would be needed "
        "to report full diagrams."
    )
    lines.append(
        "The finite sliced Euler vector is computed from Euler characteristics of directional slices of the "
        "same superlevel sets. The code checks slice-wise Euler-chamber invariance by comparing "
        "every finite sliced Euler signature coordinate with its known-potential-outcome counterpart. In runs where that check fails, sliced Euler signature "
        "is reported descriptively rather than interpreted as a matched finite structural signature."
    )
    lines.append("")
    lines.append(r"\subsection*{Structural Benchmark Interpretation}")
    lines.append(
        "The first two datasets are semi-synthetic structural-equation benchmarks, "
        "so the relevant mechanism is known. "
        "Treatment depends on the observed balancing score and the unobserved binary variable $U$, which makes standard exchangeability "
        "false. The outcome maps, however, were constructed so that $U$ only reweights and mildly shifts latent "
        "density inside the same finite superlevel topological chamber: control outcomes retain one connected "
        "high-density region and no one-dimensional hole, while treated outcomes retain two connected high-density "
        "regions and no one-dimensional hole over the reported level grid. Therefore the full conditional laws can "
        "change with treatment selection, and mean targets can be biased, while the selected noninjective signature "
        "$\\Psi$ is constructed to be invariant within the score strata used by the estimator. The reported diagnostics "
        "compare the observed estimators with the known-potential-outcome finite Betti0, Betti1, and Euler superlevel signatures. "
        "For sliced Euler signature, the comparison is descriptive unless the slice-wise chamber-invariance table reports exact "
        "global and balancing-bin agreement."
    )
    lines.append("")
    lines.append(
        "The chamber-check CSV files report empirical diagnostics for this benchmark construction: reference Betti0, Betti1, "
        "and Euler effects are compared between oracle and observed analyses globally and within balancing bins, "
        "with level-margin proxies included to flag proximity to finite-grid chamber changes. sliced Euler signature is checked "
        "coordinate by coordinate using finite directional slices of the superlevel filtration."
    )
    lines.append("")
    lines.append(r"\subsection*{Replication Script Scope}")
    lines.append(
        r"The canonical replication script is \texttt{finite\_superlevel\_topology\_pipeline.py}. "
        "It builds the synthetic exact benchmark, the breast-cancer semi-synthetic benchmark, the JOBS II randomized "
        "application, and optional eICU analysis. The companion notebook calls this same script so both entry points "
        "produce the same outputs."
    )
    output = output_dir / "superlevel_results_summary.tex"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finite superlevel-set topological diagnostics for structural benchmarks and empirical applications.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python finite_superlevel_topology_pipeline.py --outdir superlevel_outputs "
            "--bootstrap 30 --diagnostic-permutations 99 --skip-eicu\n"
            "  python finite_superlevel_topology_pipeline.py --outdir superlevel_outputs_with_eicu "
            "--eicu-dir data/eicu\n\n"
            "Outputs:\n"
            "  Dataset-specific CSV and PNG files are written under --outdir/<dataset-name>/.\n"
            "  The root output directory also contains superlevel_results_summary.tex and output_manifest.csv."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("superlevel_outputs"),
        help="Output directory. Relative paths are resolved from the current working directory.",
    )
    parser.add_argument("--bootstrap", type=int, default=30, help="Number of nonparametric bootstrap replicates for sensitivity summaries.")
    parser.add_argument(
        "--diagnostic-permutations",
        type=int,
        default=99,
        help=(
            "Permutation count for design balance, distributional, and topological-signature diagnostics. "
            "Use larger values such as 499 or 999 for a final slow archival run."
        ),
    )
    parser.add_argument(
        "--eicu-dir",
        type=Path,
        default=DEFAULT_EICU_DIRECTORY,
        help="Directory containing the eICU Collaborative CSV.GZ extract.",
    )
    parser.add_argument(
        "--jobs-csv",
        type=Path,
        default=None,
        help="Optional local Rdatasets mediation/jobs.csv file. Supplying this file makes the JOBS II run independent of network access.",
    )
    parser.add_argument(
        "--jobs-url",
        default=JOBS_II_RDATASETS_CSV_URL,
        help="URL for the Rdatasets mediation/jobs.csv extract used for the JOBS II randomized application.",
    )
    parser.add_argument(
        "--skip-jobs",
        action="store_true",
        help="Skip the JOBS II randomized application.",
    )
    parser.add_argument(
        "--skip-eicu",
        action="store_true",
        help="Skip the observational eICU analysis even if the eICU files are present.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable timestamped progress messages during execution.",
    )
    args = parser.parse_args()
    args.outdir = args.outdir.expanduser()
    progress_enabled = not args.no_progress

    t0 = time.time()
    print_progress("Starting finite superlevel-set diagnostics pipeline", 1, 6, enabled=progress_enabled)
    print_progress("Using portable input and output paths", 2, 6, enabled=progress_enabled)
    args.outdir.mkdir(parents=True, exist_ok=True)
    print_progress(f"Output directory: {args.outdir.resolve()}", 3, 6, enabled=progress_enabled)
    print_progress("Building synthetic, semi-synthetic, randomized JOBS II, and optional eICU datasets", 4, 6, enabled=progress_enabled)
    datasets = [
        build_synthetic_structural_benchmark(n=3600, seed=DEFAULT_RANDOM_SEED),
        build_breast_cancer_semisynthetic_benchmark(seed=DEFAULT_RANDOM_SEED),
    ]
    if not args.skip_jobs:
        try:
            datasets.append(build_jobs_ii_randomized_application(args.jobs_csv, args.jobs_url))
            print_progress("Added randomized randomized JOBS II dataset with source metadata", None, None, enabled=progress_enabled)
        except Exception as exc:
            print_progress(f"JOBS II dataset could not be built and will be skipped: {exc}", None, None, enabled=progress_enabled)
    if not args.skip_eicu:
        try:
            if args.eicu_dir.exists():
                datasets.append(build_eicu_observational_application(args.eicu_dir))
                print_progress(f"Added observational eICU dataset from {args.eicu_dir}", None, None, enabled=progress_enabled)
            else:
                print_progress(f"eICU directory not found; skipping: {args.eicu_dir}", None, None, enabled=progress_enabled)
        except Exception as exc:
            print_progress(f"eICU dataset could not be built and will be skipped: {exc}", None, None, enabled=progress_enabled)
    results = []
    for idx, dataset_bundle in enumerate(datasets, 1):
        if dataset_bundle.get("mode") == "observational":
            results.append(
                analyze_observational_application(
                    dataset_bundle,
                    args.outdir,
                    n_bootstrap=args.bootstrap,
                    progress_enabled=progress_enabled,
                    dataset_index=idx,
                    n_datasets=len(datasets),
                    diagnostic_permutations=args.diagnostic_permutations,
                )
            )
        elif dataset_bundle.get("mode") == "randomized":
            results.append(
                analyze_randomized_application(
                    dataset_bundle,
                    args.outdir,
                    n_bootstrap=args.bootstrap,
                    progress_enabled=progress_enabled,
                    dataset_index=idx,
                    n_datasets=len(datasets),
                    diagnostic_permutations=args.diagnostic_permutations,
                )
            )
        else:
            results.append(
                analyze_structural_benchmark(
                    dataset_bundle,
                    args.outdir,
                    n_bootstrap=args.bootstrap,
                    progress_enabled=progress_enabled,
                    dataset_index=idx,
                    n_datasets=len(datasets),
                    diagnostic_permutations=args.diagnostic_permutations,
                )
            )

    method_description = (
        "The implementation uses density superlevel filtrations, finite level-grid signatures, "
        "balancing-bin standardization, finite-signature tuning checks, overlap diagnostics, "
        "odds-tilt sensitivity, nonparametric bootstrap sensitivity, the synthetic structural benchmark, "
        "the Wisconsin breast-cancer semi-synthetic benchmark, and the JOBS II randomized application. "
        "The optional eICU branch is reported only as an observational appendix when the required local "
        "CSV.GZ files are supplied."
    )
    print_progress("Writing TeX interpretation summary and output manifest", 5, 6, enabled=progress_enabled)
    tex_path = write_latex_summary(results, args.outdir, method_description)
    manifest = []
    for p in sorted(args.outdir.rglob("*")):
        if p.is_file():
            manifest.append({"file": str(p), "bytes": p.stat().st_size})
    pd.DataFrame(manifest).to_csv(args.outdir / "output_manifest.csv", index=False)
    elapsed = time.time() - t0
    print_progress(f"Finished all outputs in {elapsed:.1f} seconds", 6, 6, enabled=progress_enabled)
    print(f"Wrote TeX summary to {tex_path}", flush=True)
    print(f"Wrote outputs under {args.outdir.resolve()}", flush=True)
    print("Pipeline finished.", flush=True)


if __name__ == "__main__":
    main()
