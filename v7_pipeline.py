from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SUBMISSION_DIR = ROOT / "submissions"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGET = "property_organic_content"
SAMPLE_ID = "sample_id"
RANDOM_STATE = 91


@dataclass(frozen=True)
class RunConfig:
    suffix: str
    n_splits: int
    quick: bool
    n_jobs: int
    knn_ks: tuple[int, ...]
    base_cv: str
    use_te: bool
    tail_min_improvement: float = 0.01


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def finite_clip(pred: np.ndarray, upper: float) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    pred = np.nan_to_num(pred, nan=0.0, posinf=upper, neginf=0.0)
    return np.clip(pred, 0.0, upper)


def read_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    assert TARGET in train.columns
    assert TARGET not in test.columns
    assert sample[SAMPLE_ID].equals(test[SAMPLE_ID])
    return train, test, sample


def add_interactions(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    for col in raw.select_dtypes(include=["object", "string", "category"]).columns:
        raw[col] = raw[col].astype("string").fillna("NA")

    raw["geo_hierarchy"] = (
        raw["geo_zone_macro"].astype("string")
        + "|"
        + raw["geo_zone_meso"].astype("string")
        + "|"
        + raw["geo_zone_micro"].astype("string")
    )
    raw["biome_landcover"] = raw["biome"].astype("string") + "|" + raw["land_cover_type"].astype("string")
    raw["landcover_rock"] = raw["land_cover_type"].astype("string") + "|" + raw["parent_rock_type"].astype("string")
    raw["macro_biome"] = raw["geo_zone_macro"].astype("string") + "|" + raw["biome"].astype("string")
    raw["macro_landcover"] = raw["geo_zone_macro"].astype("string") + "|" + raw["land_cover_type"].astype("string")
    raw["source_bandB"] = raw["source_id"].astype("string") + "|" + raw["has_band_B_spectrum"].astype("string")
    raw["micro_landcover"] = raw["geo_zone_micro"].astype("string") + "|" + raw["land_cover_type"].astype("string")
    raw["meso_landcover"] = raw["geo_zone_meso"].astype("string") + "|" + raw["land_cover_type"].astype("string")
    return raw


def make_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_df = pd.concat([train, test], ignore_index=True, sort=False)
    X = all_df.drop(columns=[TARGET, SAMPLE_ID], errors="ignore").copy()

    band_a = [c for c in X.columns if c.startswith("spectral_band_A_PC_")]
    band_b = [c for c in X.columns if c.startswith("spectral_band_B_PC_")]
    chemistry = ["property_acidity_index", "cation_Ca", "cation_Mg", "cation_Na", "cation_exchange_capacity"]
    eps = 1e-6

    X["missing_total"] = X.isna().sum(axis=1)
    X["band_B_available_actual_num"] = X[band_b].notna().any(axis=1).astype(int)
    X["band_B_missing_count"] = X[band_b].isna().sum(axis=1)
    X["coord_available_num"] = X[["latitude", "longitude"]].notna().all(axis=1).astype(int)
    X["chem_missing_count"] = X[chemistry].isna().sum(axis=1)

    for prefix, cols in [("A", band_a), ("B", band_b)]:
        values = X[cols]
        X[f"{prefix}_mean"] = values.mean(axis=1)
        X[f"{prefix}_std"] = values.std(axis=1)
        X[f"{prefix}_min"] = values.min(axis=1)
        X[f"{prefix}_max"] = values.max(axis=1)
        X[f"{prefix}_l2"] = np.sqrt((values**2).sum(axis=1))
        X[f"{prefix}_abs_sum"] = values.abs().sum(axis=1)
        X[f"{prefix}_max_abs"] = values.abs().max(axis=1)
        X[f"{prefix}_pos_count"] = (values > 0).sum(axis=1)
        for col in cols[:5]:
            X[f"{col}_abs"] = X[col].abs()
            X[f"{col}_sq"] = X[col] ** 2

    X["particle_total"] = X["property_particle_coarse"] + X["property_particle_fine"]
    X["fine_fraction"] = X["property_particle_fine"] / (X["particle_total"].abs() + eps)
    X["fine_to_coarse"] = X["property_particle_fine"] / (X["property_particle_coarse"].abs() + eps)
    X["coarse_to_fine"] = X["property_particle_coarse"] / (X["property_particle_fine"].abs() + eps)
    X["fine_minus_coarse"] = X["property_particle_fine"] - X["property_particle_coarse"]

    X["base_cation_sum"] = X[["cation_Ca", "cation_Mg", "cation_Na"]].sum(axis=1, min_count=1)
    X["ca_mg_ratio"] = X["cation_Ca"] / (X["cation_Mg"].abs() + eps)
    X["mg_ca_ratio"] = X["cation_Mg"] / (X["cation_Ca"].abs() + eps)
    X["base_saturation_proxy"] = X["base_cation_sum"] / (X["cation_exchange_capacity"].abs() + eps)
    X["ca_cec_ratio"] = X["cation_Ca"] / (X["cation_exchange_capacity"].abs() + eps)
    X["mg_cec_ratio"] = X["cation_Mg"] / (X["cation_exchange_capacity"].abs() + eps)
    X["cec_per_fine"] = X["cation_exchange_capacity"] / (X["property_particle_fine"].abs() + eps)
    X["acidity_x_cec"] = X["property_acidity_index"] * X["cation_exchange_capacity"]
    X["acidity_per_cec"] = X["property_acidity_index"] / (X["cation_exchange_capacity"].abs() + eps)
    for col in ["cation_exchange_capacity", "cation_Ca", "cation_Mg", "property_acidity_index"]:
        X[f"log1p_{col}"] = np.log1p(X[col].clip(lower=0))

    X["abs_latitude"] = X["latitude"].abs()
    X["abs_longitude"] = X["longitude"].abs()
    X["lat_lon_sum"] = X["latitude"] + X["longitude"]
    X["lat_lon_diff"] = X["latitude"] - X["longitude"]
    X["lat_lon_prod"] = X["latitude"] * X["longitude"]
    X["lat_round_1"] = X["latitude"].round(1)
    X["lon_round_1"] = X["longitude"].round(1)
    X["latlon_grid1"] = X["lat_round_1"].astype("string").fillna("NA") + "|" + X["lon_round_1"].astype("string").fillna("NA")

    raw = add_interactions(X)
    for col in [
        "geo_hierarchy",
        "biome_landcover",
        "landcover_rock",
        "macro_biome",
        "macro_landcover",
        "source_bandB",
        "micro_landcover",
        "meso_landcover",
    ]:
        X[col] = raw[col]

    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    X_enc = X.copy()
    for col in cat_cols:
        s = X[col].astype("string").fillna("NA")
        freq = s.value_counts(dropna=False)
        X_enc[f"{col}_freq"] = s.map(freq).astype(float)
        codes = {value: idx for idx, value in enumerate(s.unique())}
        X_enc[col] = s.map(codes).astype("int32")

    X_enc = X_enc.replace([np.inf, -np.inf], np.nan)
    nunique = X_enc.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X_enc = X_enc.drop(columns=constant_cols)

    X_train = X_enc.iloc[: len(train)].reset_index(drop=True)
    X_test = X_enc.iloc[len(train) :].reset_index(drop=True)
    raw_train = raw.iloc[: len(train)].reset_index(drop=True)
    raw_test = raw.iloc[len(train) :].reset_index(drop=True)
    return X_train, X_test, raw_train, raw_test


def smooth_apply(cat_train: pd.Series, y_train: np.ndarray, cat_apply: pd.Series, m: float, global_mean: float) -> np.ndarray:
    stats = pd.DataFrame({"cat": cat_train.astype("string"), "y": y_train}).groupby("cat")["y"].agg(["count", "mean"])
    enc = (stats["mean"] * stats["count"] + global_mean * m) / (stats["count"] + m)
    return cat_apply.astype("string").map(enc).fillna(global_mean).to_numpy(dtype=float)


def smooth_loo(cat_train: pd.Series, y_train: np.ndarray, m: float, global_mean: float) -> np.ndarray:
    s = cat_train.astype("string").reset_index(drop=True)
    stats = pd.DataFrame({"cat": s, "y": y_train}).groupby("cat")["y"].agg(["count", "sum"])
    count = s.map(stats["count"]).to_numpy(dtype=float)
    total = s.map(stats["sum"]).to_numpy(dtype=float)
    enc = ((total - y_train) + global_mean * m) / np.maximum((count - 1.0) + m, 1.0)
    enc[count <= 1] = global_mean
    return enc


def make_hierarchical_prior(
    raw_train: pd.DataFrame,
    y_train: np.ndarray,
    raw_valid: pd.DataFrame,
    raw_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global_mean = float(np.mean(y_train))
    specs = [
        ("source_id", 120.0, 0.8),
        ("geo_hierarchy", 80.0, 1.2),
        ("geo_zone_micro", 80.0, 1.0),
        ("geo_zone_meso", 120.0, 0.8),
        ("geo_zone_macro", 160.0, 0.5),
        ("biome_landcover", 100.0, 0.9),
        ("landcover_rock", 140.0, 0.7),
        ("macro_biome", 140.0, 0.6),
        ("macro_landcover", 140.0, 0.6),
        ("parent_rock_type", 180.0, 0.35),
        ("land_cover_type", 180.0, 0.35),
        ("biome", 180.0, 0.35),
        ("source_bandB", 120.0, 0.45),
    ]
    prior_train = np.zeros(len(raw_train), dtype=float)
    prior_valid = np.zeros(len(raw_valid), dtype=float)
    prior_test = np.zeros(len(raw_test), dtype=float)
    weight_sum = 0.0

    for col, smoothing, weight in specs:
        if col not in raw_train.columns:
            continue
        prior_train += weight * smooth_loo(raw_train[col], y_train, smoothing, global_mean)
        prior_valid += weight * smooth_apply(raw_train[col], y_train, raw_valid[col], smoothing, global_mean)
        prior_test += weight * smooth_apply(raw_train[col], y_train, raw_test[col], smoothing, global_mean)
        weight_sum += weight

    if weight_sum == 0:
        return (
            np.full(len(raw_train), global_mean),
            np.full(len(raw_valid), global_mean),
            np.full(len(raw_test), global_mean),
        )
    return prior_train / weight_sum, prior_valid / weight_sum, prior_test / weight_sum


def make_target_encoding_features(
    raw_train: pd.DataFrame,
    y_train: np.ndarray,
    raw_valid: pd.DataFrame,
    raw_test: pd.DataFrame,
    ms: tuple[int, ...] = (50, 200),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = [
        "source_id",
        "geo_zone_macro",
        "geo_zone_meso",
        "geo_zone_micro",
        "land_cover_type",
        "biome",
        "parent_rock_type",
        "sampling_strategy",
        "geo_hierarchy",
        "biome_landcover",
        "landcover_rock",
        "macro_biome",
        "macro_landcover",
        "micro_landcover",
        "meso_landcover",
        "source_bandB",
    ]
    global_mean = float(np.mean(y_train))
    train_features: dict[str, np.ndarray] = {}
    valid_features: dict[str, np.ndarray] = {}
    test_features: dict[str, np.ndarray] = {}

    for col in cols:
        if col not in raw_train.columns:
            continue
        train_cat = raw_train[col].astype("string").reset_index(drop=True)
        valid_cat = raw_valid[col].astype("string").reset_index(drop=True)
        test_cat = raw_test[col].astype("string").reset_index(drop=True)
        stats = pd.DataFrame({"cat": train_cat, "y": y_train}).groupby("cat")["y"].agg(["count", "sum", "mean"])
        counts_train = train_cat.map(stats["count"]).to_numpy(dtype=float)
        counts_valid = valid_cat.map(stats["count"]).fillna(0).to_numpy(dtype=float)
        counts_test = test_cat.map(stats["count"]).fillna(0).to_numpy(dtype=float)
        sums_train = train_cat.map(stats["sum"]).to_numpy(dtype=float)
        train_features[f"{col}_te_count"] = np.maximum(counts_train - 1.0, 0.0)
        valid_features[f"{col}_te_count"] = counts_valid
        test_features[f"{col}_te_count"] = counts_test

        for m in ms:
            train_enc = ((sums_train - y_train) + global_mean * m) / np.maximum((counts_train - 1.0) + m, 1.0)
            train_enc[counts_train <= 1] = global_mean
            smooth_map = (stats["sum"] + global_mean * m) / (stats["count"] + m)
            train_features[f"{col}_te_m{m}"] = train_enc
            valid_features[f"{col}_te_m{m}"] = valid_cat.map(smooth_map).fillna(global_mean).to_numpy(dtype=float)
            test_features[f"{col}_te_m{m}"] = test_cat.map(smooth_map).fillna(global_mean).to_numpy(dtype=float)

    return pd.DataFrame(train_features), pd.DataFrame(valid_features), pd.DataFrame(test_features)


def knn_feature_names(prefix: str, ks: tuple[int, ...]) -> list[str]:
    names: list[str] = []
    for k in ks:
        names.extend(
            [
                f"{prefix}_k{k}_mean",
                f"{prefix}_k{k}_median",
                f"{prefix}_k{k}_std",
                f"{prefix}_k{k}_wmean",
                f"{prefix}_k{k}_min_dist",
            ]
        )
    return names


def summarize_neighbors(dist: np.ndarray, idx: np.ndarray, values: np.ndarray, prefix: str, ks: tuple[int, ...]) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for k in ks:
        kk = min(k, idx.shape[1])
        used_idx = idx[:, :kk]
        used_dist = dist[:, :kk]
        neighbor_values = values[used_idx]
        weights = 1.0 / (used_dist + 1e-3)
        data[f"{prefix}_k{k}_mean"] = np.mean(neighbor_values, axis=1)
        data[f"{prefix}_k{k}_median"] = np.median(neighbor_values, axis=1)
        data[f"{prefix}_k{k}_std"] = np.std(neighbor_values, axis=1)
        data[f"{prefix}_k{k}_wmean"] = np.sum(neighbor_values * weights, axis=1) / np.sum(weights, axis=1)
        data[f"{prefix}_k{k}_min_dist"] = np.min(used_dist, axis=1)
    return pd.DataFrame(data)


def compute_knn_block(
    X_fit: pd.DataFrame,
    values: np.ndarray,
    X_apply: pd.DataFrame,
    feature_cols: list[str],
    prefix: str,
    ks: tuple[int, ...],
    train_apply: bool,
) -> pd.DataFrame:
    names = knn_feature_names(prefix, ks)
    if len(X_fit) < 2 or not feature_cols:
        return pd.DataFrame({name: np.nan for name in names}, index=X_apply.index)

    max_k = min(max(ks) + int(train_apply), len(X_fit))
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    fit_matrix = scaler.fit_transform(imputer.fit_transform(X_fit[feature_cols]))
    apply_matrix = scaler.transform(imputer.transform(X_apply[feature_cols]))

    nn = NearestNeighbors(n_neighbors=max_k, metric="euclidean")
    nn.fit(fit_matrix)
    dist, idx = nn.kneighbors(apply_matrix, return_distance=True)

    if train_apply:
        clean_dist = np.empty((idx.shape[0], max_k - 1), dtype=float)
        clean_idx = np.empty((idx.shape[0], max_k - 1), dtype=int)
        for row in range(idx.shape[0]):
            keep = idx[row] != row
            if keep.sum() < max_k - 1:
                keep = np.ones(idx.shape[1], dtype=bool)
                keep[0] = False
            clean_dist[row] = dist[row][keep][: max_k - 1]
            clean_idx[row] = idx[row][keep][: max_k - 1]
        dist, idx = clean_dist, clean_idx

    return summarize_neighbors(dist, idx, np.asarray(values, dtype=float), prefix, ks).set_index(X_apply.index)


def compute_knn_features(
    X_train: pd.DataFrame,
    raw_train: pd.DataFrame,
    y_train: np.ndarray,
    residual_train: np.ndarray,
    X_valid: pd.DataFrame,
    raw_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    raw_test: pd.DataFrame,
    ks: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    band_a = [c for c in X_train.columns if c.startswith("spectral_band_A_PC_")]
    band_b = [c for c in X_train.columns if c.startswith("spectral_band_B_PC_")]
    blocks_train: list[pd.DataFrame] = []
    blocks_valid: list[pd.DataFrame] = []
    blocks_test: list[pd.DataFrame] = []

    for values, label in [(y_train, "target"), (residual_train, "resid")]:
        blocks_train.append(compute_knn_block(X_train, values, X_train, band_a, f"knn_A_{label}", ks, train_apply=True))
        blocks_valid.append(compute_knn_block(X_train, values, X_valid, band_a, f"knn_A_{label}", ks, train_apply=False))
        blocks_test.append(compute_knn_block(X_train, values, X_test, band_a, f"knn_A_{label}", ks, train_apply=False))

    b_train_mask = raw_train["has_band_B_spectrum"].astype("string").eq("YES") & X_train[band_b].notna().all(axis=1)
    b_valid_mask = raw_valid["has_band_B_spectrum"].astype("string").eq("YES") & X_valid[band_b].notna().all(axis=1)
    b_test_mask = raw_test["has_band_B_spectrum"].astype("string").eq("YES") & X_test[band_b].notna().all(axis=1)
    ab_cols = band_a + band_b

    for values, label in [(y_train, "target"), (residual_train, "resid")]:
        prefix = f"knn_AB_{label}"
        names = knn_feature_names(prefix, ks)
        train_block = pd.DataFrame({name: np.nan for name in names}, index=X_train.index)
        valid_block = pd.DataFrame({name: np.nan for name in names}, index=X_valid.index)
        test_block = pd.DataFrame({name: np.nan for name in names}, index=X_test.index)

        if int(b_train_mask.sum()) > max(ks):
            fit_X = X_train.loc[b_train_mask, ab_cols].reset_index(drop=True)
            fit_values = values[b_train_mask.to_numpy()]
            train_ab = compute_knn_block(fit_X, fit_values, fit_X, ab_cols, prefix, ks, train_apply=True)
            train_block.loc[b_train_mask, names] = train_ab.to_numpy()
            if int(b_valid_mask.sum()) > 0:
                valid_ab = compute_knn_block(
                    fit_X,
                    fit_values,
                    X_valid.loc[b_valid_mask, ab_cols],
                    ab_cols,
                    prefix,
                    ks,
                    train_apply=False,
                )
                valid_block.loc[b_valid_mask, names] = valid_ab.to_numpy()
            if int(b_test_mask.sum()) > 0:
                test_ab = compute_knn_block(
                    fit_X,
                    fit_values,
                    X_test.loc[b_test_mask, ab_cols],
                    ab_cols,
                    prefix,
                    ks,
                    train_apply=False,
                )
                test_block.loc[b_test_mask, names] = test_ab.to_numpy()

        blocks_train.append(train_block)
        blocks_valid.append(valid_block)
        blocks_test.append(test_block)

    return (
        pd.concat(blocks_train, axis=1).reset_index(drop=True),
        pd.concat(blocks_valid, axis=1).reset_index(drop=True),
        pd.concat(blocks_test, axis=1).reset_index(drop=True),
    )


def append_fold_features(
    X_train_base: pd.DataFrame,
    X_valid_base: pd.DataFrame,
    X_test_base: pd.DataFrame,
    prior_train: np.ndarray,
    prior_valid: np.ndarray,
    prior_test: np.ndarray,
    te_train: pd.DataFrame,
    te_valid: pd.DataFrame,
    te_test: pd.DataFrame,
    knn_train: pd.DataFrame,
    knn_valid: pd.DataFrame,
    knn_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = pd.concat([X_train_base.reset_index(drop=True), te_train.reset_index(drop=True), knn_train], axis=1)
    X_valid = pd.concat([X_valid_base.reset_index(drop=True), te_valid.reset_index(drop=True), knn_valid], axis=1)
    X_test = pd.concat([X_test_base.reset_index(drop=True), te_test.reset_index(drop=True), knn_test], axis=1)
    for frame, prior in [(X_train, prior_train), (X_valid, prior_valid), (X_test, prior_test)]:
        frame["hier_prior"] = prior
        frame["prior_log1p"] = np.log1p(np.clip(prior, 0, None))
    return X_train, X_valid, X_test


def fit_predict_lgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
    params: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    model = LGBMRegressor(**params, random_state=seed)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[early_stopping(150, verbose=False), log_evaluation(0)],
    )
    return model.predict(X_valid), model.predict(X_test), model.best_iteration_


def fit_predict_xgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
    params: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    model = XGBRegressor(**params, random_state=seed)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return model.predict(X_valid), model.predict(X_test), getattr(model, "best_iteration", None)


def fit_predict_cat(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
    params: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    model = CatBoostRegressor(**params, random_seed=seed)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=180, verbose=False)
    return model.predict(X_valid), model.predict(X_test), model.get_best_iteration()


def fit_predict_ridge(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, None]:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    model.fit(X_train, y_train)
    return model.predict(X_valid), model.predict(X_test), None


def fit_predict_extra_trees(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    params: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, None]:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", ExtraTreesRegressor(**params, random_state=seed)),
        ]
    )
    model.fit(X_train, y_train)
    return model.predict(X_valid), model.predict(X_test), None


def fit_predict_segmented_lgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
    segment_train: np.ndarray,
    segment_valid: np.ndarray,
    segment_test: np.ndarray,
    params: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | None]]:
    valid_pred = np.zeros(len(X_valid), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    info: dict[str, int | None] = {}
    global_model = LGBMRegressor(**params, random_state=seed)
    global_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[early_stopping(120, verbose=False), log_evaluation(0)],
    )
    fallback_valid = global_model.predict(X_valid)
    fallback_test = global_model.predict(X_test)
    valid_pred[:] = fallback_valid
    test_pred[:] = fallback_test
    info["global"] = global_model.best_iteration_

    for seg in [0, 1]:
        tr_mask = segment_train == seg
        va_mask = segment_valid == seg
        te_mask = segment_test == seg
        if int(tr_mask.sum()) < 80:
            info[f"segment_{seg}"] = None
            continue
        eval_mask = va_mask if int(va_mask.sum()) >= 20 else np.ones(len(X_valid), dtype=bool)
        model = LGBMRegressor(**params, random_state=seed + 10 + seg)
        model.fit(
            X_train.loc[tr_mask],
            y_train[tr_mask],
            eval_set=[(X_valid.loc[eval_mask], y_valid[eval_mask])],
            callbacks=[early_stopping(120, verbose=False), log_evaluation(0)],
        )
        if int(va_mask.sum()) > 0:
            valid_pred[va_mask] = model.predict(X_valid.loc[va_mask])
        if int(te_mask.sum()) > 0:
            test_pred[te_mask] = model.predict(X_test.loc[te_mask])
        info[f"segment_{seg}"] = model.best_iteration_
    return valid_pred, test_pred, info


def model_params(config: RunConfig) -> dict[str, dict]:
    if config.quick:
        return {
            "lgb_base": dict(n_estimators=180, num_leaves=31, learning_rate=0.05, min_child_samples=20, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=0.3, objective="regression", n_jobs=config.n_jobs, verbosity=-1),
            "lgb_wide": dict(n_estimators=180, num_leaves=63, learning_rate=0.04, min_child_samples=15, subsample=0.85, colsample_bytree=0.7, reg_alpha=0.05, reg_lambda=0.2, objective="regression", n_jobs=config.n_jobs, verbosity=-1),
            "lgb_segmented": dict(n_estimators=160, num_leaves=31, learning_rate=0.05, min_child_samples=15, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=0.3, objective="regression", n_jobs=config.n_jobs, verbosity=-1),
            "xgb_d4": dict(n_estimators=180, max_depth=4, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0, objective="reg:squarederror", tree_method="hist", n_jobs=config.n_jobs, early_stopping_rounds=80, eval_metric="rmse"),
            "cat_d6": dict(iterations=180, depth=6, learning_rate=0.05, l2_leaf_reg=10, random_strength=1.0, bagging_temperature=0.4, bootstrap_type="Bayesian", loss_function="RMSE", eval_metric="RMSE", allow_writing_files=False, verbose=False, thread_count=config.n_jobs),
            "extra_trees": dict(n_estimators=100, min_samples_leaf=2, max_features=0.8, n_jobs=config.n_jobs),
        }
    return {
        "lgb_base": dict(n_estimators=2600, num_leaves=31, learning_rate=0.025, min_child_samples=20, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=0.3, objective="regression", n_jobs=config.n_jobs, verbosity=-1),
        "lgb_wide": dict(n_estimators=3000, num_leaves=95, learning_rate=0.015, min_child_samples=16, subsample=0.85, colsample_bytree=0.65, reg_alpha=0.05, reg_lambda=0.2, objective="regression", n_jobs=config.n_jobs, verbosity=-1),
        "lgb_segmented": dict(n_estimators=2200, num_leaves=39, learning_rate=0.025, min_child_samples=14, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=0.3, objective="regression", n_jobs=config.n_jobs, verbosity=-1),
        "xgb_d4": dict(n_estimators=2600, max_depth=4, learning_rate=0.025, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.0, objective="reg:squarederror", tree_method="hist", n_jobs=config.n_jobs, early_stopping_rounds=180, eval_metric="rmse"),
        "cat_d6": dict(iterations=2600, depth=6, learning_rate=0.032, l2_leaf_reg=10, random_strength=1.2, bagging_temperature=0.4, bootstrap_type="Bayesian", loss_function="RMSE", eval_metric="RMSE", allow_writing_files=False, verbose=False, thread_count=config.n_jobs),
        "extra_trees": dict(n_estimators=550, min_samples_leaf=2, max_features=0.8, n_jobs=config.n_jobs),
    }


def base_model_names() -> list[str]:
    return [
        "lgb_target",
        "cat_target",
        "extra_trees_target",
        "lgb_base",
        "lgb_wide",
        "lgb_segmented",
        "xgb_d4",
        "cat_d6",
        "ridge",
        "extra_trees",
    ]


def fold_run(
    fold: int,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    raw: pd.DataFrame,
    raw_test: pd.DataFrame,
    y: np.ndarray,
    sample_upper: float,
    params: dict[str, dict],
    config: RunConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict]]:
    started = time.perf_counter()
    X_tr0 = X.iloc[tr_idx].reset_index(drop=True)
    X_va0 = X.iloc[va_idx].reset_index(drop=True)
    X_te0 = X_test.reset_index(drop=True)
    raw_tr = raw.iloc[tr_idx].reset_index(drop=True)
    raw_va = raw.iloc[va_idx].reset_index(drop=True)
    raw_te = raw_test.reset_index(drop=True)
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    prior_tr, prior_va, prior_te = make_hierarchical_prior(raw_tr, y_tr, raw_va, raw_te)
    resid_tr = y_tr - prior_tr
    resid_va = y_va - prior_va
    if config.use_te:
        te_tr, te_va, te_te = make_target_encoding_features(raw_tr, y_tr, raw_va, raw_te)
    else:
        te_tr = pd.DataFrame(index=X_tr0.index)
        te_va = pd.DataFrame(index=X_va0.index)
        te_te = pd.DataFrame(index=X_te0.index)
    knn_tr, knn_va, knn_te = compute_knn_features(X_tr0, raw_tr, y_tr, resid_tr, X_va0, raw_va, X_te0, raw_te, config.knn_ks)
    X_tr, X_va, X_te = append_fold_features(X_tr0, X_va0, X_te0, prior_tr, prior_va, prior_te, te_tr, te_va, te_te, knn_tr, knn_va, knn_te)

    segment_tr = raw_tr["has_band_B_spectrum"].astype("string").eq("YES").to_numpy(dtype=int)
    segment_va = raw_va["has_band_B_spectrum"].astype("string").eq("YES").to_numpy(dtype=int)
    segment_te = raw_te["has_band_B_spectrum"].astype("string").eq("YES").to_numpy(dtype=int)

    oof_fold: dict[str, np.ndarray] = {}
    test_fold: dict[str, np.ndarray] = {}
    rows: list[dict] = []

    def store(name: str, valid_resid: np.ndarray, test_resid: np.ndarray, best_iter: int | None) -> None:
        valid_pred = finite_clip(prior_va + valid_resid, sample_upper)
        test_pred = finite_clip(prior_te + test_resid, sample_upper)
        oof_fold[name] = valid_pred
        test_fold[name] = test_pred
        rows.append(
            {
                "fold": fold,
                "model": name,
                "rmse": rmse(y_va, valid_pred),
                "mae": mean_absolute_error(y_va, valid_pred),
                "best_iter": best_iter,
                "seconds_elapsed": time.perf_counter() - started,
            }
        )
        print(f"fold={fold} model={name} rmse={rows[-1]['rmse']:.5f} elapsed={rows[-1]['seconds_elapsed']:.1f}s", flush=True)

    def store_direct(name: str, valid_pred_raw: np.ndarray, test_pred_raw: np.ndarray, best_iter: int | None) -> None:
        valid_pred = finite_clip(valid_pred_raw, sample_upper)
        test_pred = finite_clip(test_pred_raw, sample_upper)
        oof_fold[name] = valid_pred
        test_fold[name] = test_pred
        rows.append(
            {
                "fold": fold,
                "model": name,
                "rmse": rmse(y_va, valid_pred),
                "mae": mean_absolute_error(y_va, valid_pred),
                "best_iter": best_iter,
                "seconds_elapsed": time.perf_counter() - started,
            }
        )
        print(f"fold={fold} model={name} rmse={rows[-1]['rmse']:.5f} elapsed={rows[-1]['seconds_elapsed']:.1f}s", flush=True)

    pred_va, pred_te, best_iter = fit_predict_lgb(X_tr, y_tr, X_va, y_va, X_te, params["lgb_base"], RANDOM_STATE + 700 + fold)
    store_direct("lgb_target", pred_va, pred_te, best_iter)

    pred_va, pred_te, best_iter = fit_predict_cat(X_tr, y_tr, X_va, y_va, X_te, params["cat_d6"], RANDOM_STATE + 800 + fold)
    store_direct("cat_target", pred_va, pred_te, best_iter)

    pred_va, pred_te, best_iter = fit_predict_extra_trees(X_tr, y_tr, X_va, X_te, params["extra_trees"], RANDOM_STATE + 900 + fold)
    store_direct("extra_trees_target", pred_va, pred_te, best_iter)

    pred_va, pred_te, best_iter = fit_predict_lgb(X_tr, resid_tr, X_va, resid_va, X_te, params["lgb_base"], RANDOM_STATE + fold)
    store("lgb_base", pred_va, pred_te, best_iter)

    pred_va, pred_te, best_iter = fit_predict_lgb(X_tr, resid_tr, X_va, resid_va, X_te, params["lgb_wide"], RANDOM_STATE + 100 + fold)
    store("lgb_wide", pred_va, pred_te, best_iter)

    pred_va, pred_te, seg_info = fit_predict_segmented_lgb(
        X_tr,
        resid_tr,
        X_va,
        resid_va,
        X_te,
        segment_tr,
        segment_va,
        segment_te,
        params["lgb_segmented"],
        RANDOM_STATE + 200 + fold,
    )
    store("lgb_segmented", pred_va, pred_te, seg_info.get("global"))

    pred_va, pred_te, best_iter = fit_predict_xgb(X_tr, resid_tr, X_va, resid_va, X_te, params["xgb_d4"], RANDOM_STATE + 300 + fold)
    store("xgb_d4", pred_va, pred_te, best_iter)

    pred_va, pred_te, best_iter = fit_predict_cat(X_tr, resid_tr, X_va, resid_va, X_te, params["cat_d6"], RANDOM_STATE + 400 + fold)
    store("cat_d6", pred_va, pred_te, best_iter)

    pred_va, pred_te, best_iter = fit_predict_ridge(X_tr, resid_tr, X_va, X_te, alpha=80.0)
    store("ridge", pred_va, pred_te, best_iter)

    pred_va, pred_te, best_iter = fit_predict_extra_trees(X_tr, resid_tr, X_va, X_te, params["extra_trees"], RANDOM_STATE + 500 + fold)
    store("extra_trees", pred_va, pred_te, best_iter)

    gc.collect()
    return oof_fold, test_fold, rows


def fit_weights_group_aware(O: np.ndarray, y: np.ndarray, groups: np.ndarray, names: list[str]) -> np.ndarray:
    n_models = O.shape[1]
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    splits = list(gkf.split(O, y, groups))

    def objective(w: np.ndarray) -> float:
        fold_scores = [rmse(y[va], O[va].dot(w)) for _, va in splits]
        uniform = np.ones(n_models) / n_models
        return float(np.mean(fold_scores) + 1e-3 * np.sum((w - uniform) ** 2))

    res = minimize(
        objective,
        np.ones(n_models) / n_models,
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        method="SLSQP",
        options={"maxiter": 500},
    )
    if not res.success:
        print(f"Weight optimization warning: {res.message}", flush=True)
    weights = np.maximum(res.x, 0.0)
    weights = weights / weights.sum()
    print("Final group-aware weights:", flush=True)
    for name, weight in sorted(zip(names, weights), key=lambda item: -item[1]):
        print(f"  {name:15s} {weight:.4f}", flush=True)
    return weights


def evaluate_tail_candidates(
    oof: np.ndarray,
    pred: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    upper: float,
    min_improvement: float,
) -> tuple[str, np.ndarray, np.ndarray, pd.DataFrame]:
    candidates: list[tuple[str, np.ndarray, np.ndarray]] = [("none", oof, pred)]
    for threshold in [60.0, 75.0, 90.0]:
        for alpha in [0.02, 0.04, 0.06, 0.08]:
            adj_oof = finite_clip(oof + alpha * np.maximum(oof - threshold, 0.0), upper)
            adj_pred = finite_clip(pred + alpha * np.maximum(pred - threshold, 0.0), upper)
            candidates.append((f"tail_lift_t{threshold:.0f}_a{alpha:.2f}", adj_oof, adj_pred))
    mean_y = float(np.mean(y))
    for factor in [1.01, 1.02, 1.03, 1.05]:
        adj_oof = finite_clip(mean_y + factor * (oof - mean_y), upper)
        adj_pred = finite_clip(mean_y + factor * (pred - mean_y), upper)
        candidates.append((f"mean_stretch_{factor:.2f}", adj_oof, adj_pred))

    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    rows = []
    for name, cand_oof, _ in candidates:
        fold_scores = [rmse(y[va], cand_oof[va]) for _, va in gkf.split(cand_oof, y, groups)]
        rows.append({"tail_strategy": name, "group_rmse": float(np.mean(fold_scores)), "overall_rmse": rmse(y, cand_oof)})
    summary = pd.DataFrame(rows).sort_values("group_rmse").reset_index(drop=True)
    none_score = float(summary.loc[summary["tail_strategy"] == "none", "group_rmse"].iloc[0])
    best_name = str(summary.loc[0, "tail_strategy"])
    best_score = float(summary.loc[0, "group_rmse"])
    if best_name != "none" and none_score - best_score >= min_improvement:
        for name, cand_oof, cand_pred in candidates:
            if name == best_name:
                return name, cand_oof, cand_pred, summary
    return "none", oof, pred, summary


def diagnostic_rows(y: np.ndarray, pred: np.ndarray, raw: pd.DataFrame, label: str) -> list[dict]:
    rows = [
        {"slice": f"{label}:all", "count": len(y), "rmse": rmse(y, pred), "mae": mean_absolute_error(y, pred)},
    ]
    band_b = raw["has_band_B_spectrum"].astype("string").eq("YES").to_numpy()
    for name, mask in [("bandB_yes", band_b), ("bandB_no", ~band_b), ("tail_ge_75", y >= 75), ("tail_ge_100", y >= 100)]:
        if int(mask.sum()) == 0:
            continue
        rows.append(
            {
                "slice": f"{label}:{name}",
                "count": int(mask.sum()),
                "rmse": rmse(y[mask], pred[mask]),
                "mae": mean_absolute_error(y[mask], pred[mask]),
            }
        )
    top5_threshold = float(np.percentile(y, 95))
    top5 = y >= top5_threshold
    rows.append(
        {
            "slice": f"{label}:top5pct_ge_{top5_threshold:.3f}",
            "count": int(top5.sum()),
            "rmse": rmse(y[top5], pred[top5]),
            "mae": mean_absolute_error(y[top5], pred[top5]),
        }
    )
    return rows


def random_stratified_diagnostic(y: np.ndarray, pred: np.ndarray) -> dict:
    bins = pd.qcut(y, 10, labels=False, duplicates="drop")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = [rmse(y[va], pred[va]) for _, va in cv.split(pred, bins)]
    return {"diagnostic": "random_stratified_existing_oof", "rmse_mean": float(np.mean(fold_scores)), "rmse_folds": fold_scores}


def group_proxy_rmse(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> float:
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    return float(np.mean([rmse(y[va], pred[va]) for _, va in gkf.split(pred, y, groups)]))


def write_outputs(
    suffix: str,
    sample: pd.DataFrame,
    y: np.ndarray,
    raw: pd.DataFrame,
    oofs: dict[str, np.ndarray],
    preds: dict[str, np.ndarray],
    ensemble_oof: np.ndarray,
    ensemble_pred: np.ndarray,
    weights: np.ndarray,
    names: list[str],
    fold_rows: list[dict],
    tail_summary: pd.DataFrame,
    tail_strategy: str,
    upper: float,
    config: RunConfig,
    groups: np.ndarray,
) -> None:
    submission = sample.copy()
    submission[TARGET] = finite_clip(ensemble_pred, upper)
    submission_path = SUBMISSION_DIR / f"submission_{suffix}.csv"
    submission.to_csv(submission_path, index=False)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(DATA_DIR / f"fold_metrics_{suffix}.csv", index=False)

    model_rows = []
    for name in names:
        model_rows.append(
            {
                "name": name,
                "rmse": rmse(y, oofs[name]),
                "group_proxy_rmse": group_proxy_rmse(y, oofs[name], groups),
                "mae": mean_absolute_error(y, oofs[name]),
                "r2": r2_score(y, oofs[name]),
                "weight": float(weights[names.index(name)]),
            }
        )
    model_rows.append(
        {
            "name": f"ensemble_{suffix}",
            "rmse": rmse(y, ensemble_oof),
            "group_proxy_rmse": group_proxy_rmse(y, ensemble_oof, groups),
            "mae": mean_absolute_error(y, ensemble_oof),
            "r2": r2_score(y, ensemble_oof),
            "weight": 1.0,
        }
    )
    model_summary = pd.DataFrame(model_rows).sort_values("rmse")
    diagnostics = pd.DataFrame(diagnostic_rows(y, ensemble_oof, raw, f"ensemble_{suffix}"))
    model_summary.to_csv(DATA_DIR / f"model_summary_{suffix}.csv", index=False)
    diagnostics.to_csv(DATA_DIR / f"diagnostics_{suffix}.csv", index=False)
    tail_summary.to_csv(DATA_DIR / f"tail_summary_{suffix}.csv", index=False)

    checkpoint_payload = {
        **{f"oof_{name}": oofs[name] for name in names},
        **{f"pred_{name}": preds[name] for name in names},
        "ensemble_oof": ensemble_oof,
        "ensemble_pred": ensemble_pred,
        "weights": weights,
        "model_names": np.array(names),
    }
    np.savez(DATA_DIR / f"checkpoint_{suffix}.npz", **checkpoint_payload)

    metadata = {
        "suffix": suffix,
        "quick": config.quick,
        "n_splits": config.n_splits,
        "base_cv": config.base_cv,
        "use_te": config.use_te,
        "knn_ks": list(config.knn_ks),
        "tail_strategy": tail_strategy,
        "submission": str(submission_path),
        "ensemble_rmse": rmse(y, ensemble_oof),
        "ensemble_group_proxy_rmse": group_proxy_rmse(y, ensemble_oof, groups),
        "ensemble_mae": mean_absolute_error(y, ensemble_oof),
        "random_stratified_diagnostic": random_stratified_diagnostic(y, ensemble_oof),
    }
    (DATA_DIR / f"metadata_{suffix}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print()
    print(f"Wrote {submission_path}", flush=True)
    print(f"Wrote {DATA_DIR / f'model_summary_{suffix}.csv'}", flush=True)
    print(f"Wrote {DATA_DIR / f'checkpoint_{suffix}.npz'}", flush=True)
    print(f"Tail strategy: {tail_strategy}", flush=True)
    print(model_summary.to_string(index=False), flush=True)
    print(diagnostics.to_string(index=False), flush=True)


def run(config: RunConfig) -> None:
    started = time.perf_counter()
    train, test, sample = read_data()
    y = train[TARGET].to_numpy(dtype=float)
    groups = train["source_id"].to_numpy()
    upper = float(np.max(y))
    X, X_test, raw, raw_test = make_features(train, test)
    params = model_params(config)
    names = base_model_names()
    print(f"Features: train={X.shape}, test={X_test.shape}", flush=True)
    print(f"Base CV: {config.base_cv}; folds: {config.n_splits}; group-aware selection metric enabled", flush=True)
    print(f"Models: {names}", flush=True)

    oofs = {name: np.zeros(len(train), dtype=float) for name in names}
    preds = {name: np.zeros(len(test), dtype=float) for name in names}
    fold_rows: list[dict] = []

    if config.base_cv == "group":
        splits = list(GroupKFold(n_splits=config.n_splits).split(X, y, groups))
    else:
        bins = pd.qcut(y, 10, labels=False, duplicates="drop")
        splits = list(StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=RANDOM_STATE).split(X, bins))

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        print(f"\n=== fold {fold}/{config.n_splits}: train={len(tr_idx)} valid={len(va_idx)} ===", flush=True)
        oof_fold, test_fold, rows = fold_run(
            fold,
            tr_idx,
            va_idx,
            X,
            X_test,
            raw,
            raw_test,
            y,
            upper,
            params,
            config,
        )
        for name in names:
            oofs[name][va_idx] = oof_fold[name]
            preds[name] += test_fold[name] / config.n_splits
        fold_rows.extend(rows)

        np.savez(
            DATA_DIR / f"checkpoint_{config.suffix}_partial.npz",
            **{f"oof_{name}": oofs[name] for name in names},
            **{f"pred_{name}": preds[name] for name in names},
        )

    O = np.vstack([oofs[name] for name in names]).T
    P = np.vstack([preds[name] for name in names]).T
    weights = fit_weights_group_aware(O, y, groups, names)
    ensemble_oof = finite_clip(O.dot(weights), upper)
    ensemble_pred = finite_clip(P.dot(weights), upper)
    tail_strategy, ensemble_oof, ensemble_pred, tail_summary = evaluate_tail_candidates(
        ensemble_oof,
        ensemble_pred,
        y,
        groups,
        upper,
        config.tail_min_improvement,
    )

    write_outputs(
        config.suffix,
        sample,
        y,
        raw,
        oofs,
        preds,
        ensemble_oof,
        ensemble_pred,
        weights,
        names,
        fold_rows,
        tail_summary,
        tail_strategy,
        upper,
        config,
        groups,
    )
    print(f"Total runtime: {(time.perf_counter() - started) / 60:.1f} minutes", flush=True)


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Run the V7 grouped-CV soil organic content experiment.")
    parser.add_argument("--suffix", default="v7", help="Output suffix for submission/checkpoint/summary files.")
    parser.add_argument("--quick", action="store_true", help="Use fewer folds and smaller models for smoke testing.")
    parser.add_argument("--folds", type=int, default=None, help="Override GroupKFold split count.")
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs/threads for supported models.")
    parser.add_argument(
        "--base-cv",
        choices=["stratified", "group"],
        default="stratified",
        help="Fold strategy for base OOF models. Blending/tail selection remains group-aware.",
    )
    parser.add_argument(
        "--use-te",
        action="store_true",
        help="Include fold-safe target-encoding feature columns. Disabled by default because smoke tests showed instability.",
    )
    args = parser.parse_args()

    if args.quick:
        n_splits = args.folds or 2
        knn_ks = (5, 15)
        suffix = args.suffix if args.suffix != "v7" else "v7_quick"
    else:
        n_splits = args.folds or 5
        knn_ks = (8, 32)
        suffix = args.suffix
    return RunConfig(
        suffix=suffix,
        n_splits=n_splits,
        quick=args.quick,
        n_jobs=args.n_jobs,
        knn_ks=knn_ks,
        base_cv=args.base_cv,
        use_te=args.use_te,
    )


if __name__ == "__main__":
    run(parse_args())
