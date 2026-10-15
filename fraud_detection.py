"""Leakage-aware PaySim fraud-detection pipeline.

The module deliberately does not download data or invent a fallback dataset.  A caller
must provide the PaySim CSV path explicitly (or through ``PAYSIM_CSV``).  Training is
optional and imports the heavyweight model libraries only when requested.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RANDOM_SEED = 42
ARTIFACT_VERSION = 1
DEFAULT_DATA_PATH = Path("data/PS_20174392719_1491204439457_log.csv")
REQUIRED_COLUMNS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "isFraud",
}
DROP_COLUMNS = {
    "isFlaggedFraud",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
}
FEATURE_COLUMNS = [
    "amount",
    "amount_log",
    "hour",
    "is_night",
    "is_high_amount",
    "is_very_high_amount",
    "dest_is_merchant",
    "user_tx_count",
    "user_avg_amount",
    "last_amount",
    "amount_vs_avg",
    "amount_ratio_avg",
    "amount_ratio_last",
    "dest_popularity",
    "is_new_destination",
    "type_TRANSFER",
    "type_CASH_OUT",
    "high_transfer_risk",
]
ALLOWED_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
NUMERIC_COLUMNS = ("step", "amount", "isFraud")


@dataclass(frozen=True)
class FeatureState:
    """Statistics fit on the training period only."""

    amount_95: float
    amount_99: float
    user_avg_amount: dict[str, float]
    last_amount: dict[str, float]
    user_tx_count: dict[str, int]
    dest_popularity: dict[str, int]


def _normalise_path(path: str | os.PathLike[str] | None) -> Path:
    return Path(path or os.environ.get("PAYSIM_CSV", DEFAULT_DATA_PATH)).expanduser()


def _validated_frame(
    frame: pd.DataFrame,
    *,
    context: str,
    require_both_classes: bool = True,
) -> pd.DataFrame:
    """Return a normalized copy after enforcing the PaySim data contract."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{context} must be a pandas DataFrame")
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{context} is empty")

    normalized = frame.copy()
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(normalized[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{context}.{column} must contain finite numeric values")
        normalized[column] = values
    if (normalized["step"] < 0).any() or (normalized["step"] % 1 != 0).any():
        raise ValueError(f"{context}.step must contain non-negative integers")
    if (normalized["amount"] < 0).any():
        raise ValueError(f"{context}.amount must contain non-negative values")
    labels = set(normalized["isFraud"].tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{context}.isFraud must contain only 0/1 labels")
    if require_both_classes and normalized["isFraud"].nunique() < 2:
        raise ValueError(f"{context} must contain both fraud and non-fraud labels")

    for column in ("type", "nameOrig", "nameDest"):
        if normalized[column].isna().any():
            raise ValueError(f"{context}.{column} must not contain missing values")
        normalized[column] = normalized[column].astype(str).str.strip()
        if (normalized[column] == "").any():
            raise ValueError(f"{context}.{column} must not contain empty values")
    invalid_types = sorted(set(normalized["type"]) - ALLOWED_TYPES)
    if invalid_types:
        raise ValueError(f"{context}.type contains unsupported values: {invalid_types}")
    normalized["step"] = normalized["step"].astype(np.int64)
    normalized["isFraud"] = normalized["isFraud"].astype(np.int8)
    return normalized


def load_paysim(path: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """Load and validate a PaySim CSV, failing clearly when it is unavailable."""

    csv_path = _normalise_path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"PaySim CSV not found: {csv_path}. Download the dataset from "
            "https://www.kaggle.com/datasets/ealaxi/paysim1 and place the CSV at "
            "that path, or pass --data/PAYSIM_CSV. No data was fabricated."
        )

    frame = pd.read_csv(csv_path)
    return _validated_frame(frame, context=f"Invalid PaySim data in {csv_path}")


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return chronological train/validation/test periods without shuffling."""

    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("train_fraction and validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    ordered = _validated_frame(frame, context="Frame")
    ordered = ordered.sort_values("step", kind="mergesort").reset_index(drop=True)
    steps = ordered["step"].drop_duplicates().to_numpy()
    train_end = int(len(steps) * train_fraction)
    validation_end = train_end + int(len(steps) * validation_fraction)
    if min(train_end, validation_end - train_end, len(steps) - validation_end) < 1:
        raise ValueError("Dataset is too small for three non-empty step periods")
    train_steps = set(steps[:train_end])
    validation_steps = set(steps[train_end:validation_end])
    test_steps = set(steps[validation_end:])
    return (
        ordered[ordered["step"].isin(train_steps)].copy(),
        ordered[ordered["step"].isin(validation_steps)].copy(),
        ordered[ordered["step"].isin(test_steps)].copy(),
    )


def fit_feature_state(train_frame: pd.DataFrame) -> FeatureState:
    """Fit all target-independent feature statistics on training data only."""

    train_frame = _validated_frame(train_frame, context="Training frame")
    ordered = train_frame.sort_values(["nameOrig", "step"], kind="mergesort")
    user_groups = ordered.groupby("nameOrig", sort=False)
    return FeatureState(
        amount_95=float(train_frame["amount"].quantile(0.95)),
        amount_99=float(train_frame["amount"].quantile(0.99)),
        user_avg_amount={
            str(k): float(v) for k, v in user_groups["amount"].mean().items()
        },
        last_amount={
            str(k): float(v) for k, v in user_groups["amount"].last().items()
        },
        user_tx_count={str(k): int(v) for k, v in user_groups.size().items()},
        dest_popularity={
            str(k): int(v) for k, v in train_frame["nameDest"].value_counts().items()
        },
    )


def _previous_user_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate prior-transaction features without looking ahead within a period."""

    ordered = frame.sort_values(["nameOrig", "step"], kind="mergesort").copy()
    groups = ordered.groupby("nameOrig", sort=False)
    ordered["user_tx_count"] = groups.cumcount()
    ordered["user_avg_amount"] = (
        groups["amount"].transform(lambda values: values.shift().expanding().mean())
    )
    ordered["last_amount"] = groups["amount"].shift()
    return ordered.sort_index()


def transform_features(
    frame: pd.DataFrame,
    state: FeatureState,
    *,
    training_period: bool = False,
) -> pd.DataFrame:
    """Create the fixed model matrix using statistics learned from train only.

    For training rows, user history is strictly prior-only.  Validation and test rows
    use frozen training-period aggregates; this is conservative and prevents future
    rows from influencing an earlier prediction.
    """

    source = _validated_frame(frame, context="Frame", require_both_classes=False)
    if training_period:
        source = _previous_user_features(source)
    else:
        source["user_tx_count"] = source["nameOrig"].map(state.user_tx_count).fillna(0)
        source["user_avg_amount"] = source["nameOrig"].map(state.user_avg_amount).fillna(0)
        source["last_amount"] = source["nameOrig"].map(state.last_amount).fillna(0)

    source["user_avg_amount"] = source["user_avg_amount"].fillna(0)
    source["last_amount"] = source["last_amount"].fillna(0)
    source["hour"] = source["step"].astype(np.int64) % 24
    source["is_night"] = (source["hour"] < 6).astype(np.int8)
    source["amount_log"] = np.log1p(source["amount"].clip(lower=0))
    source["is_high_amount"] = (source["amount"] > state.amount_95).astype(np.int8)
    source["is_very_high_amount"] = (source["amount"] > state.amount_99).astype(np.int8)
    source["dest_is_merchant"] = source["nameDest"].astype(str).str.startswith("M").astype(np.int8)
    source["amount_vs_avg"] = source["amount"] - source["user_avg_amount"]
    source["amount_ratio_avg"] = source["amount"] / (source["user_avg_amount"] + 1e-6)
    source["amount_ratio_last"] = source["amount"] / (source["last_amount"] + 1e-6)
    source["dest_popularity"] = source["nameDest"].map(state.dest_popularity).fillna(0)
    source["is_new_destination"] = (source["dest_popularity"] == 0).astype(np.int8)
    source["type_TRANSFER"] = (source["type"] == "TRANSFER").astype(np.int8)
    source["type_CASH_OUT"] = (source["type"] == "CASH_OUT").astype(np.int8)
    source["high_transfer_risk"] = source["type_TRANSFER"] * source["is_high_amount"]
    return source[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)


def sample_training_data(
    features: pd.DataFrame,
    labels: pd.Series,
    non_fraud_fraction: float = 0.05,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Downsample only training negatives; validation/test remain untouched."""

    if not 0 < non_fraud_fraction <= 1:
        raise ValueError("non_fraud_fraction must be in (0, 1]")
    if len(features) != len(labels) or not features.index.equals(labels.index):
        raise ValueError("features and labels must have the same unique index and length")
    if not features.index.is_unique:
        raise ValueError("features and labels must have a unique index")
    labels = pd.to_numeric(labels, errors="coerce")
    if labels.isna().any() or not set(labels.tolist()).issubset({0, 1}):
        raise ValueError("Training labels must contain only finite 0/1 values")
    labels = labels.astype(np.int8)
    positives = labels == 1
    if positives.sum() == 0 or (~positives).sum() == 0:
        raise ValueError("Training data must contain both fraud and non-fraud labels")
    rng = np.random.RandomState(random_seed)
    negative_indices = labels.index[~positives]
    keep_count = max(1, round(len(negative_indices) * non_fraud_fraction))
    kept_negative = rng.choice(negative_indices.to_numpy(), size=keep_count, replace=False)
    selected = np.concatenate([labels.index[positives].to_numpy(), kept_negative])
    selected = rng.permutation(selected)
    return features.loc[selected], labels.loc[selected]


def select_threshold(y_validation: pd.Series, probabilities: np.ndarray) -> float:
    """Select F1 threshold on validation only; never on the final test period."""

    from sklearn.metrics import precision_recall_curve

    y_validation = pd.to_numeric(y_validation, errors="coerce")
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_validation) != len(probabilities) or len(y_validation) == 0:
        raise ValueError("Validation labels and probabilities must have the same non-zero length")
    if y_validation.isna().any() or not set(y_validation.tolist()).issubset({0, 1}):
        raise ValueError("Validation labels must contain only 0/1 values")
    if y_validation.nunique() < 2:
        raise ValueError("Validation labels must contain both fraud and non-fraud classes")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Validation probabilities must be finite values in [0, 1]")
    precision, recall, thresholds = precision_recall_curve(y_validation, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def evaluate_probabilities(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Return threshold-independent and operating-point metrics."""

    from sklearn.metrics import (
        average_precision_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    probabilities = np.asarray(probabilities, dtype=float)
    y_true = pd.to_numeric(y_true, errors="coerce")
    if len(y_true) != len(probabilities) or len(y_true) == 0:
        raise ValueError("Labels and probabilities must have the same non-zero length")
    if y_true.isna().any() or not set(y_true.tolist()).issubset({0, 1}):
        raise ValueError("Evaluation labels must contain only 0/1 values")
    if y_true.nunique() < 2:
        raise ValueError("Evaluation labels must contain both fraud and non-fraud classes")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Evaluation probabilities must be finite values in [0, 1]")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite value in [0, 1]")
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "threshold": float(threshold),
    }


def train_ensemble(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Train the documented ensemble with deterministic seeds.

    Heavy libraries are imported here so validation and feature tests work without
    installing the model stack.  The returned mapping contains fitted estimators.
    """

    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    positives = max(1, int((y_train == 1).sum()))
    negatives = max(1, int((y_train == 0).sum()))
    weight = negatives / positives
    return {
        "xgb": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            scale_pos_weight=weight,
            reg_alpha=1,
            reg_lambda=2,
            eval_metric="auc",
            random_state=random_seed,
            n_jobs=1,
        ).fit(x_train, y_train),
        "lgb": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=20,
            subsample=0.7,
            colsample_bytree=0.7,
            scale_pos_weight=weight,
            random_state=random_seed,
            n_jobs=1,
            verbosity=-1,
        ).fit(x_train, y_train),
        "cat": CatBoostClassifier(
            iterations=300,
            depth=5,
            learning_rate=0.05,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            verbose=False,
            random_seed=random_seed,
            thread_count=1,
        ).fit(x_train, y_train),
    }


def ensemble_probability(models: dict[str, Any], features: pd.DataFrame) -> np.ndarray:
    """Predict a weighted ensemble probability in the same order as training."""

    if list(features.columns) != FEATURE_COLUMNS:
        raise ValueError("Feature columns do not match the trained FEATURE_COLUMNS contract")
    required_models = ("xgb", "lgb", "cat")
    missing = [name for name in required_models if name not in models]
    if missing:
        raise ValueError(f"Missing ensemble models: {missing}")
    model_probabilities = []
    for name in required_models:
        output = np.asarray(models[name].predict_proba(features), dtype=float)
        if output.shape != (len(features), 2):
            raise ValueError(f"Model {name} returned invalid predict_proba shape: {output.shape}")
        model_probabilities.append(output[:, 1])
    probabilities = (
        0.3 * model_probabilities[0]
        + 0.3 * model_probabilities[1]
        + 0.4 * model_probabilities[2]
    )
    if not np.isfinite(probabilities).all():
        raise ValueError("Ensemble returned non-finite probabilities")
    return np.clip(probabilities, 0, 1)


def _feature_state_from_metadata(metadata: dict[str, Any]) -> FeatureState:
    """Validate and reconstruct feature state stored beside model artifacts."""

    raw = metadata.get("feature_state")
    if not isinstance(raw, dict):
        raise TypeError("Artifact metadata has no valid feature_state")
    try:
        state = FeatureState(
            amount_95=float(raw["amount_95"]),
            amount_99=float(raw["amount_99"]),
            user_avg_amount={str(k): float(v) for k, v in raw["user_avg_amount"].items()},
            last_amount={str(k): float(v) for k, v in raw["last_amount"].items()},
            user_tx_count={str(k): int(v) for k, v in raw["user_tx_count"].items()},
            dest_popularity={str(k): int(v) for k, v in raw["dest_popularity"].items()},
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Artifact metadata has an invalid feature_state") from exc
    scalar_values = (state.amount_95, state.amount_99)
    if not np.isfinite(scalar_values).all() or any(v < 0 for v in scalar_values):
        raise ValueError("Artifact metadata has invalid amount quantiles")
    return state


def load_artifacts(output_dir: str | os.PathLike[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate the complete model bundle written by :func:`run_training`."""

    import joblib

    directory = Path(output_dir).expanduser()
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Artifact metadata not found: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read artifact metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise TypeError("Artifact metadata must contain a JSON object")
    if metadata.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported or missing artifact_version")
    if metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Artifact feature_columns do not match this pipeline")
    _feature_state_from_metadata(metadata)
    try:
        threshold = metadata["test_metrics"]["threshold"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Artifact metadata has no selected threshold") from exc
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("Artifact threshold must be a finite value in [0, 1]") from exc
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Artifact threshold must be a finite value in [0, 1]")
    models = {}
    for name in ("xgb", "lgb", "cat"):
        model_path = directory / f"{name}.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Artifact model not found: {model_path}")
        try:
            model = joblib.load(model_path)
        except (EOFError, ImportError, OSError, ValueError) as exc:
            raise ValueError(f"Could not load artifact model {name}: {model_path}") from exc
        if not hasattr(model, "predict_proba"):
            raise ValueError(f"Artifact model {name} does not implement predict_proba")
        models[name] = model
    return models, metadata


def predict_with_artifacts(
    frame: pd.DataFrame,
    output_dir: str | os.PathLike[str],
) -> pd.DataFrame:
    """Score new rows with a complete, metadata-validated artifact bundle."""

    models, metadata = load_artifacts(output_dir)
    state = _feature_state_from_metadata(metadata)
    features = transform_features(frame, state)
    probabilities = ensemble_probability(models, features)
    threshold = float(metadata["test_metrics"]["threshold"])
    return pd.DataFrame(
        {
            "probability": probabilities,
            "isFraud": (probabilities >= threshold).astype(np.int8),
        },
        index=frame.index,
    )


def run_training(
    data_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run the complete leakage-aware workflow and save models/metadata locally."""

    import joblib

    frame = load_paysim(data_path)
    train, validation, test = chronological_split(frame)
    state = fit_feature_state(train)
    x_train_full = transform_features(train, state, training_period=True)
    x_validation = transform_features(validation, state)
    x_test = transform_features(test, state)
    y_train = train["isFraud"].astype(int)
    y_validation = validation["isFraud"].astype(int)
    y_test = test["isFraud"].astype(int)
    x_train, y_train_sampled = sample_training_data(
        x_train_full, y_train, random_seed=random_seed
    )
    models = train_ensemble(x_train, y_train_sampled, random_seed=random_seed)
    validation_probability = ensemble_probability(models, x_validation)
    threshold = select_threshold(y_validation, validation_probability)
    test_probability = ensemble_probability(models, x_test)
    metrics = evaluate_probabilities(y_test, test_probability, threshold)

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, output_dir / f"{name}.joblib")
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "artifact_version": ARTIFACT_VERSION,
                "model_names": ["xgb", "lgb", "cat"],
                "feature_columns": FEATURE_COLUMNS,
                "feature_state": asdict(state),
                "threshold_selected_on": "validation",
                "test_metrics": metrics,
                "random_seed": random_seed,
            },
            indent=2,
        )
        + "\n"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or train the PaySim fraud pipeline")
    parser.add_argument("--data", default=None, help="Path to the PaySim CSV (or set PAYSIM_CSV)")
    parser.add_argument("--train", action="store_true", help="Train models; otherwise validate only")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if not args.train:
        frame = load_paysim(args.data)
        print(f"Validated {len(frame):,} rows; no model training was requested.")
        return
    metrics = run_training(args.data, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
