import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection import (
    ARTIFACT_VERSION,
    FEATURE_COLUMNS,
    chronological_split,
    evaluate_probabilities,
    fit_feature_state,
    load_artifacts,
    load_paysim,
    sample_training_data,
    select_threshold,
    transform_features,
)

REQUIRED = {
    "step": [1, 2, 3, 4, 5, 6],
    "type": ["TRANSFER", "CASH_OUT", "PAYMENT", "TRANSFER", "CASH_OUT", "PAYMENT"],
    "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    "nameOrig": ["C1", "C1", "C2", "C2", "C3", "C3"],
    "nameDest": ["M1", "D1", "D2", "D2", "D3", "D3"],
    "isFraud": [0, 1, 0, 0, 1, 0],
}


def tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(REQUIRED)


def test_missing_data_fails_with_actionable_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="PaySim CSV not found"):
        load_paysim(tmp_path / "missing.csv")


def test_schema_failure_names_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"step": [1], "isFraud": [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_paysim(path)


def test_schema_failure_rejects_invalid_numeric_values(tmp_path: Path) -> None:
    invalid = tiny_frame()
    invalid.loc[0, "amount"] = -1
    path = tmp_path / "invalid.csv"
    invalid.to_csv(path, index=False)
    with pytest.raises(ValueError, match="amount must contain non-negative"):
        load_paysim(path)


def test_split_is_chronological_and_non_overlapping() -> None:
    train, validation, test = chronological_split(tiny_frame(), 0.5, 0.25)
    assert train["step"].max() < validation["step"].min()
    assert validation["step"].max() < test["step"].min()
    assert len(train) + len(validation) + len(test) == len(REQUIRED["step"])


def test_split_never_divides_rows_with_the_same_step() -> None:
    frame = tiny_frame()
    frame = pd.concat([frame] * 2, ignore_index=True)
    frame["step"] = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    train, validation, test = chronological_split(frame)
    assert train["step"].max() < validation["step"].min()
    assert validation["step"].max() < test["step"].min()
    assert len(train) + len(validation) + len(test) == len(frame)


def test_features_are_fixed_and_fit_from_training_only() -> None:
    frame = tiny_frame()
    train, validation, test = chronological_split(frame, 0.5, 0.25)
    state = fit_feature_state(train)
    training_features = transform_features(train, state, training_period=True)
    features = transform_features(validation, state)
    test_features = transform_features(test, state)
    assert training_features.index.equals(train.index)
    assert np.isfinite(training_features.to_numpy()).all()
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.shape == (len(validation), len(FEATURE_COLUMNS))
    # Validation destinations absent from training must remain unseen (zero count).
    assert test_features["dest_popularity"].iloc[-1] == 0
    assert np.isfinite(features.to_numpy()).all()


def test_threshold_selection_returns_probability_cutoff() -> None:
    threshold = select_threshold(pd.Series([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert 0.0 <= threshold <= 1.0


def test_metric_contract_rejects_one_class_and_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="both fraud and non-fraud"):
        select_threshold(pd.Series([0, 0]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="finite values"):
        evaluate_probabilities(pd.Series([0, 1]), np.array([np.nan, 0.8]), 0.5)


def test_sampling_requires_aligned_indices() -> None:
    features = pd.DataFrame({"x": [1, 2]}, index=[10, 11])
    labels = pd.Series([0, 1], index=[0, 1])
    with pytest.raises(ValueError, match="same unique index"):
        sample_training_data(features, labels)


def test_artifact_loader_rejects_incompatible_metadata(tmp_path: Path) -> None:
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "feature_columns": ["wrong"],
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="feature_columns"):
        load_artifacts(tmp_path)
