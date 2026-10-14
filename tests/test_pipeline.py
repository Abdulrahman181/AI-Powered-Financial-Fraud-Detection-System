from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection import (
    FEATURE_COLUMNS,
    chronological_split,
    fit_feature_state,
    load_paysim,
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


def test_split_is_chronological_and_non_overlapping() -> None:
    train, validation, test = chronological_split(tiny_frame(), 0.5, 0.25)
    assert train["step"].max() < validation["step"].min()
    assert validation["step"].max() < test["step"].min()
    assert len(train) + len(validation) + len(test) == len(REQUIRED["step"])


def test_features_are_fixed_and_fit_from_training_only() -> None:
    frame = tiny_frame()
    train, validation, test = chronological_split(frame, 0.5, 0.25)
    state = fit_feature_state(train)
    features = transform_features(validation, state)
    test_features = transform_features(test, state)
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.shape == (len(validation), len(FEATURE_COLUMNS))
    # Validation destinations absent from training must remain unseen (zero count).
    assert test_features["dest_popularity"].iloc[-1] == 0
    assert np.isfinite(features.to_numpy()).all()


def test_threshold_selection_returns_probability_cutoff() -> None:
    threshold = select_threshold(pd.Series([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert 0.0 <= threshold <= 1.0
