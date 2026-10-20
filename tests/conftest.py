import numpy as np
import pandas as pd
import pytest

from heartrisk.features import RAW_FEATURES


@pytest.fixture(scope="session")
def synthetic_patients() -> tuple[pd.DataFrame, pd.Series]:
    """Small, valid-looking dataset so model tests run without downloading anything."""
    rng = np.random.default_rng(7)
    n = 120
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({
        "age": rng.integers(30, 77, n),
        "sex": rng.integers(0, 2, n),
        "cp": np.where(y == 1, 4, rng.choice([1, 2, 3], n)),
        "trestbps": rng.normal(131, 17, n).round(),
        "chol": rng.normal(247, 50, n).round(),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.choice([0, 1, 2], n),
        "thalach": np.where(y == 1, rng.normal(139, 22, n), rng.normal(158, 19, n)).round(),
        "exang": np.where(y == 1, rng.integers(0, 2, n), 0),
        "oldpeak": np.abs(rng.normal(0.6 + y, 0.8, n)).round(1),
        "slope": rng.choice([1, 2, 3], n),
        "ca": rng.choice([0, 1, 2, 3], n),
        "thal": rng.choice([3, 6, 7], n),
    })[RAW_FEATURES].astype(float)
    return X, pd.Series(y, name="target")


@pytest.fixture
def patient() -> dict:
    return {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
            "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}
