from __future__ import annotations

import numpy as np
import pandas as pd


def feature_quantile_slices(
    frame: pd.DataFrame, feature: str = "f0", bins: int = 5
) -> dict[str, np.ndarray]:
    labels = pd.qcut(frame[feature], bins, duplicates="drop")
    return {f"{feature}:{value}": (labels == value).to_numpy() for value in labels.cat.categories}
