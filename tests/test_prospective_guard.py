from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_scale.data.prospective_guard import OutcomeAccessError, assert_feature_only
from causal_scale.evidence.seal import ProspectiveState, transition


def features() -> pd.DataFrame:
    return pd.DataFrame({"source_row_id": [1], **{f"f{i}": np.zeros(1) for i in range(12)}})


def test_feature_only_guard_blocks_every_outcome_field() -> None:
    assert_feature_only(features())
    for forbidden in ("visit", "conversion", "exposure", "treatment"):
        with pytest.raises(OutcomeAccessError):
            assert_feature_only(features().assign(**{forbidden: 0}))


def test_state_machine_blocks_skipping_the_seal() -> None:
    with pytest.raises(ValueError):
        transition(ProspectiveState.CREATED, ProspectiveState.OUTCOME_ACCESS_STARTED)
    assert (
        transition(ProspectiveState.CREATED, ProspectiveState.IMPLEMENTATION_FROZEN)
        == ProspectiveState.IMPLEMENTATION_FROZEN
    )
