from math import floor
from typing import Literal

import pandas
import pytest

import convoys.export
import convoys.utils


@pytest.mark.parametrize(
    "model",
    [
        "kaplan-meier",
        "weibull",
        "exponential",
    ],
)
def test_export_cohorts(
    model: Literal[
        "kaplan-meier", "exponential", "weibull", "gamma", "generalized-gamma"
    ],
    weibull_df: pandas.DataFrame,
) -> None:
    df = weibull_df
    unit, groups, (G, B, T) = convoys.utils.get_arrays(df, unit="days")
    assert groups is not None

    result_df = convoys.export.export_cohorts(
        G, B, T, model=model, ci=0.95, groups=groups
    )

    assert set(result_df["group"]) == set(groups)

    assert result_df["t"].dtype == int
    assert max(T) - 1 <= max(result_df["t"]) == floor(max(T))
    assert min(result_df["t"]) == 0

    assert 0 <= min(result_df["prediction_value"])
    assert max(result_df["prediction_value"]) <= 1


def test_export_cohorts_bad_model_raises(weibull_df: pandas.DataFrame) -> None:
    df = weibull_df
    unit, groups, (G, B, T) = convoys.utils.get_arrays(df)

    with pytest.raises(ValueError):
        convoys.export.export_cohorts(G, B, T, model="bad", groups=groups)  # type: ignore[arg-type]


def test_export_cohorts_bad_groups_raises(weibull_df: pandas.DataFrame) -> None:
    df = weibull_df
    unit, groups, (G, B, T) = convoys.utils.get_arrays(df)

    with pytest.raises(ValueError):
        convoys.export.export_cohorts(
            G, B, T, model="kaplan-meier", groups=groups, specific_groups=["Nonsense"]
        )
