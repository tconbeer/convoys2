from __future__ import annotations

from math import floor
from typing import TYPE_CHECKING, Callable, Hashable, Literal

import numpy
import pandas

import convoys.multi

if TYPE_CHECKING:
    pass

__all__ = ["export_cohorts"]


_models: dict[str, Callable[[bool], convoys.multi.MultiModel]] = {
    "kaplan-meier": lambda _: convoys.multi.KaplanMeier(),
    "exponential": lambda ci: convoys.multi.Exponential(mcmc=ci),
    "weibull": lambda ci: convoys.multi.Weibull(mcmc=ci),
    "gamma": lambda ci: convoys.multi.Gamma(mcmc=ci),
    "generalized-gamma": lambda ci: convoys.multi.GeneralizedGamma(mcmc=ci),
}


def export_cohorts(
    G: numpy.ndarray,
    B: numpy.ndarray,
    T: numpy.ndarray,
    t_max: int | float | None = None,
    model: Literal[
        "kaplan-meier", "exponential", "weibull", "gamma", "generalized-gamma"
    ]
    | convoys.multi.MultiModel = "kaplan-meier",
    ci: float | None = None,
    groups: list[Hashable] | None = None,
    specific_groups: list[Hashable] | None = None,
) -> pandas.DataFrame:
    """Helper function to fit data using a model and then
    export the model predictions as a DataFrame. The Dataframe will
    have the columns "group", "t", and "prediction_value"; if ci is not None,
    it will also have the columns "ci_low" and "ci_high". "t" will be integers
    in the range from 0 to t_max (or max(T) if t_max is None). This
    format enables easy storage in a database and plotting with BI tools.

    :param G: numpy array of shape :math:`n`, containing integers representing group
        assignments.
    :param B: numpy array of shape :math:`n`, containing booleans representing
        whether or not the subject 'converted' at time delta :math:`t`.
    :param T: numpy array of shape :math:`n`, containing floats representing the
        time delta :math:`t` between creation and either conversion or censoring.
    :param t_max: (optional) max value for x axis
    :param model: (optional, default is kaplan-meier) model to fit.
        Can be an instance of :class:`multi.MultiModel` or a string
        identifying the model. One of 'kaplan-meier', 'exponential',
        'weibull', 'gamma', or 'generalized-gamma'.
    :param ci: confidence interval, value from 0-1, or None (default) if
        no confidence interval is to be plotted
    :param groups: list of group labels
    :param specific_groups: subset of groups to plot

    See  :meth:`convoys.utils.get_arrays` which is handy for converting
    a Pandas dataframe into arrays `G`, `B`, `T`.
    """

    if model not in _models.keys():
        if not isinstance(model, convoys.multi.MultiModel):
            raise ValueError("model incorrectly specified")

    if groups is None:
        groups = list(set(G))

    # Set x scale
    if t_max is None:
        t_max = max(T)
    if not isinstance(model, convoys.multi.MultiModel):
        # Fit model
        m = _models[model](bool(ci))
        m.fit(G, B, T)
    else:
        m = model

    if specific_groups is None:
        specific_groups = groups

    if len(set(specific_groups).intersection(groups)) != len(specific_groups):
        raise ValueError("specific_groups not a subset of groups!")

    RESULT_LENGTH = floor(t_max + 1)
    t = numpy.arange(RESULT_LENGTH)
    data: list[pandas.DataFrame] = []

    for group in specific_groups:
        j = groups.index(group)  # matching index of group

        if ci is not None:
            result = m.predict_ci(j, t, ci=ci)
            result_df = pandas.DataFrame(
                data=result, columns=["prediction_value", "ci_low", "ci_high"]
            )
        else:
            result = m.predict(j, t)
            result_df = pandas.DataFrame(data=result, columns=["prediction_value"])
        result_df["t"] = t
        result_df["group"] = group
        data.append(result_df)

    unioned_data = pandas.concat(data)
    return unioned_data
