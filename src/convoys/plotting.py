from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Hashable, Literal

import numpy
from matplotlib import pyplot

import convoys.multi

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = ["plot_cohorts"]


_models: dict[str, Callable[[bool], convoys.multi.MultiModel]] = {
    "kaplan-meier": lambda _: convoys.multi.KaplanMeier(),
    "exponential": lambda ci: convoys.multi.Exponential(mcmc=ci),
    "weibull": lambda ci: convoys.multi.Weibull(mcmc=ci),
    "gamma": lambda ci: convoys.multi.Gamma(mcmc=ci),
    "generalized-gamma": lambda ci: convoys.multi.GeneralizedGamma(mcmc=ci),
}


def plot_cohorts(
    G: numpy.ndarray,
    B: numpy.ndarray,
    T: numpy.ndarray,
    t_max: int | float | None = None,
    model: Literal[
        "kaplan-meier", "exponential", "weibull", "gamma", "generalized-gamma"
    ]
    | convoys.multi.MultiModel = "kaplan-meier",
    ci: float | None = None,
    ax: "Axes" | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    plot_ci_kwargs: dict[str, Any] | None = None,
    groups: list[Hashable] | None = None,
    specific_groups: list[Hashable] | None = None,
    label_fmt: str = "%(group)s (n=%(n).0f, k=%(k).0f)",
) -> convoys.multi.MultiModel:
    """Helper function to fit data using a model and then plot the cohorts.

    :param G: list with group assignment
    :param B: list with group assignment
    :param T: list with group assignment
    :param t_max: (optional) max value for x axis
    :param model: (optional, default is kaplan-meier) model to fit.
        Can be an instance of :class:`multi.MultiModel` or a string
        identifying the model. One of 'kaplan-meier', 'exponential',
        'weibull', 'gamma', or 'generalized-gamma'.
    :param ci: confidence interval, value from 0-1, or None (default) if
        no confidence interval is to be plotted
    :param ax: custom pyplot axis to plot on
    :param plot_kwargs: extra arguments to pyplot for the lines
    :param plot_ci_kwargs: extra arguments to pyplot for the confidence
        intervals
    :param groups: list of group labels
    :param specific_groups: subset of groups to plot
    :param label_fmt: custom format for the labels to use in the legend

    See  :meth:`convoys.utils.get_arrays` which is handy for converting
    a Pandas dataframe into arrays `G`, `B`, `T`.
    """

    if model not in _models.keys():
        if not isinstance(model, convoys.multi.MultiModel):
            raise ValueError("model incorrectly specified")

    if groups is None:
        groups = list(set(G))

    if ax is None:
        ax = pyplot.gca()

    # Set x scale
    if t_max is None:
        _, t_max = ax.get_xlim()
        t_max = max(t_max, max(T))
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

    # Plot
    t = numpy.linspace(0, t_max, 1000)  # type: ignore[arg-type]
    _, y_max = ax.get_ylim()
    # Reset to first color
    ax.set_prop_cycle(None)  # type:ignore[call-overload]
    for group in specific_groups:
        j = groups.index(group)  # matching index of group

        n = numpy.sum(G == j)
        k = numpy.sum(B[G == j])
        label = label_fmt % dict(group=group, n=n, k=k)

        if ci is not None:
            p_y, p_y_lo, p_y_hi = m.predict_ci(j, t, ci=ci).T
            merged_plot_ci_kwargs = {"alpha": 0.2}
            if plot_ci_kwargs is not None:
                merged_plot_ci_kwargs.update(plot_ci_kwargs)
            p = ax.fill_between(
                t,
                100.0 * p_y_lo,
                100.0 * p_y_hi,
                **merged_plot_ci_kwargs,  # type: ignore[arg-type]
            )
            color = p.get_facecolor()[0]  # reuse color for the line
        else:
            p_y = m.predict(j, t).T
            color = None

        merged_plot_kwargs = {"color": color, "linewidth": 1.5, "alpha": 0.7}
        if plot_kwargs is not None:
            merged_plot_kwargs.update(plot_kwargs)
        ax.plot(t, 100.0 * p_y, label=label, **merged_plot_kwargs)  # type: ignore[arg-type]
        y_max = max(y_max, 110.0 * max(p_y))

    ax.set_xlim(0, t_max)
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Conversion rate %")
    ax.grid(True)
    return m
