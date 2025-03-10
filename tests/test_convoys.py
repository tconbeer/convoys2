import datetime
import random
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import flaky
import matplotlib
import numpy
import pandas
import pytest
import scipy.stats  # type: ignore[import-untyped]
from numpy.typing import ArrayLike

matplotlib.use("Agg")  # Needed for matplotlib to run in Travis
import convoys
import convoys.multi
import convoys.plotting
import convoys.regression
import convoys.single
import convoys.utils

if TYPE_CHECKING:
    from tests.conftest import Utilities


def test_kaplan_meier_model() -> None:
    data = [(2, 0), (3, 0), (6, 1), (6, 1), (7, 1), (10, 0)]
    now = pandas.Timestamp("2019-01-22")  # fix now end date for easier testing
    created_array = [now - pandas.DateOffset(n=t) for t, _ in data]
    converted_array = [
        ts + pandas.DateOffset(n=t) if e == 1 else numpy.nan
        for ts, (t, e) in zip(created_array, data, strict=False)
    ]
    df = pandas.DataFrame(
        {"created_at": created_array, "converted_at": converted_array, "group": 1}
    )
    df["now"] = now
    unit, groups, (G, B, T) = convoys.utils.get_arrays(
        df, converted="converted_at", created="created_at", unit="days"
    )
    m = convoys.multi.KaplanMeier()
    m.fit(G, B, T)
    assert m.predict(0, 9) == 0.75


def test_output_shapes(
    utilities: "Utilities",
    c: float = 0.3,
    lambd: float = 0.1,
    n: int = 1000,
    k: int = 5,
) -> None:
    X = numpy.random.randn(n, k)
    C = scipy.stats.bernoulli.rvs(c, size=(n,))
    N = scipy.stats.uniform.rvs(scale=5.0 / lambd, size=(n,))
    E = scipy.stats.expon.rvs(scale=1.0 / lambd, size=(n,))
    B, T = utilities.generate_censored_data(N, E, C)

    # Fit model with ci
    model = convoys.regression.Exponential(mcmc=True)
    model.fit(X, B, T)

    # Generate output without ci
    assert model.predict(X[0], 0).shape == ()
    assert model.predict([X[0], X[1]], 0).shape == (2,)
    assert model.predict([X[0]], [0, 1, 2, 3]).shape == (4,)
    assert model.predict([X[0], X[1], X[2]], [0, 1, 2]).shape == (3,)
    assert model.predict([[X[0], X[1]]], [[0], [1], [2]]).shape == (3, 2)
    assert model.predict([[X[0]], [X[1]]], [[0, 1, 2]]).shape == (2, 3)

    # Generate output with ci (same as above plus (3,))
    assert model.predict_ci(X[0], 0, ci=0.8).shape == (3,)
    assert model.predict_ci([X[0], X[1]], 0, ci=0.8).shape == (2, 3)
    assert model.predict_ci([X[0]], [0, 1, 2, 3], ci=0.8).shape == (4, 3)
    assert model.predict_ci([X[0], X[1], X[2]], [0, 1, 2], ci=0.8).shape == (3, 3)
    assert model.predict_ci([[X[0], X[1]]], [[0], [1], [2]], ci=0.8).shape == (3, 2, 3)
    assert model.predict_ci([[X[0]], [X[1]]], [[0, 1, 2]], ci=0.8).shape == (2, 3, 3)

    # Fit model without ci (should be the same)
    model = convoys.regression.Exponential(mcmc=False)
    model.fit(X, B, T)
    assert model.predict(X[0], 0).shape == ()
    assert model.predict([X[0], X[1]], [0, 1]).shape == (2,)


@flaky.flaky
def test_exponential_regression_model(
    utilities: "Utilities", c: float = 0.3, lambd: float = 0.1, n: int = 10000
) -> None:
    X = numpy.ones((n, 1))
    C: numpy.ndarray = scipy.stats.bernoulli.rvs(c, size=(n,))  # did it convert
    N: numpy.ndarray = scipy.stats.uniform.rvs(scale=5.0 / lambd, size=(n,))  # time now
    E: numpy.ndarray = scipy.stats.expon.rvs(
        scale=1.0 / lambd, size=(n,)
    )  # time of event
    B, T = utilities.generate_censored_data(N, E, C)
    model = convoys.regression.Exponential(mcmc=True)
    model.fit(X, B, T)
    assert 0.80 * c < model.predict([1], float("inf")) < 1.30 * c
    for t in [1, 3, 10]:
        d = 1 - numpy.exp(-lambd * t)
        assert 0.80 * c * d < model.predict([1], t) < 1.30 * c * d

    # Check the confidence intervals
    assert model.predict_ci([1], float("inf"), ci=0.95).shape == (3,)
    assert model.predict_ci([1], [0, 1, 2, 3], ci=0.95).shape == (4, 3)
    y, y_lo, y_hi = model.predict_ci([1], float("inf"), ci=0.95)
    assert 0.80 * c < y < 1.30 * c

    # Check the random variates
    will_convert, convert_at = model.rvs([1], n_curves=10000, n_samples=1)
    assert 0.80 * c < numpy.mean(will_convert) < 1.30 * c
    convert_times = convert_at[will_convert]
    for t in [1, 3, 10]:
        d = 1 - numpy.exp(-lambd * t)
        assert 0.70 * d < (convert_times < t).mean() < 1.30 * d

    # Fit a linear model
    model = convoys.regression.Exponential(mcmc=False, flavor="linear")
    model.fit(X, B, T)
    model_c = model.params["map"]["b"] + model.params["map"]["beta"][0]
    assert 0.9 * c < model_c < 1.1 * c
    for t in [1, 3, 10]:
        d = 1 - numpy.exp(-lambd * t)
        assert 0.80 * c * d < model.predict([1], t) < 1.30 * c * d


@flaky.flaky
def test_weibull_regression_model(
    utilities: "Utilities",
    cs: tuple[float, float, float] = (0.3, 0.5, 0.7),
    lambd: float = 0.1,
    k: float = 0.5,
    n: int = 10000,
) -> None:
    X = numpy.array([[r % len(cs) == j for j in range(len(cs))] for r in range(n)])
    C = numpy.array([bool(random.random() < cs[r % len(cs)]) for r in range(n)])
    N = scipy.stats.uniform.rvs(scale=5.0 / lambd, size=(n,))
    E = numpy.array([utilities.sample_weibull(k, lambd) for r in range(n)])
    B, T = utilities.generate_censored_data(N, E, C)

    model = convoys.regression.Weibull()
    model.fit(X, B, T)

    # Validate shape of results
    x: "ArrayLike" = numpy.ones((len(cs),))
    assert model.predict(x, float("inf")).shape == ()
    assert model.predict(x, 1).shape == ()
    assert model.predict(x, [1, 2, 3, 4]).shape == (4,)

    # Check results
    for r, c in enumerate(cs):
        x = [int(r == j) for j in range(len(cs))]
        assert 0.80 * c < model.predict(x, float("inf")) < 1.30 * c

    # Fit a linear model
    model = convoys.regression.Weibull(mcmc=False, flavor="linear")
    model.fit(X, B, T)
    model_cs = model.params["map"]["b"] + model.params["map"]["beta"]
    for model_c, c in zip(model_cs, cs, strict=False):
        assert 0.8 * c < model_c < 1.2 * c


@flaky.flaky
def test_gamma_regression_model(
    utilities: "Utilities",
    c: float = 0.3,
    lambd: float = 0.1,
    k: float = 3.0,
    n: int = 10000,
) -> None:
    # TODO: this one seems very sensitive to large values for N (i.e. less censoring)
    X = numpy.ones((n, 1))
    C = scipy.stats.bernoulli.rvs(c, size=(n,))
    N = scipy.stats.uniform.rvs(scale=20.0 / lambd, size=(n,))
    E = scipy.stats.gamma.rvs(a=k, scale=1.0 / lambd, size=(n,))
    B, T = utilities.generate_censored_data(N, E, C)

    model = convoys.regression.Gamma()
    model.fit(X, B, T)
    assert 0.80 * c < model.predict([1], float("inf")) < 1.30 * c
    assert 0.80 * k < numpy.mean(model.params["map"]["k"]) < 1.30 * k

    # Fit a linear model
    model = convoys.regression.Gamma(mcmc=False, flavor="linear")
    model.fit(X, B, T)
    model_c = model.params["map"]["b"] + model.params["map"]["beta"][0]
    assert 0.9 * c < model_c < 1.1 * c


@flaky.flaky
def test_linear_model(
    utilities: "Utilities",
    n: int = 10000,
    m: int = 5,
    k: float = 3.0,
    lambd: float = 0.1,
) -> None:
    # Generate data with little censoring
    # The coefficients should be quite close to their actual value
    cs = numpy.random.dirichlet(numpy.ones(m))
    X = numpy.random.binomial(n=1, p=0.5, size=(n, m))
    C = numpy.random.rand(n) < numpy.dot(X, cs.T)
    N = scipy.stats.uniform.rvs(scale=20.0 / lambd, size=(n,))
    E = numpy.array([utilities.sample_weibull(k, lambd) for r in range(n)])
    B, T = utilities.generate_censored_data(N, E, C)

    model = convoys.regression.Weibull(mcmc=False, flavor="linear")
    model.fit(X, B, T)

    # Check the fitted parameters
    model_cs = model.params["map"]["b"] + model.params["map"]["beta"]
    for model_c, c in zip(model_cs, cs, strict=False):
        assert c - 0.03 < model_c < c + 0.03
    model_lambds = numpy.exp(model.params["map"]["a"] + model.params["map"]["alpha"])
    for model_lambd in model_lambds:
        assert 0.95 * lambd < model_lambd < 1.05 * lambd

    # Check predictions
    for i, c in enumerate(cs):
        x = numpy.array([float(j == i) for j in range(m)])
        p = model.predict(x, float("inf"))
        assert c - 0.03 < p < c + 0.03
        t = 10.0
        p = model.predict(x, t)
        f = 1 - numpy.exp(-((t * lambd) ** k))
        assert c * f - 0.03 < p < c * f + 0.03


@flaky.flaky
def test_exponential_pooling(
    utilities: "Utilities",
    c: "ArrayLike" = 0.5,
    lambd: float = 0.01,
    n: int = 10000,
    ks: list[int] | tuple[int, ...] = (1, 2, 3),
) -> None:
    # Generate one series of n observations with c conversion rate
    # Then k1...kn series with zero conversion
    # The predicted conversion rates should go towards c for the small cohorts
    G = numpy.zeros(n + sum(ks))
    C = numpy.zeros(n + sum(ks))
    N = numpy.zeros(n + sum(ks))
    E = numpy.zeros(n + sum(ks))
    offset = 0
    for i, k in enumerate([n] + list(ks)):
        G[offset : offset + k] = i
        offset += k
    C[:n] = scipy.stats.bernoulli.rvs(c, size=(n,))
    N[:] = 1000.0
    E[:n] = scipy.stats.expon.rvs(scale=1.0 / lambd, size=(n,))
    B, T = utilities.generate_censored_data(N, E, C)

    # Fit model
    model = convoys.multi.Exponential()
    model.fit(G, B, T)

    # Generate predictions for each cohort
    c = numpy.array([model.predict(i, float("inf")) for i in range(1 + len(ks))])
    assert numpy.all(c[1:] > 0.25)  # rough check
    assert numpy.all(c[1:] < 0.50)  # same
    assert numpy.all(numpy.diff(c) < 0)  # c should be monotonically decreasing


def test_convert_dataframe(weibull_df: pandas.DataFrame) -> None:
    df = weibull_df
    unit, groups, (G, B, T) = convoys.utils.get_arrays(df)
    assert G.shape == B.shape == T.shape == (len(df),)


def test_convert_dataframe_features(weibull_df: pandas.DataFrame) -> None:
    df = weibull_df
    df["features"] = [
        tuple(numpy.random.randn() for z in range(3)) for g in df["group"]
    ]
    df = df.drop("group", axis=1)
    unit, groups, (X, B, T) = convoys.utils.get_arrays(df)
    assert X.shape == (len(df), 3)


def test_convert_dataframe_features_multi_cols(weibull_df: pandas.DataFrame) -> None:
    # Generate from multiple columns
    df = weibull_df
    df["feature_1"] = [numpy.random.randn() for g in df["group"]]
    df["feature_2"] = [numpy.random.randn() for g in df["group"]]
    df = df.drop("group", axis=1)
    unit, groups, (X, B, T) = convoys.utils.get_arrays(
        df, features=("feature_1", "feature_2")
    )
    assert X.shape == (len(df), 2)


def test_convert_dataframe_infer_now(weibull_df: pandas.DataFrame) -> None:
    df = weibull_df
    df = df.drop("now", axis=1)

    unit, groups, (G1, B1, T1) = convoys.utils.get_arrays(df, unit="days")

    # Now, let's make the timezone-naive objects timezone aware
    utc = datetime.timezone.utc
    local = datetime.datetime.now(utc).astimezone().tzinfo
    df[["created", "converted"]] = df[["created", "converted"]].map(
        lambda z: z.replace(tzinfo=local)
    )
    unit, groups, (G2, B2, T2) = convoys.utils.get_arrays(df, unit="days")

    # Convert everything to UTC and make sure it's still the same
    df[["created", "converted"]] = df[["created", "converted"]].map(
        lambda z: z.tz_convert(utc)
    )
    unit, groups, (G3, B3, T3) = convoys.utils.get_arrays(df, unit="days")

    # Let's check that all deltas are the same
    # There will be some slight clock drift, so let's accept up to 3s
    for t1, t2, t3 in zip(T1, T2, T3, strict=False):
        assert 0 <= t2 - t1 < 3.0 / (24 * 60 * 60)
        assert 0 <= t3 - t1 < 3.0 / (24 * 60 * 60)


def test_convert_dataframe_timedeltas(weibull_df: pandas.DataFrame) -> None:
    df = weibull_df

    unit, groups, (G1, B1, T1) = convoys.utils.get_arrays(df, unit="days")
    df2 = pandas.DataFrame(
        {
            "group": df["group"],
            "converted": df["converted"] - df["created"],
            "now": df["now"] - df["created"],
        }
    )
    unit, groups, (G2, B2, T2) = convoys.utils.get_arrays(df2, unit="days")

    for t1, t2 in zip(T1, T2, strict=False):
        assert 0 <= t2 - t1 < 3.0 / (24 * 60 * 60)


def test_convert_dataframe_more_args(weibull_df: pandas.DataFrame) -> None:
    df = weibull_df
    unit, groups, (G, B, T) = convoys.utils.get_arrays(df, max_groups=2)
    assert groups is not None
    assert len(groups) <= 2
    unit, groups, (G, B, T) = convoys.utils.get_arrays(df, group_min_size=9999)
    assert G.shape == (0,)


def test_convert_dataframe_created_at_nan(weibull_df: pandas.DataFrame) -> None:
    df = weibull_df
    df.loc[df.index[0], "created"] = None
    unit, groups, (G, B, T) = convoys.utils.get_arrays(df)
    assert numpy.issubdtype(G.dtype, numpy.integer)
    assert numpy.issubdtype(B.dtype, numpy.bool_)
    assert numpy.issubdtype(T.dtype, numpy.number)


@pytest.mark.parametrize(
    "model,extra_model",
    [
        ("kaplan-meier", None),
        ("weibull", None),
        ("exponential", None),
        ("kaplan-meier", "weibull"),
    ],
)
@flaky.flaky
def test_plot_cohorts(
    model: Literal[
        "kaplan-meier", "exponential", "weibull", "gamma", "generalized-gamma"
    ],
    extra_model: Literal[
        "kaplan-meier", "exponential", "weibull", "gamma", "generalized-gamma"
    ]
    | None,
    weibull_df: pandas.DataFrame,
) -> None:
    df = weibull_df
    unit, groups, (G, B, T) = convoys.utils.get_arrays(df)
    matplotlib.pyplot.clf()
    convoys.plotting.plot_cohorts(G, B, T, model=model, ci=0.95, groups=groups)
    matplotlib.pyplot.legend()
    if extra_model:
        convoys.plotting.plot_cohorts(
            G, B, T, model=extra_model, plot_kwargs=dict(linestyle="--", alpha=0.1)
        )
    here = Path(__file__)
    snapshots_dir = here.parent / "snapshots"
    snapshots_dir.mkdir(exist_ok=True)
    matplotlib.pyplot.savefig(
        snapshots_dir / f"{model}-{extra_model}.png"
        if extra_model is not None
        else snapshots_dir / f"{model}.png"
    )


def test_plot_cohorts_bad_model_raises(weibull_df: pandas.DataFrame) -> None:
    df = weibull_df
    unit, groups, (G, B, T) = convoys.utils.get_arrays(df)

    with pytest.raises(ValueError):
        convoys.plotting.plot_cohorts(G, B, T, model="bad", groups=groups)  # type: ignore[arg-type]


def test_plot_cohorts_bad_groups_raises(weibull_df: pandas.DataFrame) -> None:
    df = weibull_df
    unit, groups, (G, B, T) = convoys.utils.get_arrays(df)

    with pytest.raises(ValueError):
        convoys.plotting.plot_cohorts(
            G, B, T, model="kaplan-meier", groups=groups, specific_groups=["Nonsense"]
        )


def test_plot_cohorts_subplots(weibull_df: pandas.DataFrame) -> None:
    df = weibull_df
    unit, groups, (G, B, T) = convoys.utils.get_arrays(df)
    matplotlib.pyplot.clf()
    fix, axes = matplotlib.pyplot.subplots(nrows=2, ncols=2)
    for ax in axes.flatten():
        convoys.plotting.plot_cohorts(G, B, T, groups=groups, ax=ax)
        ax.legend()
    here = Path(__file__)
    snapshots_dir = here.parent / "snapshots"
    snapshots_dir.mkdir(exist_ok=True)
    matplotlib.pyplot.savefig(snapshots_dir / "subplots.png")


@pytest.fixture
def add_examples_to_path() -> None:
    import sys

    root = Path(__file__).parent.parent
    sys.path.append(str(root))


def test_marriage_example(add_examples_to_path: None) -> None:
    from examples.marriage import run

    run()


def test_dob_violations_example(add_examples_to_path: None) -> None:
    from examples.dob_violations import run

    run()
