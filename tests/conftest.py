import datetime
import random

import matplotlib
import numpy
import pandas
import pytest
import scipy.stats  # type: ignore[import-untyped]

matplotlib.use("Agg")  # Needed for matplotlib to run in Travis


class Utilities:
    @staticmethod
    def sample_weibull(k: float, lambd: float) -> float:
        # scipy.stats is garbage for this
        # exp(-(x * lambda)^k) = y
        base: numpy.float64 = -numpy.log(random.random())
        return base ** (1.0 / k) / lambd

    @staticmethod
    def generate_censored_data(
        N: numpy.ndarray, E: numpy.ndarray, C: numpy.ndarray
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        B = numpy.array([c and e < n for n, e, c in zip(N, E, C, strict=False)])
        T = numpy.array([e if b else n for e, b, n in zip(E, B, N, strict=False)])
        return B, T


@pytest.fixture(scope="session")
def utilities() -> Utilities:
    return Utilities()


def generate_weibull_df() -> pandas.DataFrame:
    cs = (0.3, 0.5, 0.7)
    k = 0.5
    lambd = 0.1
    n = 1000
    groups = [r % len(cs) for r in range(n)]
    C = numpy.array([bool(random.random() < cs[g]) for g in groups])
    N = scipy.stats.expon.rvs(scale=10.0 / lambd, size=(n,))
    E = numpy.array([Utilities.sample_weibull(k, lambd) for r in range(n)])
    B, T = Utilities.generate_censored_data(N, E, C)

    def x2t(x: int) -> datetime.datetime:
        return datetime.datetime(2000, 1, 1) + datetime.timedelta(days=x)

    return pandas.DataFrame(
        data=dict(
            group=["Group %d" % g for g in groups],
            created=[x2t(0) for g in groups],
            converted=[x2t(t) if b else None for t, b in zip(T, B, strict=False)],
            now=[x2t(n) for n in N],
        )
    )


weibull_data = generate_weibull_df()


@pytest.fixture(scope="function")
def weibull_df() -> pandas.DataFrame:
    return weibull_data.copy()
