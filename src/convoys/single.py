import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy
import scipy.stats  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = ["KaplanMeier"]


class SingleModel(ABC):
    @abstractmethod
    def fit(self, B: "ArrayLike", T: "ArrayLike") -> None:
        raise NotImplementedError("Need to implement fit")

    @abstractmethod
    def predict(self, t: "ArrayLike") -> numpy.ndarray:
        raise NotImplementedError("Need to implement predict")

    @abstractmethod
    def predict_ci(self, t: "ArrayLike", ci: float) -> numpy.ndarray:
        raise NotImplementedError("Need to implement predict_ci")


class KaplanMeier(SingleModel):
    """Implementation of the Kaplan-Meier nonparametric method."""

    def fit(self, B: "ArrayLike", T: "ArrayLike") -> None:
        """Fits the model

        :param B: numpy vector of shape :math:`n`
        :param T: numpy vector of shape :math:`n`
        """
        if not isinstance(B, numpy.ndarray):
            B = numpy.array(B)
        if not isinstance(T, numpy.ndarray):
            T = numpy.array(T)
        # See https://www.math.wustl.edu/~sawyer/handouts/greenwood.pdf
        BT = [
            (b, t) for b, t in zip(B, T, strict=False) if t >= 0 and 0 <= float(b) <= 1
        ]
        if len(BT) < len(B):
            n_removed = len(B) - len(BT)
            warnings.warn(
                "Warning! Removed %d/%d entries from inputs where "
                "T < 0 or B not 0/1" % (n_removed, len(B)),
                stacklevel=2,
            )
        B, T = ([z[i] for z in BT] for i in range(2))
        n = len(T)
        self._ts = [0.0]
        self._ss = [1.0]
        self._vs = [0.0]
        sum_var_terms = 0.0
        prod_s_terms = 1.0
        for t, b in sorted(zip(T, B, strict=False)):
            d = float(b)
            self._ts.append(t)
            prod_s_terms *= 1 - d / n
            self._ss.append(prod_s_terms)
            if d == n == 1:
                sum_var_terms = float("inf")
            else:
                sum_var_terms += d / (n * (n - d))
            if sum_var_terms > 0:
                self._vs.append(1 / numpy.log(prod_s_terms) ** 2 * sum_var_terms)
            else:
                self._vs.append(0)
            n -= 1

        # Just prevent overflow warning when computing the confidence interval
        eps = 1e-9
        self._ss_clipped = numpy.clip(self._ss, eps, 1.0 - eps)

    def predict(self, t: "ArrayLike") -> numpy.ndarray:
        """Returns the predicted values."""
        t = numpy.array(t)
        res = numpy.zeros(t.shape)
        for indexes, value in numpy.ndenumerate(t):
            j = numpy.searchsorted(self._ts, value, side="right") - 1
            if j >= len(self._ts) - 1:
                # Make the plotting stop at the last value of t
                res[indexes] = float("nan")
            else:
                res[indexes] = 1 - self._ss[j]
        return res

    def predict_ci(self, t: "ArrayLike", ci: float = 0.8) -> numpy.ndarray:
        """Returns the predicted values with a confidence interval."""
        t = numpy.array(t)
        res = numpy.zeros(t.shape + (3,))
        for indexes, value in numpy.ndenumerate(t):
            j = numpy.searchsorted(self._ts, value, side="right") - 1
            if j >= len(self._ts) - 1:
                # Make the plotting stop at the last value of t
                res[indexes] = [float("nan")] * 3
            else:
                z_lo, z_hi = scipy.stats.norm.ppf([(1 - ci) / 2, (1 + ci) / 2])
                res[indexes] = (
                    1 - self._ss[j],
                    1
                    - numpy.exp(
                        -numpy.exp(
                            numpy.log(-numpy.log(self._ss_clipped[j]))
                            + z_hi * self._vs[j] ** 0.5
                        )
                    ),
                    1
                    - numpy.exp(
                        -numpy.exp(
                            numpy.log(-numpy.log(self._ss_clipped[j]))
                            + z_lo * self._vs[j] ** 0.5
                        )
                    ),
                )
        return res
