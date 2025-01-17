from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Generic, Hashable, TypeVar

import numpy

from convoys import regression, single

if TYPE_CHECKING:
    from numpy import ndarray
    from numpy.typing import ArrayLike

T_Group = TypeVar("T_Group", "ArrayLike", Hashable)

__all__ = ["KaplanMeier", "Exponential", "Weibull", "Gamma", "GeneralizedGamma"]


class MultiModel(ABC, Generic[T_Group]):
    _base_model_cls: type[regression.RegressionModel | single.SingleModel]

    @abstractmethod
    def fit(self, G: "ArrayLike", B: "ArrayLike", T: "ArrayLike") -> None:
        raise NotImplementedError("Need to implement fit")

    @abstractmethod
    def predict(self, group: T_Group, t: "ArrayLike") -> "ndarray":
        raise NotImplementedError("Need to implement predict")

    @abstractmethod
    def predict_ci(self, group: T_Group, t: "ArrayLike", ci: float) -> "ndarray":
        raise NotImplementedError("Need to implement predict_ci")


class RegressionToMulti(MultiModel["ArrayLike"]):
    _base_model_cls: type[regression.RegressionModel]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.base_model = self._base_model_cls(*args, **kwargs)

    def fit(self, G: "ArrayLike", B: "ArrayLike", T: "ArrayLike") -> None:
        """Fits the model

        :param G: numpy vector of shape :math:`n`
        :param B: numpy vector of shape :math:`n`
        :param T: numpy vector of shape :math:`n`
        """
        G = numpy.array(G, dtype=int)
        (n,) = G.shape
        self._n_groups = max(G) + 1
        X = numpy.zeros((n, self._n_groups), dtype=numpy.bool)
        for i, group in enumerate(G):
            X[i, group] = 1
        self.base_model.fit(X, B, T)

    def _get_x(self, group: "ArrayLike") -> "ndarray":
        x = numpy.zeros(self._n_groups)
        g = numpy.array(group)
        x[g] = 1
        return x

    def predict(self, group: "ArrayLike", t: "ArrayLike") -> "ndarray":
        return self.base_model.predict(self._get_x(group), t)

    def predict_ci(self, group: "ArrayLike", t: "ArrayLike", ci: float) -> "ndarray":
        return self.base_model.predict_ci(self._get_x(group), t, ci)

    def rvs(
        self, group: "ArrayLike", *args: Any, **kwargs: Any
    ) -> tuple["ndarray", "ndarray"]:
        return self.base_model.rvs(self._get_x(group), *args, **kwargs)


class SingleToMulti(MultiModel[Hashable]):
    _base_model_cls: type[single.SingleModel]

    def __init__(self, *args: Any, **kwargs: Any):
        self.base_model_init: Callable[[], single.SingleModel] = (
            lambda: self._base_model_cls(*args, **kwargs)
        )

    def fit(self, G: "ArrayLike", B: "ArrayLike", T: "ArrayLike") -> None:
        """Fits the model

        :param G: numpy vector of shape :math:`n`
        :param B: numpy vector of shape :math:`n`
        :param T: numpy vector of shape :math:`n`
        """
        G = numpy.array(G)
        B = numpy.array(B)
        T = numpy.array(T)
        group2bt: dict[Hashable, list[tuple]] = {}
        for g, b, t in zip(G, B, T, strict=False):
            group2bt.setdefault(g, []).append((b, t))
        self._group2model: dict[Hashable, single.SingleModel] = {}
        for g, BT in group2bt.items():
            self._group2model[g] = self.base_model_init()
            self._group2model[g].fit([b for b, t in BT], [t for b, t in BT])

    def predict(self, group: Hashable, t: "ArrayLike") -> "ndarray":
        return self._group2model[group].predict(t)

    def predict_ci(self, group: Hashable, t: "ArrayLike", ci: float) -> "ndarray":
        return self._group2model[group].predict_ci(t, ci)


class Exponential(RegressionToMulti):
    """Multi-group version of :class:`convoys.regression.Exponential`."""

    _base_model_cls = regression.Exponential


class Weibull(RegressionToMulti):
    """Multi-group version of :class:`convoys.regression.Weibull`."""

    _base_model_cls = regression.Weibull


class Gamma(RegressionToMulti):
    """Multi-group version of :class:`convoys.regression.Gamma`."""

    _base_model_cls = regression.Gamma


class GeneralizedGamma(RegressionToMulti):
    """Multi-group version of :class:`convoys.regression.GeneralizedGamma`."""

    _base_model_cls = regression.GeneralizedGamma


class KaplanMeier(SingleToMulti):
    """Multi-group version of :class:`convoys.single.KaplanMeier`."""

    _base_model_cls = single.KaplanMeier
