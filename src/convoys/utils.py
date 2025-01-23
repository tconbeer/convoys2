from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, Callable, Hashable, Literal, Sequence

import numpy
import pandas

if TYPE_CHECKING:
    Unit = Literal["years", "days", "hours", "minutes", "seconds"]

__all__ = ["get_arrays"]


def get_timescale(
    t: datetime.timedelta | pandas.Timedelta, unit: "Unit" | None
) -> tuple["Unit" | None, Callable[[Any], float]]:
    """Take a datetime or a numerical type, return two things:

    1. A unit
    2. A function that converts it to numerical form
    """

    def get_timedelta_converter(
        t_factor: int | float,
    ) -> Callable[[datetime.timedelta | pandas.Timedelta], float]:
        return lambda td: td.total_seconds() * t_factor

    if not isinstance(t, datetime.timedelta) or not isinstance(t, pandas.Timedelta):
        # Assume numeric type
        return None, lambda x: float(x)
    for u, f in [
        ("years", 365.25 * 24 * 60 * 60),
        ("days", 24 * 60 * 60),
        ("hours", 60 * 60),
        ("minutes", 60),
        ("seconds", 1),
    ]:
        if u == unit or (unit is None and t >= datetime.timedelta(seconds=f)):
            return u, get_timedelta_converter(1.0 / f)  # type: ignore[return-value]
    raise ValueError(f"Could not find unit for {t} and {unit}")


def get_groups(
    data: pandas.Series, group_min_size: int, max_groups: int
) -> list[Hashable]:
    """Picks the top groups out of a dataset

    1. Remove groups with too few data points
    2. Pick the top groups
    3. Sort groups lexicographically
    """
    group2count: dict[Hashable, int] = {}
    for group in data:
        group2count[group] = group2count.get(group, 0) + 1

    groups = [group for group, count in group2count.items() if count >= group_min_size]
    if max_groups >= 0:
        groups = sorted(
            groups,
            key=group2count.get,  # type: ignore[arg-type]
            reverse=True,
        )[:max_groups]
    return sorted(groups, key=lambda g: (g is None, g))  # Put Nones last


def _sub(a: datetime.datetime | Any, b: Any) -> Any:
    # Computes a - b for a bunch of different cases
    if isinstance(a, datetime.datetime) and a.tzinfo is not None:
        return a.astimezone(b.tzinfo) - b
    else:
        # Either naive timestamps or numerical type
        return a - b


def get_arrays(
    data: pandas.DataFrame,
    features: str | Sequence[str] | None = None,
    groups: str | None = None,
    created: str | None = None,
    converted: str | None = None,
    now: str | None = None,
    unit: "Unit" | None = None,
    group_min_size: int = 0,
    max_groups: int = -1,
) -> tuple[
    "Unit" | None,
    list[Hashable] | None,
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray],
]:
    """Converts a dataframe to a list of numpy arrays.

    Generates either feature data, or group data.

    :param data: Pandas dataframe
    :param features: string or Sequence[str] (optional), refers to a column
        in the dataframe containing features, each being a 1d-vector or list
        of features. If not provided, then it it will look for a column in the
        dataframe named "features". This argument can also be a list of columns.
    :param groups: string (optional), refers to a column in the dataframe
        containing the groups for each row. If not provided, then it will
        look for a column in the dataframe named "groups".
    :param created: string (optional), refers to a column in the dataframe
        containing timestamps of when each item was "created". If not
        provided, then it will look for a column in the dataframe named
        "created".
    :param converted: string, refers to a column in the dataframe
        containing timestamps of when each item converted. If there is no
        column containing creation values, then the converted values should
        be timedeltas denoting time until conversion. If this argument is
        not provided, then it will look for a column in the dataframe named
        "created".
    :param now: string (optional), refers to a column in the dataframe
        containing the point in time up until which we have observed
        non-conversion. If there is no column containing creation value,
        then these values should be timedeltas. If this argument is not
        provided, the current timestamp will be used.
    :param unit: string (optional), time unit to use when converting to
        numerical values. Has to be one of "years", "days", "hours",
        "minutes", or "seconds". If not provided, then a choice will be
        made based on the largest time interval in the inputs.
    :param group_min_size: integer (optional), only include groups that
        has at least this many observations
    :param max_groups: integer (optional), only include the `n` largest
        groups
    :returns: tuple (unit, groups, arrays)

        `unit` is the unit chosen. Will be one of "years", "days", "hours",
        "minutes", or "seconds". If the `unit` parameter is passed, this
        will be the same.

        `groups` is a list of strings containing the groups. Will be `None`
         if `groups` is not set.

        `arrays` is a tuple of numpy arrays `(G, B, T)` or `(X, B, T)`
        containing the transformed input in numerical format. `G`, `B`, `T`
        will all be 1D numpy arrays. `X` will be a 2D numpy array.
    """
    # First, construct either the `X` or the `G` array
    if features is None and groups is None:
        if "group" in data.columns:
            groups = "group"
        elif "features" in data.columns:
            features = "features"
        else:
            raise Exception(
                "Neither of the `features` or `group` parameters"
                " was provided, and there was no `features` or"
                " `groups` dataframe column"
            )
    if groups is not None:
        groups_list = get_groups(data[groups], group_min_size, max_groups)
        group2j = dict((group, j) for j, group in enumerate(groups_list))
        # Remove rows for rare groups
        data = data[data[groups].isin(group2j.keys())]
        G = data[groups].apply(lambda g: group2j.get(g, -1)).to_numpy()
        retval = G
    else:
        groups_list = None
        if isinstance(features, tuple):
            features = list(features)  # Otherwise sad Panda
        # this creates an array of shape (n, k), whether features is a
        # single Series containing tuples of length k, or features is a
        # list of columns of length k.
        X = numpy.array([numpy.array(z) for z in data[features].values])
        retval = X

    # Next, construct the `B` and `T` arrays
    if converted is None:
        if "converted" in data.columns:
            converted = "converted"
        else:
            raise Exception(
                "The `converted` parameter was not provided"
                " and there was no `converted` dataframe column"
            )
    if now is None and "now" in data.columns:
        now = "now"
    if created is None and "created" in data.columns:
        created = "created"
    B = ~pandas.isnull(data[converted]).to_numpy()

    def _calculate_T(row: pandas.Series) -> Any:
        if not pandas.isnull(row[converted]):
            if created is not None:
                return _sub(row[converted], row[created])
            else:
                return row[converted]
        else:
            if created is not None:
                if now is not None:
                    return _sub(row[now], row[created])
                else:
                    return datetime.datetime.now(tz=row[created].tzinfo) - row[created]
            elif now is not None:
                return row[now]
            else:
                return datetime.datetime.now()

    T_deltas = data.apply(_calculate_T, axis=1)
    max_T_delta = T_deltas.max()
    unit, converter = get_timescale(max_T_delta, unit)
    T = T_deltas.apply(converter).to_numpy()

    return unit, groups_list, (retval, B, T)
