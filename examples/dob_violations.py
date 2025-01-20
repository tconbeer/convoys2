import datetime
from pathlib import Path

import pandas
from matplotlib import pyplot

import convoys.plotting
import convoys.utils

here = Path(__file__)


def run() -> None:
    print("loading data")
    df = pandas.read_pickle("examples/dob_violations.pickle")
    print(df["issue_date"])
    print(df["issue_date"].dtype)
    print(df["issue_date"] < datetime.date(2018, 1, 1))
    df = df[df["issue_date"] < datetime.date(2018, 1, 1)]

    print("converting to arrays")
    unit, groups, (G, B, T) = convoys.utils.get_arrays(
        df,
        groups="type",
        created="issue_date",
        converted="disposition_date",
        unit="years",
        group_min_size=100,
    )

    for model in ["kaplan-meier", "weibull"]:
        print("plotting", model)
        pyplot.figure(figsize=(9, 6))
        convoys.plotting.plot_cohorts(
            G,
            B,
            T,
            model=model,  # type: ignore[arg-type]
            ci=0.95,
            groups=groups,
            t_max=30,
        )
        pyplot.legend()
        assert unit is not None
        pyplot.xlabel(unit)
        fig_path = here.parent / f"dob-violations-{model}.png"
        pyplot.savefig(fig_path)

    pyplot.figure(figsize=(9, 6))
    df["bucket"] = df["issue_date"].apply(
        lambda d: "%d-%d" % (5 * (d.year // 5), 5 * (d.year // 5) + 4)
    )
    unit, groups, (G, B, T) = convoys.utils.get_arrays(
        df,
        groups="bucket",
        created="issue_date",
        converted="disposition_date",
        unit="years",
        group_min_size=500,
    )
    convoys.plotting.plot_cohorts(
        G, B, T, model="kaplan-meier", groups=groups, t_max=30, ci=0.95
    )
    convoys.plotting.plot_cohorts(
        G,
        B,
        T,
        model="weibull",
        groups=groups,
        t_max=30,
        plot_kwargs={"linestyle": "--"},
    )
    pyplot.legend()
    assert unit is not None
    pyplot.xlabel(unit)
    fig_path = here.parent / "dob-violations-combined.png"
    pyplot.savefig(fig_path)


if __name__ == "__main__":
    run()
