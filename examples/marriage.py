from pathlib import Path

import pandas
from matplotlib import pyplot

import convoys.plotting
import convoys.utils


def run() -> None:
    print("loading data")
    df = pandas.read_pickle("examples/marriage.pickle")
    df = df.sample(1000)  # speed up
    print(df)

    _, groups, (G, B, T) = convoys.utils.get_arrays(
        df, groups="sex", created="born", converted="married"
    )

    pyplot.figure(figsize=(6, 6))
    convoys.plotting.plot_cohorts(G, B, T, model="generalized-gamma", groups=groups)
    pyplot.legend()
    pyplot.xlabel("Age of marriage")
    convoys.plotting.plot_cohorts(
        G, B, T, model="kaplan-meier", groups=groups, plot_kwargs={"linestyle": "--"}
    )
    fig_path = Path(__file__).parent / "marriage-combined.png"
    pyplot.savefig(fig_path)


if __name__ == "__main__":
    run()
