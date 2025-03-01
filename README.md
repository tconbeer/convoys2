Convoys2
=======

![pic](docs/images/dob-violations-combined.png)

Convoys is a simple library that fits a few statistical model useful for modeling time-lagged conversions.
There is a lot more info if you head over to the [documentation](https://tconbeer.github.io/convoys2/).

[The original blog post](https://better.engineering/2019/07/29/modeling-conversion-rates-and-saving-millions-of-dollars-using-kaplan-meier-and-gamma-distributions/) about Convoys provides more motivation and background on survival analysis.

This package is an updated and maintained fork of the original, which hasn't seen a commit since 2021.

Installation
------------

You can install from PyPI using any compatible tool (pip, uv, poetry, etc.):

```bash
pip install convoys2
```

You can then import the package using the name `convoys`:

```py
import convoys
```

More info
---------

Convoys was built by [Erik Bernhardsson](https://github.com/erikbern) and has the MIT license. The original repo lives [here](https://github.com/better/convoys).

In 2025, it was updated for breaking changes caused by Numpy 2.0 and other dependencies by [Ted Conbeer](https://tedconbeer.com). It has since added new features.
