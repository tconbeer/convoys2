# Convoys2 CHANGELOG

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.4.0] - 2025-01-23

### Features

- Adds the `convoys.export` module, which includes the `export_cohorts` function. `export_cohorts`
  samples from the model predictions to create a DataFrame that can be easily inserted into a
  database table and plotted with a BI tool.

## [0.3.0] - 2025-01-20

### Breaking Changes

- Removes the deprecated `MultiModel.cdf()` method. Use `predict()` or `predict_ci()` instead.

[unreleased]: https://github.com/tconbeer/convoys2/compare/0.4.0...HEAD
[0.4.0]: https://github.com/tconbeer/convoys2/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/tconbeer/convoys2/compare/2223d1e343ef32cf067489e1f661f9415b9e9222...0.3.0
