# Educational C++ Machine Learning Library

This project is a from-scratch, object-oriented machine learning library written in modern C++17 with only the standard library. It is designed for learning first and performance second.

## Structure

- `include/`: public headers for the library.
- `src/`: implementations.
- `tests/`: assert-based unit tests on small synthetic datasets.
- `docs/`: student-oriented algorithm notes.
- `examples/`: compact usage examples.

## Included Modules

- Classical ML models for regression, classification, ensembles, and clustering
- Educational neural-network and transformer-style components
- A data plumbing layer with CSV parsing, scaling, encoding, and dataset splitting

## Multiclass Classification

The library now supports multiclass classification across the main classification stack:

- `ml::LogisticRegression` uses sigmoid for binary targets and softmax for `K > 2` classes.
- `ml::LinearSVM` uses binary margin scoring for two classes and one-vs-rest heads for multiclass problems.
- `ml::MLPClassifier`, `ml::SimpleCNN`, `ml::SimpleRNN`, `ml::SimpleLSTM`, and `ml::TransformerClassifier` now expose multiclass output heads.

User-facing behavior is consistent across these classifiers:

- `fit(x, y)` expects `y` as an `N x 1` matrix of integer class ids.
- `predict(x)` returns an `N x 1` matrix of predicted class ids.
- `predict_proba(x)` returns:
  - `N x 1` probabilities for binary models
  - `N x K` probabilities for multiclass models, ordered by `model.classes()`
- `classes()` returns the original class labels used during training and `num_classes()` reports `K`.

This makes the library usable on datasets such as Iris without collapsing labels into a binary task.

## Metrics

`ml/metrics/Metrics.hpp` now includes classification, ranking, and regression helpers that work with the library's current prediction APIs:

- hard-label classification metrics: `accuracy_score`, `confusion_matrix`, `classification_report`, `precision_score`, `recall_score`, `f1_score`
- score-based classification metrics: `roc_auc_score`, `roc_auc_report`
- regression metrics: `mean_squared_error`, `mean_absolute_error`, `root_mean_squared_error`, `r2_score`, `regression_report`

Example:

```cpp
#include "ml/metrics/Metrics.hpp"

ml::Matrix predictions = model.predict(x_test);
ml::ClassificationReport report = ml::classification_report(predictions, y_test);

double macro_f1 = report.macro_f1;
double micro_precision = report.micro_precision;

double auc = ml::roc_auc_score(model.predict_proba(x_test), y_test, model.classes());
ml::RegressionReport regression = ml::regression_report(model.predict(x_test), y_test);
```

For binary classifiers, `roc_auc_score` expects the library's existing `N x 1` score/probability output. For multiclass classifiers, it expects `N x K` scores ordered by `model.classes()`.

## Build

If `cmake` is available:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

If you only have a compiler:

```bash
c++ -std=c++17 -Iinclude -Itests src/linear/*.cpp src/probabilistic/*.cpp src/optimization/*.cpp src/unsupervised/*.cpp src/deep/*.cpp src/modern/*.cpp tests/test_linear_regression.cpp -o test_linear_regression
./test_linear_regression
```

## Design Notes

- `ml::Model` provides a shared interface with `fit`, `predict`, `save`, and `load`.
- `ml::Matrix` is a small custom dense matrix class used across all models.
- Deep-learning and transformer components are intentionally simplified so the code stays readable for students.
- The sequence, CNN, and transformer models still train their final classification heads only; they are educational encoders with trainable readouts rather than full end-to-end deep-learning implementations.

## Data Pipeline

See [docs/data_preprocessing/README.md](/Users/dksingh/src/ml/docs/data_preprocessing/README.md) for the CSV-to-model workflow and [data_pipeline.cpp](/Users/dksingh/src/ml/examples/data_pipeline.cpp) for a compact end-to-end example.

The Iris example now runs as a true 3-class classification pipeline:

1. Read the CSV fixture.
2. Label-encode species into integer ids.
3. Scale numeric features.
4. Train multiclass logistic regression.
5. Decode predictions back to species labels.

## Tests

The project uses small assert-based tests that are fast enough to run locally on every change:

```bash
ctest --test-dir build --output-on-failure
```

There is dedicated multiclass coverage in [test_multiclass_models.cpp](/Users/dksingh/src/ml/tests/test_multiclass_models.cpp), including:

- softmax logistic regression on a 3-class dataset
- one-vs-rest linear SVM on a 3-class dataset
- multiclass probability-shape checks for MLP, CNN, RNN, LSTM, and transformer classifiers
- save/load round trips for the upgraded classifiers

## Fetch, Train, Plot

For a full educational workflow:

1. Fetch datasets:
   `python3 scripts/fetch_datasets.py`
2. Compile and run your C++ experiment so it writes metrics to `output/` using [MetricsLogger.hpp](/Users/dksingh/src/ml/include/MetricsLogger.hpp).
3. Generate plots:
   `python3 tools/plot_results.py --metrics output/metrics.csv --confusion output/confusion_pairs.csv --grid output/prediction_grid.csv`

The fetcher writes clean CSVs to `data/`, the C++ code writes run artifacts to `output/`, and the plotting tool saves PNGs to `output/plots/`.

## Publish To GitHub

This repository is ready to push to GitHub with standard git commands:

```bash
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

If you prefer SSH:

```bash
git remote add origin git@github.com:<your-user>/<your-repo>.git
git push -u origin main
```

This repo also includes a GitHub Actions workflow at `.github/workflows/ci.yml` so pushes and pull requests automatically build the project and run the test suite.
