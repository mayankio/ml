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

## Data Pipeline

See [docs/data_preprocessing/README.md](/Users/dksingh/src/ml/docs/data_preprocessing/README.md) for the CSV-to-model workflow and [data_pipeline.cpp](/Users/dksingh/src/ml/examples/data_pipeline.cpp) for a compact end-to-end example.

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
