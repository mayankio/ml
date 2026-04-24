# High-Level Design

## 1. Purpose and System Scope

This repository is an educational machine learning library written primarily in C++17 with a strict bias toward minimal dependencies and readable implementations. The system is designed to teach:

- core machine learning algorithms,
- basic numerical linear algebra,
- preprocessing and data ingestion,
- model evaluation and experiment logging,
- dataset acquisition and visualization workflows,
- test-driven exploration of small synthetic problems.

The codebase intentionally favors clarity over performance, completeness, and industrial robustness.

At the highest level, the repository contains six cooperating subsystems:

1. Core numerical and model abstractions
2. Data ingestion and preprocessing
3. Machine learning algorithm implementations
4. Experiment logging and visualization pipeline
5. Example programs and documentation
6. Build, test, and CI automation

## 2. Architectural Style

The architecture is a layered library-plus-tooling design.

- The bottom layer is a custom dense matrix type and math helpers.
- Above that is a narrow `Model` interface that standardizes `fit`, `predict`, `save`, and `load`.
- Data preprocessing utilities adapt external CSV/tabular data into `Matrix` objects and encoded labels.
- Algorithm classes consume `Matrix` inputs and produce `Matrix` outputs.
- Logging/export components write training artifacts into `output/` for later plotting.
- Python scripts operate outside the C++ runtime as companion tools for dataset acquisition and visualization.

This is not a plugin system, service-oriented system, or template-heavy framework. It is a small, explicit, file-organized educational codebase.

## 3. Repository Layout and Responsibility Map

### 3.1 Top-Level Build and Project Files

- [CMakeLists.txt](/Users/dksingh/src/ml/CMakeLists.txt)
  Defines the library target, test executables, C++ standard settings, and CTest integration.
- [README.md](/Users/dksingh/src/ml/README.md)
  Primary entry point for users. Describes build, data pipeline, plotting pipeline, and GitHub publishing.
- [.github/workflows/ci.yml](/Users/dksingh/src/ml/.github/workflows/ci.yml)
  GitHub Actions CI workflow that configures, builds, and runs tests on pushes and pull requests.

### 3.2 Public Headers

- [/Users/dksingh/src/ml/include/ml/core](/Users/dksingh/src/ml/include/ml/core)
  Core abstractions used by nearly every other component.
- [/Users/dksingh/src/ml/include/ml/data](/Users/dksingh/src/ml/include/ml/data)
  DataFrame, CSV, preprocessing, encoding, and dataset splitting interfaces.
- [/Users/dksingh/src/ml/include/ml/linear](/Users/dksingh/src/ml/include/ml/linear)
  Linear and distance-based supervised models.
- [/Users/dksingh/src/ml/include/ml/probabilistic](/Users/dksingh/src/ml/include/ml/probabilistic)
  Probabilistic and tree-based learners.
- [/Users/dksingh/src/ml/include/ml/optimization](/Users/dksingh/src/ml/include/ml/optimization)
  Margin-based and boosting models.
- [/Users/dksingh/src/ml/include/ml/unsupervised](/Users/dksingh/src/ml/include/ml/unsupervised)
  Clustering and dimensionality reduction.
- [/Users/dksingh/src/ml/include/ml/deep](/Users/dksingh/src/ml/include/ml/deep)
  Educational neural-network building blocks and sequence/image models.
- [/Users/dksingh/src/ml/include/ml/modern](/Users/dksingh/src/ml/include/ml/modern)
  Self-attention and transformer-style classifier.
- [MetricsLogger.hpp](/Users/dksingh/src/ml/include/MetricsLogger.hpp)
  Standalone experiment artifact logger for metrics, confusion pairs, and prediction grids.

### 3.3 C++ Implementations

- [/Users/dksingh/src/ml/src](/Users/dksingh/src/ml/src)
  Concrete implementations of all models and data-processing classes.

### 3.4 Python Companion Tooling

- [/Users/dksingh/src/ml/scripts](/Users/dksingh/src/ml/scripts)
  Dataset acquisition scripts.
- [/Users/dksingh/src/ml/tools](/Users/dksingh/src/ml/tools)
  Plot generation tools.

### 3.5 Validation and Fixtures

- [/Users/dksingh/src/ml/tests](/Users/dksingh/src/ml/tests)
  Assert-based executable unit tests for models and utility code.
- [/Users/dksingh/src/ml/tests/data](/Users/dksingh/src/ml/tests/data)
  CSV fixtures used to validate parsing and preprocessing.

### 3.6 Student Documentation

- [/Users/dksingh/src/ml/docs](/Users/dksingh/src/ml/docs)
  Per-algorithm educational notes, intuition, complexity, and usage guidance.

## 4. Core Design Principles

### 4.1 Minimal Dependencies

The central C++ library relies on the standard library and a custom dense matrix class. There is no Eigen, BLAS, OpenCV, protobuf, or external ML runtime.

Python is used only for:

- downloading public datasets,
- plotting experiment outputs.

### 4.2 Uniform Model Interface

Most learners inherit from [Model.hpp](/Users/dksingh/src/ml/include/ml/core/Model.hpp), which provides a common lifecycle:

- `fit(features, targets)`
- `predict(features)`
- `save(path)`
- `load(path)`

This gives the codebase a consistent mental model even though the algorithms are very different internally.

### 4.3 Matrix-Centric Data Flow

The primary data representation across the library is `ml::Matrix`.

- Preprocessing modules eventually emit `Matrix`
- Models consume `Matrix`
- Metrics compare `Matrix`
- Logging utilities can export matrix-derived artifacts

This keeps the entire system coherent and avoids per-module tensor abstractions.

### 4.4 Educational Explicitness

Most algorithms are written with:

- explicit loops,
- direct parameter updates,
- plain recursion for trees,
- plain vectors for hidden states,
- small serialization formats.

This makes the internal computation graph visible to students.

## 5. Subsystem Design

## 5.1 Core Numerical Subsystem

The numerical subsystem is implemented primarily in [Matrix.hpp](/Users/dksingh/src/ml/include/ml/core/Matrix.hpp).

It provides:

- a dense row-major matrix container,
- basic element access,
- row and column extraction,
- transpose,
- elementwise transform application,
- matrix arithmetic,
- matrix multiplication,
- distance and dot-product helpers,
- activation helpers such as sigmoid, ReLU, and softmax,
- very simple serialization helpers for matrices.

Architecturally, this file acts as both:

- a data container definition,
- and a collection of low-level numerical utility functions.

This is intentionally compact, but it means many responsibilities are concentrated in a single header.

## 5.2 Model Abstraction Subsystem

[Model.hpp](/Users/dksingh/src/ml/include/ml/core/Model.hpp) defines the polymorphic contract for trainable predictors. This contract standardizes usage across:

- linear regression,
- logistic regression,
- KNN,
- naive Bayes,
- tree/forest models,
- SVM,
- gradient boosting,
- KMeans and PCA,
- MLP/CNN/RNN/LSTM,
- self-attention and transformer classifier.

Not every algorithm uses the full abstraction equally well. For example:

- `KMeans` ignores targets during `fit`
- `PCA::predict` is semantically a transform
- `SelfAttention::fit` is a no-op

The shared interface is pedagogically convenient, but it is broader than some algorithms naturally require.

## 5.3 Data Plumbing Subsystem

This subsystem spans:

- [CSVReader.hpp](/Users/dksingh/src/ml/include/ml/data/CSVReader.hpp) / [CSVReader.cpp](/Users/dksingh/src/ml/src/data/CSVReader.cpp)
- [DataFrame.hpp](/Users/dksingh/src/ml/include/ml/data/DataFrame.hpp) / [DataFrame.cpp](/Users/dksingh/src/ml/src/data/DataFrame.cpp)
- [Transformers.hpp](/Users/dksingh/src/ml/include/ml/data/Transformers.hpp) / [Transformers.cpp](/Users/dksingh/src/ml/src/data/Transformers.cpp)
- [Split.hpp](/Users/dksingh/src/ml/include/ml/data/Split.hpp) / [Split.cpp](/Users/dksingh/src/ml/src/data/Split.cpp)

It is responsible for:

- parsing delimited text into a structured table,
- mapping named columns to positions,
- converting numeric columns into `Matrix`,
- normalizing numeric features,
- encoding categorical/string labels,
- splitting datasets into train/test partitions.

The conceptual flow is:

`CSV file -> DataFrame -> column selection/encoding -> Matrix -> model`

This subsystem is the bridge between real-world tabular data and the matrix-oriented model layer.

## 5.4 Classical ML Subsystem

The classical ML layer includes:

- Linear Regression
- Logistic Regression
- KNN
- Gaussian Naive Bayes
- Decision Tree Classifier / Regressor
- Random Forest Classifier
- Linear SVM
- Gradient Boosting Regressor
- KMeans
- PCA

These models are independent concrete classes, but they share patterns:

- constructor-configured hyperparameters,
- `fit` mutates internal learned state,
- `predict` returns a new `Matrix`,
- simple text-based serialization,
- explicit iterative or recursive learning logic.

There is intentionally no optimizer abstraction, no trainer abstraction, and no model factory. Each algorithm stands alone.

## 5.5 Neural and Sequence Subsystem

The neural subsystem includes:

- [MLP.hpp](/Users/dksingh/src/ml/include/ml/deep/MLP.hpp)
- [CNN.hpp](/Users/dksingh/src/ml/include/ml/deep/CNN.hpp)
- [RNN.hpp](/Users/dksingh/src/ml/include/ml/deep/RNN.hpp)

These classes are designed to demonstrate:

- dense layers and backpropagation,
- convolution plus pooled responses,
- recurrent state propagation,
- gated recurrent memory with LSTM-like equations.

They are not general-purpose deep-learning layers. They are small, fixed-topology educational models. Training is simplified heavily:

- only portions of the parameter sets are updated in some models,
- there is no full backpropagation through time,
- no batching abstraction exists,
- no optimizer state exists,
- no tensor dimension generalization exists.

## 5.6 Attention and Transformer Subsystem

The modern architecture subsystem includes:

- [SelfAttention.hpp](/Users/dksingh/src/ml/include/ml/modern/SelfAttention.hpp)
- [Transformer.hpp](/Users/dksingh/src/ml/include/ml/modern/Transformer.hpp)

This layer demonstrates:

- query/key/value projection,
- scaled dot-product attention,
- attention-weighted value mixing,
- feed-forward encoding,
- pooling to a single sequence embedding,
- binary classification from pooled embeddings.

Its role in the architecture is mostly pedagogical. It introduces the shape and idea of transformer blocks without attempting a production-grade implementation.

## 5.7 Experiment Logging and Visualization Subsystem

The experiment artifact path is:

`C++ training loop -> MetricsLogger CSV files -> Python plotting tool -> PNG figures`

Components:

- [MetricsLogger.hpp](/Users/dksingh/src/ml/include/MetricsLogger.hpp)
- [plot_results.py](/Users/dksingh/src/ml/tools/plot_results.py)

Supported artifact types:

- epoch-level learning curves,
- confusion-pair CSV for building confusion matrices,
- prediction-grid CSV for drawing decision boundaries.

The logging format is deliberately simple and filesystem-based so students can inspect outputs directly.

## 5.8 Dataset Acquisition Subsystem

[fetch_datasets.py](/Users/dksingh/src/ml/scripts/fetch_datasets.py) downloads and normalizes public teaching datasets into `data/`.

Its architectural purpose is to keep:

- data sourcing,
- data cleaning into header-based CSV,
- and downstream C++ usage

separate from the main C++ build. This avoids embedding network access or non-standard parsing logic into the C++ library.

## 5.9 Example and Documentation Subsystem

Examples:

- [basic_usage.cpp](/Users/dksingh/src/ml/examples/basic_usage.cpp)
- [data_pipeline.cpp](/Users/dksingh/src/ml/examples/data_pipeline.cpp)

Documentation:

- per-algorithm READMEs in `/docs`
- preprocessing pipeline README in `/docs/data_preprocessing`

This subsystem is important architecturally because the repository is explicitly teaching-oriented. Documentation is not an afterthought; it is part of the product surface.

## 6. Data Flow Views

## 6.1 Training Flow for Supervised Models

1. Source data is loaded from CSV into `DataFrame`
2. Chosen numeric columns are converted to `Matrix`
3. Labels are encoded into numeric form if needed
4. Optional scaling/encoding is applied
5. `train_test_split` partitions the dataset
6. Model `fit` is called
7. `predict` generates outputs
8. `accuracy_score` or `r2_score` evaluates results
9. `MetricsLogger` exports training artifacts
10. Python plotting tool visualizes results

## 6.2 Training Flow for Unsupervised Models

1. CSV -> `DataFrame`
2. `DataFrame` -> numeric `Matrix`
3. Optional scaling
4. `fit(features, empty_targets)`
5. `predict` or `transform`
6. Optional logging/visualization outside the core unsupervised implementation

## 6.3 Decision-Boundary Visualization Flow

1. A C++ experiment computes a regular 2D prediction grid
2. The experiment exports grid points and predicted labels using `MetricsLogger::export_prediction_grid_to_csv`
3. The same export can optionally include original sample points and labels
4. `tools/plot_results.py` reconstructs the mesh and writes a contour-based PNG

## 7. Build and Execution Design

The build is intentionally simple.

- `edu_ml` is a single library target built from all `src/*.cpp`
- each file in `/tests` becomes an executable
- each test executable links against `edu_ml`
- `ctest` runs these executables

This design is easy to understand and maintain, but it couples every test to the full library and does not optimize compilation time.

## 8. Testing Strategy

Testing is assert-based and executable-driven.

Coverage areas include:

- core regression/classification behavior on tiny synthetic datasets,
- clustering and PCA sanity checks,
- parser row-count and quoted-field handling,
- scaling math,
- label/one-hot encoding,
- metrics logger CSV format.

The tests emphasize:

- API correctness,
- numerical plausibility,
- artifact shape and format,
- educational reproducibility.

The tests do not aim for:

- exhaustive edge-case coverage,
- stress/performance testing,
- fuzzing,
- floating-point stability analysis,
- serialization round-trip depth for recursive models.

## 9. Persistence Strategy

Most models expose `save` and `load`, implemented as plain text serialization.

This is simple and readable, but not uniform in completeness:

- linear models serialize learned parameters directly,
- KNN serializes training data,
- PCA and KMeans serialize learned components/centroids,
- neural models serialize selected learned matrices,
- tree-based ensemble save/load currently persists mostly hyperparameters rather than the full learned tree structure.

This is an important architectural boundary: the API suggests complete persistence support, but the implementation quality varies by model family.

## 10. Key Strengths of the Current Design

- Consistent public model interface
- Very low conceptual overhead
- Strong alignment with educational use cases
- Self-contained C++ core with standard library focus
- Integrated preprocessing and data-to-plot workflow
- Easy-to-read source due to explicit loops and small files
- Clear directory-level separation by algorithm family

## 11. Key Constraints and Tradeoffs

### 11.1 Performance Tradeoffs

- frequent row extraction copies data,
- recursive tree building repeatedly copies subsets,
- KNN stores full training matrices and computes brute-force distances,
- no vectorization or optimized BLAS usage,
- no batching or parallel training.

### 11.2 API Tradeoffs

- `Model` is slightly too generic for some algorithms,
- some models reset parameters each `fit`,
- persistence semantics are inconsistent across implementations,
- matrix operations and utility functions live together in one header.

### 11.3 Numerical and ML Tradeoffs

- some deep-learning models use partial/simplified training rules,
- no regularization abstractions,
- no probability calibration,
- no multiclass logistic regression abstraction,
- no robust missing-value handling beyond simple normalization and `0.0` conversion in `DataFrame::numeric_matrix`.

## 12. Intended Audience and Design Fit

This design is well matched to:

- students learning ML algorithms from scratch,
- developers exploring how models can be implemented with only basic C++,
- instructors wanting small, inspectable examples,
- experimentation on tiny tabular and toy sequence/image datasets.

It is not designed for:

- large-scale training,
- production deployment,
- high-dimensional optimized numerical workloads,
- GPU execution,
- feature-complete model lifecycle management.

## 13. File Coverage Summary

The design described above is implemented across all major repository files:

- core: `Matrix`, `Model`, metrics helpers
- data: `CSVReader`, `DataFrame`, transformers, split utilities
- models: linear, probabilistic, optimization, unsupervised, deep, modern
- tooling: `MetricsLogger`, dataset fetcher, plotting tool
- user entry points: examples and README
- validation: tests and fixtures
- automation: CMake and GitHub Actions

In other words, the repository is architected as a complete educational pipeline:

`data acquisition -> data plumbing -> model training -> evaluation -> artifact export -> visualization`

That end-to-end shape is the most important high-level design characteristic of the system.
