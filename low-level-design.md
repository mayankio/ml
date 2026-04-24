# Low-Level Design

## 1. Overview

This document describes the repository at file, class, method, and data-format level. It focuses on concrete implementation details rather than conceptual grouping.

## 2. Build and Runtime Files

## 2.1 [CMakeLists.txt](/Users/dksingh/src/ml/CMakeLists.txt)

Purpose:

- defines the `edu_ml` library from every `src/*.cpp`,
- enables testing through `enable_testing()`,
- auto-discovers all `tests/*.cpp`,
- creates one test executable per test source,
- registers each executable with CTest.

Low-level implications:

- tests are standalone programs, not a shared framework-based suite,
- any new `.cpp` file under `tests/` automatically becomes a test target,
- include path is global via `include_directories(include)`.

## 2.2 [.github/workflows/ci.yml](/Users/dksingh/src/ml/.github/workflows/ci.yml)

Pipeline steps:

1. checkout
2. `cmake -S . -B build`
3. `cmake --build build --parallel`
4. `ctest --test-dir build --output-on-failure`

This CI assumes:

- a Linux environment,
- working CMake on the runner,
- no extra dependencies for the core C++ build.

## 3. Core Layer

## 3.1 [Matrix.hpp](/Users/dksingh/src/ml/include/ml/core/Matrix.hpp)

### 3.1.1 Class Layout

`ml::Matrix` stores:

- `rows_`
- `cols_`
- `std::vector<double> data_`

The storage is row-major because indexing is implemented as:

`row * cols_ + col`

### 3.1.2 Constructors

- default constructor builds an empty matrix,
- `(rows, cols, value)` fills a dense matrix with a constant,
- initializer-list constructor supports literal construction like `{{1,2},{3,4}}`.

The initializer-list constructor validates rectangularity and throws on ragged inputs.

### 3.1.3 Accessors and Views

- `rows()`, `cols()`, `empty()`
- `operator()(row, col)` with bounds checking through `std::vector::at`
- `data()` const/non-const access
- `row(index)` returns a new `1 x cols` matrix
- `col(index)` returns a new `rows x 1` matrix
- `row_vector(index)` returns a copied `std::vector<double>`

Important implementation note:

- row/column extraction copies data; there are no views or spans.

### 3.1.4 Matrix Operations

Defined inline in the header:

- addition and subtraction with shape checks
- scalar multiply and divide
- Hadamard product
- matrix multiplication
- row-vector bias addition
- transpose
- elementwise `apply(fn)`

### 3.1.5 Statistical and Activation Helpers

Also in the same header:

- `column_means`
- `column_stds`
- `dot`
- `euclidean_distance`
- `mean_squared_error`
- `sigmoid`
- `relu`
- `relu_derivative`
- `softmax`
- `argmax`

### 3.1.6 Serialization Helpers

Simple text helpers:

- `save_matrix(std::ofstream&, const Matrix&)`
- `load_matrix(std::ifstream&)`

Format:

1. first line: `rows cols`
2. second line: flat sequence of values

### 3.1.7 Known Low-Level Constraints

- no broadcasting beyond explicit `add_row_vector`
- no slicing or views
- no sparse support
- no matrix inversion/eigendecomposition utilities beyond what PCA builds manually
- `apply` uses `std::function`, which is convenient but not the fastest choice

## 3.2 [Model.hpp](/Users/dksingh/src/ml/include/ml/core/Model.hpp)

Defines abstract base class `ml::Model`.

Methods:

- `virtual void fit(const Matrix&, const Matrix&) = 0`
- `virtual Matrix predict(const Matrix&) const = 0`
- `virtual void save(const std::string&) const = 0`
- `virtual void load(const std::string&) = 0`

This base class is used uniformly by nearly every model class, including unsupervised and non-trainable-in-practice classes like `SelfAttention`.

## 3.3 [Metrics.hpp](/Users/dksingh/src/ml/include/ml/metrics/Metrics.hpp)

Inline functions:

- `accuracy_score`
- `r2_score`

Assumptions:

- classification targets are single-column matrices,
- predictions are already discrete or close enough to discrete to round safely,
- regression is single-output.

## 4. Metrics Export Layer

## 4.1 [MetricsLogger.hpp](/Users/dksingh/src/ml/include/MetricsLogger.hpp)

Header-only standalone utility, not namespaced under `ml`.

### 4.1.1 Stored State

- `loss_per_epoch_`
- `training_accuracy_`
- `validation_accuracy_`

### 4.1.2 Mutation APIs

- `set_loss_per_epoch`
- `set_training_accuracy`
- `set_validation_accuracy`
- `log_epoch`

`log_epoch` appends one full metric tuple.

### 4.1.3 CSV Export APIs

`export_to_csv(filename)`

- writes to `output/<filename>`
- creates `output/` if missing
- validates equal vector lengths
- output schema:
  `epoch,loss,training_accuracy,validation_accuracy`

`export_confusion_pairs_to_csv(filename, truth, predicted)`

- schema:
  `true_label,predicted_label`

`export_prediction_grid_to_csv(filename, points, predictions, samples, sample_labels)`

- requires 2D points
- schema:
  `kind,x1,x2,label`
- `kind` is either `grid` or `sample`

### 4.1.4 Low-Level Constraints

- uses `std::filesystem`, so it is not part of the strictest “tiny headers only” subset
- no append mode
- no timestamped run directories
- no JSON or binary output

## 5. Data Plumbing Layer

## 5.1 [DataFrame.hpp](/Users/dksingh/src/ml/include/ml/data/DataFrame.hpp) and [DataFrame.cpp](/Users/dksingh/src/ml/src/data/DataFrame.cpp)

### 5.1.1 Internal Representation

State:

- `columns_`: ordered column names
- `column_to_index_`: name-to-index map
- `rows_`: `std::vector<std::vector<std::string>>`

This is row-oriented storage.

### 5.1.2 Behavior

- constructor builds the index map
- `add_row` checks row width
- `column(name)` copies an entire column into a new vector
- `column_index(name)` validates name existence
- `numeric_matrix(selected_columns)` converts chosen columns to `Matrix`

### 5.1.3 Numeric Conversion Rules

If a cell is:

- `"NaN"`
- or empty

then it becomes `0.0` in the numeric matrix.

This is a simple imputation policy embedded into conversion rather than a separate strategy object.

## 5.2 [CSVReader.hpp](/Users/dksingh/src/ml/include/ml/data/CSVReader.hpp) and [CSVReader.cpp](/Users/dksingh/src/ml/src/data/CSVReader.cpp)

### 5.2.1 State

- `delimiter_`

### 5.2.2 `read(path, has_header)`

Detailed flow:

1. open file
2. iterate line by line
3. trim trailing `\r`
4. skip blank lines
5. parse line via `parse_line`
6. either treat first row as header or synthesize `column_0`, `column_1`, ...
7. verify row width consistency
8. normalize missing values
9. add row to `DataFrame`

### 5.2.3 `parse_line`

Implements a manual state machine:

- tracks `in_quotes`
- supports escaped quotes via doubled `""`
- splits only on delimiters seen outside quoted segments

### 5.2.4 Missing-Value Rules

`normalize_missing` maps:

- `""`
- `"NA"`
- `"N/A"`
- `"null"`

to `"NaN"`.

## 5.3 [Transformers.hpp](/Users/dksingh/src/ml/include/ml/data/Transformers.hpp) and [Transformers.cpp](/Users/dksingh/src/ml/src/data/Transformers.cpp)

### 5.3.1 `MatrixTransformer`

Abstract preprocessing interface:

- `fit`
- `transform`
- `fit_transform`

This interface is only for matrix-to-matrix transformers.

### 5.3.2 `StandardScaler`

Stored state:

- `means_`
- `stds_`

Fit:

- uses `Matrix::column_means`
- uses `Matrix::column_stds(0.0)`
- replaces any zero std with `1.0`

Transform:

- computes `(x - mean) / std` elementwise
- throws if called before fit

### 5.3.3 `MinMaxScaler`

Stored state:

- `feature_min_`
- `feature_max_`
- `mins_`
- `maxs_`

Fit:

- scans each column to find minima and maxima

Transform:

- normalizes each value by column range
- maps to requested output range
- if range is zero, outputs `feature_min_`

### 5.3.4 `OneHotEncoder`

Stored state:

- `categories_`
- `index_`

Fit:

- preserves order of first appearance

Transform:

- returns `rows = values.size()`, `cols = categories_.size()`
- sets exactly one position to `1.0` per row
- throws on unknown category

### 5.3.5 `LabelEncoder`

Stored state:

- `classes_`
- `to_id_`

Fit:

- preserves order of first appearance

Transform:

- maps strings to int IDs

Inverse transform:

- maps IDs back to strings
- validates ID range

## 5.4 [Split.hpp](/Users/dksingh/src/ml/include/ml/data/Split.hpp) and [Split.cpp](/Users/dksingh/src/ml/src/data/Split.cpp)

### 5.4.1 `MatrixSplit`

Plain aggregate:

- `x_train`
- `x_test`
- `y_train`
- `y_test`

### 5.4.2 Shuffle and Split Logic

`train_test_split`:

- validates matching row counts
- validates `test_ratio in (0,1)`
- builds an index vector
- shuffles it using a simple linear-congruential generator helper `next_random`
- allocates result matrices
- copies rows into train and test partitions

Important detail:

- this does not use `<random>`-based `shuffle`; it uses a custom deterministic integer generator.

## 6. Model Implementations

## 6.1 Linear Models

### 6.1.1 [LinearRegression.hpp](/Users/dksingh/src/ml/include/ml/linear/LinearRegression.hpp) and [LinearRegression.cpp](/Users/dksingh/src/ml/src/linear/LinearRegression.cpp)

State:

- `learning_rate_`
- `epochs_`
- `weights_`
- `bias_`

Fit:

- reinitializes weights and bias to zero every call
- performs batch gradient descent over MSE

Predict:

- computes `dot(weights_, row) + bias_`

Serialization:

- saves hyperparameters, bias, weight count, then weights

### 6.1.2 [LogisticRegression.hpp](/Users/dksingh/src/ml/include/ml/linear/LogisticRegression.hpp) and [LogisticRegression.cpp](/Users/dksingh/src/ml/src/linear/LogisticRegression.cpp)

Very similar shape to linear regression.

Differences:

- output uses sigmoid
- `predict_proba` exposes probabilities
- `predict` thresholds at `0.5`
- loss is not explicitly computed or stored

### 6.1.3 [KNN.hpp](/Users/dksingh/src/ml/include/ml/linear/KNN.hpp) and [KNN.cpp](/Users/dksingh/src/ml/src/linear/KNN.cpp)

State:

- `k_`
- `train_features_`
- `train_targets_`

Fit:

- stores training matrices directly

Predict:

- computes all pairwise distances to training data
- uses `std::nth_element` to partially order neighbors
- votes via `std::map<int, int>`

Serialization:

- saves `k` and both stored training matrices

## 6.2 Probabilistic and Tree-Based Models

### 6.2.1 [NaiveBayes.hpp](/Users/dksingh/src/ml/include/ml/probabilistic/NaiveBayes.hpp) and [NaiveBayes.cpp](/Users/dksingh/src/ml/src/probabilistic/NaiveBayes.cpp)

State:

- `classes_`
- `means_`
- `variances_`
- `priors_`

Fit:

- builds class set from targets
- groups row indices per class
- computes Gaussian mean and variance per feature per class

Predict:

- evaluates log posterior per class
- chooses argmax

Serialization:

- persists priors, feature count, means, variances

### 6.2.2 [DecisionTree.hpp](/Users/dksingh/src/ml/include/ml/probabilistic/DecisionTree.hpp) and [DecisionTree.cpp](/Users/dksingh/src/ml/src/probabilistic/DecisionTree.cpp)

Contains two classes:

- `DecisionTreeClassifier`
- `DecisionTreeRegressor`

Both use private recursive `Node` structs containing:

- leaf flag
- prediction value
- split feature
- threshold
- left/right child pointers

Classifier specifics:

- impurity measure: Gini
- leaf prediction: majority class
- optional random feature subsampling via `max_features_`

Regressor specifics:

- impurity measure: variance
- leaf prediction: mean target

Shared low-level pattern:

- evaluate every feature-threshold candidate from observed values
- split rows into left/right index lists
- copy subsets using helper `gather_rows`
- recurse until stopping conditions

Persistence note:

- only hyperparameters are saved, not the tree structure

### 6.2.3 [RandomForest.hpp](/Users/dksingh/src/ml/include/ml/probabilistic/RandomForest.hpp) and [RandomForest.cpp](/Users/dksingh/src/ml/src/probabilistic/RandomForest.cpp)

State:

- forest hyperparameters
- `std::vector<DecisionTreeClassifier> trees_`

Fit:

- computes `max_features = sqrt(num_features)`
- draws bootstrap samples with replacement
- trains one decision tree per bootstrap sample

Predict:

- predicts each sample with each tree
- majority votes with `std::map<int,int>`

Persistence note:

- only forest hyperparameters are saved; trained trees are not serialized

## 6.3 Optimization-Based Models

### 6.3.1 [SVM.hpp](/Users/dksingh/src/ml/include/ml/optimization/SVM.hpp) and [SVM.cpp](/Users/dksingh/src/ml/src/optimization/SVM.cpp)

State:

- `learning_rate_`
- `epochs_`
- `c_`
- `hard_margin_`
- `weights_`
- `bias_`

Fit:

- uses label remapping to `{-1, +1}`
- hard margin is approximated by using a large `C` value
- updates weights according to whether the margin constraint is satisfied

Predict:

- `decision_function` returns raw score
- `predict` thresholds score at zero

### 6.3.2 [GradientBoosting.hpp](/Users/dksingh/src/ml/include/ml/optimization/GradientBoosting.hpp) and [GradientBoosting.cpp](/Users/dksingh/src/ml/src/optimization/GradientBoosting.cpp)

State:

- `n_estimators_`
- `learning_rate_`
- `max_depth_`
- `init_prediction_`
- `trees_` of `DecisionTreeRegressor`

Fit:

- initializes prediction to target mean
- computes residuals
- fits shallow regression tree to residuals
- adds scaled tree outputs to current prediction

Predict:

- starts at `init_prediction_`
- adds each tree contribution multiplied by learning rate

Persistence note:

- only top-level hyperparameters are saved; tree ensemble itself is not serialized

## 6.4 Unsupervised Models

### 6.4.1 [KMeans.hpp](/Users/dksingh/src/ml/include/ml/unsupervised/KMeans.hpp) and [KMeans.cpp](/Users/dksingh/src/ml/src/unsupervised/KMeans.cpp)

State:

- `n_clusters_`
- `max_iters_`
- `seed_`
- `centroids_`

Fit:

- random initialization from existing rows
- repeated assignment and centroid recomputation
- skips empty clusters instead of reseeding them

Predict:

- nearest-centroid assignment

### 6.4.2 [PCA.hpp](/Users/dksingh/src/ml/include/ml/unsupervised/PCA.hpp) and [PCA.cpp](/Users/dksingh/src/ml/src/unsupervised/PCA.cpp)

State:

- `n_components_`
- `power_iterations_`
- `means_`
- `components_`

Fit:

- centers data
- computes covariance
- uses power iteration per component
- uses deflation to estimate successive components

Predict:

- delegated to `transform`

Serialization:

- saves component count, iteration count, means, and projection matrix

## 6.5 Neural Models

### 6.5.1 [MLP.hpp](/Users/dksingh/src/ml/include/ml/deep/MLP.hpp) and [MLP.cpp](/Users/dksingh/src/ml/src/deep/MLP.cpp)

Architecture:

- one hidden layer
- sigmoid hidden activation
- sigmoid output

State:

- dimensions
- learning rate and epoch count
- `w1_`, `w2_`
- `b1_`, `b2_`

Fit:

- sample-wise forward pass
- computes scalar output error
- computes hidden gradients from output error
- updates both layers

### 6.5.2 [CNN.hpp](/Users/dksingh/src/ml/include/ml/deep/CNN.hpp) and [CNN.cpp](/Users/dksingh/src/ml/src/deep/CNN.cpp)

Architecture:

- fixed-size image input
- several trainable convolution filters
- ReLU activation
- global max over each filter response map
- dense binary output layer

State:

- geometry and hyperparameters
- `filters_`
- `dense_weights_`
- `dense_bias_`

Important detail:

- `fit` updates only the dense output layer; convolution filters are initialized randomly and then kept fixed

### 6.5.3 [RNN.hpp](/Users/dksingh/src/ml/include/ml/deep/RNN.hpp) and [RNN.cpp](/Users/dksingh/src/ml/src/deep/RNN.cpp)

Contains:

- `SimpleRNN`
- `SimpleLSTM`

#### `SimpleRNN`

State:

- sequence dimensions
- `wx_`, `wh_`, `wy_`
- hidden bias vector `bh_`
- output bias `by_`

Fit:

- unrolls sequence manually
- updates only the output weights and output bias
- recurrent/input weights stay fixed after initialization

#### `SimpleLSTM`

State:

- gate matrices `wf_`, `wi_`, `wo_`, `wc_`
- output matrix `wy_`
- gate bias vectors
- output bias

Fit:

- manually computes forget/input/output/candidate gates
- updates only the final output layer
- gate weights remain fixed after initialization

These are sequence encoders with a trainable readout more than full recurrent learners.

## 6.6 Attention and Transformer Models

### 6.6.1 [SelfAttention.hpp](/Users/dksingh/src/ml/include/ml/modern/SelfAttention.hpp) and [SelfAttention.cpp](/Users/dksingh/src/ml/src/modern/SelfAttention.cpp)

State:

- sequence length
- embedding dimension
- projection dimension
- `wq_`, `wk_`, `wv_`, `wo_`

Forward path:

1. project inputs into Q, K, V
2. compute attention scores row by row
3. apply softmax per query row
4. compute weighted sum of V
5. apply output projection

`fit` is intentionally empty.

### 6.6.2 [Transformer.hpp](/Users/dksingh/src/ml/include/ml/modern/Transformer.hpp) and [Transformer.cpp](/Users/dksingh/src/ml/src/modern/Transformer.cpp)

State:

- sequence and embedding hyperparameters
- internal `SelfAttention attention_`
- feed-forward matrices `ff1_`, `ff2_`
- feed-forward biases
- output matrix `out_`
- output bias

`encode`:

1. attention
2. feed-forward hidden with ReLU
3. feed-forward output
4. mean pooling over tokens

Fit:

- reshapes each flattened sample into sequence matrix
- encodes sequence
- updates only final output layer

Again, the architecture demonstrates the pattern but does not train the entire transformer stack.

## 7. Example Programs

## 7.1 [basic_usage.cpp](/Users/dksingh/src/ml/examples/basic_usage.cpp)

Shows three independent mini examples:

- linear regression
- logistic regression
- KMeans

This file is the shortest entry point for new users.

## 7.2 [data_pipeline.cpp](/Users/dksingh/src/ml/examples/data_pipeline.cpp)

Shows a small end-to-end preprocessing workflow:

1. read CSV
2. select numeric features
3. label-encode string species
4. build binary target matrix
5. standardize features
6. train/test split
7. fit logistic regression
8. print predictions

## 8. Python Pipeline Files

## 8.1 [fetch_datasets.py](/Users/dksingh/src/ml/scripts/fetch_datasets.py)

Functions:

- `download_text`
- `download_bytes`
- `write_csv`
- `fetch_iris`
- `fetch_breast_cancer`
- `fetch_tiny_mnist`
- `main`

Output datasets:

- `data/iris.csv`
- `data/breast_cancer_wisconsin_diagnostic.csv`
- `data/mnist_tiny.csv`

Implementation details:

- uses `urllib.request`
- uses `zipfile` for compressed MNIST source
- writes normalized CSVs with explicit headers

## 8.2 [plot_results.py](/Users/dksingh/src/ml/tools/plot_results.py)

Functions:

- `read_metrics`
- `plot_learning_curves`
- `read_confusion_pairs`
- `plot_confusion_matrix`
- `read_grid`
- `plot_decision_boundaries`
- `main`

Dependencies:

- `matplotlib`
- `numpy`
- `seaborn`

Expected input files:

- `output/metrics.csv`
- `output/confusion_pairs.csv`
- `output/prediction_grid.csv`

Generated files:

- `output/plots/learning_curves.png`
- `output/plots/confusion_matrix.png`
- `output/plots/decision_boundary.png`

## 9. Tests and Fixtures

## 9.1 [TestUtils.hpp](/Users/dksingh/src/ml/tests/TestUtils.hpp)

Provides `assert_close(actual, expected, tolerance)` for floating-point comparisons.

## 9.2 Parser and Preprocessing Fixtures

- [Iris.csv](/Users/dksingh/src/ml/tests/data/Iris.csv)
  150-row iris-style fixture used to validate parser row count and matrix conversion.
- [mixed.csv](/Users/dksingh/src/ml/tests/data/mixed.csv)
  Exercises quoted strings, embedded commas, and missing values.

## 9.3 Representative Tests

[test_data_module.cpp](/Users/dksingh/src/ml/tests/test_data_module.cpp)

- validates row counts
- validates quoted parsing
- validates missing-value normalization
- validates standard-scaler and min-max math
- validates encoders
- validates split sizes

[test_metrics_logger.cpp](/Users/dksingh/src/ml/tests/test_metrics_logger.cpp)

- validates metrics CSV header and first row
- validates confusion-pair export schema
- validates prediction-grid export header

[test_linear_regression.cpp](/Users/dksingh/src/ml/tests/test_linear_regression.cpp)

- validates fit quality using `r2_score`
- validates held-out point prediction

Other test files follow the same pattern:

- build a tiny synthetic dataset,
- fit a model,
- assert accuracy, shape, or output range.

Files covered this way include:

- `test_logistic_regression.cpp`
- `test_knn.cpp`
- `test_naive_bayes.cpp`
- `test_decision_tree.cpp`
- `test_random_forest.cpp`
- `test_svm.cpp`
- `test_gradient_boosting.cpp`
- `test_kmeans.cpp`
- `test_pca.cpp`
- `test_mlp.cpp`
- `test_cnn.cpp`
- `test_rnn.cpp`
- `test_lstm.cpp`
- `test_self_attention.cpp`
- `test_transformer.cpp`

## 10. Documentation Files

The `/docs` subtree contains student-facing algorithm notes. These files are not part of the compiled runtime, but they are part of the repository design and mirror the implemented model families:

- linear: regression, logistic regression, KNN
- probabilistic: naive Bayes, decision tree, random forest
- optimization: SVM, gradient boosting
- unsupervised: KMeans, PCA
- deep: MLP, CNN, RNN, LSTM
- modern: self-attention, transformer
- preprocessing: CSV/scaler/encoder pipeline

Their purpose is to connect source code with mathematical explanations and usage examples.

## 11. Serialization and Artifact Formats

### 11.1 Matrix Text Format

Used by multiple model save/load paths:

```text
<rows> <cols>
<flat_value_0> <flat_value_1> ...
```

### 11.2 Metrics CSV Format

```csv
epoch,loss,training_accuracy,validation_accuracy
1,0.9,0.6,0.55
```

### 11.3 Confusion Pair CSV Format

```csv
true_label,predicted_label
0,0
1,0
```

### 11.4 Prediction Grid CSV Format

```csv
kind,x1,x2,label
grid,0.0,0.0,0
sample,0.5,0.5,1
```

## 12. Known Implementation Boundaries

These are important low-level realities of the current code:

- many `fit` methods reinitialize state and are not incremental
- some “deep” models update only the final readout layer
- tree/forest/boosting serialization is incomplete
- `DataFrame::numeric_matrix` hardcodes missing numeric values to `0.0`
- `SelfAttention::fit` is empty
- `PCA::predict` is a synonym for projection
- there is no shared RNG service or configuration object
- there is no exception hierarchy beyond standard exceptions

## 13. File-by-File Intent Summary

This repository’s files can be understood as:

- `include/ml/core/*`: data model and universal contracts
- `include/ml/data/*`, `src/data/*`: ingest and prepare external data
- `include/ml/<family>/*`, `src/<family>/*`: concrete ML implementations by family
- `include/MetricsLogger.hpp`: run artifact export
- `examples/*`: compact usage demonstrations
- `scripts/*`: external dataset preparation
- `tools/*`: output visualization
- `tests/*`: executable verification programs
- `.github/workflows/*`: CI automation
- `README.md` and `/docs/*`: teaching and operational documentation

That file-level separation is clean and consistent enough that new contributors can usually infer where new code belongs just from its concern:

- new preprocessing feature -> `include/ml/data`, `src/data`
- new model -> model family directory under `include/ml` and `src`
- new experiment utility -> top-level `include/`, `scripts/`, or `tools`
- new validation -> `tests`
