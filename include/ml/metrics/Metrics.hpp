#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ml/core/ClassificationUtils.hpp"
#include "ml/core/Matrix.hpp"

namespace ml {

enum class Average {
    Macro,
    Micro
};

struct ConfusionMatrix {
    std::vector<int> labels;
    std::vector<std::vector<std::size_t>> counts;
};

struct ClassMetrics {
    int label = 0;
    std::size_t support = 0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
};

struct ClassificationReport {
    double accuracy = 0.0;
    ConfusionMatrix confusion;
    std::vector<ClassMetrics> per_class;
    double macro_precision = 0.0;
    double macro_recall = 0.0;
    double macro_f1 = 0.0;
    double micro_precision = 0.0;
    double micro_recall = 0.0;
    double micro_f1 = 0.0;
};

struct RocAucReport {
    std::vector<int> labels;
    std::vector<double> per_class_ovr;
    double macro_auc = 0.0;
    double micro_auc = 0.0;
};

struct RegressionReport {
    double mse = 0.0;
    double mae = 0.0;
    double rmse = 0.0;
    double r2 = 0.0;
};

namespace detail {

inline void validate_label_matrix(const Matrix& values, const char* name) {
    if (values.rows() == 0) {
        throw std::invalid_argument(std::string(name) + " must be non-empty");
    }
    if (values.cols() != 1) {
        throw std::invalid_argument(std::string(name) + " must have exactly one column");
    }
}

inline std::vector<int> matrix_to_labels(const Matrix& values, const char* name) {
    validate_label_matrix(values, name);
    std::vector<int> labels(values.rows());
    for (std::size_t i = 0; i < values.rows(); ++i) {
        labels[i] = checked_class_label(values(i, 0));
    }
    return labels;
}

inline void validate_prediction_target_rows(const Matrix& predictions, const Matrix& targets) {
    validate_label_matrix(predictions, "predictions");
    validate_label_matrix(targets, "targets");
    if (predictions.rows() != targets.rows()) {
        throw std::invalid_argument("predictions and targets must have the same number of rows");
    }
}

inline void validate_regression_inputs(const Matrix& predictions, const Matrix& targets, const char* metric_name) {
    if (predictions.rows() == 0 || predictions.cols() == 0 || targets.rows() == 0 || targets.cols() == 0) {
        throw std::invalid_argument(std::string(metric_name) + " requires non-empty predictions and targets");
    }
    if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols()) {
        throw std::invalid_argument(std::string(metric_name) + " requires predictions and targets to have the same shape");
    }
    for (double value : predictions.data()) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(std::string(metric_name) + " requires finite prediction values");
        }
    }
    for (double value : targets.data()) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(std::string(metric_name) + " requires finite target values");
        }
    }
}

inline void validate_labels_argument(const std::vector<int>& labels, const char* name) {
    std::set<int> unique(labels.begin(), labels.end());
    if (unique.size() != labels.size()) {
        throw std::invalid_argument(std::string(name) + " must not contain duplicate labels");
    }
}

inline std::vector<int> resolve_confusion_labels(const std::vector<int>& truth,
                                                 const std::vector<int>& predicted,
                                                 const std::vector<int>& labels) {
    if (labels.empty()) {
        std::set<int> inferred(truth.begin(), truth.end());
        inferred.insert(predicted.begin(), predicted.end());
        return std::vector<int>(inferred.begin(), inferred.end());
    }

    validate_labels_argument(labels, "labels");
    std::set<int> allowed(labels.begin(), labels.end());
    for (int label : truth) {
        if (allowed.count(label) == 0) {
            throw std::invalid_argument("labels must include every target label");
        }
    }
    for (int label : predicted) {
        if (allowed.count(label) == 0) {
            throw std::invalid_argument("labels must include every predicted label");
        }
    }
    return labels;
}

inline std::map<int, std::size_t> build_label_to_index(const std::vector<int>& labels) {
    std::map<int, std::size_t> label_to_index;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        label_to_index.emplace(labels[i], i);
    }
    return label_to_index;
}

inline double safe_ratio(double numerator, double denominator) {
    return denominator == 0.0 ? 0.0 : numerator / denominator;
}

inline double f1_from_precision_recall(double precision, double recall) {
    const double sum = precision + recall;
    return sum == 0.0 ? 0.0 : (2.0 * precision * recall) / sum;
}

inline std::vector<int> validate_roc_auc_classes(const std::vector<int>& classes) {
    if (classes.size() < 2) {
        throw std::invalid_argument("roc_auc_score requires at least two classes");
    }
    validate_labels_argument(classes, "classes");
    return classes;
}

inline std::vector<int> validate_roc_auc_targets(const Matrix& scores,
                                                 const Matrix& targets,
                                                 const std::vector<int>& classes) {
    if (scores.rows() == 0) {
        throw std::invalid_argument("roc_auc_score requires non-empty scores");
    }
    if (targets.rows() == 0) {
        throw std::invalid_argument("roc_auc_score requires non-empty targets");
    }
    if (targets.cols() != 1) {
        throw std::invalid_argument("roc_auc_score requires targets to have exactly one column");
    }
    if (scores.rows() != targets.rows()) {
        throw std::invalid_argument("roc_auc_score requires scores and targets to have the same number of rows");
    }
    if (classes.size() == 2) {
        if (scores.cols() != 1) {
            throw std::invalid_argument("binary roc_auc_score expects scores with exactly one column");
        }
    } else if (scores.cols() != classes.size()) {
        throw std::invalid_argument("multiclass roc_auc_score expects one score column per class");
    }
    for (double score : scores.data()) {
        if (!std::isfinite(score)) {
            throw std::invalid_argument("roc_auc_score requires finite score values");
        }
    }

    std::vector<int> labels = matrix_to_labels(targets, "targets");
    std::set<int> observed(labels.begin(), labels.end());
    std::set<int> allowed(classes.begin(), classes.end());
    for (int label : observed) {
        if (allowed.count(label) == 0) {
            throw std::invalid_argument("targets contain labels that are missing from classes");
        }
    }
    for (int label : classes) {
        if (observed.count(label) == 0) {
            throw std::invalid_argument("roc_auc_score requires every class to appear in targets");
        }
    }
    return labels;
}

inline double binary_auc_from_pairs(const std::vector<double>& scores, const std::vector<int>& truth) {
    if (scores.size() != truth.size()) {
        throw std::invalid_argument("binary_auc_from_pairs requires score and truth vectors of the same size");
    }
    if (scores.empty()) {
        throw std::invalid_argument("binary_auc_from_pairs requires at least one sample");
    }

    std::vector<std::pair<double, int>> ranked(scores.size());
    std::size_t positives = 0;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (truth[i] != 0 && truth[i] != 1) {
            throw std::invalid_argument("binary_auc_from_pairs expects binary truth labels");
        }
        positives += static_cast<std::size_t>(truth[i] == 1);
        ranked[i] = {scores[i], truth[i]};
    }
    const std::size_t negatives = scores.size() - positives;
    if (positives == 0 || negatives == 0) {
        throw std::invalid_argument("roc_auc_score is undefined when only one class is present");
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const std::pair<double, int>& lhs, const std::pair<double, int>& rhs) {
                  return lhs.first < rhs.first;
              });

    double positive_rank_sum = 0.0;
    std::size_t index = 0;
    while (index < ranked.size()) {
        std::size_t end = index + 1;
        while (end < ranked.size() && ranked[end].first == ranked[index].first) {
            ++end;
        }
        const double start_rank = static_cast<double>(index + 1);
        const double end_rank = static_cast<double>(end);
        const double average_rank = (start_rank + end_rank) / 2.0;
        for (std::size_t i = index; i < end; ++i) {
            if (ranked[i].second == 1) {
                positive_rank_sum += average_rank;
            }
        }
        index = end;
    }

    const double positive_count = static_cast<double>(positives);
    const double negative_count = static_cast<double>(negatives);
    return (positive_rank_sum - (positive_count * (positive_count + 1.0) / 2.0)) / (positive_count * negative_count);
}

inline std::vector<int> binary_truth_for_label_int(const std::vector<int>& truth, int positive_label) {
    std::vector<int> output(truth.size(), 0);
    for (std::size_t i = 0; i < truth.size(); ++i) {
        output[i] = truth[i] == positive_label ? 1 : 0;
    }
    return output;
}

inline std::vector<double> score_column(const Matrix& scores, std::size_t column) {
    std::vector<double> values(scores.rows());
    for (std::size_t i = 0; i < scores.rows(); ++i) {
        values[i] = scores(i, column);
    }
    return values;
}

inline std::vector<double> binary_positive_scores(const Matrix& scores) {
    return score_column(scores, 0);
}

}  // namespace detail

inline ConfusionMatrix confusion_matrix(const Matrix& predictions,
                                        const Matrix& targets,
                                        const std::vector<int>& labels = {}) {
    detail::validate_prediction_target_rows(predictions, targets);
    const std::vector<int> predicted_labels = detail::matrix_to_labels(predictions, "predictions");
    const std::vector<int> truth_labels = detail::matrix_to_labels(targets, "targets");
    const std::vector<int> ordered_labels = detail::resolve_confusion_labels(truth_labels, predicted_labels, labels);
    const std::map<int, std::size_t> label_to_index = detail::build_label_to_index(ordered_labels);

    ConfusionMatrix result;
    result.labels = ordered_labels;
    result.counts.assign(ordered_labels.size(), std::vector<std::size_t>(ordered_labels.size(), 0));
    for (std::size_t i = 0; i < truth_labels.size(); ++i) {
        const std::size_t truth_index = label_to_index.at(truth_labels[i]);
        const std::size_t predicted_index = label_to_index.at(predicted_labels[i]);
        ++result.counts[truth_index][predicted_index];
    }
    return result;
}

inline ClassificationReport classification_report(const Matrix& predictions,
                                                  const Matrix& targets,
                                                  const std::vector<int>& labels = {}) {
    const ConfusionMatrix confusion = confusion_matrix(predictions, targets, labels);

    ClassificationReport report;
    report.confusion = confusion;
    report.per_class.reserve(confusion.labels.size());

    double correct = 0.0;
    double total = 0.0;
    double macro_precision_sum = 0.0;
    double macro_recall_sum = 0.0;
    double macro_f1_sum = 0.0;
    double total_tp = 0.0;
    double total_fp = 0.0;
    double total_fn = 0.0;

    for (std::size_t i = 0; i < confusion.labels.size(); ++i) {
        const double tp = static_cast<double>(confusion.counts[i][i]);
        double predicted_total = 0.0;
        double actual_total = 0.0;
        for (std::size_t row = 0; row < confusion.labels.size(); ++row) {
            predicted_total += static_cast<double>(confusion.counts[row][i]);
        }
        for (std::size_t col = 0; col < confusion.labels.size(); ++col) {
            actual_total += static_cast<double>(confusion.counts[i][col]);
        }

        const double precision = detail::safe_ratio(tp, predicted_total);
        const double recall = detail::safe_ratio(tp, actual_total);
        const double f1 = detail::f1_from_precision_recall(precision, recall);

        report.per_class.push_back(ClassMetrics{
            confusion.labels[i],
            static_cast<std::size_t>(actual_total),
            precision,
            recall,
            f1,
        });

        macro_precision_sum += precision;
        macro_recall_sum += recall;
        macro_f1_sum += f1;
        total_tp += tp;
        total_fp += predicted_total - tp;
        total_fn += actual_total - tp;
        correct += tp;
        total += actual_total;
    }

    const double class_count = static_cast<double>(confusion.labels.size());
    report.accuracy = detail::safe_ratio(correct, total);
    report.macro_precision = detail::safe_ratio(macro_precision_sum, class_count);
    report.macro_recall = detail::safe_ratio(macro_recall_sum, class_count);
    report.macro_f1 = detail::safe_ratio(macro_f1_sum, class_count);
    report.micro_precision = detail::safe_ratio(total_tp, total_tp + total_fp);
    report.micro_recall = detail::safe_ratio(total_tp, total_tp + total_fn);
    report.micro_f1 = detail::f1_from_precision_recall(report.micro_precision, report.micro_recall);
    return report;
}

inline double accuracy_score(const Matrix& predictions, const Matrix& targets) {
    detail::validate_prediction_target_rows(predictions, targets);
    const std::vector<int> predicted_labels = detail::matrix_to_labels(predictions, "predictions");
    const std::vector<int> truth_labels = detail::matrix_to_labels(targets, "targets");

    std::size_t correct = 0;
    for (std::size_t i = 0; i < predicted_labels.size(); ++i) {
        correct += static_cast<std::size_t>(predicted_labels[i] == truth_labels[i]);
    }
    return static_cast<double>(correct) / static_cast<double>(predicted_labels.size());
}

inline double precision_score(const Matrix& predictions,
                              const Matrix& targets,
                              Average average = Average::Macro,
                              const std::vector<int>& labels = {}) {
    const ClassificationReport report = classification_report(predictions, targets, labels);
    return average == Average::Macro ? report.macro_precision : report.micro_precision;
}

inline double recall_score(const Matrix& predictions,
                           const Matrix& targets,
                           Average average = Average::Macro,
                           const std::vector<int>& labels = {}) {
    const ClassificationReport report = classification_report(predictions, targets, labels);
    return average == Average::Macro ? report.macro_recall : report.micro_recall;
}

inline double f1_score(const Matrix& predictions,
                       const Matrix& targets,
                       Average average = Average::Macro,
                       const std::vector<int>& labels = {}) {
    const ClassificationReport report = classification_report(predictions, targets, labels);
    return average == Average::Macro ? report.macro_f1 : report.micro_f1;
}

inline RocAucReport roc_auc_report(const Matrix& scores,
                                   const Matrix& targets,
                                   const std::vector<int>& classes) {
    const std::vector<int> validated_classes = detail::validate_roc_auc_classes(classes);
    const std::vector<int> truth_labels = detail::validate_roc_auc_targets(scores, targets, validated_classes);

    RocAucReport report;
    if (validated_classes.size() == 2) {
        const std::vector<double> positive_scores = detail::binary_positive_scores(scores);
        const std::vector<int> binary_truth = detail::binary_truth_for_label_int(truth_labels, validated_classes[1]);
        const double auc = detail::binary_auc_from_pairs(positive_scores, binary_truth);
        report.labels = {validated_classes[1]};
        report.per_class_ovr = {auc};
        report.macro_auc = auc;
        report.micro_auc = auc;
        return report;
    }

    report.labels = validated_classes;
    report.per_class_ovr.reserve(validated_classes.size());

    double auc_sum = 0.0;
    std::vector<double> micro_scores;
    std::vector<int> micro_truth;
    micro_scores.reserve(scores.rows() * scores.cols());
    micro_truth.reserve(scores.rows() * scores.cols());

    for (std::size_t class_index = 0; class_index < validated_classes.size(); ++class_index) {
        const std::vector<double> class_scores = detail::score_column(scores, class_index);
        const std::vector<int> class_truth = detail::binary_truth_for_label_int(truth_labels, validated_classes[class_index]);
        const double auc = detail::binary_auc_from_pairs(class_scores, class_truth);
        report.per_class_ovr.push_back(auc);
        auc_sum += auc;

        micro_scores.insert(micro_scores.end(), class_scores.begin(), class_scores.end());
        micro_truth.insert(micro_truth.end(), class_truth.begin(), class_truth.end());
    }

    report.macro_auc = auc_sum / static_cast<double>(validated_classes.size());
    report.micro_auc = detail::binary_auc_from_pairs(micro_scores, micro_truth);
    return report;
}

inline double roc_auc_score(const Matrix& scores,
                            const Matrix& targets,
                            const std::vector<int>& classes,
                            Average average = Average::Macro) {
    const RocAucReport report = roc_auc_report(scores, targets, classes);
    return average == Average::Macro ? report.macro_auc : report.micro_auc;
}

inline double mean_absolute_error(const Matrix& predictions, const Matrix& targets) {
    detail::validate_regression_inputs(predictions, targets, "mean_absolute_error");
    double sum = 0.0;
    for (std::size_t i = 0; i < predictions.data().size(); ++i) {
        sum += std::fabs(predictions.data()[i] - targets.data()[i]);
    }
    return sum / static_cast<double>(predictions.data().size());
}

inline double root_mean_squared_error(const Matrix& predictions, const Matrix& targets) {
    detail::validate_regression_inputs(predictions, targets, "root_mean_squared_error");
    return std::sqrt(mean_squared_error(predictions, targets));
}

inline double r2_score(const Matrix& predictions, const Matrix& targets) {
    detail::validate_regression_inputs(predictions, targets, "r2_score");

    const double target_mean =
        std::accumulate(targets.data().begin(), targets.data().end(), 0.0) / static_cast<double>(targets.data().size());

    double ss_tot = 0.0;
    double ss_res = 0.0;
    for (std::size_t i = 0; i < targets.data().size(); ++i) {
        const double diff_mean = targets.data()[i] - target_mean;
        const double diff_pred = targets.data()[i] - predictions.data()[i];
        ss_tot += diff_mean * diff_mean;
        ss_res += diff_pred * diff_pred;
    }

    if (ss_tot == 0.0) {
        return ss_res == 0.0 ? 1.0 : 0.0;
    }
    return 1.0 - (ss_res / ss_tot);
}

inline RegressionReport regression_report(const Matrix& predictions, const Matrix& targets) {
    detail::validate_regression_inputs(predictions, targets, "regression_report");
    const double mse = mean_squared_error(predictions, targets);
    return RegressionReport{
        mse,
        mean_absolute_error(predictions, targets),
        std::sqrt(mse),
        r2_score(predictions, targets),
    };
}

}  // namespace ml
