#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ml/core/Matrix.hpp"

class MetricsLogger {
public:
    void set_loss_per_epoch(std::vector<double> values) {
        loss_per_epoch_ = std::move(values);
    }

    void set_training_accuracy(std::vector<double> values) {
        training_accuracy_ = std::move(values);
    }

    void set_validation_accuracy(std::vector<double> values) {
        validation_accuracy_ = std::move(values);
    }

    void log_epoch(double loss, double training_accuracy, double validation_accuracy) {
        loss_per_epoch_.push_back(loss);
        training_accuracy_.push_back(training_accuracy);
        validation_accuracy_.push_back(validation_accuracy);
    }

    const std::vector<double>& loss_per_epoch() const {
        return loss_per_epoch_;
    }

    const std::vector<double>& training_accuracy() const {
        return training_accuracy_;
    }

    const std::vector<double>& validation_accuracy() const {
        return validation_accuracy_;
    }

    void export_to_csv(const std::string& filename) const {
        if (!(loss_per_epoch_.size() == training_accuracy_.size() &&
              training_accuracy_.size() == validation_accuracy_.size())) {
            throw std::invalid_argument("metrics vectors must have the same length");
        }

        const std::filesystem::path output_dir("output");
        std::filesystem::create_directories(output_dir);
        std::ofstream out(output_dir / filename);
        if (!out) {
            throw std::runtime_error("failed to open metrics output file");
        }

        out << "epoch,loss,training_accuracy,validation_accuracy\n";
        for (std::size_t i = 0; i < loss_per_epoch_.size(); ++i) {
            out << (i + 1) << ',' << loss_per_epoch_[i] << ',' << training_accuracy_[i] << ',' << validation_accuracy_[i] << '\n';
        }
    }

    void export_confusion_pairs_to_csv(const std::string& filename, const std::vector<int>& truth, const std::vector<int>& predicted) const {
        if (truth.size() != predicted.size()) {
            throw std::invalid_argument("truth and predicted vectors must have the same length");
        }

        const std::filesystem::path output_dir("output");
        std::filesystem::create_directories(output_dir);
        std::ofstream out(output_dir / filename);
        if (!out) {
            throw std::runtime_error("failed to open confusion output file");
        }

        out << "true_label,predicted_label\n";
        for (std::size_t i = 0; i < truth.size(); ++i) {
            out << truth[i] << ',' << predicted[i] << '\n';
        }
    }

    void export_prediction_grid_to_csv(const std::string& filename,
                                       const ml::Matrix& points,
                                       const std::vector<int>& predictions,
                                       const ml::Matrix& samples = ml::Matrix{},
                                       const std::vector<int>& sample_labels = {}) const {
        if (points.cols() != 2) {
            throw std::invalid_argument("prediction grid points must have exactly 2 columns");
        }
        if (points.rows() != predictions.size()) {
            throw std::invalid_argument("grid predictions size must match number of grid points");
        }
        if ((!samples.empty() && samples.cols() != 2) || (!sample_labels.empty() && samples.rows() != sample_labels.size())) {
            throw std::invalid_argument("sample points must be 2D and sample labels must match sample rows");
        }

        const std::filesystem::path output_dir("output");
        std::filesystem::create_directories(output_dir);
        std::ofstream out(output_dir / filename);
        if (!out) {
            throw std::runtime_error("failed to open decision boundary output file");
        }

        out << "kind,x1,x2,label\n";
        for (std::size_t i = 0; i < points.rows(); ++i) {
            out << "grid," << points(i, 0) << ',' << points(i, 1) << ',' << predictions[i] << '\n';
        }
        for (std::size_t i = 0; i < samples.rows(); ++i) {
            const int label = sample_labels.empty() ? 0 : sample_labels[i];
            out << "sample," << samples(i, 0) << ',' << samples(i, 1) << ',' << label << '\n';
        }
    }

private:
    std::vector<double> loss_per_epoch_;
    std::vector<double> training_accuracy_;
    std::vector<double> validation_accuracy_;
};
