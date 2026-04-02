#include "ml/probabilistic/NaiveBayes.hpp"

#include <cmath>
#include <fstream>
#include <map>
#include <set>

namespace ml {

void GaussianNaiveBayes::fit(const Matrix& features, const Matrix& targets) {
    std::set<int> class_set;
    for (std::size_t i = 0; i < targets.rows(); ++i) {
        class_set.insert(static_cast<int>(targets(i, 0)));
    }
    classes_.assign(class_set.begin(), class_set.end());
    for (int cls : classes_) {
        std::vector<std::size_t> indices;
        for (std::size_t i = 0; i < targets.rows(); ++i) {
            if (static_cast<int>(targets(i, 0)) == cls) {
                indices.push_back(i);
            }
        }
        priors_[cls] = static_cast<double>(indices.size()) / static_cast<double>(targets.rows());
        means_[cls].assign(features.cols(), 0.0);
        variances_[cls].assign(features.cols(), 0.0);
        for (std::size_t j = 0; j < features.cols(); ++j) {
            for (std::size_t index : indices) {
                means_[cls][j] += features(index, j);
            }
            means_[cls][j] /= static_cast<double>(indices.size());
            for (std::size_t index : indices) {
                const double diff = features(index, j) - means_[cls][j];
                variances_[cls][j] += diff * diff;
            }
            variances_[cls][j] = variances_[cls][j] / static_cast<double>(indices.size()) + 1e-8;
        }
    }
}

Matrix GaussianNaiveBayes::predict(const Matrix& features) const {
    Matrix output(features.rows(), 1);
    for (std::size_t i = 0; i < features.rows(); ++i) {
        double best_score = -1e18;
        int best_class = classes_.front();
        for (int cls : classes_) {
            double log_prob = std::log(priors_.at(cls));
            for (std::size_t j = 0; j < features.cols(); ++j) {
                const double mean = means_.at(cls)[j];
                const double variance = variances_.at(cls)[j];
                const double diff = features(i, j) - mean;
                log_prob += -0.5 * std::log(2.0 * M_PI * variance) - (diff * diff) / (2.0 * variance);
            }
            if (log_prob > best_score) {
                best_score = log_prob;
                best_class = cls;
            }
        }
        output(i, 0) = best_class;
    }
    return output;
}

void GaussianNaiveBayes::save(const std::string& path) const {
    std::ofstream out(path);
    out << classes_.size() << '\n';
    for (int cls : classes_) {
        out << cls << ' ' << priors_.at(cls) << ' ' << means_.at(cls).size() << '\n';
        for (double value : means_.at(cls)) {
            out << value << ' ';
        }
        out << '\n';
        for (double value : variances_.at(cls)) {
            out << value << ' ';
        }
        out << '\n';
    }
}

void GaussianNaiveBayes::load(const std::string& path) {
    std::ifstream in(path);
    std::size_t count = 0;
    in >> count;
    classes_.resize(count);
    means_.clear();
    variances_.clear();
    priors_.clear();
    for (std::size_t i = 0; i < count; ++i) {
        int cls = 0;
        double prior = 0.0;
        std::size_t feature_count = 0;
        in >> cls >> prior >> feature_count;
        classes_[i] = cls;
        priors_[cls] = prior;
        means_[cls].assign(feature_count, 0.0);
        variances_[cls].assign(feature_count, 0.0);
        for (double& value : means_[cls]) {
            in >> value;
        }
        for (double& value : variances_[cls]) {
            in >> value;
        }
    }
}

}  // namespace ml
