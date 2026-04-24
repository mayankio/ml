#include "ml/deep/RNN.hpp"

#include <fstream>
#include <sstream>

#include "ml/core/ClassificationUtils.hpp"

namespace ml {

SimpleRNN::SimpleRNN(std::size_t sequence_length, std::size_t input_dim, std::size_t hidden_dim, double learning_rate, std::size_t epochs)
    : sequence_length_(sequence_length),
      input_dim_(input_dim),
      hidden_dim_(hidden_dim),
      learning_rate_(learning_rate),
      epochs_(epochs),
      wx_(Matrix::random(input_dim, hidden_dim, -0.4, 0.4, 31)),
      wh_(Matrix::random(hidden_dim, hidden_dim, -0.4, 0.4, 32)),
      wy_(Matrix::random(hidden_dim, 1, -0.4, 0.4, 33)),
      bh_(hidden_dim, 0.0),
      by_(1, 0.0) {}

void SimpleRNN::fit(const Matrix& features, const Matrix& targets) {
    const ClassificationTargetInfo target_info = parse_classification_targets(features, targets);
    classes_ = target_info.classes;
    const std::size_t output_dim = output_dimension_for_classes(classes_);
    if (wy_.rows() != hidden_dim_ || wy_.cols() != output_dim) {
        wy_ = Matrix::random(hidden_dim_, output_dim, -0.4, 0.4, 33);
    }
    if (by_.size() != output_dim) {
        by_.assign(output_dim, 0.0);
    }

    for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
        for (std::size_t sample = 0; sample < features.rows(); ++sample) {
            std::vector<double> h(hidden_dim_, 0.0);
            for (std::size_t t = 0; t < sequence_length_; ++t) {
                std::vector<double> next_h(hidden_dim_, 0.0);
                for (std::size_t k = 0; k < hidden_dim_; ++k) {
                    double sum = bh_[k];
                    for (std::size_t j = 0; j < input_dim_; ++j) {
                        sum += features(sample, t * input_dim_ + j) * wx_(j, k);
                    }
                    for (std::size_t j = 0; j < hidden_dim_; ++j) {
                        sum += h[j] * wh_(j, k);
                    }
                    next_h[k] = std::tanh(sum);
                }
                h = next_h;
            }

            if (is_binary_classes(classes_)) {
                double logit = by_[0];
                for (std::size_t k = 0; k < hidden_dim_; ++k) {
                    logit += h[k] * wy_(k, 0);
                }
                const double prediction = sigmoid(logit);
                const double error = prediction - (target_info.indices[sample] == 1 ? 1.0 : 0.0);
                for (std::size_t k = 0; k < hidden_dim_; ++k) {
                    wy_(k, 0) -= learning_rate_ * error * h[k];
                }
                by_[0] -= learning_rate_ * error;
                continue;
            }

            std::vector<double> logits(classes_.size(), 0.0);
            for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
                logits[cls] = by_[cls];
                for (std::size_t k = 0; k < hidden_dim_; ++k) {
                    logits[cls] += h[k] * wy_(k, cls);
                }
            }
            const std::vector<double> probabilities = softmax(logits);
            for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
                const double error = probabilities[cls] - (target_info.indices[sample] == cls ? 1.0 : 0.0);
                for (std::size_t k = 0; k < hidden_dim_; ++k) {
                    wy_(k, cls) -= learning_rate_ * error * h[k];
                }
                by_[cls] -= learning_rate_ * error;
            }
        }
    }
}

Matrix SimpleRNN::predict_proba(const Matrix& features) const {
    if (classes_.empty()) {
        throw std::logic_error("SimpleRNN must be fit before predict_proba");
    }
    Matrix output(features.rows(), output_dimension_for_classes(classes_));
    for (std::size_t sample = 0; sample < features.rows(); ++sample) {
        std::vector<double> h(hidden_dim_, 0.0);
        for (std::size_t t = 0; t < sequence_length_; ++t) {
            std::vector<double> next_h(hidden_dim_, 0.0);
            for (std::size_t k = 0; k < hidden_dim_; ++k) {
                double sum = bh_[k];
                for (std::size_t j = 0; j < input_dim_; ++j) {
                    sum += features(sample, t * input_dim_ + j) * wx_(j, k);
                }
                for (std::size_t j = 0; j < hidden_dim_; ++j) {
                    sum += h[j] * wh_(j, k);
                }
                next_h[k] = std::tanh(sum);
            }
            h = next_h;
        }
        if (is_binary_classes(classes_)) {
            double logit = by_[0];
            for (std::size_t k = 0; k < hidden_dim_; ++k) {
                logit += h[k] * wy_(k, 0);
            }
            output(sample, 0) = sigmoid(logit);
            continue;
        }

        std::vector<double> logits(classes_.size(), 0.0);
        for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
            logits[cls] = by_[cls];
            for (std::size_t k = 0; k < hidden_dim_; ++k) {
                logits[cls] += h[k] * wy_(k, cls);
            }
        }
        const std::vector<double> probabilities = softmax(logits);
        for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
            output(sample, cls) = probabilities[cls];
        }
    }
    return output;
}

Matrix SimpleRNN::predict(const Matrix& features) const {
    const Matrix probabilities = predict_proba(features);
    if (is_binary_classes(classes_)) {
        return decode_binary_predictions(classes_, probabilities, 0.5);
    }
    return decode_multiclass_predictions(classes_, probabilities);
}

const std::vector<int>& SimpleRNN::classes() const {
    return classes_;
}

std::size_t SimpleRNN::num_classes() const {
    return classes_.size();
}

void SimpleRNN::save(const std::string& path) const {
    std::ofstream out(path);
    out << "v2\n";
    out << sequence_length_ << ' ' << input_dim_ << ' ' << hidden_dim_ << ' ' << learning_rate_ << ' ' << epochs_ << '\n';
    save_vector(out, classes_);
    save_vector(out, bh_);
    save_vector(out, by_);
    save_matrix(out, wx_);
    save_matrix(out, wh_);
    save_matrix(out, wy_);
}

void SimpleRNN::load(const std::string& path) {
    std::ifstream in(path);
    std::string version;
    in >> version;
    if (version == "v2") {
        in >> sequence_length_ >> input_dim_ >> hidden_dim_ >> learning_rate_ >> epochs_;
        classes_ = load_vector<int>(in);
        bh_ = load_vector<double>(in);
        by_ = load_vector<double>(in);
        wx_ = load_matrix(in);
        wh_ = load_matrix(in);
        wy_ = load_matrix(in);
        return;
    }

    std::istringstream header(version);
    header >> sequence_length_;
    double legacy_bias = 0.0;
    in >> input_dim_ >> hidden_dim_ >> learning_rate_ >> epochs_ >> legacy_bias;
    wx_ = load_matrix(in);
    wh_ = load_matrix(in);
    wy_ = load_matrix(in);
    classes_ = {0, 1};
    bh_.assign(hidden_dim_, 0.0);
    by_.assign(1, legacy_bias);
}

SimpleLSTM::SimpleLSTM(std::size_t sequence_length, std::size_t input_dim, std::size_t hidden_dim, double learning_rate, std::size_t epochs)
    : sequence_length_(sequence_length),
      input_dim_(input_dim),
      hidden_dim_(hidden_dim),
      learning_rate_(learning_rate),
      epochs_(epochs),
      wf_(Matrix::random(input_dim + hidden_dim, hidden_dim, -0.3, 0.3, 41)),
      wi_(Matrix::random(input_dim + hidden_dim, hidden_dim, -0.3, 0.3, 42)),
      wo_(Matrix::random(input_dim + hidden_dim, hidden_dim, -0.3, 0.3, 43)),
      wc_(Matrix::random(input_dim + hidden_dim, hidden_dim, -0.3, 0.3, 44)),
      wy_(Matrix::random(hidden_dim, 1, -0.3, 0.3, 45)),
      bf_(hidden_dim, 0.0),
      bi_(hidden_dim, 0.0),
      bo_(hidden_dim, 0.0),
      bc_(hidden_dim, 0.0),
      by_(1, 0.0) {}

void SimpleLSTM::fit(const Matrix& features, const Matrix& targets) {
    const ClassificationTargetInfo target_info = parse_classification_targets(features, targets);
    classes_ = target_info.classes;
    const std::size_t output_dim = output_dimension_for_classes(classes_);
    if (wy_.rows() != hidden_dim_ || wy_.cols() != output_dim) {
        wy_ = Matrix::random(hidden_dim_, output_dim, -0.3, 0.3, 45);
    }
    if (by_.size() != output_dim) {
        by_.assign(output_dim, 0.0);
    }

    for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
        for (std::size_t sample = 0; sample < features.rows(); ++sample) {
            std::vector<double> h(hidden_dim_, 0.0);
            std::vector<double> c(hidden_dim_, 0.0);
            for (std::size_t t = 0; t < sequence_length_; ++t) {
                std::vector<double> concat(input_dim_ + hidden_dim_, 0.0);
                for (std::size_t j = 0; j < input_dim_; ++j) {
                    concat[j] = features(sample, t * input_dim_ + j);
                }
                for (std::size_t j = 0; j < hidden_dim_; ++j) {
                    concat[input_dim_ + j] = h[j];
                }
                for (std::size_t k = 0; k < hidden_dim_; ++k) {
                    double f = bf_[k];
                    double in_gate = bi_[k];
                    double o = bo_[k];
                    double candidate = bc_[k];
                    for (std::size_t j = 0; j < concat.size(); ++j) {
                        f += concat[j] * wf_(j, k);
                        in_gate += concat[j] * wi_(j, k);
                        o += concat[j] * wo_(j, k);
                        candidate += concat[j] * wc_(j, k);
                    }
                    f = sigmoid(f);
                    in_gate = sigmoid(in_gate);
                    o = sigmoid(o);
                    candidate = std::tanh(candidate);
                    c[k] = f * c[k] + in_gate * candidate;
                    h[k] = o * std::tanh(c[k]);
                }
            }

            if (is_binary_classes(classes_)) {
                double logit = by_[0];
                for (std::size_t k = 0; k < hidden_dim_; ++k) {
                    logit += h[k] * wy_(k, 0);
                }
                const double error = sigmoid(logit) - (target_info.indices[sample] == 1 ? 1.0 : 0.0);
                for (std::size_t k = 0; k < hidden_dim_; ++k) {
                    wy_(k, 0) -= learning_rate_ * error * h[k];
                }
                by_[0] -= learning_rate_ * error;
                continue;
            }

            std::vector<double> logits(classes_.size(), 0.0);
            for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
                logits[cls] = by_[cls];
                for (std::size_t k = 0; k < hidden_dim_; ++k) {
                    logits[cls] += h[k] * wy_(k, cls);
                }
            }
            const std::vector<double> probabilities = softmax(logits);
            for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
                const double error = probabilities[cls] - (target_info.indices[sample] == cls ? 1.0 : 0.0);
                for (std::size_t k = 0; k < hidden_dim_; ++k) {
                    wy_(k, cls) -= learning_rate_ * error * h[k];
                }
                by_[cls] -= learning_rate_ * error;
            }
        }
    }
}

Matrix SimpleLSTM::predict_proba(const Matrix& features) const {
    if (classes_.empty()) {
        throw std::logic_error("SimpleLSTM must be fit before predict_proba");
    }
    Matrix output(features.rows(), output_dimension_for_classes(classes_));
    for (std::size_t sample = 0; sample < features.rows(); ++sample) {
        std::vector<double> h(hidden_dim_, 0.0);
        std::vector<double> c(hidden_dim_, 0.0);
        for (std::size_t t = 0; t < sequence_length_; ++t) {
            std::vector<double> concat(input_dim_ + hidden_dim_, 0.0);
            for (std::size_t j = 0; j < input_dim_; ++j) {
                concat[j] = features(sample, t * input_dim_ + j);
            }
            for (std::size_t j = 0; j < hidden_dim_; ++j) {
                concat[input_dim_ + j] = h[j];
            }
            for (std::size_t k = 0; k < hidden_dim_; ++k) {
                double f = bf_[k];
                double in_gate = bi_[k];
                double o = bo_[k];
                double candidate = bc_[k];
                for (std::size_t j = 0; j < concat.size(); ++j) {
                    f += concat[j] * wf_(j, k);
                    in_gate += concat[j] * wi_(j, k);
                    o += concat[j] * wo_(j, k);
                    candidate += concat[j] * wc_(j, k);
                }
                f = sigmoid(f);
                in_gate = sigmoid(in_gate);
                o = sigmoid(o);
                candidate = std::tanh(candidate);
                c[k] = f * c[k] + in_gate * candidate;
                h[k] = o * std::tanh(c[k]);
            }
        }
        if (is_binary_classes(classes_)) {
            double logit = by_[0];
            for (std::size_t k = 0; k < hidden_dim_; ++k) {
                logit += h[k] * wy_(k, 0);
            }
            output(sample, 0) = sigmoid(logit);
            continue;
        }

        std::vector<double> logits(classes_.size(), 0.0);
        for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
            logits[cls] = by_[cls];
            for (std::size_t k = 0; k < hidden_dim_; ++k) {
                logits[cls] += h[k] * wy_(k, cls);
            }
        }
        const std::vector<double> probabilities = softmax(logits);
        for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
            output(sample, cls) = probabilities[cls];
        }
    }
    return output;
}

Matrix SimpleLSTM::predict(const Matrix& features) const {
    const Matrix probabilities = predict_proba(features);
    if (is_binary_classes(classes_)) {
        return decode_binary_predictions(classes_, probabilities, 0.5);
    }
    return decode_multiclass_predictions(classes_, probabilities);
}

const std::vector<int>& SimpleLSTM::classes() const {
    return classes_;
}

std::size_t SimpleLSTM::num_classes() const {
    return classes_.size();
}

void SimpleLSTM::save(const std::string& path) const {
    std::ofstream out(path);
    out << "v2\n";
    out << sequence_length_ << ' ' << input_dim_ << ' ' << hidden_dim_ << ' ' << learning_rate_ << ' ' << epochs_ << '\n';
    save_vector(out, classes_);
    save_vector(out, bf_);
    save_vector(out, bi_);
    save_vector(out, bo_);
    save_vector(out, bc_);
    save_vector(out, by_);
    save_matrix(out, wf_);
    save_matrix(out, wi_);
    save_matrix(out, wo_);
    save_matrix(out, wc_);
    save_matrix(out, wy_);
}

void SimpleLSTM::load(const std::string& path) {
    std::ifstream in(path);
    std::string version;
    in >> version;
    if (version == "v2") {
        in >> sequence_length_ >> input_dim_ >> hidden_dim_ >> learning_rate_ >> epochs_;
        classes_ = load_vector<int>(in);
        bf_ = load_vector<double>(in);
        bi_ = load_vector<double>(in);
        bo_ = load_vector<double>(in);
        bc_ = load_vector<double>(in);
        by_ = load_vector<double>(in);
        wf_ = load_matrix(in);
        wi_ = load_matrix(in);
        wo_ = load_matrix(in);
        wc_ = load_matrix(in);
        wy_ = load_matrix(in);
        return;
    }

    std::istringstream header(version);
    header >> sequence_length_;
    double legacy_bias = 0.0;
    in >> input_dim_ >> hidden_dim_ >> learning_rate_ >> epochs_ >> legacy_bias;
    wf_ = load_matrix(in);
    wi_ = load_matrix(in);
    wo_ = load_matrix(in);
    wc_ = load_matrix(in);
    wy_ = load_matrix(in);
    classes_ = {0, 1};
    bf_.assign(hidden_dim_, 0.0);
    bi_.assign(hidden_dim_, 0.0);
    bo_.assign(hidden_dim_, 0.0);
    bc_.assign(hidden_dim_, 0.0);
    by_.assign(1, legacy_bias);
}

}  // namespace ml
