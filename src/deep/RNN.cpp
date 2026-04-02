#include "ml/deep/RNN.hpp"

#include <fstream>

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
      bh_(hidden_dim, 0.0) {}

void SimpleRNN::fit(const Matrix& features, const Matrix& targets) {
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
            double logit = by_;
            for (std::size_t k = 0; k < hidden_dim_; ++k) {
                logit += h[k] * wy_(k, 0);
            }
            const double prediction = sigmoid(logit);
            const double error = prediction - targets(sample, 0);
            for (std::size_t k = 0; k < hidden_dim_; ++k) {
                wy_(k, 0) -= learning_rate_ * error * h[k];
            }
            by_ -= learning_rate_ * error;
        }
    }
}

Matrix SimpleRNN::predict_proba(const Matrix& features) const {
    Matrix output(features.rows(), 1);
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
        double logit = by_;
        for (std::size_t k = 0; k < hidden_dim_; ++k) {
            logit += h[k] * wy_(k, 0);
        }
        output(sample, 0) = sigmoid(logit);
    }
    return output;
}

Matrix SimpleRNN::predict(const Matrix& features) const {
    Matrix probabilities = predict_proba(features);
    for (std::size_t i = 0; i < probabilities.rows(); ++i) {
        probabilities(i, 0) = probabilities(i, 0) >= 0.5 ? 1.0 : 0.0;
    }
    return probabilities;
}

void SimpleRNN::save(const std::string& path) const {
    std::ofstream out(path);
    out << sequence_length_ << ' ' << input_dim_ << ' ' << hidden_dim_ << ' ' << learning_rate_ << ' ' << epochs_ << ' ' << by_ << '\n';
    save_matrix(out, wx_);
    save_matrix(out, wh_);
    save_matrix(out, wy_);
}

void SimpleRNN::load(const std::string& path) {
    std::ifstream in(path);
    in >> sequence_length_ >> input_dim_ >> hidden_dim_ >> learning_rate_ >> epochs_ >> by_;
    wx_ = load_matrix(in);
    wh_ = load_matrix(in);
    wy_ = load_matrix(in);
    bh_.assign(hidden_dim_, 0.0);
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
      bc_(hidden_dim, 0.0) {}

void SimpleLSTM::fit(const Matrix& features, const Matrix& targets) {
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
            double logit = by_;
            for (std::size_t k = 0; k < hidden_dim_; ++k) {
                logit += h[k] * wy_(k, 0);
            }
            const double error = sigmoid(logit) - targets(sample, 0);
            for (std::size_t k = 0; k < hidden_dim_; ++k) {
                wy_(k, 0) -= learning_rate_ * error * h[k];
            }
            by_ -= learning_rate_ * error;
        }
    }
}

Matrix SimpleLSTM::predict_proba(const Matrix& features) const {
    Matrix output(features.rows(), 1);
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
        double logit = by_;
        for (std::size_t k = 0; k < hidden_dim_; ++k) {
            logit += h[k] * wy_(k, 0);
        }
        output(sample, 0) = sigmoid(logit);
    }
    return output;
}

Matrix SimpleLSTM::predict(const Matrix& features) const {
    Matrix probabilities = predict_proba(features);
    for (std::size_t i = 0; i < probabilities.rows(); ++i) {
        probabilities(i, 0) = probabilities(i, 0) >= 0.5 ? 1.0 : 0.0;
    }
    return probabilities;
}

void SimpleLSTM::save(const std::string& path) const {
    std::ofstream out(path);
    out << sequence_length_ << ' ' << input_dim_ << ' ' << hidden_dim_ << ' ' << learning_rate_ << ' ' << epochs_ << ' ' << by_ << '\n';
    save_matrix(out, wf_);
    save_matrix(out, wi_);
    save_matrix(out, wo_);
    save_matrix(out, wc_);
    save_matrix(out, wy_);
}

void SimpleLSTM::load(const std::string& path) {
    std::ifstream in(path);
    in >> sequence_length_ >> input_dim_ >> hidden_dim_ >> learning_rate_ >> epochs_ >> by_;
    wf_ = load_matrix(in);
    wi_ = load_matrix(in);
    wo_ = load_matrix(in);
    wc_ = load_matrix(in);
    wy_ = load_matrix(in);
}

}  // namespace ml
