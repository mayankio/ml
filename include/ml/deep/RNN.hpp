#pragma once

#include <vector>

#include "ml/core/Model.hpp"

namespace ml {

class SimpleRNN : public Model {
public:
    SimpleRNN(std::size_t sequence_length, std::size_t input_dim, std::size_t hidden_dim, double learning_rate = 0.05, std::size_t epochs = 800);

    void fit(const Matrix& features, const Matrix& targets) override;
    Matrix predict(const Matrix& features) const override;
    Matrix predict_proba(const Matrix& features) const;
    [[nodiscard]] const std::vector<int>& classes() const;
    [[nodiscard]] std::size_t num_classes() const;
    void save(const std::string& path) const override;
    void load(const std::string& path) override;

private:
    std::size_t sequence_length_;
    std::size_t input_dim_;
    std::size_t hidden_dim_;
    double learning_rate_;
    std::size_t epochs_;
    Matrix wx_;
    Matrix wh_;
    Matrix wy_;
    std::vector<double> bh_;
    std::vector<double> by_;
    std::vector<int> classes_;
};

class SimpleLSTM : public Model {
public:
    SimpleLSTM(std::size_t sequence_length, std::size_t input_dim, std::size_t hidden_dim, double learning_rate = 0.03, std::size_t epochs = 1000);

    void fit(const Matrix& features, const Matrix& targets) override;
    Matrix predict(const Matrix& features) const override;
    Matrix predict_proba(const Matrix& features) const;
    [[nodiscard]] const std::vector<int>& classes() const;
    [[nodiscard]] std::size_t num_classes() const;
    void save(const std::string& path) const override;
    void load(const std::string& path) override;

private:
    std::size_t sequence_length_;
    std::size_t input_dim_;
    std::size_t hidden_dim_;
    double learning_rate_;
    std::size_t epochs_;
    Matrix wf_;
    Matrix wi_;
    Matrix wo_;
    Matrix wc_;
    Matrix wy_;
    std::vector<double> bf_;
    std::vector<double> bi_;
    std::vector<double> bo_;
    std::vector<double> bc_;
    std::vector<double> by_;
    std::vector<int> classes_;
};

}  // namespace ml
