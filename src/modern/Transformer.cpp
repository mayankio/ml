#include "ml/modern/Transformer.hpp"

#include <fstream>
#include <sstream>

#include "ml/core/ClassificationUtils.hpp"

namespace ml {

TransformerClassifier::TransformerClassifier(std::size_t sequence_length, std::size_t embedding_dim, std::size_t projection_dim, std::size_t hidden_dim, double learning_rate, std::size_t epochs)
    : sequence_length_(sequence_length),
      embedding_dim_(embedding_dim),
      projection_dim_(projection_dim),
      hidden_dim_(hidden_dim),
      learning_rate_(learning_rate),
      epochs_(epochs),
      attention_(sequence_length, embedding_dim, projection_dim),
      ff1_(Matrix::random(embedding_dim, hidden_dim, -0.2, 0.2, 61)),
      ff2_(Matrix::random(hidden_dim, embedding_dim, -0.2, 0.2, 62)),
      b1_(hidden_dim, 0.0),
      b2_(embedding_dim, 0.0),
      out_(Matrix::random(embedding_dim, 1, -0.2, 0.2, 63)),
      out_bias_(1, 0.0) {}

Matrix TransformerClassifier::encode(const Matrix& features) const {
    Matrix attended = attention_.predict(features);
    Matrix ff_hidden = add_row_vector(matmul(attended, ff1_), b1_).apply([](double value) { return relu(value); });
    Matrix ff_output = add_row_vector(matmul(ff_hidden, ff2_), b2_);
    Matrix pooled(1, embedding_dim_, 0.0);
    for (std::size_t i = 0; i < ff_output.rows(); ++i) {
        for (std::size_t j = 0; j < ff_output.cols(); ++j) {
            pooled(0, j) += ff_output(i, j);
        }
    }
    for (std::size_t j = 0; j < pooled.cols(); ++j) {
        pooled(0, j) /= static_cast<double>(ff_output.rows());
    }
    return pooled;
}

void TransformerClassifier::fit(const Matrix& features, const Matrix& targets) {
    const ClassificationTargetInfo target_info = parse_classification_targets(features, targets);
    classes_ = target_info.classes;
    const std::size_t output_dim = output_dimension_for_classes(classes_);
    if (out_.rows() != embedding_dim_ || out_.cols() != output_dim) {
        out_ = Matrix::random(embedding_dim_, output_dim, -0.2, 0.2, 63);
    }
    if (out_bias_.size() != output_dim) {
        out_bias_.assign(output_dim, 0.0);
    }

    for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
        for (std::size_t sample = 0; sample < features.rows(); ++sample) {
            Matrix sequence(sequence_length_, embedding_dim_);
            for (std::size_t t = 0; t < sequence_length_; ++t) {
                for (std::size_t d = 0; d < embedding_dim_; ++d) {
                    sequence(t, d) = features(sample, t * embedding_dim_ + d);
                }
            }
            Matrix pooled = encode(sequence);
            Matrix logits_matrix = add_row_vector(matmul(pooled, out_), out_bias_);
            if (is_binary_classes(classes_)) {
                const double prediction = sigmoid(logits_matrix(0, 0));
                const double error = prediction - (target_info.indices[sample] == 1 ? 1.0 : 0.0);
                for (std::size_t j = 0; j < embedding_dim_; ++j) {
                    out_(j, 0) -= learning_rate_ * error * pooled(0, j);
                }
                out_bias_[0] -= learning_rate_ * error;
                continue;
            }

            const std::vector<double> probabilities = softmax(logits_matrix.row_vector(0));
            for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
                const double error = probabilities[cls] - (target_info.indices[sample] == cls ? 1.0 : 0.0);
                for (std::size_t j = 0; j < embedding_dim_; ++j) {
                    out_(j, cls) -= learning_rate_ * error * pooled(0, j);
                }
                out_bias_[cls] -= learning_rate_ * error;
            }
        }
    }
}

Matrix TransformerClassifier::predict_proba(const Matrix& features) const {
    if (classes_.empty()) {
        throw std::logic_error("TransformerClassifier must be fit before predict_proba");
    }
    Matrix output(features.rows(), output_dimension_for_classes(classes_));
    for (std::size_t sample = 0; sample < features.rows(); ++sample) {
        Matrix sequence(sequence_length_, embedding_dim_);
        for (std::size_t t = 0; t < sequence_length_; ++t) {
            for (std::size_t d = 0; d < embedding_dim_; ++d) {
                sequence(t, d) = features(sample, t * embedding_dim_ + d);
            }
        }
        Matrix pooled = encode(sequence);
        Matrix logits_matrix = add_row_vector(matmul(pooled, out_), out_bias_);
        if (is_binary_classes(classes_)) {
            output(sample, 0) = sigmoid(logits_matrix(0, 0));
            continue;
        }

        const std::vector<double> probabilities = softmax(logits_matrix.row_vector(0));
        for (std::size_t cls = 0; cls < classes_.size(); ++cls) {
            output(sample, cls) = probabilities[cls];
        }
    }
    return output;
}

Matrix TransformerClassifier::predict(const Matrix& features) const {
    const Matrix probabilities = predict_proba(features);
    if (is_binary_classes(classes_)) {
        return decode_binary_predictions(classes_, probabilities, 0.5);
    }
    return decode_multiclass_predictions(classes_, probabilities);
}

const std::vector<int>& TransformerClassifier::classes() const {
    return classes_;
}

std::size_t TransformerClassifier::num_classes() const {
    return classes_.size();
}

void TransformerClassifier::save(const std::string& path) const {
    std::ofstream out(path);
    out << "v2\n";
    out << sequence_length_ << ' ' << embedding_dim_ << ' ' << projection_dim_ << ' ' << hidden_dim_ << ' ' << learning_rate_ << ' ' << epochs_ << '\n';
    save_vector(out, classes_);
    save_vector(out, b1_);
    save_vector(out, b2_);
    save_vector(out, out_bias_);
    attention_.save(out);
    save_matrix(out, ff1_);
    save_matrix(out, ff2_);
    save_matrix(out, out_);
}

void TransformerClassifier::load(const std::string& path) {
    std::ifstream in(path);
    std::string version;
    in >> version;
    if (version == "v2") {
        in >> sequence_length_ >> embedding_dim_ >> projection_dim_ >> hidden_dim_ >> learning_rate_ >> epochs_;
        classes_ = load_vector<int>(in);
        b1_ = load_vector<double>(in);
        b2_ = load_vector<double>(in);
        out_bias_ = load_vector<double>(in);
        attention_.load(in);
        ff1_ = load_matrix(in);
        ff2_ = load_matrix(in);
        out_ = load_matrix(in);
        return;
    }

    std::istringstream header(version);
    header >> sequence_length_;
    double legacy_bias = 0.0;
    in >> embedding_dim_ >> projection_dim_ >> hidden_dim_ >> learning_rate_ >> epochs_ >> legacy_bias;
    attention_ = SelfAttention(sequence_length_, embedding_dim_, projection_dim_);
    ff1_ = load_matrix(in);
    ff2_ = load_matrix(in);
    out_ = load_matrix(in);
    classes_ = {0, 1};
    b1_.assign(hidden_dim_, 0.0);
    b2_.assign(embedding_dim_, 0.0);
    out_bias_.assign(1, legacy_bias);
}

}  // namespace ml
