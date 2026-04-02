#include "ml/unsupervised/PCA.hpp"

#include <fstream>

namespace ml {

PCA::PCA(std::size_t n_components, std::size_t power_iterations)
    : n_components_(n_components), power_iterations_(power_iterations) {}

void PCA::fit(const Matrix& features, const Matrix&) {
    means_ = features.column_means();
    Matrix centered(features.rows(), features.cols());
    for (std::size_t i = 0; i < features.rows(); ++i) {
        for (std::size_t j = 0; j < features.cols(); ++j) {
            centered(i, j) = features(i, j) - means_[j];
        }
    }
    Matrix covariance = matmul(centered.transpose(), centered) / static_cast<double>(features.rows() - 1);
    components_ = Matrix(features.cols(), n_components_);
    Matrix working = covariance;

    for (std::size_t component = 0; component < n_components_; ++component) {
        Matrix vector = Matrix::random(features.cols(), 1, -1.0, 1.0, 100 + static_cast<std::uint32_t>(component));
        for (std::size_t iter = 0; iter < power_iterations_; ++iter) {
            vector = matmul(working, vector);
            double norm = 0.0;
            for (double value : vector.data()) {
                norm += value * value;
            }
            norm = std::sqrt(norm) + 1e-12;
            vector = vector / norm;
        }
        for (std::size_t i = 0; i < features.cols(); ++i) {
            components_(i, component) = vector(i, 0);
        }
        Matrix eigen_outer(features.cols(), features.cols());
        Matrix vt = vector.transpose();
        const Matrix outer = matmul(vector, vt);
        const Matrix projected = matmul(vt, matmul(working, vector));
        working = working - outer * projected(0, 0);
    }
}

Matrix PCA::transform(const Matrix& features) const {
    Matrix centered(features.rows(), features.cols());
    for (std::size_t i = 0; i < features.rows(); ++i) {
        for (std::size_t j = 0; j < features.cols(); ++j) {
            centered(i, j) = features(i, j) - means_[j];
        }
    }
    return matmul(centered, components_);
}

Matrix PCA::predict(const Matrix& features) const {
    return transform(features);
}

const Matrix& PCA::components() const {
    return components_;
}

void PCA::save(const std::string& path) const {
    std::ofstream out(path);
    out << n_components_ << ' ' << power_iterations_ << ' ' << means_.size() << '\n';
    for (double mean : means_) {
        out << mean << ' ';
    }
    out << '\n';
    save_matrix(out, components_);
}

void PCA::load(const std::string& path) {
    std::ifstream in(path);
    std::size_t mean_size = 0;
    in >> n_components_ >> power_iterations_ >> mean_size;
    means_.assign(mean_size, 0.0);
    for (double& mean : means_) {
        in >> mean;
    }
    components_ = load_matrix(in);
}

}  // namespace ml
