#include "ml/unsupervised/KMeans.hpp"

#include <fstream>
#include <limits>
#include <random>

namespace ml {

KMeans::KMeans(std::size_t n_clusters, std::size_t max_iters, std::uint32_t seed)
    : n_clusters_(n_clusters), max_iters_(max_iters), seed_(seed) {}

void KMeans::fit(const Matrix& features, const Matrix&) {
    std::mt19937 rng(seed_);
    std::uniform_int_distribution<std::size_t> dist(0, features.rows() - 1);
    centroids_ = Matrix(n_clusters_, features.cols());
    for (std::size_t cluster = 0; cluster < n_clusters_; ++cluster) {
        const auto sample = features.row_vector(dist(rng));
        for (std::size_t j = 0; j < features.cols(); ++j) {
            centroids_(cluster, j) = sample[j];
        }
    }
    for (std::size_t iter = 0; iter < max_iters_; ++iter) {
        Matrix sums(n_clusters_, features.cols(), 0.0);
        std::vector<std::size_t> counts(n_clusters_, 0);
        for (std::size_t i = 0; i < features.rows(); ++i) {
            const auto row = features.row_vector(i);
            std::size_t best_cluster = 0;
            double best_distance = std::numeric_limits<double>::infinity();
            for (std::size_t cluster = 0; cluster < n_clusters_; ++cluster) {
                const double distance = euclidean_distance(row, centroids_.row_vector(cluster));
                if (distance < best_distance) {
                    best_distance = distance;
                    best_cluster = cluster;
                }
            }
            counts[best_cluster]++;
            for (std::size_t j = 0; j < features.cols(); ++j) {
                sums(best_cluster, j) += features(i, j);
            }
        }
        for (std::size_t cluster = 0; cluster < n_clusters_; ++cluster) {
            if (counts[cluster] == 0) {
                continue;
            }
            for (std::size_t j = 0; j < features.cols(); ++j) {
                centroids_(cluster, j) = sums(cluster, j) / static_cast<double>(counts[cluster]);
            }
        }
    }
}

Matrix KMeans::predict(const Matrix& features) const {
    Matrix labels(features.rows(), 1);
    for (std::size_t i = 0; i < features.rows(); ++i) {
        const auto row = features.row_vector(i);
        std::size_t best_cluster = 0;
        double best_distance = std::numeric_limits<double>::infinity();
        for (std::size_t cluster = 0; cluster < centroids_.rows(); ++cluster) {
            const double distance = euclidean_distance(row, centroids_.row_vector(cluster));
            if (distance < best_distance) {
                best_distance = distance;
                best_cluster = cluster;
            }
        }
        labels(i, 0) = static_cast<double>(best_cluster);
    }
    return labels;
}

const Matrix& KMeans::centroids() const {
    return centroids_;
}

void KMeans::save(const std::string& path) const {
    std::ofstream out(path);
    out << n_clusters_ << ' ' << max_iters_ << ' ' << seed_ << '\n';
    save_matrix(out, centroids_);
}

void KMeans::load(const std::string& path) {
    std::ifstream in(path);
    in >> n_clusters_ >> max_iters_ >> seed_;
    centroids_ = load_matrix(in);
}

}  // namespace ml
