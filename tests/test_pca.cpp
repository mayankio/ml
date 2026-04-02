#include <cassert>

#include "ml/unsupervised/PCA.hpp"

int main() {
    ml::Matrix x{{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}, {4.0, 8.0}};
    ml::PCA pca(1, 100);
    pca.fit(x, ml::Matrix{});
    const ml::Matrix projected = pca.transform(x);
    assert(projected.rows() == 4);
    assert(projected.cols() == 1);
}
