#include <cassert>

#include "ml/unsupervised/KMeans.hpp"

int main() {
    ml::Matrix x{{0.0, 0.0}, {0.1, 0.2}, {5.0, 5.0}, {5.2, 4.9}};
    ml::KMeans model(2, 20, 3);
    model.fit(x, ml::Matrix{});
    const ml::Matrix labels = model.predict(x);
    assert(labels(0, 0) == labels(1, 0));
    assert(labels(2, 0) == labels(3, 0));
    assert(labels(0, 0) != labels(2, 0));
}
