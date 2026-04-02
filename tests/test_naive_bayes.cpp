#include <cassert>

#include "ml/metrics/Metrics.hpp"
#include "ml/probabilistic/NaiveBayes.hpp"

int main() {
    ml::Matrix x{{1.0, 20.0}, {1.2, 18.0}, {4.0, 80.0}, {4.2, 78.0}};
    ml::Matrix y{{0.0}, {0.0}, {1.0}, {1.0}};
    ml::GaussianNaiveBayes model;
    model.fit(x, y);
    assert(ml::accuracy_score(model.predict(x), y) >= 0.99);
}
