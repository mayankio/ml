#include <cassert>

#include "ml/metrics/Metrics.hpp"
#include "ml/optimization/SVM.hpp"

int main() {
    ml::Matrix x{{0.0, 0.0}, {0.5, 0.2}, {2.0, 2.0}, {2.5, 2.2}};
    ml::Matrix y{{0.0}, {0.0}, {1.0}, {1.0}};
    ml::LinearSVM model(0.01, 1500, 1.0, false);
    model.fit(x, y);
    assert(ml::accuracy_score(model.predict(x), y) >= 0.99);
}
