#include <cassert>

#include "ml/deep/MLP.hpp"
#include "ml/metrics/Metrics.hpp"

int main() {
    ml::Matrix x{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    ml::Matrix y{{0.0}, {1.0}, {1.0}, {1.0}};
    ml::MLPClassifier model(2, 4, 0.1, 3000);
    model.fit(x, y);
    assert(ml::accuracy_score(model.predict(x), y) >= 0.99);
}
