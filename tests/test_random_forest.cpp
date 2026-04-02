#include <cassert>

#include "ml/metrics/Metrics.hpp"
#include "ml/probabilistic/RandomForest.hpp"

int main() {
    ml::Matrix x{{0.0, 0.0}, {0.1, 0.2}, {1.0, 1.0}, {1.1, 0.9}, {0.0, 1.0}, {1.0, 0.0}};
    ml::Matrix y{{0.0}, {0.0}, {1.0}, {1.0}, {0.0}, {1.0}};
    ml::RandomForestClassifier model(7, 4, 2, 5);
    model.fit(x, y);
    assert(ml::accuracy_score(model.predict(x), y) >= 0.83);
}
