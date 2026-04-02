#include <cassert>

#include "ml/metrics/Metrics.hpp"
#include "ml/probabilistic/DecisionTree.hpp"

int main() {
    ml::Matrix x{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    ml::Matrix y{{0.0}, {0.0}, {1.0}, {1.0}};
    ml::DecisionTreeClassifier model(3, 2);
    model.fit(x, y);
    assert(ml::accuracy_score(model.predict(x), y) >= 0.99);
}
