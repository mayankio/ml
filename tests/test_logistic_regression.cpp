#include <cassert>

#include "ml/linear/LogisticRegression.hpp"
#include "ml/metrics/Metrics.hpp"

int main() {
    ml::Matrix x{{0.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
    ml::Matrix y{{0.0}, {0.0}, {0.0}, {1.0}, {1.0}, {1.0}};
    ml::LogisticRegression model(0.2, 3000);
    model.fit(x, y);
    assert(ml::accuracy_score(model.predict(x), y) >= 0.99);
}
