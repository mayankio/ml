#include <cassert>

#include "ml/metrics/Metrics.hpp"
#include "ml/optimization/GradientBoosting.hpp"

int main() {
    ml::Matrix x{{0.0}, {1.0}, {2.0}, {3.0}, {4.0}};
    ml::Matrix y{{1.0}, {2.0}, {5.0}, {10.0}, {17.0}};
    ml::GradientBoostingRegressor model(30, 0.1, 2);
    model.fit(x, y);
    assert(ml::r2_score(model.predict(x), y) > 0.9);
}
