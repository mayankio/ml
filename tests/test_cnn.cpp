#include <cassert>

#include "ml/deep/CNN.hpp"

int main() {
    ml::Matrix x{
        {1, 1, 0, 0, 1, 1, 0, 0, 0},
        {1, 0, 1, 1, 0, 1, 1, 0, 1},
        {0, 0, 1, 0, 0, 1, 1, 1, 0},
        {1, 1, 1, 0, 0, 0, 1, 1, 1}};
    ml::Matrix y{{1.0}, {1.0}, {0.0}, {1.0}};
    ml::SimpleCNN model(3, 3, 3, 2, 0.05, 800);
    model.fit(x, y);
    const ml::Matrix probs = model.predict_proba(x);
    for (std::size_t i = 0; i < probs.rows(); ++i) {
        assert(probs(i, 0) >= 0.0 && probs(i, 0) <= 1.0);
    }
}
