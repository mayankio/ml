#include <cassert>

#include "ml/deep/RNN.hpp"

int main() {
    ml::Matrix x{{0, 0, 0, 1}, {1, 1, 1, 0}, {0, 1, 0, 1}, {1, 0, 1, 0}};
    ml::Matrix y{{0.0}, {1.0}, {0.0}, {1.0}};
    ml::SimpleRNN model(4, 1, 4, 0.05, 1200);
    model.fit(x, y);
    const ml::Matrix probs = model.predict_proba(x);
    for (std::size_t i = 0; i < probs.rows(); ++i) {
        assert(probs(i, 0) >= 0.0 && probs(i, 0) <= 1.0);
    }
}
