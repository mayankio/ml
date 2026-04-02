#include <cassert>

#include "ml/linear/KNN.hpp"

int main() {
    ml::Matrix x{{0.0, 0.0}, {0.2, 0.1}, {1.0, 1.0}, {0.9, 1.1}};
    ml::Matrix y{{0.0}, {0.0}, {1.0}, {1.0}};
    ml::KNNClassifier model(3);
    model.fit(x, y);
    assert(model.predict(ml::Matrix{{0.1, 0.0}})(0, 0) == 0.0);
    assert(model.predict(ml::Matrix{{0.95, 0.95}})(0, 0) == 1.0);
}
