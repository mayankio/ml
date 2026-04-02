#include <cassert>

#include "TestUtils.hpp"
#include "ml/linear/LinearRegression.hpp"
#include "ml/metrics/Metrics.hpp"

int main() {
    ml::Matrix x{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};
    ml::Matrix y{{3.0}, {5.0}, {7.0}, {9.0}, {11.0}};
    ml::LinearRegression model(0.01, 4000);
    model.fit(x, y);
    const ml::Matrix predictions = model.predict(x);
    assert(ml::r2_score(predictions, y) > 0.99);
    assert_close(model.predict(ml::Matrix{{6.0}})(0, 0), 13.0, 0.4);
}
