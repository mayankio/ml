#include <iostream>

#include "ml/linear/LinearRegression.hpp"
#include "ml/linear/LogisticRegression.hpp"
#include "ml/unsupervised/KMeans.hpp"

int main() {
    ml::Matrix regression_x{{1.0}, {2.0}, {3.0}, {4.0}};
    ml::Matrix regression_y{{3.0}, {5.0}, {7.0}, {9.0}};
    ml::LinearRegression linear;
    linear.fit(regression_x, regression_y);
    std::cout << "Linear prediction at x=5: " << linear.predict(ml::Matrix{{5.0}})(0, 0) << '\n';

    ml::Matrix classifier_x{{0.0}, {1.0}, {2.0}, {3.0}, {4.0}};
    ml::Matrix classifier_y{{0.0}, {0.0}, {0.0}, {1.0}, {1.0}};
    ml::LogisticRegression logistic;
    logistic.fit(classifier_x, classifier_y);
    std::cout << "Logistic class at x=2.5: " << logistic.predict(ml::Matrix{{2.5}})(0, 0) << '\n';

    ml::Matrix points{{0.0, 0.0}, {0.2, 0.1}, {5.0, 5.0}, {5.1, 4.9}};
    ml::KMeans kmeans(2);
    kmeans.fit(points, ml::Matrix{});
    std::cout << "Cluster for (0.1, 0.1): " << kmeans.predict(ml::Matrix{{0.1, 0.1}})(0, 0) << '\n';
}
