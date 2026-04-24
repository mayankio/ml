#include <cassert>
#include <fstream>
#include <string>

#include "MetricsLogger.hpp"

int main() {
    MetricsLogger logger;
    logger.log_epoch(0.9, 0.6, 0.55);
    logger.log_epoch(0.5, 0.8, 0.75);
    logger.export_to_csv("metrics_logger_test.csv");

    std::ifstream metrics("output/metrics_logger_test.csv");
    assert(metrics.good());
    std::string line;
    std::getline(metrics, line);
    assert(line == "epoch,loss,training_accuracy,validation_accuracy");
    std::getline(metrics, line);
    assert(line == "1,0.9,0.6,0.55");

    std::vector<int> truth{0, 1, 1};
    std::vector<int> predicted{0, 1, 0};
    logger.export_confusion_pairs_to_csv("confusion_logger_test.csv", truth, predicted);

    std::ifstream confusion("output/confusion_logger_test.csv");
    assert(confusion.good());
    std::getline(confusion, line);
    assert(line == "true_label,predicted_label");

    ml::Matrix grid{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    ml::Matrix samples{{0.25, 0.25}, {0.75, 0.75}};
    logger.export_prediction_grid_to_csv("grid_logger_test.csv", grid, std::vector<int>{0, 0, 1, 1}, samples, std::vector<int>{0, 1});

    std::ifstream grid_file("output/grid_logger_test.csv");
    assert(grid_file.good());
    std::getline(grid_file, line);
    assert(line == "kind,x1,x2,label");
}
