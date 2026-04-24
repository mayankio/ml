#include <cassert>
#include <cmath>
#include <string>
#include <vector>

#include "TestUtils.hpp"
#include "ml/deep/CNN.hpp"
#include "ml/deep/MLP.hpp"
#include "ml/deep/RNN.hpp"
#include "ml/linear/LogisticRegression.hpp"
#include "ml/metrics/Metrics.hpp"
#include "ml/modern/Transformer.hpp"
#include "ml/optimization/SVM.hpp"

namespace {

ml::Matrix labels_to_matrix(const std::vector<int>& values) {
    ml::Matrix output(values.size(), 1);
    for (std::size_t i = 0; i < values.size(); ++i) {
        output(i, 0) = static_cast<double>(values[i]);
    }
    return output;
}

void assert_same_matrix(const ml::Matrix& lhs, const ml::Matrix& rhs, double tolerance = 1e-5) {
    assert(lhs.rows() == rhs.rows());
    assert(lhs.cols() == rhs.cols());
    for (std::size_t i = 0; i < lhs.rows(); ++i) {
        for (std::size_t j = 0; j < lhs.cols(); ++j) {
            assert_close(lhs(i, j), rhs(i, j), tolerance);
        }
    }
}

}  // namespace

int main() {
    {
        ml::Matrix x{
            {-3.0, -3.0},
            {-2.7, -2.4},
            {-3.4, -2.1},
            {0.0, 3.0},
            {0.4, 2.7},
            {-0.6, 3.3},
            {3.0, -1.0},
            {2.5, -1.7},
            {3.4, -0.4}};
        ml::Matrix y = labels_to_matrix({2, 2, 2, 4, 4, 4, 7, 7, 7});

        ml::LogisticRegression model(0.1, 5000);
        model.fit(x, y);
        assert(model.num_classes() == 3);
        assert(model.classes() == std::vector<int>({2, 4, 7}));
        assert(ml::accuracy_score(model.predict(x), y) >= 0.99);
        const ml::Matrix probs = model.predict_proba(x);
        assert_probability_rows(probs, 3);

        const std::string path = "multiclass_logistic.tmp";
        model.save(path);
        ml::LogisticRegression loaded;
        loaded.load(path);
        assert(loaded.classes() == std::vector<int>({2, 4, 7}));
        assert_same_matrix(loaded.predict_proba(x), probs);
    }

    {
        ml::Matrix x{
            {-3.0, -3.0},
            {-2.5, -2.2},
            {-3.2, -1.8},
            {0.0, 3.2},
            {0.6, 2.4},
            {-0.4, 3.1},
            {3.1, -1.1},
            {2.6, -1.8},
            {3.5, -0.2}};
        ml::Matrix y = labels_to_matrix({2, 2, 2, 4, 4, 4, 7, 7, 7});

        ml::LinearSVM model(0.01, 3000, 1.0, false);
        model.fit(x, y);
        assert(model.num_classes() == 3);
        assert(model.classes() == std::vector<int>({2, 4, 7}));
        assert(ml::accuracy_score(model.predict(x), y) >= 0.99);
        const ml::Matrix scores = model.decision_function(x);
        assert(scores.cols() == 3);

        const std::string path = "multiclass_svm.tmp";
        model.save(path);
        ml::LinearSVM loaded;
        loaded.load(path);
        assert(loaded.classes() == std::vector<int>({2, 4, 7}));
        assert_same_matrix(loaded.decision_function(x), scores);
    }

    {
        ml::Matrix x{
            {-3.0, -3.0},
            {-2.8, -2.5},
            {-3.5, -1.9},
            {0.0, 3.0},
            {0.4, 2.6},
            {-0.5, 3.4},
            {3.2, -1.0},
            {2.4, -1.6},
            {3.5, -0.1}};
        ml::Matrix y = labels_to_matrix({2, 2, 2, 4, 4, 4, 7, 7, 7});

        ml::MLPClassifier model(2, 6, 0.1, 4000);
        model.fit(x, y);
        assert(model.num_classes() == 3);
        assert(model.classes() == std::vector<int>({2, 4, 7}));
        assert(ml::accuracy_score(model.predict(x), y) >= 0.99);
        const ml::Matrix probs = model.predict_proba(x);
        assert_probability_rows(probs, 3);

        const std::string path = "multiclass_mlp.tmp";
        model.save(path);
        ml::MLPClassifier loaded(2, 6, 0.1, 1);
        loaded.load(path);
        assert_same_matrix(loaded.predict_proba(x), probs);
    }

    {
        ml::Matrix x{
            {1, 1, 0, 1, 1, 0, 0, 0, 0},
            {1, 1, 0, 1, 0, 0, 0, 0, 0},
            {0, 1, 0, 0, 1, 0, 0, 1, 0},
            {0, 1, 0, 0, 1, 0, 0, 1, 1},
            {0, 0, 0, 0, 1, 1, 0, 1, 1},
            {0, 0, 0, 1, 1, 1, 0, 1, 1}};
        ml::Matrix y = labels_to_matrix({10, 10, 20, 20, 30, 30});

        ml::SimpleCNN model(3, 3, 3, 2, 0.05, 1200);
        model.fit(x, y);
        assert(model.num_classes() == 3);
        const ml::Matrix probs = model.predict_proba(x);
        assert_probability_rows(probs, 3);
        assert_labels_in_set(model.predict(x), {10, 20, 30});

        const std::string path = "multiclass_cnn.tmp";
        model.save(path);
        ml::SimpleCNN loaded(3, 3, 3, 2, 0.05, 1);
        loaded.load(path);
        assert_same_matrix(loaded.predict_proba(x), probs);
    }

    {
        ml::Matrix x{
            {0, 0, 0, 0},
            {0, 0, 0, 1},
            {1, 1, 0, 0},
            {1, 1, 1, 0},
            {0, 1, 0, 1},
            {1, 0, 1, 0}};
        ml::Matrix y = labels_to_matrix({2, 2, 4, 4, 7, 7});

        ml::SimpleRNN rnn(4, 1, 4, 0.05, 1200);
        rnn.fit(x, y);
        assert(rnn.num_classes() == 3);
        const ml::Matrix rnn_probs = rnn.predict_proba(x);
        assert_probability_rows(rnn_probs, 3);
        assert_labels_in_set(rnn.predict(x), {2, 4, 7});

        const std::string rnn_path = "multiclass_rnn.tmp";
        rnn.save(rnn_path);
        ml::SimpleRNN loaded_rnn(4, 1, 4, 0.05, 1);
        loaded_rnn.load(rnn_path);
        assert_same_matrix(loaded_rnn.predict_proba(x), rnn_probs);

        ml::SimpleLSTM lstm(4, 1, 4, 0.03, 800);
        lstm.fit(x, y);
        assert(lstm.num_classes() == 3);
        const ml::Matrix lstm_probs = lstm.predict_proba(x);
        assert_probability_rows(lstm_probs, 3);
        assert_labels_in_set(lstm.predict(x), {2, 4, 7});

        const std::string lstm_path = "multiclass_lstm.tmp";
        lstm.save(lstm_path);
        ml::SimpleLSTM loaded_lstm(4, 1, 4, 0.03, 1);
        loaded_lstm.load(lstm_path);
        assert_same_matrix(loaded_lstm.predict_proba(x), lstm_probs);
    }

    {
        ml::Matrix x{
            {1, 0, 1, 0, 1, 0},
            {1, 0, 1, 0, 0.9, 0.1},
            {0, 1, 0, 1, 0, 1},
            {0, 1, 0.1, 0.9, 0, 1},
            {1, 1, 1, 1, 1, 1},
            {0.8, 0.9, 1, 1, 0.9, 0.8}};
        ml::Matrix y = labels_to_matrix({1, 1, 3, 3, 5, 5});

        ml::TransformerClassifier model(3, 2, 2, 4, 0.05, 900);
        model.fit(x, y);
        assert(model.num_classes() == 3);
        const ml::Matrix probs = model.predict_proba(x);
        assert_probability_rows(probs, 3);
        assert_labels_in_set(model.predict(x), {1, 3, 5});

        const std::string path = "multiclass_transformer.tmp";
        model.save(path);
        ml::TransformerClassifier loaded(3, 2, 2, 4, 0.05, 1);
        loaded.load(path);
        assert_same_matrix(loaded.predict_proba(x), probs);
    }
}
