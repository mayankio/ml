#include <cassert>

#include "ml/modern/SelfAttention.hpp"

int main() {
    ml::SelfAttention attention(3, 2, 2);
    ml::Matrix sequence{{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    const ml::Matrix output = attention.predict(sequence);
    assert(output.rows() == 3);
    assert(output.cols() == 2);
}
