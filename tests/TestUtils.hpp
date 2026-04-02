#pragma once

#include <cassert>
#include <cmath>

inline void assert_close(double actual, double expected, double tolerance) {
    assert(std::fabs(actual - expected) <= tolerance);
}
