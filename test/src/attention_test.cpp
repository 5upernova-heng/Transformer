#include <matrix.h>
#include <attention.h>
#include "gtest/gtest.h"

namespace {
    TEST(ModuleTest, TransformerTest) {
        Mat2D input, output;
        Attention attention(input.sizes.first, input.sizes.second, 64, 64, 8, false);
        attention.MultiheadAttention(input, output);
    }
}
