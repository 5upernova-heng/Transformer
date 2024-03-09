#include <matrix.h>
#include <attention.h>
#include "gtest/gtest.h"

namespace {
    TEST(ModuleTest, TransformerTest) {
        Mat2D input, output;
        Attention attention(input, output);
        attention.MultiheadAttention();
    }
}
