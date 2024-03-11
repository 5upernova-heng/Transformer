#include "matrix/matrix.h"
#include "module/attention.h"


int main() {
    Mat2D input(10, 500), output(10, 500);
    Attention attention(input.sizes.first, input.sizes.second, 64, 64, 8, false);
    attention.MultiheadAttention(input, output);
    return 0;
}

