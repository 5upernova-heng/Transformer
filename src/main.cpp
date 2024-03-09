#include "matrix/matrix.h"
#include "module/attention.h"


int main() {
    Mat2D input(10, 500), output(10, 500);
    Attention attention(input, output);
    attention.MultiheadAttention();
    return 0;
}

