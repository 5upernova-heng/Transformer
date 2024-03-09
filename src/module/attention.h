#ifndef TRANSFORMER_ATTENTION_H
#define TRANSFORMER_ATTENTION_H

#define DK 64
#define DV 64
#define H 8

#include "../matrix/matrix.h"

class Attention {
public:
    int n, d_model;
    std::vector<std::shared_ptr<Mat2D>> Wq, Wk, Wv, Q, K, V;
    Mat2D &input, output;
    std::shared_ptr<Mat2D> Wo;

    Attention(Mat2D &input, Mat2D &output);

    void SingleHeadAttention(Mat2D &in, Mat2D &out, int h);

    void MultiheadAttention();

    ~Attention();

};

#endif //TRANSFORMER_ATTENTION_H
