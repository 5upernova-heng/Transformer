#ifndef TRANSFORMER_ATTENTION_H
#define TRANSFORMER_ATTENTION_H

#include "../matrix/matrix.h"

class Attention {
public:
    int n, d_model, dk, dv, h;
    std::vector<std::shared_ptr<Mat2D>> Wq, Wk, Wv, Q, K, V;
    std::shared_ptr<Mat2D> Wo;
    bool mask;

    Attention(int n, int d_model, int dk, int dv, int h, bool mask = false);

    void SingleHeadAttention(Mat2D &q, Mat2D &k, Mat2D &v, Mat2D &out, int index);

    void MultiheadAttention(Mat2D &input, Mat2D &output);

    void MultiheadAttention(Mat2D &q, Mat2D &k, Mat2D &v, Mat2D &output);

    ~Attention();

};

#endif //TRANSFORMER_ATTENTION_H
