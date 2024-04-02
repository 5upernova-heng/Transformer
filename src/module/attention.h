#ifndef TRANSFORMER_ATTENTION_H
#define TRANSFORMER_ATTENTION_H

#include "../matrix/matrix.h"

class Attention {
public:
    int n, m, d_model, dk, dv, h;
    std::vector<std::shared_ptr<Mat2D>> Q, K, V, outputs;
    std::shared_ptr<Mat2D> Wo, Wq, Wk, Wv;

    Attention(int n, int m, int d_model, int dk, int dv, int h);

    void SingleHeadAttention(Mat2D &out, int index, bool mask);

    void MultiheadAttention(Mat2D &input, Mat2D &output, bool mask);

    void MultiheadAttention(Mat2D &q, Mat2D &k, Mat2D &v, Mat2D &output, bool mask);

    ~Attention();

private:
    void split(Mat2D &mat, const std::vector<std::shared_ptr<Mat2D>>& dests);
};

#endif //TRANSFORMER_ATTENTION_H
