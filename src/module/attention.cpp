#include "attention.h"

Attention::Attention(int n, int m, int d_model, int dk, int dv, int h)
        : n(n), m(m), d_model(d_model), dk(dk), dv(dv), h(h) {
    for (int i = 0; i < h; i++) {
        Wq.push_back(std::make_shared<Mat2D>(d_model, dk));
        Wk.push_back(std::make_shared<Mat2D>(d_model, dk));
        Wv.push_back(std::make_shared<Mat2D>(d_model, dv));
        Q.push_back(std::make_shared<Mat2D>(n, dk));
        K.push_back(std::make_shared<Mat2D>(m, dk));
        V.push_back(std::make_shared<Mat2D>(m, dv));
        outputs.push_back(std::make_shared<Mat2D>(n, dv));
    }
    Wo = std::make_shared<Mat2D>(h * dv, d_model);
}


/*
 * (n, d_model) -> (n, d_v)
 */
void Attention::SingleHeadAttention(Mat2D &q, Mat2D &k, Mat2D &v, Mat2D &out, int index, bool mask) {
    if (index < 0 or index >= Q.size()) {
        printf("Channel number h out of range\n");
    }
    Mat2D::multiply(q, *Wq[index], *Q[index]);
    Mat2D::multiply(k, *Wk[index], *K[index]);
    Mat2D::multiply(v, *Wv[index], *V[index]);
    Mat2D attention_matrix(n, m), K_T(dk, m);
    Mat2D::transpose(*K[index], K_T);
    Mat2D::multiply(*Q[index], K_T, attention_matrix);
    attention_matrix /= sqrt(dk);
    if (mask)
        attention_matrix.mask();
    attention_matrix.softmax();
    Mat2D::multiply(attention_matrix, *V[index], out);
}

void Attention::MultiheadAttention(Mat2D &input, Mat2D &output, bool mask) {
    Attention::MultiheadAttention(input, input, input, output, mask);
}

void Attention::MultiheadAttention(Mat2D &q, Mat2D &k, Mat2D &v, Mat2D &output, bool mask) {
    for (int i = 0; i < h; i++) {
        SingleHeadAttention(q, k, v, *outputs[i], i, mask);
    }
    Mat2D concatenated(n, dv * h);
    Mat2D::concat(outputs, concatenated);
    Mat2D::multiply(concatenated, *Wo, output);
}

Attention::~Attention() = default;


