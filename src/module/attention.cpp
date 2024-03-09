#include "attention.h"

Attention::Attention(Mat2D &input, Mat2D &output) : input(input), output(output) {
    n = input.sizes.first;
    d_model = input.sizes.second;
    for (int i = 0; i < H; i++) {
        Wq.push_back(std::make_shared<Mat2D>(d_model, DK));
        Wk.push_back(std::make_shared<Mat2D>(d_model, DK));
        Wv.push_back(std::make_shared<Mat2D>(d_model, DV));
        Q.push_back(std::make_shared<Mat2D>(n, DK));
        K.push_back(std::make_shared<Mat2D>(n, DK));
        V.push_back(std::make_shared<Mat2D>(n, DV));
    }
    Wo = std::make_shared<Mat2D>(H * DV, d_model);
}


/*
 * (n, d_model) -> (n, d_v)
 */
void Attention::SingleHeadAttention(Mat2D &in, Mat2D &out, int h) {
    if (h < 0 or h >= Q.size()) {
        printf("Channel number h out of range\n");
    }
    Mat2D::multiply(in, *Wq[h], *Q[h]);
    Mat2D::multiply(in, *Wk[h], *K[h]);
    Mat2D::multiply(in, *Wv[h], *V[h]);
    Mat2D attention_matrix(n, n), K_T(DK, n);
    Mat2D::transpose(*K[h], K_T);
    Mat2D::multiply(*Q[h], K_T, attention_matrix);
    attention_matrix /= sqrt(d_model);
    attention_matrix.softmax();
    Mat2D::multiply(attention_matrix, *V[h], out);
}

void Attention::MultiheadAttention() {
    std::vector<std::shared_ptr<Mat2D>> outputs;
    for (int i = 0; i < H; i++) {
        outputs.push_back(std::make_shared<Mat2D>(n, DV));
        SingleHeadAttention(input, *outputs[i], i);
    }
    Mat2D concatenated(n, DV * H);
    Mat2D::concat(outputs, concatenated);
    Mat2D::multiply(concatenated, *Wo, output);
}

Attention::~Attention() = default;


