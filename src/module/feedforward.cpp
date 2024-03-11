#include "feedforward.h"

FeedForward::FeedForward(int d_model, int dff) {
    W1 = std::make_shared<Mat2D>(d_model, dff);
    W2 = std::make_shared<Mat2D>(dff, d_model);
}

void FeedForward::forward(Mat2D &input, Mat2D &output) const {
    Mat2D hidden(input.sizes.first, W1->sizes.second);
    Mat2D::multiply(input, *W1, hidden);
    hidden.ReLU();
    Mat2D::multiply(hidden, *W2, output);
    output.ReLU();
}

FeedForward::~FeedForward() = default;
