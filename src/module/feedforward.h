#ifndef TRANSFORMER_FEEDFORWARD_H
#define TRANSFORMER_FEEDFORWARD_H

#include "../matrix/matrix.h"

class FeedForward {
public:
    std::shared_ptr<Mat2D> W1, W2;

    FeedForward(int d_model, int dff);

    void forward(Mat2D &input, Mat2D &output) const;

    ~FeedForward();

};

#endif //TRANSFORMER_FEEDFORWARD_H
