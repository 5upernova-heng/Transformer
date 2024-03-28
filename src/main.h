#ifndef TRANSFORMER_MAIN_H
#define TRANSFORMER_MAIN_H

#include <torch/torch.h>
#include <iostream>
#include "utils.h"
#include "matrix/matrix.h"
#include "module/attention.h"
#include "module/feedforward.h"


void encoder(Mat2D &input, Mat2D &output);

void decoder(Mat2D &input, Mat2D &output, Mat2D &encoder_out);

#endif //TRANSFORMER_MAIN_H
