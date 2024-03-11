#include "matrix/matrix.h"
#include "module/attention.h"
#include "module/feedforward.h"

int d_model = 512;
int dff = 2048;
int N = 6;
int dk = 64, dv = 64, h = 8;

void encoder(Mat2D &input, Mat2D &output) {
    input.positionalEncode();
    int n = input.sizes.first;
    FeedForward ffn(d_model, dff);
    Attention attention(input.sizes.first, d_model,
                        dk, dv, h);
    Mat2D temp(n, d_model);
    for (int i = 0; i < N; i++) {
        // attention
        attention.MultiheadAttention(input, temp, false);
        // add & norm
        Mat2D::add(input, temp, input);
        input.layerNorm();
        // ffn
        ffn.forward(input, temp);
        // add & norm
        Mat2D::add(input, temp, input);
        input.layerNorm();
    }
    output = input;
}

void decoder(Mat2D &input, Mat2D &output, Mat2D &encoder_out) {
    input.positionalEncode();
    int n = input.sizes.first;
    FeedForward ffn(d_model, dff);
    Attention attention(input.sizes.first, d_model,
                        dk, dv, h);
    Mat2D temp(n, d_model);
    for (int i = 0; i < N; i++) {
        // attention
        attention.MultiheadAttention(input, temp, false);
        // add & norm
        Mat2D::add(input, temp, input);
        input.layerNorm();
        // masked attention
        attention.MultiheadAttention(input, encoder_out, encoder_out, temp, true);
        // add & norm
        Mat2D::add(input, temp, input);
        input.layerNorm();
        // ffn
        ffn.forward(input, temp);
        // add & norm
        Mat2D::add(input, temp, input);
        input.layerNorm();
    }
    Mat2D linear(input.sizes.second, d_model);
    Mat2D::multiply(input, linear, output);
}


int main() {
    Mat2D encoder_input(10, d_model), encoder_output(10, d_model);
    Mat2D decoder_input(10, d_model), decoder_output(10, d_model);
    encoder(encoder_input, encoder_output);
    decoder(decoder_input, decoder_output, encoder_output);
    return 0;
}

