#include "main.h"

int d_model = 512;
int dff = 2048;
int N = 6;
int dk = 64, dv = 64, h = 8;


void encoder(Mat2D &input, Mat2D &output) {
//    input.positionalEncode();
    int n = input.sizes.first;
    Mat2D temp(n, d_model);
    for (int i = 0; i < N; i++) {
        FeedForward ffn(d_model, dff);
        Attention attention(input.sizes.first, input.sizes.first,
                            d_model, dk, dv, h);
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
//    input.positionalEncode();
    int n = input.sizes.first;
    FeedForward ffn(d_model, dff);
    Attention self_attention(input.sizes.first, input.sizes.first,
                             d_model, dk, dv, h);
    Attention cross_attention(input.sizes.first, encoder_out.sizes.first,
                              d_model, dk, dv, h);
    Mat2D temp(n, d_model);
    for (int i = 0; i < N; i++) {
        // attention
        self_attention.MultiheadAttention(input, temp, false);
        // add & norm
        Mat2D::add(input, temp, input);
        input.layerNorm();
        // masked attention
        cross_attention.MultiheadAttention(input, encoder_out, encoder_out, temp, true);
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
    auto source = std::make_shared<Mat2D>();
    auto target = std::make_shared<Mat2D>();
    auto enc_src = std::make_shared<Mat2D>();
    read_matrix("source", source);
    read_matrix("target", target);
    read_matrix("enc_src", enc_src);
    Mat2D encoder_output(source->sizes.first, source->sizes.second);
    Mat2D decoder_output(target->sizes.first, target->sizes.second);
    printf("Source: %s\n", pair2String(source->sizes).c_str());
    printf("Target: %s\n", pair2String(target->sizes).c_str());
    encoder(*source, encoder_output);
    write_matrix("enc_src_c", encoder_output);
    std::cout << (encoder_output == *enc_src) << std::endl;
    decoder(*target, decoder_output, encoder_output);
//    decoder_output.print(6);
    return 0;
}

