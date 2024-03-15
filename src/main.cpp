#include <iostream>
#include <fstream>
#include <regex>

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
    Attention attention(input.sizes.first, input.sizes.first,
                        d_model, dk, dv, h);
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

void readin_matrix(const std::string &filename, const std::shared_ptr<Mat2D> &mat) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    std::string line;
    std::getline(file, line);

    std::regex pattern(R"(\((\d+),\s*(\d+)\))");
    std::smatch matches;
    int m, n;
    std::regex_search(line, matches, pattern);

    if (matches.size() != 3) {
        std::cerr << "[Read in matrix] Invalid matrix size" << std::endl;
        std::cerr << "Line: " << line << std::endl;
        std::cerr << "Matches: " << matches.size() << std::endl;
        std::cerr << matches[0] << std::endl;
        return;
    }
    m = std::stoi(matches[1]);
    n = std::stoi(matches[2]);
    mat->sizes = std::make_pair(m, n);
    mat->data = std::make_shared<std::shared_ptr<double[]>[]>(m);
    for (int i = 0; i < m; i++) {
        mat->data[i] = std::make_shared<double[]>(n);
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            file >> mat->data[i][j];
        }
    }
    file.close();
}


int main() {
    const std::string path("../../python/pytorch-transformer/data");
    auto source = std::make_shared<Mat2D>();
    auto target = std::make_shared<Mat2D>();
    auto output = std::make_shared<Mat2D>();
    readin_matrix(path + "/source", source);
    readin_matrix(path + "/target", target);
    readin_matrix(path + "/output", output);
    Mat2D encoder_output(source->sizes.first, source->sizes.second);
    Mat2D decoder_output(target->sizes.first, target->sizes.second);
    printf("Source: %s\n", pair2String(source->sizes).c_str());
    printf("Target: %s\n", pair2String(target->sizes).c_str());
    printf("Output: %s\n", pair2String(output->sizes).c_str());
    encoder(*source, encoder_output);
    decoder(*target, decoder_output, encoder_output);
    decoder_output.print(6);
    return 0;
}

