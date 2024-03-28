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
        self_attention.MultiheadAttention(input, temp, true);
        // add & norm
        Mat2D::add(input, temp, input);
        input.layerNorm();
        // masked attention
        cross_attention.MultiheadAttention(input, encoder_out, encoder_out, temp, false);
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


std::vector<char> get_the_bytes(const std::string& filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

torch::Tensor pt2tensor(const std::string& filename) {
    std::vector<char> f = get_the_bytes(TENSOR_PATH + filename);
    torch::IValue x = torch::pickle_load(f);
    torch::Tensor tensor = x.toTensor();
    tensor = tensor.to(torch::kFloat32);
    std::cout << tensor.sizes() << std::endl;
    return tensor;
}

std::shared_ptr<Mat2D> tensor2mat(const torch::Tensor& tensor) {
    int w = tensor.size(0), h = tensor.size(1);
    auto data = tensor.const_data_ptr<float>();
    auto mat = std::make_shared<Mat2D>(w, h);
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            mat->data[i][j] = data[i * w + j];
        }
    }
    return mat;
}

std::shared_ptr<Mat2D> tensor2mat2(const torch::Tensor& tensor) {
    int w = tensor.size(0), h = tensor.size(1);
    auto mat = std::make_shared<Mat2D>(w, h);
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            mat->data[i][j] = tensor[i][j].item<float>();
        }
    }
    return mat;
}

int main() {
    auto a = pt2tensor("a.pt");
    auto b = pt2tensor("b.pt");
    auto c = pt2tensor("c.pt");
    torch::Tensor dest = torch::matmul(a, b);
    std::cout << torch::all(dest.eq(c)) << std::endl;

    auto mat1 = tensor2mat(a);
    auto mat2 = tensor2mat(a);
//    auto mat3 = tensor2mat(c);
//    Mat2D mat_dest(mat1->sizes);
//    Mat2D::multiply(*mat1, *mat2, mat_dest);
//    print_matrix(*mat1, 3, 3, 4);
//    print_matrix(*mat2, 3, 3, 4);
//    print_matrix(*mat3, 3, 3, 4);
//    print_matrix(mat_dest, 3, 3, 4);
    std::cout << (*mat1 == *mat2) << std::endl;

    return 0;
//    auto source = pt2matrix("source.pt");
//    auto target = pt2matrix("target.pt");
//    auto enc_src = pt2matrix("enc_src.pt");
//    auto output = pt2matrix("output.pt");
//    read_matrix("source", source);
//    write_matrix("source_c", *source);
//    read_matrix("target", target);
//    read_matrix("enc_src", enc_src);
//    read_matrix("output", output);
//    Mat2D encoder_output(source->sizes.first, source->sizes.second);
//    Mat2D decoder_output(target->sizes.first, target->sizes.second);
//    printf("Source: %s\n", pair2String(source->sizes).c_str());
//    printf("Target: %s\n", pair2String(target->sizes).c_str());
//    encoder(*source, encoder_output);
//    std::cout << (encoder_output == *enc_src) << std::endl;
//    decoder(*target, decoder_output, encoder_output);
//    std::cout << (decoder_output == *output) << std::endl;
//    return 0;
}

