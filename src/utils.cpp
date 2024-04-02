#include "utils.h"

std::string pair2String(int a, int b) {
    return std::format("({}, {})", a, b);
}

std::string pair2String(const std::pair<int, int> &p) {
    return std::format("({}, {})", p.first, p.second);
}

void read_matrix(const std::string &filename, const std::shared_ptr<Mat2D> &mat) {
    std::ifstream file(TENSOR_PATH + filename);
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
    mat->data = std::make_shared<std::shared_ptr<float[]>[]>(m);
    for (int i = 0; i < m; i++) {
        mat->data[i] = std::make_shared<float[]>(n);
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            file >> mat->data[i][j];
        }
    }
    file.close();
}


void write_matrix(const std::string &filename, const Mat2D &mat) {
    std::ofstream file(TENSOR_PATH + filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    file << std::scientific;
    file << std::setprecision(18);
    file << "# (" << mat.sizes.first << ", " << mat.sizes.second << ")" << std::endl;

    for (int i = 0; i < mat.sizes.first; ++i) {
        for (int j = 0; j < mat.sizes.second; ++j) {
            file << mat.data[i][j] << " ";
        }
        file << std::endl;
    }

    file.close();
}

void print_matrix(const Mat2D &mat, int row, int col, int precision) {
    printf("%s, (%d, %d)\n", pair2String(mat.sizes).c_str(), row, col);
    for (int i = 0; i < std::min(mat.sizes.first, row); i++) {
        for (int j = 0; j < std::min(mat.sizes.second, col); j++) {
            printf("%.*e ", precision, mat.data[i][j]);
        }
        printf("\n");
    }
}

void print_matrix(const std::shared_ptr<Mat2D> &mat, int row, int col, int precision) {
    printf("%s, (%d, %d)\n", pair2String(mat->sizes).c_str(), row, col);
    for (int i = 0; i < std::min(mat->sizes.first, row); i++) {
        for (int j = 0; j < std::min(mat->sizes.second, col); j++) {
            printf("%.*e ", precision, mat->data[i][j]);
        }
        printf("\n");
    }
}

void print_diff(const Mat2D &mat1, const Mat2D &mat2) {
    Mat2D diff(mat1.sizes);
    Mat2D::minus(mat1, mat2, diff);
    double max_diff = 0.0f, min_diff = 0.0f, sum = 0.0f;
    for (int i = 0; i < diff.sizes.first; i++) {
        for (int j = 0; j < diff.sizes.second; j++) {
            double temp = std::abs(diff.data[i][j]);
            max_diff = max_diff > temp ? max_diff : temp;
            min_diff = min_diff < temp ? min_diff : temp;
            sum += temp;
        }
    }
    printf("Max diff: %e\n", max_diff);
    printf("Min diff: %e\n", min_diff);
    printf("Avg diff: %e\n", sum / (diff.sizes.first * diff.sizes.second));
}

std::vector<char> get_the_bytes(const std::string &filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

torch::Tensor pt2tensor(const std::string &filename) {
    std::vector<char> f = get_the_bytes(TENSOR_PATH + filename);
    torch::IValue x = torch::pickle_load(f);
    torch::Tensor tensor = x.toTensor();
    tensor = tensor.to(torch::kFloat32);
    std::cout << tensor.sizes() << std::endl;
    return tensor;
}

std::shared_ptr<Mat2D> tensor2mat(const torch::Tensor &tensor) {
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

std::shared_ptr<Mat2D> tensor2mat2(const torch::Tensor &tensor) {
    int w = tensor.size(0), h = tensor.size(1);
    auto mat = std::make_shared<Mat2D>(w, h);
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            mat->data[i][j] = tensor[i][j].item<float>();
        }
    }
    return mat;
}

std::shared_ptr<Mat2D> pt2matrix(const std::string &filename) {
    return tensor2mat2(pt2tensor(filename));
}

bool compare_matrix(const Mat2D &mat, const std::string &tensor) {
    std::shared_ptr<Mat2D> mat2 = pt2matrix(tensor);
    print_matrix(mat, 10, 10, 4);
    print_matrix(mat2, 10, 10, 4);
    return mat == *mat2;
}