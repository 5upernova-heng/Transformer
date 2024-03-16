#include "utils.h"


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
