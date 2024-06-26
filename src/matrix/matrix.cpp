#include "matrix.h"

std::string pair2String(int a, int b) {
    return std::format("({}, {})", a, b);
}

std::string pair2String(const std::pair<int, int> &p) {
    return std::format("({}, {})", p.first, p.second);
}

Mat2D::Mat2D() = default;


Mat2D::Mat2D(int w, int h) {
    sizes.first = w;
    sizes.second = h;
    initData();
}

Mat2D::Mat2D(Mat2D const &mat) : sizes(mat.sizes), data(mat.data) {}

Mat2D::Mat2D(const std::pair<int, int> &sizes) : sizes(sizes) {
    initData();
}

Mat2D::Mat2D(std::initializer_list<std::initializer_list<float>> array) {
    sizes.first = (int) array.size();
    sizes.second = (int) array.begin()->size();
    for (auto &row: array) {
        if (sizes.second != row.size()) {
            printf("Initializer list given is not in a form of matrix\n");
            return;
        }
    }
    initData();
    int i = 0;
    for (auto &row: array) {
        int j = 0;
        for (float e: row) {
            data[i][j++] = e;
        }
        i++;
    }
}

Mat2D::~Mat2D() = default;

void Mat2D::add(const Mat2D &mat1, const Mat2D &mat2, Mat2D &dest) {
    if (mat1.sizes != mat2.sizes) {
        printf("[Add] Two matrix operands have different size: %s, %s\n",
               pair2String(mat1.sizes).c_str(), pair2String(mat2.sizes).c_str());
        return;
    }
    if (mat1.sizes != dest.sizes) {
        printf("[Add] Unmatched size of dest: expect %s, got %s\n",
               pair2String(mat1.sizes).c_str(), pair2String(dest.sizes).c_str());
        return;
    }
    for (int i = 0; i < mat1.sizes.first; i++) {
        for (int j = 0; j < mat1.sizes.second; j++) {
            dest.data[i][j] = mat1.data[i][j] + mat2.data[i][j];
        }
    }
}

void Mat2D::minus(const Mat2D &mat1, const Mat2D &mat2, Mat2D &dest) {
    if (mat1.sizes != mat2.sizes) {
        printf("[Add] Two matrix operands have different size: %s, %s\n",
               pair2String(mat1.sizes).c_str(), pair2String(mat2.sizes).c_str());
        return;
    }
    if (mat1.sizes != dest.sizes) {
        printf("[Add] Unmatched size of dest: expect %s, got %s\n",
               pair2String(mat1.sizes).c_str(), pair2String(dest.sizes).c_str());
        return;
    }
    for (int i = 0; i < mat1.sizes.first; i++) {
        for (int j = 0; j < mat1.sizes.second; j++) {
            dest.data[i][j] = mat1.data[i][j] - mat2.data[i][j];
        }
    }
}

void Mat2D::multiply(const Mat2D &mat1, const Mat2D &mat2, Mat2D &dest) {
    if (mat1.sizes.second != mat2.sizes.first) {
        printf("[Multiply] Matrix sizes are not match, can not perform multiplication.\n");
        printf("[Multiply] Try to multiply a matrix with size %s by another matrix with size %s.\n",
               pair2String(mat1.sizes).c_str(), pair2String(mat2.sizes).c_str());
        return;
    }
    if (mat1.sizes.first != dest.sizes.first or mat2.sizes.second != dest.sizes.second) {
        printf("[Multiply] Unmatched size of dest: expect %s, got %s\n",
               pair2String(mat1.sizes.first, mat2.sizes.second).c_str(),
               pair2String(dest.sizes).c_str());
        return;
    }
    Mat2D::matmul(mat1, mat2, dest);
}

void copyout(const Mat2D &mat, float *data) {
    for (int i = 0; i < mat.sizes.first; i++) {
        for (int j = 0; j < mat.sizes.second; j++) {
            data[i * mat.sizes.second + j] = mat.data[i][j];
        }
    }
}

void copyback(const Mat2D &mat, float *data) {
    for (int i = 0; i < mat.sizes.first; i++) {
        for (int j = 0; j < mat.sizes.second; j++) {
            mat.data[i][j] = data[i * mat.sizes.second + j];
        }
    }
}

void Mat2D::matmul(const Mat2D &mat1, const Mat2D &mat2, Mat2D &dest) {
    int row = mat1.sizes.first;
    float* data1, *data2, *dest_data;
    data1 = (float *) malloc(sizeof(float) * mat1.sizes.first * mat1.sizes.second);
    data2 = (float *) malloc(sizeof(float) * mat2.sizes.first * mat2.sizes.second);
    dest_data = (float *) malloc(sizeof(float) * dest.sizes.first * dest.sizes.second);
    copyout(mat1, data1);
    copyout(mat2, data2);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            mat1.sizes.first, mat2.sizes.second, mat1.sizes.second, 1.0f,
            data1, mat1.sizes.second,
            data2, mat2.sizes.second,
            0.0f, dest_data, mat2.sizes.second);
    copyback(dest, dest_data);
    free(data1);
    free(data2);
    free(dest_data);
}

/*
 * Concat on the second dimension
 */
void Mat2D::concat(const Mat2D &mat1, const Mat2D &mat2, Mat2D &dest) {
    if (mat1.sizes.first != mat2.sizes.first) {
        printf("[Concat] Matrices given can not be concatenated: %s, %s\n",
               pair2String(mat1.sizes).c_str(), pair2String(mat2.sizes).c_str());
        return;
    }

    if (dest.sizes.first != mat1.sizes.first or
        dest.sizes.second != mat1.sizes.second + mat2.sizes.second) {
        printf("[Concat] dest matrix size does not match: expect %s, got %s\n",
               pair2String(mat1.sizes.first, mat1.sizes.second + mat2.sizes.second).c_str(),
               pair2String(dest.sizes).c_str());
        return;
    }
    for (int i = 0; i < mat1.sizes.first; i++) {
        for (int j = 0; j < mat1.sizes.second; j++) {
            dest.data[i][j] = mat1.data[i][j];
        }
    }
    for (int i = 0; i < mat2.sizes.first; i++) {
        for (int j = 0; j < mat2.sizes.second; j++) {
            dest.data[i][mat1.sizes.second + j] = mat2.data[i][j];
        }
    }
}

/*
 * Concat on the second dimension
 */
void Mat2D::concat(const std::vector<std::shared_ptr<Mat2D>> &mats, Mat2D &dest) {
    if (mats.empty()) {
        printf("[Concat] mats is empty.\n");
        return;
    }
    int first = mats[0]->sizes.first;
    int second = mats[0]->sizes.second;
    for (auto &mat: mats) {
        if (mat->sizes.first != first) {
            printf("[Concat] Unmatched matrix size in mat: %s\n", pair2String(mat->sizes).c_str());
            return;
        }
    }


    if (first != dest.sizes.first or second * mats.size() != dest.sizes.second) {
        printf("[Concat] dest matrix size does not match: expect %s, got %s\n",
               pair2String(first, second * (int) mats.size()).c_str(),
               pair2String(dest.sizes).c_str());
        return;
    }
    for (int i = 0; i < first; i++) {
        for (int j = 0; j < mats.size(); j++) {
            for (int k = 0; k < second; k++) {
                dest.data[i][j * second + k] = mats[j]->data[i][k];
            }
        }
    }
}


void Mat2D::transpose(const Mat2D &mat1, const Mat2D &dest) {
    if (mat1.sizes.first != dest.sizes.second or
        mat1.sizes.second != dest.sizes.first) {
        printf("dest matrix size does not match: expect %s, got %s\n",
               pair2String(mat1.sizes.second, mat1.sizes.first).c_str(),
               pair2String(dest.sizes).c_str());
    }
    for (int i = 0; i < mat1.sizes.first; i++) {
        for (int j = 0; j < mat1.sizes.second; j++) {
            dest.data[j][i] = mat1.data[i][j];
        }
    }
}

void Mat2D::initData() {
    data = std::make_shared<std::shared_ptr<float[]>[]>(sizes.first);
    for (int i = 0; i < sizes.first; i++) {
        data[i] = std::make_shared<float[]>(sizes.second);
    }
    for (int i = 0; i < sizes.first; i++) {
        for (int j = 0; j < sizes.second; j++) {
            data[i][j] = WEIGHT_INIT;
        }
    }
}


void Mat2D::positionalEncode() const {
    auto *bias = new float[sizes.second];
    for (int i = 0; i < sizes.second; i++) {
        int i_ = i / 2 * 2;
        bias[i] = pow(POSITIONAL_ENCODING_BASE, (float) i_ / sizes.second);
    }
    for (int i = 0; i < sizes.first; i++) {
        for (int j = 0; j < sizes.second; j++) {
            if (j % 2 == 0) {
                data[i][j] += sin(i / bias[j]);
            } else {
                data[i][j] += cos(i / bias[j]);
            }
        }
    }
    delete[] bias;
}

void normalize(const std::shared_ptr<float[]> &data, int length) {
    double sum = 0.0f;
    for (int i = 0; i < length; i++) {
        sum += data[i];
    }
    double mean = sum / (double) length;
    double var = 0;
    for (int i = 0; i < length; i++) {
        var += (data[i] - mean) * (data[i] - mean);
    }
    var /= (double) length;
    double std = sqrt(var);
    for (int i = 0; i < length; i++) {
        data[i] = (data[i] - mean) / std;
    }
}

void Mat2D::layerNorm() const {
    for (int i = 0; i < sizes.first; i++) {
        normalize(data[i], sizes.second);
    }
}

void Mat2D::mask() const {
    for (int i = 0; i < sizes.first; i++) {
        for (int j = i + 1; j < sizes.second; j++) {
            data[i][j] = -(float) MASK_INF;
        }
    }
}

/*
 * Perform 1D softmax along the first axis.
 */
void Mat2D::softmax() const {
    for (int i = 0; i < sizes.first; i++) {
        float sum = 0;
        for (int j = 0; j < sizes.second; j++) {
            if (data[i][j] > 100) {
                printf("Warning: matrix element greater than 100, softmax might not work\n");
            }
            if (data[i][j] < -100) {
                data[i][j] = 0;
            } else {
                data[i][j] = exp(data[i][j]);
            }
            sum += data[i][j];
        }
        for (int j = 0; j < sizes.second; j++) {
            data[i][j] /= sum;
        }
    }
}

void Mat2D::ReLU() const {
    for (int i = 0; i < sizes.first; i++) {
        for (int j = 0; j < sizes.second; j++) {
            data[i][j] = std::max(data[i][j], (float) 0.0);
        }
    }
}

void Mat2D::operator+=(float bias) const {
    for (int i = 0; i < sizes.first; i++) {
        for (int j = 0; j < sizes.second; j++) {
            data[i][j] += bias;
        }
    }
}

void Mat2D::operator*=(float scale) const {
    for (int i = 0; i < sizes.first; i++) {
        for (int j = 0; j < sizes.second; j++) {
            data[i][j] *= scale;
        }
    }
}

void Mat2D::operator/=(float scale) const {
    for (int i = 0; i < sizes.first; i++) {
        for (int j = 0; j < sizes.second; j++) {
            data[i][j] /= scale;
        }
    }
}

bool Mat2D::operator==(const Mat2D &mat) const {
    if (mat.sizes != sizes) {
        printf("Warning: comparing two matrix with different sizes.\n");
        return false;
    }
    for (int i = 0; i < sizes.first; i++) {
        for (int j = 0; j < sizes.second; j++) {
            if (std::abs(data[i][j] - mat.data[i][j]) > 1e-6) {
                return false;
            }
        }
    }
    return true;
}


void Mat2D::print(int n) const {
    for (int i = 0; i < sizes.first; i++) {
        for (int j = 0; j < sizes.second; j++) {
            printf("%.*e ", n, data[i][j]);
        }
        printf("\n");
    }
    printf("Sizes: %s\n", pair2String(sizes).c_str());
}

