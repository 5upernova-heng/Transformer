#ifndef C_TRANSFORMER_MATRIX_H
#define C_TRANSFORMER_MATRIX_H
#define POSITIONAL_ENCODING_BASE 10000
#define WEIGHT_INIT 0.001
#define MASK_INF 10000

#include <algorithm>
#include <vector>
#include <string>
#include <format>

std::string pair2String(int a, int b);

std::string pair2String(const std::pair<int, int> &p);

class Mat2D {
public:
    Mat2D();

    Mat2D(int w, int h);

    Mat2D(Mat2D const &mat);

    explicit Mat2D(const std::pair<int, int> &sizes);

    Mat2D(std::initializer_list<std::initializer_list<double>> array);

    ~Mat2D();

    static void add(const Mat2D &mat1, const Mat2D &mat2, Mat2D &dest);

    static void multiply(const Mat2D &mat1, const Mat2D &mat2, Mat2D &dest);

    static void concat(const Mat2D &mat1, const Mat2D &mat2, Mat2D &dest);

    static void concat(const std::vector<std::shared_ptr<Mat2D>> &mats, Mat2D &dest);

    static void transpose(const Mat2D &mat1, const Mat2D &dest);

    void initData();

    void positionalEncode() const;

    void layerNorm() const;

    void mask() const;

    void softmax() const;

    void ReLU() const;

    void print(int n = 2) const;

    void operator+=(double bias) const;

    void operator*=(double scale) const;

    void operator/=(double scale) const;

    bool operator==(const Mat2D &mat) const;

    std::pair<int, int> sizes;
    std::shared_ptr<std::shared_ptr<double[]>[]> data;
};

#endif //C_TRANSFORMER_MATRIX_H
