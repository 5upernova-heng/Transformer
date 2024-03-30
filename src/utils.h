#ifndef TRANSFORMER_UTILS_H
#define TRANSFORMER_UTILS_H
#define TENSOR_PATH std::string("./python/pytorch-transformer/data/")

#include <iostream>
#include <iomanip>
#include <fstream>
#include <regex>
#include "matrix.h"

std::string pair2String(int a, int b);

std::string pair2String(const std::pair<int, int> &p);

void read_matrix(const std::string &filename, const std::shared_ptr<Mat2D> &mat);

void write_matrix(const std::string &filename, const Mat2D &mat);

void print_matrix(const Mat2D &mat, int row, int col, int precision);

void print_matrix(const std::shared_ptr<Mat2D> &mat, int row, int col, int precision);

#endif //TRANSFORMER_UTILS_H
