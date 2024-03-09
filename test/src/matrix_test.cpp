#include <matrix.h>
#include "gtest/gtest.h"

namespace {
    TEST(MatrixTest, AddTest) {
        Mat2D mat1({{1, 1},
                    {1, 1}});
        Mat2D mat2({{2, 2},
                    {2, 2}});
        Mat2D result(2, 2);
        Mat2D::add(mat1, mat2, result);
        EXPECT_EQ(result, Mat2D({{3, 3},
                                 {3, 3}}));
    }

    TEST(MatrixTest, MulTest) {
        Mat2D mat1({{1, 1},
                    {1, 1}});
        Mat2D mat2({{2, 2},
                    {2, 2}});
        Mat2D result(2, 2);
        Mat2D::multiply(mat1, mat2, result);
        EXPECT_EQ(result, Mat2D({{4, 4},
                                 {4, 4}}));
    }

    TEST(MatrixTest, ConcatTest) {
        Mat2D mat1({{1, 1},
                    {1, 1}});
        Mat2D mat2({{2, 2},
                    {2, 2}});
        Mat2D result(2, 4);
        Mat2D::concat(mat1, mat2, result);
        EXPECT_EQ(result, Mat2D({{1, 1, 2, 2},
                                 {1, 1, 2, 2},}));
    }

    TEST(MatrixTest, ArrayConcatTest) {
        std::vector<std::shared_ptr<Mat2D>> mats;
        Mat2D mat1({{1, 1},
                    {1, 1}});
        Mat2D mat2({{2, 2},
                    {2, 2}});
        Mat2D mat3({{3, 3},
                    {3, 3}});
        mats.push_back(std::make_shared<Mat2D>(mat1));
        mats.push_back(std::make_shared<Mat2D>(mat2));
        mats.push_back(std::make_shared<Mat2D>(mat3));
        Mat2D result(2, 6);
        Mat2D::concat(mats, result);
        EXPECT_EQ(result, Mat2D({{1, 1, 2, 2, 3, 3},
                                 {1, 1, 2, 2, 3, 3}}));
    }

    TEST(MatrixTest, EqTest) {
        Mat2D mat1({{1, 1},
                    {1, 1}});
        Mat2D mat2({{1, 1},
                    {1, 1}});
        EXPECT_EQ(mat1, mat2);
    }

    TEST(MatrixTest, BiasTest) {
        Mat2D mat1({{1, 0},
                    {0, 1}});
        mat1 += 3;
        EXPECT_EQ(mat1, Mat2D({{4, 3},
                               {3, 4}}));
    }

    TEST(MatrixTest, ScaleTest) {
        Mat2D mat1({{1, 1},
                    {1, 1}});
        Mat2D mat2({{2, 2},
                    {2, 2}});
        mat1 *= 2;
        EXPECT_EQ(mat1, mat2);
    }

    TEST(MatrixTest, SoftmaxTest) {
        Mat2D mat1({{2, 0, 1},
                    {0, 2, 1},
                    {1, 1, 2}});
        mat1.softmax();
        EXPECT_EQ(mat1, Mat2D({{0.665241, 0.090031, 0.244728},
                               {0.090031, 0.665241, 0.244728},
                               {0.211942, 0.211942, 0.576117}}));
    }

    TEST(MatrixTest, MaskTest) {
        Mat2D mat1({{0, 0, 0},
                    {0, 0, 0},
                    {0, 0, 0}});
        mat1.mask();
        auto inf = (double) INFINITY;
        EXPECT_EQ(mat1, Mat2D({{0, -inf, -inf},
                               {0, 0,    -inf},
                               {0, 0,    0}}));
    }
}
