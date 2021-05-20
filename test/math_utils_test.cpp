/*
 * COPYRIGHT AND PERMISSION NOTICE
 * UCSD Software ORCVIO
 * Copyright (C) 2021 
 * All rights reserved.
 */

#include <gtest/gtest.h>
#include <iostream>

#include <Eigen/Dense>

#include "orcvio/utils.h"

using namespace std;
using namespace Eigen;
using namespace orcvio;

TEST(MathUtilsTest, skewSymmetric) {
    Vector3d w(1.0, 2.0, 3.0);
    Matrix3d w_hat = utils::skewSymmetric(w);
    Vector3d zero_vector = w_hat * w;

    FullPivLU<Matrix3d> lu_helper(w_hat);
    EXPECT_EQ(lu_helper.rank(), 2);
    EXPECT_DOUBLE_EQ(zero_vector.norm(), 0.0);
    return;
}


int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
