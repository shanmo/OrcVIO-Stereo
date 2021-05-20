#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>


namespace orcvio {

class MotionEstimator
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool solveRelativeRT(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &corres, Eigen::Matrix3d &R, Eigen::Vector3d &T);
};

}

