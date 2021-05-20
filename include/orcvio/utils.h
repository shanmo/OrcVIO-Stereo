/*
 * COPYRIGHT AND PERMISSION NOTICE
 * UCSD Software ORCVIO
 * Copyright (C) 2021 
 * All rights reserved.
 */

#pragma once 

#include <ros/ros.h>

#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <string>
#include <cmath>

#include <opencv2/core/core.hpp>

namespace orcvio {

/*
 * @brief utilities for orcvio
 */
namespace utils {

Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& w) noexcept;

Eigen::Isometry3d getTransformEigen(const ros::NodeHandle &nh, const std::string &field) noexcept;

cv::Mat getTransformCV(const ros::NodeHandle &nh, const std::string &field);

cv::Mat getVec16Transform(const ros::NodeHandle &nh, const std::string &field);

cv::Mat getKalibrStyleTransform(const ros::NodeHandle &nh, const std::string &field);

/*
functions used in propagation start 
*/

Eigen::Matrix3d Hl_operator(const Eigen::Vector3d& gyro) noexcept;
Eigen::Matrix3d Jl_operator(const Eigen::Vector3d& gyro) noexcept;

/*
functions used in propagation end 
*/


/*
functions used in update start 
*/

Eigen::Matrix<double, 6, 6> get_cam_wrt_imu_se3_jacobian(const Eigen::Matrix3d& R_b2c, const Eigen::Vector3d& t_c_b, const Eigen::Matrix3d& R_w2c) noexcept;
Eigen::Matrix<double, 4, 6> odotOperator(const Eigen::Vector4d& x) noexcept;

/*
functions used in update end 
*/

}  // namespace utils

}  // namespace orcvio