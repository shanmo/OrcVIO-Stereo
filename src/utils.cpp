/*
 * COPYRIGHT AND PERMISSION NOTICE
 * UCSD Software ORCVIO
 * Copyright (C) 2021 
 * All rights reserved.
 */

#include <vector>

#include "orcvio/utils.h"

namespace orcvio {
namespace utils {

/*
 *  @brief Create a skew-symmetric matrix from a 3-element vector.
 *  @note Performs the operation:
 *  w   ->  [  0 -w3  w2]
 *          [ w3   0 -w1]
 *          [-w2  w1   0]
 */
Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& w) noexcept {
    Eigen::Matrix3d w_hat;
    w_hat(0, 0) = 0;
    w_hat(0, 1) = -w(2);
    w_hat(0, 2) = w(1);
    w_hat(1, 0) = w(2);
    w_hat(1, 1) = 0;
    w_hat(1, 2) = -w(0);
    w_hat(2, 0) = -w(1);
    w_hat(2, 1) = w(0);
    w_hat(2, 2) = 0;
    return w_hat;
}

Eigen::Isometry3d getTransformEigen(const ros::NodeHandle &nh, const std::string &field) noexcept {

    Eigen::Isometry3d T;
    cv::Mat c = getTransformCV(nh, field);

    T.linear()(0, 0) = c.at<double>(0, 0);
    T.linear()(0, 1) = c.at<double>(0, 1);
    T.linear()(0, 2) = c.at<double>(0, 2);
    T.linear()(1, 0) = c.at<double>(1, 0);
    T.linear()(1, 1) = c.at<double>(1, 1);
    T.linear()(1, 2) = c.at<double>(1, 2);
    T.linear()(2, 0) = c.at<double>(2, 0);
    T.linear()(2, 1) = c.at<double>(2, 1);
    T.linear()(2, 2) = c.at<double>(2, 2);
    T.translation()(0) = c.at<double>(0, 3);
    T.translation()(1) = c.at<double>(1, 3);
    T.translation()(2) = c.at<double>(2, 3);
    return T;
}

cv::Mat getTransformCV(const ros::NodeHandle &nh, const std::string &field) {
    cv::Mat T;
    try {
        // first try reading kalibr format
        T = getKalibrStyleTransform(nh, field);
    } catch (std::runtime_error &e) {
        // maybe it's the old style format?
        ROS_DEBUG_STREAM("cannot read transform " << field << " in kalibr format, trying old one!");
        try {
            T = getVec16Transform(nh, field);
        } catch (std::runtime_error &e) {
            std::string msg = "cannot read transform " + field + " error: " + e.what();
            ROS_ERROR_STREAM(msg);
            throw std::runtime_error(msg);
        }
    }
    return T;
}

cv::Mat getVec16Transform(const ros::NodeHandle &nh, const std::string &field) {
    std::vector<double> v;
    nh.getParam(field, v);
    if (v.size() != 16) {
        throw std::runtime_error("invalid vec16!");
    }
    cv::Mat T = cv::Mat(v).clone().reshape(1, 4);  // one channel 4 rows
    return T;
}

cv::Mat getKalibrStyleTransform(const ros::NodeHandle &nh, const std::string &field) {
    cv::Mat T = cv::Mat::eye(4, 4, CV_64FC1);
    XmlRpc::XmlRpcValue lines;
    if (!nh.getParam(field, lines)) {
        throw(std::runtime_error("cannot find transform " + field));
    }
    if (lines.size() != 4 || lines.getType() != XmlRpc::XmlRpcValue::TypeArray) {
        throw(std::runtime_error("invalid transform " + field));
    }
    for (int i = 0; i < lines.size(); i++) {
        if (lines.size() != 4 || lines.getType() != XmlRpc::XmlRpcValue::TypeArray) {
            throw(std::runtime_error("bad line for transform " + field));
        }
        for (int j = 0; j < lines[i].size(); j++) {
            if (lines[i][j].getType() != XmlRpc::XmlRpcValue::TypeDouble) {
                throw(std::runtime_error("bad value for transform " + field));
            } else {
                T.at<double>(i, j) = static_cast<double>(lines[i][j]);
            }
        }
    }
    return T;
}

/*
functions used in propagation start 
*/


Eigen::Matrix3d Hl_operator(const Eigen::Vector3d& gyro) noexcept {

    double gyro_norm = gyro.norm();

    Eigen::Matrix3d term1 = 0.5 * Eigen::Matrix3d::Identity();

    // handle the case when the input is close to 0 
    if (gyro_norm < 1.0e-5)
    {
        return term1;
    }

    Eigen::Matrix3d term2 = ((gyro_norm - sin(gyro_norm)) / pow(gyro_norm, 3)) * skewSymmetric(gyro);
    Eigen::Matrix3d term3 = ((2*(cos(gyro_norm) - 1) + pow(gyro_norm, 2)) / (2*pow(gyro_norm, 4))) * (skewSymmetric(gyro) * skewSymmetric(gyro));

    Eigen::Matrix3d Hl = term1 + term2 + term3; 

    return Hl;

}

Eigen::Matrix3d Jl_operator(const Eigen::Vector3d& gyro) noexcept {

    double gyro_norm = gyro.norm();

    Eigen::Matrix3d term1 = Eigen::Matrix3d::Identity();

    // handle the case when the input is close to 0 
    if (gyro_norm < 1.0e-5)
    {
        return term1;
    }

    Eigen::Matrix3d term2 = ((1 - cos(gyro_norm)) / pow(gyro_norm, 2)) * skewSymmetric(gyro);
    Eigen::Matrix3d term3 = ((gyro_norm - sin(gyro_norm)) / pow(gyro_norm, 3)) * skewSymmetric(gyro) * skewSymmetric(gyro);

    Eigen::Matrix3d Jl = term1 + term2 + term3; 

    return Jl;
  
}

/*
functions used in propagation end 
*/




/*
functions used in update start 
*/

/**
 * @brief Computes the derivative of camera se3 wrt IMU se3 
 * @param [in] R_b2c : rotation of body frame to camera frame 
 * @param [in] t_c_b : position of camera frame in body frame 
 * @param [in] R_w2c : rotation of world frame to camera frame 
 *
 * @return Jacobians (6 x 6)
 */
Eigen::Matrix<double, 6, 6> get_cam_wrt_imu_se3_jacobian(const Eigen::Matrix3d& R_b2c, const Eigen::Vector3d& t_c_b, const Eigen::Matrix3d& R_w2c) noexcept
{
    Eigen::Matrix<double, 6, 6> p_cxi_p_ixi = Eigen::Matrix<double, 6, 6>::Zero();

    p_cxi_p_ixi.block<3, 3>(0, 0) = -1 * R_b2c * skewSymmetric(t_c_b);
    p_cxi_p_ixi.block<3, 3>(3, 0) = R_b2c;
    p_cxi_p_ixi.block<3, 3>(0, 3) = R_w2c;

    return p_cxi_p_ixi;
}


/**
 * @brief odot operator
 *
 * \f{align*}{
 *  \underline{x}^{\odot} = \begin{bmatrix} I_{3\times 3} & -x_{\times} \\ 0 & 0\end{bmatrix}
 @f}
 *
 * @param ph = 4 = point in homogeneous coordinates
 *
 * @return : odot(ph) = 4 x 6
 */
Eigen::Matrix<double, 4, 6> odotOperator(const Eigen::Vector4d& x) noexcept {

    Eigen::Matrix<double, 4, 6> temp;
    temp.setZero();

    temp.block(0, 3, 3, 3) = -1 * skewSymmetric(x.head(3));

    temp(0, 0) = x(3);
    temp(1, 1) = x(3);
    temp(2, 2) = x(3);

    return temp;

 }


/*
functions used in update end 
*/




}  // namespace utils

}  // namespace orcvio