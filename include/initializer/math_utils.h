#pragma once 

#include <iostream>
#include <cmath>
#include <Eigen/Dense>

namespace orcvio {

/*
 * @brief Convert the vector part of a quaternion to a
 *    full quaternion.
 * @note This function is useful to convert delta quaternion
 *    which is usually a 3x1 vector to a full quaternion.
 *    For more details, check Section 3.2 "Kalman Filter Update" in
 *    "Indirect Kalman Filter for 3D Attitude Estimation:
 *    A Tutorial for quaternion Algebra".
 */
inline Eigen::Quaterniond getSmallAngleQuaternion(
    const Eigen::Vector3d& dtheta) {

    Eigen::Vector3d dq = dtheta / 2.0;
    Eigen::Quaterniond q;
    double dq_square_norm = dq.squaredNorm();

    if (dq_square_norm <= 1) {
        q.x() = dq(0);
        q.y() = dq(1);
        q.z() = dq(2);
        q.w() = std::sqrt(1-dq_square_norm);
    } else {
        q.x() = dq(0);
        q.y() = dq(1);
        q.z() = dq(2);
        q.w() = 1;
        q.normalize();
    }

    return q;
    
}


/*
 * @brief Convert a quaternion to the corresponding rotation matrix
 * @note Pay attention to the convention used. The function follows the
 *    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
 *    A Tutorial for Quaternion Algebra", Equation (78).
 *
 *    The input quaternion should be in the form
 *      [q1, q2, q3, q4(scalar)]^T
 */
inline Eigen::Matrix3d quaternionToRotation(
    const Eigen::Vector4d& q) {
    // Hamilton
    const double& qw = q(3);
    const double& qx = q(0);
    const double& qy = q(1);
    const double& qz = q(2);
    Eigen::Matrix3d R;
    R(0, 0) = 1-2*(qy*qy+qz*qz);  R(0, 1) =   2*(qx*qy-qw*qz);  R(0, 2) =   2*(qx*qz+qw*qy);
    R(1, 0) =   2*(qx*qy+qw*qz);  R(1, 1) = 1-2*(qx*qx+qz*qz);  R(1, 2) =   2*(qy*qz-qw*qx);
    R(2, 0) =   2*(qx*qz-qw*qy);  R(2, 1) =   2*(qy*qz+qw*qx);  R(2, 2) = 1-2*(qx*qx+qy*qy);

    return R;
}


}