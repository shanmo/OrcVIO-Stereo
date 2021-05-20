/*
 * COPYRIGHT AND PERMISSION NOTICE
 * UCSD Software ORCVIO
 * Copyright (C) 2021 
 * All rights reserved.
 */

 #include "orcvio/feature.h"

namespace orcvio {

void Feature::cost(const Eigen::Isometry3d &T_c0_ci,
                   const Eigen::Vector3d &x, 
                   const Eigen::Vector2d &z, 
                   double &e) const noexcept {
    
    // ref https://intra.ece.ucr.edu/~mourikis/papers/MourikisRoumeliotis-ICRA07.pdf
    // Compute hi1, hi2, and hi3 as Equation (37).
    const double &alpha = x(0);
    const double &beta  = x(1);
    const double &rho   = x(2);

    Eigen::Vector3d h = T_c0_ci.linear() * Eigen::Vector3d(alpha, beta, 1.0) + rho * T_c0_ci.translation();
    double &h1 = h(0);
    double &h2 = h(1);
    double &h3 = h(2);

    // Predict the feature observation in ci frame.
    Eigen::Vector2d z_hat(h1 / h3, h2 / h3);

    // Compute the residual.
    e = (z_hat - z).squaredNorm();

    return;
}


void Feature::jacobian(const Eigen::Isometry3d &T_c0_ci,
                        const Eigen::Vector3d &x, const Eigen::Vector2d &z,
                        Eigen::Matrix<double, 2, 3> &J, Eigen::Vector2d &r,
                        double &w) const noexcept {

    // Compute hi1, hi2, and hi3 as Equation (37).
    const double &alpha = x(0);
    const double &beta = x(1);
    const double &rho = x(2);

    Eigen::Vector3d h = T_c0_ci.linear() * Eigen::Vector3d(alpha, beta, 1.0) + rho * T_c0_ci.translation();
    double &h1 = h(0);
    double &h2 = h(1);
    double &h3 = h(2);

    // Compute the Jacobian.
    Eigen::Matrix3d W;
    W.leftCols<2>()  = T_c0_ci.linear().leftCols<2>();
    W.rightCols<1>() = T_c0_ci.translation();

    J.row(0) = 1 / h3 * W.row(0) - h1 / (h3 * h3) * W.row(2);
    J.row(1) = 1 / h3 * W.row(1) - h2 / (h3 * h3) * W.row(2);

    // Compute the residual.
    Eigen::Vector2d z_hat(h1 / h3, h2 / h3);
    r = z_hat - z;

    // Compute the weight based on the residual.
    double e = r.norm();
    if (e <= optimization_config.huber_epsilon)
        w = 1.0;
    else
        w = std::sqrt(2.0 * optimization_config.huber_epsilon / e);

    return;
}


bool Feature::generateInitialGuess(
    const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d> > &cam_poses,
    const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > &measurements,
    Eigen::Vector3d &p) const noexcept {

    const int total_meas = cam_poses.size() / 2; 

    // Our linear system matrices
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2*total_meas, 3);
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(2*total_meas, 1);

    // Loop through each camera for this feature
    for (int i = 0; i < total_meas; i+=2) {
        // Get the pose wrt C0
        const Eigen::Matrix<double, 3, 3> R_CitoC0 = cam_poses.at(i).linear();
        const Eigen::Matrix<double, 3, 1> p_CiinC0 = cam_poses.at(i).translation();
    
        // Get the UV coordinate normal
        Eigen::Matrix<double, 3, 1> b_i;
        b_i << measurements.at(i)(0), measurements.at(i)(1), 1;

        b_i = R_CitoC0 * b_i;
        b_i = b_i / b_i.norm();

        Eigen::Matrix<double,2,3> Bperp = Eigen::Matrix<double,2,3>::Zero();
        Bperp << -b_i(2, 0), 0, b_i(0, 0), 0, b_i(2, 0), -b_i(1, 0);

        // Append to our linear system
        A.block(i, 0, 2, 3) = Bperp;
        b.block(i, 0, 2, 1).noalias() = Bperp * p_CiinC0;
    }

    // Solve the linear system
    p = A.colPivHouseholderQr().solve(b);

    // Check A and p_f
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd singularValues;
    singularValues.resize(svd.singularValues().rows(), 1);
    singularValues = svd.singularValues();
    double condA = singularValues(0, 0) / singularValues(singularValues.rows() - 1, 0);

    // If we have a bad condition number, or it is too close
    // Then set the flag for bad (i.e. set z-axis to nan)
    if (std::abs(condA) > max_cond_number || p(2,0) < min_dist || p(2,0) > max_dist || std::isnan(p.norm())) 
        return false;
    else 
        return true; 
}

bool Feature::checkMotion(const CamStateServer &cam_states) const noexcept {

    const StateIDType &first_cam_id =  observations.begin()->first;
    const StateIDType &last_cam_id = (--observations.end())->first;

    Eigen::Isometry3d first_cam_pose;
    first_cam_pose.linear() = cam_states.find(first_cam_id)->second.orientation;
    first_cam_pose.translation() = cam_states.find(first_cam_id)->second.position;

    Eigen::Isometry3d last_cam_pose;
    last_cam_pose.linear() = cam_states.find(last_cam_id)->second.orientation;
    last_cam_pose.translation() = cam_states.find(last_cam_id)->second.position;

    // Get the direction of the feature when it is first observed.
    // This direction is represented in the world frame.
    Eigen::Vector3d feature_direction(observations.begin()->second(0), observations.begin()->second(1), 1.0);
    feature_direction = feature_direction / feature_direction.norm();
    feature_direction = first_cam_pose.linear() * feature_direction;

    // Compute the translation between the first frame and the last frame.
    // We assume the first frame and the last frame will provide the largest motion to speed up the checking process.
    Eigen::Vector3d translation = last_cam_pose.translation() - first_cam_pose.translation();
    double parallel_translation = translation.transpose() * feature_direction;
    Eigen::Vector3d orthogonal_translation = translation - parallel_translation * feature_direction;

    if (orthogonal_translation.norm() > optimization_config.translation_threshold)
        return true;
    else
        return false;
}

bool Feature::initializePosition(const CamStateServer &cam_states) noexcept {

    bool is_valid_solution = true;

    // Organize camera poses and feature observations properly.
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d> > cam_poses(0);
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > measurements(0);

    for (auto &m : observations) {
        // TODO: This should be handled properly.
        //  Normally, the required camera states should all be available in the input cam_states buffer.
        auto cam_state_iter = cam_states.find(m.first);
        if (cam_state_iter == cam_states.end())
            continue;

        // Add the measurement.
        measurements.push_back(m.second.head<2>());
        measurements.push_back(m.second.tail<2>());

        // This camera pose will take a vector from this camera frame to the world frame.
        Eigen::Isometry3d cam0_pose;
        cam0_pose.linear() = cam_state_iter->second.orientation;
        cam0_pose.translation() = cam_state_iter->second.position;

        Eigen::Isometry3d cam1_pose;
        cam1_pose = cam0_pose * CAMState::T_cam0_cam1.inverse();

        cam_poses.push_back(cam0_pose);
        cam_poses.push_back(cam1_pose);
    }

    // All camera poses should be modified such that it takes a
    // vector from the first camera frame in the buffer to this camera frame.
    Eigen::Isometry3d T_c0_w = cam_poses[0];
    std::for_each(cam_poses.begin(), cam_poses.end(),
        [&T_c0_w] (auto& pose) {
            pose = pose.inverse() * T_c0_w;
        });

    // Generate initial guess
    Eigen::Vector3d initial_position(0.0, 0.0, 0.0); // c0 frame
    is_valid_solution = generateInitialGuess(cam_poses, measurements, initial_position);
    if (!is_valid_solution)
        return is_valid_solution; 

    Eigen::Vector3d solution(
            initial_position(0) / initial_position(2),
            initial_position(1) / initial_position(2),
            1.0 / initial_position(2));

    // Apply Levenberg-Marquart method to solve for the 3d position.
    double lambda = optimization_config.initial_damping;
    int inner_loop_cntr = 0;
    int outer_loop_cntr = 0;
    bool is_cost_reduced = false;
    double delta_norm = 0;

    // Compute the initial cost.
    double total_cost = 0.0;
    for (int i = 0; i < cam_poses.size(); ++i) {
        double this_cost = 0.0;
        cost(cam_poses[i], solution, measurements[i], this_cost);
        total_cost += this_cost;
    }

    // Outer loop.
    do {
        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();

        for (int i = 0; i < cam_poses.size(); ++i) {
            Eigen::Matrix<double, 2, 3> J;
            Eigen::Vector2d r;
            double w;

            jacobian(cam_poses[i], solution, measurements[i], J, r, w);

            if (w == 1) {
                A += J.transpose() * J;
                b += J.transpose() * r;
            } else {
                double w_square = w * w;
                A += w_square * J.transpose() * J;
                b += w_square * J.transpose() * r;
            }
        }

        // Inner loop.
        // Solve for the delta that can reduce the total cost.
        do {
            Eigen::Matrix3d damper = lambda * Eigen::Matrix3d::Identity();
            Eigen::Vector3d delta = (A + damper).ldlt().solve(b);
            Eigen::Vector3d new_solution = solution - delta;
            delta_norm = delta.norm();

            double new_cost = 0.0;
            for (int i = 0; i < cam_poses.size(); ++i) {
                double this_cost = 0.0;
                cost(cam_poses[i], new_solution, measurements[i], this_cost);
                new_cost += this_cost;
            }

            if (new_cost < total_cost) {
                is_cost_reduced = true;
                solution = new_solution;
                total_cost = new_cost;
                lambda = lambda / 10 > 1e-10 ? lambda / 10 : 1e-10;
            } else {
                is_cost_reduced = false;
                lambda = lambda * 10 < 1e12 ? lambda * 10 : 1e12;
            }

        } while (inner_loop_cntr++ < optimization_config.inner_loop_max_iteration && !is_cost_reduced);

        inner_loop_cntr = 0;

    } while (outer_loop_cntr++ < optimization_config.outer_loop_max_iteration &&
                delta_norm > optimization_config.estimation_precision);

    // Covert the feature position from inverse depth representation to its 3d coordinate.
    Eigen::Vector3d final_position(solution(0) / solution(2), solution(1) / solution(2), 1.0 / solution(2));

    // Check if the solution is valid. Make sure the feature is in front of every camera frame observing it.
    for (const auto &pose : cam_poses) {
        Eigen::Vector3d position = pose.linear() * final_position + pose.translation();
        if (position(2) <= 0) {
            is_valid_solution = false;
            return is_valid_solution;
        }
    }

    // Get tangent plane to x_hat
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(final_position);
    Eigen::MatrixXd Q = qr.householderQ();

    // Max baseline we have between poses
    double base_line_max = 0.0;

    // Check maximum baseline
    // Loop through each camera for this feature
    // Loop through the other clones to see what the max baseline is
    for (const auto &pose : cam_poses) {
        // Dot product camera pose and nullspace
        double base_line = ((Q.block(0,1,3,2)).transpose() * pose.translation()).norm();
        if (base_line > base_line_max) base_line_max = base_line;
    }

    // Check if this feature is bad or not
    // 1. If the feature is too close
    // 2. If the feature is invalid
    // 3. If the baseline ratio is large
    if(final_position(2) < min_dist
        || final_position(2) > max_dist
        || (final_position.norm() / base_line_max) > max_baseline
        || std::isnan(final_position.norm())) {
        is_valid_solution = false; 
        return is_valid_solution;
    }

    // Convert the feature position to the world frame.
    position = T_c0_w.linear() * final_position + T_c0_w.translation();

    if (is_valid_solution)
        is_initialized = true;

    return is_valid_solution;
}


} // namespace orcvio

    