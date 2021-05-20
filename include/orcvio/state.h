/*
 * COPYRIGHT AND PERMISSION NOTICE
 * UCSD Software ORCVIO
 * Copyright (C) 2021 
 * All rights reserved.
 */

#pragma once 

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <map>
#include <vector>

#define GRAVITY_ACCELERATION 9.81

namespace orcvio {

/**
     * @brief IMUState State for IMU
     */
struct IMUState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef long long int StateIDType;

    // An unique identifier for the IMU state.
    StateIDType id;

    // id for next IMU state
    static StateIDType next_id;

    // Time when the state is recorded
    double time;

    // Orientation q_I_G
    // Take a vector from the world frame to the IMU (body) frame.
    Eigen::Matrix3d orientation;

    // Position of the IMU (body) frame in the world frame.
    Eigen::Vector3d position;

    // Velocity of the IMU (body) frame in the world frame.
    Eigen::Vector3d velocity;

    // Bias for measured angular velocity and acceleration.
    Eigen::Vector3d gyro_bias;
    Eigen::Vector3d acc_bias;

    // Transformation between the IMU and the left camera (cam0)
    Eigen::Matrix3d R_imu_cam0;
    Eigen::Vector3d t_cam0_imu;


    // Process noise
    static double gyro_noise;
    static double acc_noise;
    static double gyro_bias_noise;
    static double acc_bias_noise;

    // Gravity vector in the world frame
    static Eigen::Vector3d gravity;

    // Transformation offset from the IMU frame to the body frame.
    // The transformation takes a vector from the IMU frame to the body frame.
    // The z axis of the body frame should point upwards.
    // Normally, this transform should be identity.
    static Eigen::Isometry3d T_imu_body;

    IMUState() : id(0), 
                 time(0), 
                 orientation(Eigen::Matrix3d::Identity()), 
                 position(Eigen::Vector3d::Zero()), 
                 velocity(Eigen::Vector3d::Zero()), 
                 gyro_bias(Eigen::Vector3d::Zero()), 
                 acc_bias(Eigen::Vector3d::Zero()) 
                 {}

    IMUState(const StateIDType& new_id) : 
                id(new_id), 
                time(0), 
                orientation(Eigen::Matrix3d::Identity()), 
                position(Eigen::Vector3d::Zero()),
                velocity(Eigen::Vector3d::Zero()), 
                gyro_bias(Eigen::Vector3d::Zero()), 
                acc_bias(Eigen::Vector3d::Zero()) 
                {}
};

typedef IMUState::StateIDType StateIDType;

/*
 * @brief CAMState Stored camera state in order to form measurement model.
 */
struct CAMState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // An unique identifier for the CAM state.
    StateIDType id;

    // Time when the state is recorded
    double time;

    // Orientation Take a vector from the camera frame to the world frame.
    Eigen::Matrix3d orientation;

    // First estimated Jacobian of orientation;
    Eigen::Matrix3d orientation_FEJ;

    // Position of the camera frame in the world frame.
    Eigen::Vector3d position;

    // First estimated Jacobian of position;
    Eigen::Vector3d position_FEJ;

    // Takes a vector from the cam0 frame to the cam1 frame.
    static Eigen::Isometry3d T_cam0_cam1;

    CAMState() : 
            id(0), 
            time(0), 
            orientation(Eigen::Matrix3d::Identity()), 
            orientation_FEJ(Eigen::Matrix3d::Identity()), 
            position(Eigen::Vector3d::Zero()), 
            position_FEJ(Eigen::Vector3d::Zero())
            {}

    CAMState(const StateIDType& new_id) : 
            id(new_id), 
            time(0), 
            orientation(Eigen::Matrix3d::Identity()), 
            orientation_FEJ(Eigen::Matrix3d::Identity()), 
            position(Eigen::Vector3d::Zero()), 
            position_FEJ(Eigen::Vector3d::Zero()) 
            {}
};

typedef std::map<StateIDType, CAMState, std::less<int>,
                 Eigen::aligned_allocator<std::pair<const StateIDType, CAMState> > >
    CamStateServer;

}  // namespace orcvio

