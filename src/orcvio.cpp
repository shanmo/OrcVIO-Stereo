/*
 * COPYRIGHT AND PERMISSION NOTICE
 * UCSD Software ORCVIO
 * Copyright (C) 2021 
 * All rights reserved.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <iterator>
#include <algorithm>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/math/distributions/chi_squared.hpp>

#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/SparseCore>
#include <Eigen/SPQRSupport>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include "orcvio/orcvio.h"
#include "orcvio/utils.h"

using namespace std;
using namespace Eigen;

namespace orcvio{

// Static member variables in IMUState class.
StateIDType IMUState::next_id = 0;
double IMUState::gyro_noise = 0.001;
double IMUState::acc_noise = 0.01;
double IMUState::gyro_bias_noise = 0.001;
double IMUState::acc_bias_noise = 0.01;
Vector3d IMUState::gravity = Vector3d(0, 0, -GRAVITY_ACCELERATION);
Isometry3d IMUState::T_imu_body = Isometry3d::Identity();

// Static member variables in CAMState class.
Isometry3d CAMState::T_cam0_cam1 = Isometry3d::Identity();

// Static member variables in Feature class.
FeatureIDType Feature::next_id = 0;
double Feature::observation_noise = 0.01;
Feature::OptimizationConfig Feature::optimization_config;

map<int, double> OrcVio::chi_squared_test_table;

OrcVio::OrcVio(ros::NodeHandle& pnh): 
    is_gravity_set(false), 
    use_FEJ_flag(true),
    is_first_img(true), 
    nh(pnh) 
    { return; }

OrcVio::~OrcVio() noexcept
{
    pose_outfile.close();
}

bool OrcVio::loadParameters() {
    // Frame id
    nh.param<string>("fixed_frame_id", fixed_frame_id, "global");
    nh.param<string>("child_frame_id", child_frame_id, "robot");
    nh.param<bool>("publish_tf", publish_tf, true);
    nh.param<double>("position_std_threshold", position_std_threshold, 8.0);

    // for saving the pose and log 
    nh.param<std::string>("output_dir_traj", output_dir_traj, "./cache/");
    nh.param<std::string>("output_dir_log", output_dir_log, "./cache/");

    nh.param<double>("rotation_threshold", rotation_threshold, 0.2618);
    nh.param<double>("translation_threshold", translation_threshold, 0.4);
    nh.param<double>("tracking_rate_threshold", tracking_rate_threshold, 0.5);

    // Feature optimization parameters
    nh.param<double>("feature/config/translation_threshold", Feature::optimization_config.translation_threshold, 0.2);

    // Noise related parameters
    nh.param<double>("noise/gyro", IMUState::gyro_noise, 0.001);
    nh.param<double>("noise/acc", IMUState::acc_noise, 0.01);
    nh.param<double>("noise/gyro_bias", IMUState::gyro_bias_noise, 0.001);
    nh.param<double>("noise/acc_bias", IMUState::acc_bias_noise, 0.01);
    nh.param<double>("noise/feature", Feature::observation_noise, 0.01);

    // Use variance instead of standard deviation.
    IMUState::gyro_noise *= IMUState::gyro_noise;
    IMUState::acc_noise *= IMUState::acc_noise;
    IMUState::gyro_bias_noise *= IMUState::gyro_bias_noise;
    IMUState::acc_bias_noise *= IMUState::acc_bias_noise;
    Feature::observation_noise *= Feature::observation_noise;

    // Set the initial IMU state.
    // The intial orientation and position will be set to the origin
    // implicitly. But the initial velocity and bias can be set by parameters.
    // TODO: is it reasonable to set the initial bias to 0?
    nh.param<double>("initial_state/velocity/x", state_server.imu_state.velocity(0), 0.0);
    nh.param<double>("initial_state/velocity/y", state_server.imu_state.velocity(1), 0.0);
    nh.param<double>("initial_state/velocity/z", state_server.imu_state.velocity(2), 0.0);

    // The initial covariance of orientation and position can be
    // set to 0. But for velocity, bias and extrinsic parameters,
    // there should be nontrivial uncertainty.
    double gyro_bias_cov, acc_bias_cov, velocity_cov;
    nh.param<double>("initial_covariance/velocity", velocity_cov, 0.25);
    nh.param<double>("initial_covariance/gyro_bias", gyro_bias_cov, 1e-4);
    nh.param<double>("initial_covariance/acc_bias", acc_bias_cov, 1e-2);

    double extrinsic_rotation_cov, extrinsic_translation_cov;
    nh.param<double>("initial_covariance/extrinsic_rotation_cov", extrinsic_rotation_cov, 3.0462e-4);
    nh.param<double>("initial_covariance/extrinsic_translation_cov", extrinsic_translation_cov, 1e-4);

    state_server.state_cov = MatrixXd::Zero(21, 21);
    for (int i = 3; i < 6; ++i)
        state_server.state_cov(i, i) = velocity_cov;
    for (int i = 9; i < 12; ++i)
        state_server.state_cov(i, i) = gyro_bias_cov;
    for (int i = 12; i < 14; ++i)
        state_server.state_cov(i, i) = acc_bias_cov;
    for (int i = 15; i < 18; ++i)
        state_server.state_cov(i, i) = extrinsic_translation_cov;
    for (int i = 18; i < 21; ++i)
        state_server.state_cov(i, i) = extrinsic_rotation_cov;

    // Transformation offsets between the frames involved.
    Isometry3d T_imu_cam0 = utils::getTransformEigen(nh, "cam0/T_cam_imu");
    Isometry3d T_cam0_imu = T_imu_cam0.inverse();

    state_server.imu_state.R_imu_cam0 = T_cam0_imu.linear().transpose();
    state_server.imu_state.t_cam0_imu = T_cam0_imu.translation();
    CAMState::T_cam0_cam1 = utils::getTransformEigen(nh, "cam1/T_cn_cnm1");
    IMUState::T_imu_body = utils::getTransformEigen(nh, "T_imu_body").inverse();

    // Maximum number of camera states to be stored
    nh.param<int>("max_cam_state_size", max_cam_state_size, 30);

    // whether to only use mono residual for testing 
    nh.param<bool>("use_mono_flag", use_mono_flag, false);

    ROS_INFO("OrcVio begin ===========================================");

    ROS_INFO("fixed frame id: %s", fixed_frame_id.c_str());
    ROS_INFO("child frame id: %s", child_frame_id.c_str());
    ROS_INFO("publish tf: %d", publish_tf);
    ROS_INFO("position std threshold: %f", position_std_threshold);
    ROS_INFO("Keyframe rotation threshold: %f", rotation_threshold);
    ROS_INFO("Keyframe translation threshold: %f", translation_threshold);
    ROS_INFO("Keyframe tracking rate threshold: %f", tracking_rate_threshold);
    ROS_INFO("gyro noise: %.10f", IMUState::gyro_noise);
    ROS_INFO("gyro bias noise: %.10f", IMUState::gyro_bias_noise);
    ROS_INFO("acc noise: %.10f", IMUState::acc_noise);
    ROS_INFO("acc bias noise: %.10f", IMUState::acc_bias_noise);
    ROS_INFO("observation noise: %.10f", Feature::observation_noise);
    ROS_INFO("initial velocity: %f, %f, %f",
             state_server.imu_state.velocity(0),
             state_server.imu_state.velocity(1),
             state_server.imu_state.velocity(2));
    ROS_INFO("initial gyro bias cov: %f", gyro_bias_cov);
    ROS_INFO("initial acc bias cov: %f", acc_bias_cov);
    ROS_INFO("initial velocity cov: %f", velocity_cov);
    ROS_INFO("initial extrinsic rotation cov: %f", extrinsic_rotation_cov);
    ROS_INFO("initial extrinsic translation cov: %f", extrinsic_translation_cov);

    cout << T_imu_cam0.linear() << endl;
    cout << T_imu_cam0.translation().transpose() << endl;

    ROS_INFO("max camera state #: %d", max_cam_state_size);

    std::cout << "T_imu_cam0:\n" << T_imu_cam0.matrix() << std::endl;
    std::cout << "T_cam0_cam1:\n" << CAMState::T_cam0_cam1.matrix() << std::endl;
    std::cout << "T_imu_body:\n" << IMUState::T_imu_body.matrix() << std::endl;

    ROS_INFO("OrcVio end ===========================================");

    return true;
}

bool OrcVio::createRosIO() noexcept {
    path_pub = nh.advertise<nav_msgs::Path>("path", 10);
    odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 10);
    posestamped_pub = nh.advertise<geometry_msgs::PoseStamped>("posestamped", 10);
    feature_pub = nh.advertise<sensor_msgs::PointCloud2>("feature_point_cloud", 10);

    reset_srv = nh.advertiseService("reset", &OrcVio::resetCallback, this);

    imu_sub = nh.subscribe("imu", 100, &OrcVio::imuCallback, this);
    feature_sub = nh.subscribe("features", 40, &OrcVio::featureCallback, this);

    mocap_odom_sub = nh.subscribe("mocap_odom", 10, &OrcVio::mocapOdomCallback, this);
    mocap_odom_pub = nh.advertise<nav_msgs::Odometry>("gt_odom", 1);

#if WITH_LC
    // TODO, should these be 2 or 1000? 
    pub_poseimu = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/ov_msckf/poseimu", 2);
    pub_keyframe_pose = nh.advertise<nav_msgs::Odometry>("/ov_msckf/keyframe_pose", 2);
    pub_keyframe_point = nh.advertise<sensor_msgs::PointCloud>("/ov_msckf/keyframe_feats", 2);    
    pub_keyframe_extrinsic = nh.advertise<nav_msgs::Odometry>("/ov_msckf/keyframe_extrinsic", 2);
    pub_keyframe_intrinsics = nh.advertise<sensor_msgs::CameraInfo>("/ov_msckf/keyframe_intrinsics", 2);
#endif    

    if (!boost::filesystem::exists(output_dir_traj))
    {
        // if this dir does not exist, create the dir
        const char *path = output_dir_traj.c_str();
        boost::filesystem::path dir(path);
        if (boost::filesystem::create_directories(dir))
        {
            std::cerr << "Directory Created: " << output_dir_traj << std::endl;
        }
    }
    pose_outfile.open((output_dir_traj+"stamped_traj_estimate.txt").c_str(), std::ofstream::trunc);
    
#if WITH_LOG
    // for logging 
    logger = spdlog::basic_logger_mt("orcvio_logger", output_dir_log+"log.txt", true);
    spdlog::set_level(spdlog::level::debug);
    // spdlog::set_level(spdlog::level::info);
    logger->flush_on(spdlog::level::debug);
    spdlog::set_pattern("[source %s] [function %!] [line %#] %v");
#endif  

    return true;
}

bool OrcVio::initialize() {
    if (!loadParameters())
        return false;
    ROS_INFO("Finish loading ROS parameters...");

    // Initialize state server
    Matrix<double, 21, 21> continuous_noise_cov =
      Matrix<double, 21, 21>::Zero();
    continuous_noise_cov.block<3, 3>(0, 0) =
      Matrix3d::Identity()*IMUState::gyro_noise;
    continuous_noise_cov.block<3, 3>(3, 3) =
      Matrix3d::Identity()*IMUState::acc_noise;
    continuous_noise_cov.block<3, 3>(9, 9) =
      Matrix3d::Identity()*IMUState::gyro_bias_noise;
    continuous_noise_cov.block<3, 3>(12, 12) =
      Matrix3d::Identity()*IMUState::acc_bias_noise;
    state_server.continuous_noise_cov = continuous_noise_cov;

    // initialize initializer
    flexInitPtr.reset(new FlexibleInitializer(sqrt(IMUState::acc_noise), sqrt(IMUState::acc_bias_noise),
        sqrt(IMUState::gyro_noise), sqrt(IMUState::gyro_bias_noise),
        state_server.imu_state.R_imu_cam0.transpose(),
        state_server.imu_state.t_cam0_imu));

    // Initialize the chi squared test table with confidence level 0.95.
    for (int i = 1; i < 100; ++i) {
        boost::math::chi_squared chi_squared_dist(i);
        chi_squared_test_table[i] =
        boost::math::quantile(chi_squared_dist, 0.05);
    }

    if (!createRosIO())
        return false;
    ROS_INFO("Finish creating ROS IO...");

    return true;
}

void OrcVio::imuCallback(const sensor_msgs::ImuConstPtr& msg) noexcept {
    // IMU msgs are pushed backed into a buffer instead of being processed immediately.
    // The IMU msgs are processed when the next image is available, in which way, we can easily handle the transfer delay.
    
    imu_msg_buffer.emplace_back(*msg);

    return;
}

bool OrcVio::resetCallback(std_srvs::Trigger::Request& req,std_srvs::Trigger::Response& res) {

    ROS_WARN("Start resetting orcvio...");
    // Temporarily shutdown the subscribers to prevent the
    // state from updating.
    feature_sub.shutdown();
    imu_sub.shutdown();

    // Reset the IMU state.
    IMUState &imu_state = state_server.imu_state;
    imu_state.time = 0.0;
    imu_state.orientation = Matrix3d::Identity();
    imu_state.position = Vector3d::Zero();
    imu_state.velocity = Vector3d::Zero();
    imu_state.gyro_bias = Vector3d::Zero();
    imu_state.acc_bias = Vector3d::Zero();


    // Remove all existing camera states.
    state_server.cam_states.clear();

    // Reset the state covariance.
    double gyro_bias_cov, acc_bias_cov, velocity_cov;
    nh.param<double>("initial_covariance/velocity", velocity_cov, 0.25);
    nh.param<double>("initial_covariance/gyro_bias", gyro_bias_cov, 1e-4);
    nh.param<double>("initial_covariance/acc_bias", acc_bias_cov, 1e-2);

    double extrinsic_rotation_cov, extrinsic_translation_cov;
    nh.param<double>("initial_covariance/extrinsic_rotation_cov", extrinsic_rotation_cov, 3.0462e-4);
    nh.param<double>("initial_covariance/extrinsic_translation_cov", extrinsic_translation_cov, 1e-4);

    state_server.state_cov = MatrixXd::Zero(21, 21);
    for (int i = 3; i < 6; ++i)
        state_server.state_cov(i, i) = velocity_cov;
    for (int i = 9; i < 12; ++i)
        state_server.state_cov(i, i) = gyro_bias_cov;
    for (int i = 12; i < 15; ++i)
        state_server.state_cov(i, i) = acc_bias_cov;
    for (int i = 15; i < 18; ++i)
        // in fact this is rho, not translation 
        state_server.state_cov(i, i) = extrinsic_translation_cov;
    for (int i = 18; i < 21; ++i)
        state_server.state_cov(i, i) = extrinsic_rotation_cov;

    // Clear all exsiting features in the map.
    map_server.clear();

    // Clear the IMU msg buffer.
    imu_msg_buffer.clear();

    // Reset the starting flags.
    is_gravity_set = false;
    is_first_img = true;

    // Restart the subscribers.
    imu_sub = nh.subscribe("imu", 100, &OrcVio::imuCallback, this);
    feature_sub = nh.subscribe("features", 40, &OrcVio::featureCallback, this);

    // TODO: When can the reset fail?
    res.success = true;
    ROS_WARN("Resetting orcvio completed...");
    return true;
}

void OrcVio::featureCallback(const CameraMeasurementConstPtr& msg) {

    // features are not utilized until receiving imu msgs ahead
    if (!bFirstFeatures) {
        if ((imu_msg_buffer.begin() != imu_msg_buffer.end()) &&
            (imu_msg_buffer.begin()->header.stamp.toSec() - msg->header.stamp.toSec() <= 0.0))
            bFirstFeatures = true;
        else
            return;
    }

    if (!is_gravity_set) {
        if (flexInitPtr->tryIncInit(imu_msg_buffer, msg, state_server.imu_state))
        {
            if (use_FEJ_flag)
                // Update FEJ imu state
                state_server.imu_state_FEJ = state_server.imu_state;
                
            is_gravity_set = true;
        }
    }

    // Return if the gravity vector has not been set.
    if (!is_gravity_set)
        return;

    // Start the system if the first image is received.
    // The frame where the first image is received will be the origin.
    if (is_first_img) {
        is_first_img = false;
        state_server.imu_state.time = msg->header.stamp.toSec();
    }

    static int critical_time_cntr = 0;
    double processing_start_time = ros::Time::now().toSec();

    // Propogate the IMU state. that are received before the image msg.
#if WITH_LOG
    ros::Time start_time = ros::Time::now();
#endif 
    batchImuProcessing(msg->header.stamp.toSec());
#if WITH_LOG
    double imu_processing_time = (ros::Time::now() - start_time).toSec();
#endif 

    // Augment the state vector.
#if WITH_LOG
    start_time = ros::Time::now();
#endif 
    stateAugmentation(msg->header.stamp.toSec());
#if WITH_LOG
    double state_augmentation_time = (ros::Time::now() - start_time).toSec();
#endif 

    // Add new observations for existing features or new features in the map server.
#if WITH_LOG
    start_time = ros::Time::now();
#endif 
    addFeatureObservations(msg);
#if WITH_LOG
    double add_observations_time = (ros::Time::now() - start_time).toSec();
#endif 

    // Perform measurement update if necessary.
#if WITH_LOG
    start_time = ros::Time::now();
#endif 
    removeLostFeatures();
#if WITH_LOG
    double remove_lost_features_time = (ros::Time::now() - start_time).toSec();
#endif 

#if WITH_LOG
    start_time = ros::Time::now();
#endif 
    pruneCamStateBuffer();
#if WITH_LOG
    double prune_cam_states_time = (ros::Time::now() - start_time).toSec();
#endif 

#if WITH_LC
    std::vector<FeatureLcPtr> featuresLC;
    for(const auto &feat : featuresLCDB) featuresLC.emplace_back(feat.second);
    featuresLCDB.clear();
    update_keyframe_historical_information(featuresLC);
    publish_keyframe_information();
#endif    

    // Publish the odometry.
#if WITH_LOG
    start_time = ros::Time::now();
#endif 
    publish(msg->header.stamp);
#if WITH_LOG
    double publish_time = (ros::Time::now() - start_time).toSec();
#endif 

    // Reset the system if necessary.
    // onlineReset();

#if WITH_LOG
    double processing_end_time = ros::Time::now().toSec();
    double processing_time = processing_end_time - processing_start_time;
    // This variable is only used to determine the timing threshold of
    // each iteration of the filter.
    const int frame_rate = 20; 
    if (processing_time > 1.0 / frame_rate) {
        ++critical_time_cntr;
        SPDLOG_LOGGER_DEBUG(logger, "Total processing time {}/{}", processing_time, critical_time_cntr);
        SPDLOG_LOGGER_DEBUG(logger, "IMU processing time: {}/{}", imu_processing_time, imu_processing_time/processing_time);
        SPDLOG_LOGGER_DEBUG(logger, "State augmentation time: {}/{}", state_augmentation_time, state_augmentation_time/processing_time);
        SPDLOG_LOGGER_DEBUG(logger, "Add observations time: {}/{}", add_observations_time, add_observations_time/processing_time);
        SPDLOG_LOGGER_DEBUG(logger, "Remove lost features time: {}/{}", remove_lost_features_time, remove_lost_features_time / processing_time);
        SPDLOG_LOGGER_DEBUG(logger, "Remove camera states time: {}/{}", prune_cam_states_time, prune_cam_states_time / processing_time);
        SPDLOG_LOGGER_DEBUG(logger, "Publish time: {}/{}", publish_time, publish_time/processing_time);  
    }
#endif   

    return;
}

void OrcVio::mocapOdomCallback(const nav_msgs::OdometryConstPtr& msg) {
  static bool first_mocap_odom_msg = true;

  // If this is the first mocap odometry messsage, set the initial frame.
  if (first_mocap_odom_msg) {
    Quaterniond orientation;
    Vector3d translation;
    tf::pointMsgToEigen(msg->pose.pose.position, translation);
    tf::quaternionMsgToEigen(msg->pose.pose.orientation, orientation);
    //tf::vectorMsgToEigen(msg->transform.translation, translation);
    //tf::quaternionMsgToEigen(msg->transform.rotation, orientation);
    mocap_initial_frame.linear() = orientation.toRotationMatrix();
    mocap_initial_frame.translation() = translation;
    first_mocap_odom_msg = false;
  }

  // Transform the ground truth.
  Quaterniond orientation;
  Vector3d translation;
  //tf::vectorMsgToEigen(msg->transform.translation, translation);
  //tf::quaternionMsgToEigen(msg->transform.rotation, orientation);
  tf::pointMsgToEigen(msg->pose.pose.position, translation);
  tf::quaternionMsgToEigen(msg->pose.pose.orientation, orientation);

  Eigen::Isometry3d T_b_v_gt;
  T_b_v_gt.linear()      = orientation.toRotationMatrix();
  T_b_v_gt.translation() = translation;
  Eigen::Isometry3d T_b_w_gt = mocap_initial_frame.inverse() * T_b_v_gt;

  //Eigen::Vector3d body_velocity_gt;
  //tf::vectorMsgToEigen(msg->twist.twist.linear, body_velocity_gt);
  //body_velocity_gt = mocap_initial_frame.linear().transpose() * body_velocity_gt;

  // Ground truth tf.
  if (publish_tf) {
    tf::Transform T_b_w_gt_tf;
    tf::transformEigenToTF(T_b_w_gt, T_b_w_gt_tf);
    tf_pub.sendTransform(tf::StampedTransform(T_b_w_gt_tf, msg->header.stamp, fixed_frame_id, child_frame_id+"_mocap"));
  }

  // Ground truth odometry.
  nav_msgs::Odometry mocap_odom_msg;
  mocap_odom_msg.header.stamp = msg->header.stamp;
  mocap_odom_msg.header.frame_id = fixed_frame_id;
  mocap_odom_msg.child_frame_id = child_frame_id+"_mocap";

  tf::poseEigenToMsg(T_b_w_gt, mocap_odom_msg.pose.pose);
  //tf::vectorEigenToMsg(body_velocity_gt, mocap_odom_msg.twist.twist.linear);

  mocap_odom_pub.publish(mocap_odom_msg);
  return;
}

void OrcVio::batchImuProcessing(const double& time_bound) {

    auto iterator = std::remove_if(imu_msg_buffer.begin(), imu_msg_buffer.end(), 
        [this, &time_bound](const auto &imu_msg) -> bool
        {
            bool used_imu_msg_flag = false; 
            double imu_time = imu_msg.header.stamp.toSec();
            if (imu_time >= state_server.imu_state.time && imu_time <= time_bound)
            {
                used_imu_msg_flag = true; 
                
                // Convert the msgs.
                Vector3d m_gyro, m_acc;
                tf::vectorMsgToEigen(imu_msg.angular_velocity, m_gyro);
                tf::vectorMsgToEigen(imu_msg.linear_acceleration, m_acc);

                // Execute process model.
                processModel(imu_time, m_gyro, m_acc);
            }
            return used_imu_msg_flag;
        });

    // Remove all used IMU msgs.
    imu_msg_buffer.erase(iterator, imu_msg_buffer.end());

    // Set the state ID for the new IMU state.
    state_server.imu_state.id = IMUState::next_id++;

    return;
}

void OrcVio::processModel(const double& time, const Vector3d& m_gyro, const Vector3d& m_acc) {
    
    // Remove the bias from the measured gyro and acceleration
    IMUState &imu_state = state_server.imu_state;
    double dtime  = time - imu_state.time;

    Matrix<double, 3, 1> gyro_hat = m_gyro - imu_state.gyro_bias;
    Matrix<double, 3, 1> acc_hat  = m_acc  - imu_state.acc_bias;

    // Propogate the state using closed form method 
    predictNewState(dtime, gyro_hat, acc_hat);

    // propagate the covariance 
    // state is R v p bg ba 
    // Compute error state transition matrix
    MatrixXd Phi;
    MatrixXd Q;

    // use right perturbation 
    // Matrix3d wRi = state_server.imu_state.orientation;
    Matrix3d wRi = (use_FEJ_flag ?
        state_server.imu_state_FEJ.orientation :
        state_server.imu_state.orientation);

    // prepare the basic terms 

    Matrix3d a_skew = utils::skewSymmetric(acc_hat);
    Matrix3d g_skew = utils::skewSymmetric(gyro_hat);
    double g_norm = gyro_hat.norm();

    Matrix3d theta_theta = Sophus::SO3d::exp(-dtime * gyro_hat).matrix();
    Matrix3d JL_plus = utils::Jl_operator(dtime * gyro_hat);
    Matrix3d JL_minus = utils::Jl_operator(-dtime * gyro_hat); 
    Matrix3d I3 = Matrix3d::Identity();
    Matrix3d Delta = -(g_skew / pow(g_norm, 2)) * (theta_theta.transpose() * (dtime * g_skew - I3) + I3);
    Matrix3d HL_plus = utils::Hl_operator(dtime * gyro_hat);
    Matrix3d HL_minus = utils::Hl_operator(-dtime * gyro_hat);

    // prepare the terms in Phi 
    Matrix3d theta_gyro = -dtime * JL_minus;

    Matrix3d v_theta = -dtime * wRi * utils::skewSymmetric(JL_plus * acc_hat);
    Matrix3d v_gyro = wRi * Delta * a_skew * (I3 + (g_skew * g_skew / pow(g_norm, 2))) + dtime * wRi * JL_plus * (a_skew * g_skew / pow(g_norm, 2)) + dtime * wRi * (gyro_hat * acc_hat.transpose() / pow(g_norm, 2)) * JL_minus - dtime * ((acc_hat.transpose() * gyro_hat).value() / pow(g_norm, 2)) * I3;
    Matrix3d v_acc = -dtime * wRi * JL_plus;

    Matrix3d p_theta = -pow(dtime, 2) * wRi * utils::skewSymmetric(HL_plus * acc_hat);
    Matrix3d p_v = dtime * I3; 
    Matrix3d p_gyro = wRi * (-g_skew * Delta - dtime * JL_plus + dtime * I3) * a_skew * (I3 + (g_skew * g_skew / pow(g_norm, 2))) * (g_skew / pow(g_norm, 2)) + pow(dtime, 2) * wRi * HL_plus * (a_skew * g_skew / pow(g_norm, 2)) + pow(dtime, 2) * wRi * (gyro_hat * acc_hat.transpose() / pow(g_norm, 2)) * HL_minus - pow(dtime, 2) * ((acc_hat.transpose() * gyro_hat).value() / (2*pow(g_norm, 2))) * wRi; 
    Matrix3d p_acc = -pow(dtime, 2) * wRi * HL_plus; 

    // for phi full 
    Phi = MatrixXd::Identity(21, 21);
    
    // theta row 
    Phi.block<3, 3>(0, 0) = theta_theta;
    Phi.block<3, 3>(0, 9) = theta_gyro;

    // v row 
    Phi.block<3, 3>(3, 0) = v_theta;  
    Phi.block<3, 3>(3, 9) = v_gyro;
    Phi.block<3, 3>(3, 12) = v_acc;

    // p row 
    Phi.block<3, 3>(6, 0) = p_theta;
    Phi.block<3, 3>(6, 3) = p_v;
    Phi.block<3, 3>(6, 9) = p_gyro;
    Phi.block<3, 3>(6, 12) = p_acc;

    // obtain Q 
    Q = Phi*state_server.continuous_noise_cov*Phi.transpose()*dtime;

    // update state covaraince 
    state_server.state_cov.block<21, 21>(0, 0) = Phi * state_server.state_cov.block<21, 21>(0, 0) * Phi.transpose() + Q;
    if (state_server.cam_states.size() > 0) {
        state_server.state_cov.block(0, 21, 21, state_server.state_cov.cols() - 21) =
                Phi * state_server.state_cov.block(0, 21, 21, state_server.state_cov.cols() - 21);
        state_server.state_cov.block(21, 0, state_server.state_cov.rows() - 21, 21) =
                state_server.state_cov.block(21, 0, state_server.state_cov.rows() - 21, 21) * Phi.transpose();
    }

    // Fix the covariance to be symmetric
    MatrixXd state_cov_fixed = (state_server.state_cov + state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;

    // Update the state info
    state_server.imu_state.time = time;

    return;
}

void OrcVio::predictNewState(const double& dt, const Vector3d& gyro, const Vector3d& acc) noexcept {

    // need to use reference since we need to change these values 
    Matrix3d& R = state_server.imu_state.orientation;
    Vector3d& v = state_server.imu_state.velocity;
    Vector3d& p = state_server.imu_state.position;

    // update position 
    Matrix3d Hl = utils::Hl_operator(dt*gyro);
    p = p + dt*v + IMUState::gravity*(pow(dt, 2)/2) + R*Hl*acc*pow(dt, 2);

    // update velocity
    Matrix3d Jl = utils::Jl_operator(dt*gyro);
    v = v + IMUState::gravity*dt + R*Jl*acc*dt;

    // update rotation
    Sophus::SO3d R_temp = Sophus::SO3d::exp(dt*gyro);
    R = R*R_temp.matrix();

    if (use_FEJ_flag)
        // Update FEJ imu state 
        state_server.imu_state_FEJ = state_server.imu_state;

    return;
}



void OrcVio::stateAugmentation(const double& time) {

    // clone a camera state from the current IMU state 
    // cRi is from IMU to camera frame 
    const Matrix3d &cRi = state_server.imu_state.R_imu_cam0;
    const Vector3d &iPc = state_server.imu_state.t_cam0_imu;

    // Add a new camera state to the state server.
    const Matrix3d &wRi = state_server.imu_state.orientation;
    const Vector3d &wPi = state_server.imu_state.position;
    Matrix3d wRc = wRi * cRi.transpose();
    Vector3d wPc = wPi + wRi * iPc;

    state_server.cam_states[state_server.imu_state.id] = CAMState(state_server.imu_state.id);
    CAMState &cam_state = state_server.cam_states[state_server.imu_state.id];

    cam_state.time              = time;
    cam_state.orientation       = wRc;
    cam_state.position          = wPc;
    if (use_FEJ_flag)
    {
        cam_state.orientation_FEJ   = wRc;
        cam_state.position_FEJ      = wPc;
    }

    // Update the covariance matrix of the state.
    Matrix<double, 6, 21> J = Matrix<double, 6, 21>::Zero();

    Eigen::Matrix<double, 6, 6> dcampose_dimupose = Eigen::Matrix<double, 6, 6>::Zero();
    dcampose_dimupose = utils::get_cam_wrt_imu_se3_jacobian(cRi, iPc, wRc.transpose());

    // rho c wrt theta i 
    J.block(0, 0, 3, 3) = dcampose_dimupose.block(0, 0, 3, 3);
    // rho c wrt p i
    J.block(0, 6, 3, 3) = dcampose_dimupose.block(0, 3, 3, 3);
    // theta c wrt theta i
    J.block(3, 0, 3, 3) = dcampose_dimupose.block(3, 0, 3, 3);
    // theta c wrt p i
    J.block(3, 6, 3, 3) = dcampose_dimupose.block(3, 3, 3, 3);

    // rho c wrt extrinsics rho 
    J.block(0, 15, 3, 3) = Matrix3d::Identity();
    // theta c wrt extrinsics theta 
    J.block(3, 18, 3, 3) = Matrix3d::Identity();

    // Resize the state covariance matrix.
    // Get old size
    size_t old_rows = state_server.state_cov.rows();
    size_t old_cols = state_server.state_cov.cols();
    state_server.state_cov.conservativeResize(old_rows+6, old_cols+6);

    // Rename some matrix blocks for convenience.
    const Matrix<double, 21, 21> &P11 = state_server.state_cov.block<21, 21>(0, 0);
    const MatrixXd &P12 = state_server.state_cov.block(0, 21, 21, old_cols - 21);

    // Fill in the augmented state covariance.
    state_server.state_cov.block(old_rows, 0, 6, old_cols) << J * P11, J * P12;
    state_server.state_cov.block(0, old_cols, old_rows, 6) =
            state_server.state_cov.block(old_rows, 0, 6, old_cols).transpose();
    state_server.state_cov.block<6, 6>(old_rows, old_cols) = J * P11 * J.transpose();

    // Fix the covariance to be symmetric
    MatrixXd state_cov_fixed = (state_server.state_cov + state_server.state_cov.transpose()) / 2.0;
    state_server.state_cov = state_cov_fixed;

    return;
}

void OrcVio::addFeatureObservations(const CameraMeasurementConstPtr& msg) noexcept {
    StateIDType state_id = state_server.imu_state.id;
    int curr_feature_num = map_server.size();
    int tracked_feature_num = 0;

    // imu_state.id <--> feature.id
    // Add new observations for existing features or new features in the map server.
    auto addFeature = [this, &tracked_feature_num, &msg, &state_id](const auto& feature) { 
        if (map_server.find(feature.id) == map_server.end()) {
            // This is a new feature.
            map_server[feature.id] = Feature(feature.id);
            map_server[feature.id].observations[state_id] = Vector4d(feature.u0, feature.v0, feature.u1, feature.v1);
#if WITH_LC            
            map_server[feature.id].observations_uvs[state_id] = Vector4d(feature.pu0, feature.pv0, feature.pu1, feature.pv1);
            map_server[feature.id].timestamp[state_id] = msg->header.stamp.toSec();
#endif            
        } else {
            // This is an old feature.
            map_server[feature.id].observations[state_id] = Vector4d(feature.u0, feature.v0, feature.u1, feature.v1);
#if WITH_LC            
            map_server[feature.id].observations_uvs[state_id] = Vector4d(feature.pu0, feature.pv0, feature.pu1, feature.pv1);
            map_server[feature.id].timestamp[state_id] = msg->header.stamp.toSec();
#endif            
            ++tracked_feature_num;
        }
    };
    std::for_each(msg->features.begin(), msg->features.end(), addFeature);

    tracking_rate = static_cast<double>(tracked_feature_num) / static_cast<double>(curr_feature_num);

    return;
}

void OrcVio::measurementJacobian(
    const StateIDType& cam_state_id, const FeatureIDType& feature_id,
    Matrix<double, 4, 6>& H_x, 
    Matrix<double, 4, 6>& H_e,
    Matrix<double, 4, 3>& H_f, 
    Vector4d& r) {

    // Prepare all the required data.
    const CAMState &cam_state = state_server.cam_states[cam_state_id];
    const Feature &feature = map_server[feature_id];

    // Cam0 pose.
    const Matrix3d &wRc0 = cam_state.orientation;
    const Vector3d &wPc0 = cam_state.position;

    // Cam1 pose.
    // T_cam0_cam1 is from camera 0 to camera 1 
    const Matrix3d &c1Rc0 = CAMState::T_cam0_cam1.linear();
    const Matrix3d &c1Rw  = CAMState::T_cam0_cam1.linear() * wRc0.transpose();
    const Vector3d &wPc1  = wPc0 - c1Rw.transpose() * CAMState::T_cam0_cam1.translation();

    // 3d feature position in the world frame. And its observation with the stereo cameras.
    const Vector3d &p_w = feature.position;
    const Vector4d &z = feature.observations.find(cam_state_id)->second;

    // Convert the feature position from the world frame to the cam0 and cam1 frame.
    const Vector3d &p_c0 = wRc0.transpose() * (p_w - wPc0);
    const Vector3d &p_c1 = c1Rw * (p_w - wPc1);

    // for fej 
    const Vector3d &p_c0_jacobi = (use_FEJ_flag ? 
            cam_state.orientation_FEJ.transpose() * (p_w - cam_state.position_FEJ) :
            p_c0
        );

    // Compute the Jacobians of the reprojection error wrt the state 

    // jacobian of inhomo coordinate wrt homo coordinate 
    Eigen::Matrix<double, 3, 4> temp_mat = Eigen::Matrix<double, 3, 4>::Zero();
    temp_mat.leftCols(3) = Eigen::Matrix<double, 3, 3>::Identity();


    // for camera 0 
    Matrix<double, 4, 3> dz_dpc0 = Matrix<double, 4, 3>::Zero();
    dz_dpc0(0, 0) = 1 / p_c0(2);
    dz_dpc0(1, 1) = 1 / p_c0(2);
    dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2) * p_c0(2));
    dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2) * p_c0(2));

    Matrix<double, 3, 6> dpc0_dxc = Matrix<double, 3, 6>::Zero();
    Eigen::Vector4d uline_l0 = Eigen::Vector4d::Zero();
    uline_l0(3) = 1;
    uline_l0.head(3) = p_c0_jacobi;
    dpc0_dxc = -1 * temp_mat * utils::odotOperator(uline_l0);


    // for camera 1 
    Matrix<double, 4, 3> dz_dpc1 = Matrix<double, 4, 3>::Zero();
    dz_dpc1(2, 0) = 1 / p_c1(2);
    dz_dpc1(3, 1) = 1 / p_c1(2);
    dz_dpc1(2, 2) = -p_c1(0) / (p_c1(2) * p_c1(2));
    dz_dpc1(3, 2) = -p_c1(1) / (p_c1(2) * p_c1(2));

    Matrix<double, 3, 6> dpc1_dxc = Matrix<double, 3, 6>::Zero();
    dpc1_dxc = -1 * temp_mat * CAMState::T_cam0_cam1.matrix() * utils::odotOperator(uline_l0);


    H_x = dz_dpc0 * dpc0_dxc + dz_dpc1 * dpc1_dxc;

    // compute jacobian wrt extrinsic 
    Eigen::Matrix<double, 6, 6> dcampose_dextpose = Eigen::Matrix<double, 6, 6>::Identity();
    H_e = H_x * dcampose_dextpose;

    // jacobian of reprojection residual wrt feature 
    Matrix3d dpc0_dpg = wRc0.transpose();
    Matrix3d dpc1_dpg = c1Rw;

    H_f = dz_dpc0 * dpc0_dpg + dz_dpc1 * dpc1_dpg;


    // Compute the residual.
    r = z - Vector4d(p_c0(0) / p_c0(2), p_c0(1) / p_c0(2), p_c1(0) / p_c1(2), p_c1(1) / p_c1(2));

    if (use_mono_flag)
    {
        // use mono residual 
        r(2) = 0;
        r(3) = 0;
    }

    return;
}

void OrcVio::featureJacobian(
    const FeatureIDType& feature_id, const std::vector<StateIDType>& cam_state_ids,
    MatrixXd& H_x, VectorXd& r) {

    const auto &feature = map_server[feature_id];

    // Check how many camera states in the provided camera id camera has actually seen this feature.
    vector<StateIDType> valid_cam_state_ids(0);
    std::copy_if(cam_state_ids.begin(), cam_state_ids.end(), std::back_inserter(valid_cam_state_ids), 
        [&feature](const auto& cam_id){
            return (feature.observations.find(cam_id) != feature.observations.end());
        });

    int jacobian_row_size = 0;
    jacobian_row_size = 4 * valid_cam_state_ids.size();

    MatrixXd H_xj = MatrixXd::Zero(jacobian_row_size, 21 + state_server.cam_states.size() * 6);
    MatrixXd H_fj = MatrixXd::Zero(jacobian_row_size, 3);
    VectorXd r_j = VectorXd::Zero(jacobian_row_size);

    int stack_cntr = 0;
    for (const auto &cam_id : valid_cam_state_ids) {
        Matrix<double, 4, 6> H_xi = Matrix<double, 4, 6>::Zero();
        Matrix<double, 4, 6> H_ei = Matrix<double, 4, 6>::Zero();
        Matrix<double, 4, 3> H_fi = Matrix<double, 4, 3>::Zero();
        Vector4d r_i = Vector4d::Zero();
        measurementJacobian(cam_id, feature.id, H_xi, H_ei, H_fi, r_i);

        auto cam_state_iter = state_server.cam_states.find(cam_id);
        int cam_state_cntr = std::distance(state_server.cam_states.begin(), cam_state_iter);

        // Stack the Jacobians.
        H_xj.block<4, 6>(stack_cntr, 21 + 6 * cam_state_cntr) = H_xi;
        H_xj.block<4, 6>(stack_cntr, 15) = H_ei;
        H_fj.block<4, 3>(stack_cntr, 0) = H_fi;
        r_j.segment<4>(stack_cntr) = r_i;
        stack_cntr += 4;
    }

    // Project the residual and Jacobians onto the nullspace of H_fj
    nullspace_project_inplace(H_fj, H_xj, r_j);
    H_x = H_xj;
    r = r_j;

    return;
}

void OrcVio::nullspace_project_inplace(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::VectorXd &res) {

    // Apply the left nullspace of H_f to all variables
    // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
    // See page 252, Algorithm 5.2.4 for how these two loops work
    // They use "matlab" index notation, thus we need to subtract 1 from all index
    Eigen::JacobiRotation<double> tempHo_GR;
    for (int n = 0; n < H_f.cols(); ++n) {
        for (int m = (int) H_f.rows() - 1; m > n; m--) {
            // Givens matrix G
            tempHo_GR.makeGivens(H_f(m - 1, n), H_f(m, n));
            // Multiply G to the corresponding lines (m-1,m) in each matrix
            // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
            //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
            (H_f.block(m - 1, n, 2, H_f.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (H_x.block(m - 1, 0, 2, H_x.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
        }
    }

    // The H_f jacobian max rank is 3 if it is a 3d position, thus size of the left nullspace is Hf.rows()-3
    // NOTE: need to eigen3 eval here since this experiences aliasing!
    //H_f = H_f.block(H_f.cols(),0,H_f.rows()-H_f.cols(),H_f.cols()).eval();
    H_x = H_x.block(H_f.cols(),0,H_x.rows()-H_f.cols(),H_x.cols()).eval();
    res = res.block(H_f.cols(),0,res.rows()-H_f.cols(),res.cols()).eval();

    // Sanity check
    assert(H_x.rows()==res.rows());
}

void OrcVio::measurement_compress_inplace(Eigen::MatrixXd &H_x, Eigen::VectorXd &res) noexcept {
    // Return if H_x is a fat matrix (there is no need to compress in this case)
    if(H_x.rows() <= H_x.cols())
        return;

    // Do measurement compression through givens rotations
    // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
    // See page 252, Algorithm 5.2.4 for how these two loops work
    // They use "matlab" index notation, thus we need to subtract 1 from all index
    Eigen::JacobiRotation<double> tempHo_GR;
    for (int n=0; n<H_x.cols(); n++) {
        for (int m=(int)H_x.rows()-1; m>n; m--) {
            // Givens matrix G
            tempHo_GR.makeGivens(H_x(m-1,n), H_x(m,n));
            // Multiply G to the corresponding lines (m-1,m) in each matrix
            // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
            //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
            (H_x.block(m-1,n,2,H_x.cols()-n)).applyOnTheLeft(0,1,tempHo_GR.adjoint());
            (res.block(m-1,0,2,1)).applyOnTheLeft(0,1,tempHo_GR.adjoint());
        }
    }

    // If H is a fat matrix, then use the rows
    // Else it should be same size as our state
    int r = std::min(H_x.rows(),H_x.cols());

    // Construct the smaller jacobian and residual after measurement compression
    assert(r<=H_x.rows());
    H_x.conservativeResize(r, H_x.cols());
    res.conservativeResize(r, res.cols());
}

void OrcVio::measurementUpdate(const MatrixXd& H, const VectorXd& r) {

    if (H.rows() == 0 || r.rows() == 0)
        return;

    // Decompose the final Jacobian matrix to reduce computational complexity as in Equation (28), (29).
    MatrixXd H_thin;
    VectorXd r_thin;

    H_thin = H;
    r_thin = r;
    if (H.rows() > H.cols()) 
    {
        // Givens QR is faster than SPQR
        measurement_compress_inplace(H_thin, r_thin);
    }

    // Compute the Kalman gain.
    const MatrixXd &P = state_server.state_cov;

    MatrixXd HP  = H_thin * P;
    MatrixXd HPH = HP * H_thin.transpose();

    MatrixXd K;
    MatrixXd R = Feature::observation_noise * MatrixXd::Identity(H_thin.rows(), H_thin.rows());
    MatrixXd S(R.rows(), R.rows());

    S = HPH + R;
    MatrixXd K_transpose = S.llt().solve(HP);
    K = K_transpose.transpose();
    
    // Compute the error of the state.
    VectorXd delta_x = K * r_thin;

    // Update the IMU state.
    const VectorXd &delta_x_imu = delta_x.head<21>();

    if (delta_x_imu.segment<3>(3).norm() > 0.5 ||
        delta_x_imu.segment<3>(6).norm() > 1.0) {
#if WITH_LOG
        SPDLOG_LOGGER_DEBUG(logger, "delta velocity: {}", delta_x_imu.segment<3>(3).norm());
        SPDLOG_LOGGER_DEBUG(logger, "delta position: {}", delta_x_imu.segment<3>(6).norm());
        SPDLOG_LOGGER_DEBUG(logger, "Update change is too large");
#endif 
        //return;
    }

    // Update IMU state
    Sophus::SO3d R_temp = Sophus::SO3d::exp(delta_x_imu.head<3>());
    state_server.imu_state.orientation = state_server.imu_state.orientation * R_temp.matrix();

    state_server.imu_state.velocity += delta_x_imu.segment<3>(3);
    state_server.imu_state.position += delta_x_imu.segment<3>(6);
    
    // update biases 
    state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(9);
    state_server.imu_state.acc_bias += delta_x_imu.segment<3>(12);

    // Update the IMU-CAM extrinsic
    Eigen::Matrix4d delta_iTc = Sophus::SE3d::exp(delta_x_imu.segment<6>(15)).matrix();
    Matrix3d &cRi = state_server.imu_state.R_imu_cam0;
    Matrix3d iRc = cRi.transpose();
    // need to update P first using old R 
    Vector3d &iPc = state_server.imu_state.t_cam0_imu;
    iPc = iRc * delta_iTc.block<3,1>(0,3) + iPc;
    // update R 
    iRc = iRc * delta_iTc.block<3,3>(0,0);
    cRi = iRc.transpose();

    // Update the camera states.
    auto cam_state_iter = state_server.cam_states.begin();
    for (int i = 0; i < state_server.cam_states.size(); ++i, ++cam_state_iter) {
        const VectorXd &delta_x_cam = delta_x.segment<6>(21 + i * 6);
        Eigen::Matrix4d delta_wTc = Sophus::SE3d::exp(delta_x_cam).matrix();

        // need to update P first using old R 
        cam_state_iter->second.position += cam_state_iter->second.orientation * delta_wTc.block<3,1>(0,3);

        // update R 
        cam_state_iter->second.orientation = cam_state_iter->second.orientation * delta_wTc.block<3,3>(0,0);
    }


    // Update state covariance.
    MatrixXd KHP = K * HP;

    state_server.state_cov.triangularView<Eigen::Upper>() -= KHP;
    state_server.state_cov = state_server.state_cov.selfadjointView<Eigen::Upper>();

    // We should check if we are not positive semi-definitate (i.e. negative diagionals is not s.p.d)
    Eigen::VectorXd diags = state_server.state_cov.diagonal();
    bool found_neg = false;
    for(int i=0; i<diags.rows(); i++) {
        if(diags(i) < 0.0) {
#if WITH_LOG
            SPDLOG_LOGGER_DEBUG(logger, "ERROR: diagonal at {} is {}", i, diags(i));
#endif 
            found_neg = true;
        }
    }
    assert(!found_neg);    

    return;
}

bool OrcVio::gatingTest(const MatrixXd& H, const VectorXd& r, const int& dof) noexcept {

    if(H.isZero(0)) return false; // check matrix empty

    MatrixXd P1 = H * state_server.state_cov * H.transpose();
    MatrixXd P2 = Feature::observation_noise * MatrixXd::Identity(H.rows(), H.rows());
    double gamma = r.transpose() * (P1 + P2).ldlt().solve(r);

    if (gamma < chi_squared_test_table[dof]) {
        //cout << "passed" << endl;
        return true;
    } else {
        //cout << "failed" << endl;
        return false;
    }
}

void OrcVio::removeLostFeatures() {
    // Remove the features that lost track.
    // BTW, find the size the final Jacobian matrix and residual vector.
    int jacobian_row_size = 0;
    vector<FeatureIDType> invalid_feature_ids(0);
    vector<FeatureIDType> processed_feature_ids(0);

    for (auto iter = map_server.begin(); iter != map_server.end(); ++iter) {
        // Rename the feature to be checked.
        auto &feature = iter->second;

        // Pass the features that are still being tracked.
        if (feature.observations.find(state_server.imu_state.id) != feature.observations.end())
            continue;
        if (feature.observations.size() < 3) {
            invalid_feature_ids.emplace_back(feature.id);
            continue;
        }

        // Check if the feature can be initialized if it has not been.
        if (!feature.is_initialized) {
            if (!feature.checkMotion(state_server.cam_states)) {
                invalid_feature_ids.emplace_back(feature.id);
                continue;
            } else {
                if (!feature.initializePosition(state_server.cam_states)) {
                    invalid_feature_ids.emplace_back(feature.id);
                    continue;
                }
            }
        }

        jacobian_row_size += 4 * feature.observations.size() - 3;
        processed_feature_ids.emplace_back(feature.id);
    }

    //cout << "invalid/processed feature #: " << invalid_feature_ids.size() << "/" << processed_feature_ids.size() << endl;
    //cout << "jacobian row #: " << jacobian_row_size << endl;

    // Remove the features that do not have enough measurements.
    std::for_each(invalid_feature_ids.begin(), invalid_feature_ids.end(),
        [this] (const auto& feature_id) {
            map_server.erase(feature_id);
        });

    // Return if there is no lost feature to be processed.
    if (processed_feature_ids.empty()) return;

    MatrixXd H_x = MatrixXd::Zero(jacobian_row_size, 21 + 6 * state_server.cam_states.size());
    VectorXd r   = VectorXd::Zero(jacobian_row_size);

    int stack_cntr = 0;

    // Process the features which lose track.
    bool size_outof_bound_flag = false; 

    for (int n=0; n< processed_feature_ids.size(); n++) 
    {
        if (!size_outof_bound_flag)
        {
            FeatureIDType feature_id = processed_feature_ids[n];
            auto &feature = map_server[feature_id];

#if WITH_LC
            update_feature(feature); // for loop closure
#endif  

            vector<StateIDType> cam_state_ids(0);
            std::for_each(feature.observations.begin(), feature.observations.end(),
                [&cam_state_ids] (const auto& measurement) {
                    cam_state_ids.emplace_back(measurement.first);
                });

            MatrixXd H_xj;
            VectorXd r_j;
            featureJacobian(feature.id, cam_state_ids, H_xj, r_j);

            if (gatingTest(H_xj, r_j, cam_state_ids.size() - 1)) {
                H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
                r.segment(stack_cntr, r_j.rows()) = r_j;
                stack_cntr += H_xj.rows();
            }

            if (stack_cntr > 1500)
                size_outof_bound_flag = true;
        }
    }

    H_x.conservativeResize(stack_cntr, H_x.cols());
    r.conservativeResize(stack_cntr);

    // Perform the measurement update step.
    measurementUpdate(H_x, r);

    // Remove all processed features from the map.
    std::for_each(processed_feature_ids.begin(), processed_feature_ids.end(),
        [this] (const auto& feature_id) {
            map_server.erase(feature_id);
        });

    return;
}

void OrcVio::findRedundantCamStates(vector<StateIDType>& rm_cam_state_ids) noexcept {
    // Move the iterator to the key position.
    auto key_cam_state_iter = state_server.cam_states.end();
    for (int i = 0; i < 4; ++i)
        --key_cam_state_iter;
    auto cam_state_iter = key_cam_state_iter;
    ++cam_state_iter;

    auto first_cam_state_iter = state_server.cam_states.begin();

    // Pose of the key camera state.
    const Vector3d key_position = key_cam_state_iter->second.position;
    const Matrix3d key_rotation = key_cam_state_iter->second.orientation;

    // Mark the camera states to be removed based on the motion between states.
    for (int i = 0; i < 2; ++i) {
        const Vector3d position = cam_state_iter->second.position;
        const Matrix3d rotation = cam_state_iter->second.orientation;

        double distance = (position - key_position).norm();
        double angle = AngleAxisd(rotation * key_rotation.transpose()).angle();

        if (angle < rotation_threshold && distance < translation_threshold && tracking_rate > tracking_rate_threshold) {
            rm_cam_state_ids.emplace_back(cam_state_iter->first);
            ++cam_state_iter;
        } else {
            rm_cam_state_ids.emplace_back(first_cam_state_iter->first);
            ++first_cam_state_iter;
        }
    }

    // Sort the elements in the output vector.
    sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());

    return;
}

void OrcVio::pruneCamStateBuffer() {
    if (state_server.cam_states.size() < max_cam_state_size)
        return;

    // Find two camera states to be removed.
    vector<StateIDType> rm_cam_state_ids(0);
    findRedundantCamStates(rm_cam_state_ids);

#if WITH_LC
    StateIDType cam_state_id_margin = std::min(rm_cam_state_ids[0], rm_cam_state_ids[1]);
    cam_state_margin.first = state_server.cam_states.find(cam_state_id_margin)->first;
    cam_state_margin.second = state_server.cam_states.find(cam_state_id_margin)->second;
    is_start_loop = true;
#endif    

    // Find the size of the Jacobian matrix.
    int jacobian_row_size = 0;
    for (auto &item : map_server) {
        auto &feature = item.second;
        // Check how many camera states to be removed are associated with this feature.
        vector<StateIDType> involved_cam_state_ids(0);
        std::copy_if(rm_cam_state_ids.begin(), rm_cam_state_ids.end(), std::back_inserter(involved_cam_state_ids), 
            [&feature](const auto& cam_id){
                return (feature.observations.find(cam_id) != feature.observations.end());
            });

        if (involved_cam_state_ids.size() == 0)
            continue;
        if (involved_cam_state_ids.size() == 1) {
            feature.observations.erase(involved_cam_state_ids[0]);
            continue;
        }

#if WITH_LC
        update_feature(feature); // for loop closure
#endif        

        if (!feature.is_initialized) {
            // Check if the feature can be initialize.
            if (!feature.checkMotion(state_server.cam_states)) {
                // If the feature cannot be initialized, just remove
                // the observations associated with the camera states to be removed.
                std::for_each(involved_cam_state_ids.begin(), involved_cam_state_ids.end(),
                    [&feature] (const auto& cam_id) {
                        feature.observations.erase(cam_id);
                    });
                continue;
            } else {
                if (!feature.initializePosition(state_server.cam_states)) {
                    std::for_each(involved_cam_state_ids.begin(), involved_cam_state_ids.end(),
                        [&feature] (const auto& cam_id) {
                            feature.observations.erase(cam_id);
                        });
                    continue;
                }
            }
        }

        jacobian_row_size += 4 * involved_cam_state_ids.size() - 3;
    }

    //cout << "jacobian row #: " << jacobian_row_size << endl;

    // Compute the Jacobian and residual.
    MatrixXd H_x = MatrixXd::Zero(jacobian_row_size, 21 + 6 * state_server.cam_states.size());
    VectorXd r   = VectorXd::Zero(jacobian_row_size);

    int stack_cntr = 0;

    for (auto &item : map_server) {
        auto &feature = item.second;
        // Check how many camera states to be removed are associated with this feature.
        vector<StateIDType> involved_cam_state_ids(0);
        std::copy_if(rm_cam_state_ids.begin(), rm_cam_state_ids.end(), std::back_inserter(involved_cam_state_ids), 
            [&feature](const auto& cam_id){
                return (feature.observations.find(cam_id) != feature.observations.end());
            });

        if (involved_cam_state_ids.size() == 0)
            continue;

        MatrixXd H_xj;
        VectorXd r_j;
        featureJacobian(feature.id, involved_cam_state_ids, H_xj, r_j);

        if (gatingTest(H_xj, r_j, involved_cam_state_ids.size())) {
            H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
            r.segment(stack_cntr, r_j.rows()) = r_j;
            stack_cntr += H_xj.rows();
        }

        std::for_each(involved_cam_state_ids.begin(), involved_cam_state_ids.end(),
            [&feature] (const auto& cam_id) {
                feature.observations.erase(cam_id);
            });
    }

    H_x.conservativeResize(stack_cntr, H_x.cols());
    r.conservativeResize(stack_cntr);


    // Perform measurement update.
    measurementUpdate(H_x, r);

    for (const auto &cam_id : rm_cam_state_ids) {
        int cam_sequence = std::distance(state_server.cam_states.begin(), state_server.cam_states.find(cam_id));
        int cam_state_start = 21 + 6 * cam_sequence;
        int cam_state_end = cam_state_start + 6;

        // Remove the corresponding rows and columns in the state covariance matrix.
        if (cam_state_end < state_server.state_cov.rows()) {
            state_server.state_cov.block(cam_state_start, 0, state_server.state_cov.rows() - cam_state_end, state_server.state_cov.cols()) =
                    state_server.state_cov.block(cam_state_end, 0, state_server.state_cov.rows() - cam_state_end, state_server.state_cov.cols());

            state_server.state_cov.block(0, cam_state_start, state_server.state_cov.rows(), state_server.state_cov.cols() - cam_state_end) =
                    state_server.state_cov.block(0, cam_state_end, state_server.state_cov.rows(), state_server.state_cov.cols() - cam_state_end);

            state_server.state_cov.conservativeResize(state_server.state_cov.rows() - 6, state_server.state_cov.cols() - 6);
        } else {
            state_server.state_cov.conservativeResize(state_server.state_cov.rows() - 6, state_server.state_cov.cols() - 6);
        }

        // Remove this camera state in the state vector.
        state_server.cam_states.erase(cam_id);
    }

    return;
}

void OrcVio::onlineReset() noexcept {
    // Never perform online reset if position std threshold is non-positive.
    if (position_std_threshold <= 0)
        return;

    static long long int online_reset_counter = 0;

    // Check the uncertainty of positions to determine if the system can be reset.
    double position_x_std = std::sqrt(state_server.state_cov(6, 6));
    double position_y_std = std::sqrt(state_server.state_cov(7, 7));
    double position_z_std = std::sqrt(state_server.state_cov(8, 8));

    if (position_x_std < position_std_threshold &&
        position_y_std < position_std_threshold &&
        position_z_std < position_std_threshold)
        return;

    ROS_WARN("Start %lld online reset procedure...", ++online_reset_counter);
    ROS_INFO("Stardard deviation in xyz: %f, %f, %f", position_x_std, position_y_std, position_z_std);

    // Remove all existing camera states.
    state_server.cam_states.clear();

    // Clear all exsiting features in the map.
    map_server.clear();

    // Reset the state covariance.
    double gyro_bias_cov, acc_bias_cov, velocity_cov;
    nh.param<double>("initial_covariance/velocity", velocity_cov, 0.25);
    nh.param<double>("initial_covariance/gyro_bias", gyro_bias_cov, 1e-4);
    nh.param<double>("initial_covariance/acc_bias", acc_bias_cov, 1e-2);

    double extrinsic_rotation_cov, extrinsic_translation_cov;
    nh.param<double>("initial_covariance/extrinsic_rotation_cov", extrinsic_rotation_cov, 3.0462e-4);
    nh.param<double>("initial_covariance/extrinsic_translation_cov", extrinsic_translation_cov, 1e-4);

    state_server.state_cov = MatrixXd::Zero(21, 21);
    for (int i = 3; i < 6; ++i)
        state_server.state_cov(i, i) = velocity_cov;
    for (int i = 9; i < 12; ++i)
        state_server.state_cov(i, i) = gyro_bias_cov;
    for (int i = 12; i < 15; ++i)
        state_server.state_cov(i, i) = acc_bias_cov;
    for (int i = 15; i < 18; ++i)
        // in fact this is rho, not translation 
        state_server.state_cov(i, i) = extrinsic_translation_cov;
    for (int i = 18; i < 21; ++i)
        state_server.state_cov(i, i) = extrinsic_rotation_cov;

    ROS_WARN("%lld online reset complete...", online_reset_counter);
    return;
}

#if WITH_LC
void OrcVio::update_feature(Feature feature) noexcept {
    const auto &id = feature.id;

    if(featuresLCDB.find(id) != featuresLCDB.end()) {
        FeatureLcPtr feat = featuresLCDB[id];
        for(const auto &obs : feature.observations) {
            const auto &cam_id = obs.first;
            feat->uvs_norm[cam_id].emplace_back(obs.second);
            Eigen::Vector4d uv = feature.observations_uvs.find(cam_id)->second;
            feat->uvs[cam_id].emplace_back(uv);
            feat->p_FinG = feature.position;
            feat->timestamps[cam_id].emplace_back(feature.timestamp.find(cam_id)->second);            
        }
        return;
    }

    FeatureLcPtr feat = std::make_shared<FeatureLC>();
    feat->featid = id;
    for(const auto &obs : feature.observations) {
        const auto &cam_id = obs.first;
        feat->uvs_norm[cam_id].emplace_back(obs.second);
        Eigen::Vector4d uv = feature.observations_uvs.find(cam_id)->second;
        feat->uvs[cam_id].emplace_back(uv);
        feat->p_FinG = feature.position;
        feat->timestamps[cam_id].emplace_back(feature.timestamp.find(cam_id)->second);
    }
    featuresLCDB.insert({id, feat});
}

void OrcVio::update_keyframe_historical_information(const std::vector<FeatureLcPtr> &features) {
    // Loop through all features that have been used in the last update
    // We want to record their historical measurements and estimates for later use
    for(const auto &feat : features) {

        // Get position of feature in the global frame of reference
        Eigen::Vector3d p_FinG = feat->p_FinG;

        // // If it is a slam feature, then get its best guess from the state
        // if(state->_features_SLAM.find(feat->featid)!=state->_features_SLAM.end()) {
        //     p_FinG = state->_features_SLAM.at(feat->featid)->get_xyz(false);
        // }

        // Push back any new measurements if we have them
        // Ensure that if the feature is already added, then just append the new measurements
        if(hist_feat_posinG.find(feat->featid)!=hist_feat_posinG.end()) {
            hist_feat_posinG.at(feat->featid) = p_FinG;
            for(const auto &cam2uv : feat->uvs) {
                if(hist_feat_uvs.at(feat->featid).find(cam2uv.first)!=hist_feat_uvs.at(feat->featid).end()) {
                    hist_feat_uvs.at(feat->featid).at(cam2uv.first).insert(hist_feat_uvs.at(feat->featid).at(cam2uv.first).end(), cam2uv.second.begin(), cam2uv.second.end());
                    hist_feat_uvs_norm.at(feat->featid).at(cam2uv.first).insert(hist_feat_uvs_norm.at(feat->featid).at(cam2uv.first).end(), feat->uvs_norm.at(cam2uv.first).begin(), feat->uvs_norm.at(cam2uv.first).end());
                    hist_feat_timestamps.at(feat->featid).at(cam2uv.first).insert(hist_feat_timestamps.at(feat->featid).at(cam2uv.first).end(), feat->timestamps.at(cam2uv.first).begin(), feat->timestamps.at(cam2uv.first).end());
                } else {
                    hist_feat_uvs.at(feat->featid).insert(cam2uv);
                    hist_feat_uvs_norm.at(feat->featid).insert({cam2uv.first,feat->uvs_norm.at(cam2uv.first)});
                    hist_feat_timestamps.at(feat->featid).insert({cam2uv.first,feat->timestamps.at(cam2uv.first)});
                }
            }
        } else {
            hist_feat_posinG.insert({feat->featid,p_FinG});
            hist_feat_uvs.insert({feat->featid,feat->uvs});
            hist_feat_uvs_norm.insert({feat->featid,feat->uvs_norm});
            hist_feat_timestamps.insert({feat->featid,feat->timestamps});
        }
    } 

    std::vector<size_t> ids_to_remove;
    for(const auto &id2feat : hist_feat_timestamps) {
        bool all_older = true;
        for(const auto &cam2time : id2feat.second) {
            for(const auto &time : cam2time.second) {
                if(time >= hist_last_marginalized_time) {
                    all_older = false;
                    break;
                }
            }
            if(!all_older) break;
        }
        if(all_older) {
            ids_to_remove.emplace_back(id2feat.first);
        }
    }

    // Remove those features!
    std::for_each(ids_to_remove.begin(), ids_to_remove.end(),
        [this] (const auto& id) {
            hist_feat_posinG.erase(id);
            hist_feat_uvs.erase(id);
            hist_feat_uvs_norm.erase(id);
            hist_feat_timestamps.erase(id);
        });

    // Remove any historical states older then the marg time
    auto it0 = hist_stateinG.begin();
    while(it0 != hist_stateinG.end()) {
        if(it0->first < hist_last_marginalized_time) 
            it0 = hist_stateinG.erase(it0);
        else 
            it0++;
    }

    // if (state_server.cam_states.size() >= max_cam_state_size)

    if (is_start_loop) {
        const auto &cam_state = cam_state_margin.second; // state_server.cam_states.begin()->second;
        hist_last_marginalized_time = cam_state.time;
        assert(hist_last_marginalized_time != INFINITY);
        Eigen::Matrix<double,7,1> state_inG = Eigen::Matrix<double,7,1>::Zero();
        Eigen::Quaterniond q_wc(cam_state.orientation);
        state_inG(0) = q_wc.x();
        state_inG(1) = q_wc.y();
        state_inG(2) = q_wc.z();
        state_inG(3) = q_wc.w();
        state_inG.tail(3) = cam_state.position;
        hist_stateinG.insert({hist_last_marginalized_time, state_inG});
    } 
}

void OrcVio::publish_keyframe_information() {

    Eigen::Matrix<double,7,1> stateinG;
    if(hist_last_marginalized_time != -1) {
        stateinG = hist_stateinG.at(hist_last_marginalized_time);
    } else {
        stateinG.setZero();
        return;
    }

    std_msgs::Header header;
    header.stamp = ros::Time(hist_last_marginalized_time);

    Isometry3d T_ci = utils::getTransformEigen(nh, "cam0/T_cam_imu");
    Eigen::Vector4d q_ItoC;
    Eigen::Quaterniond q(T_ci.linear().transpose());
    q_ItoC[0] = q.x();
    q_ItoC[1] = q.y();
    q_ItoC[2] = q.z();
    q_ItoC[3] = q.w();
    Eigen::Vector3d p_CinI = - T_ci.linear().transpose() * T_ci.translation();
    nav_msgs::Odometry odometry_calib;
    odometry_calib.header = header;
    odometry_calib.header.frame_id = "imu";
    odometry_calib.pose.pose.position.x = p_CinI(0);
    odometry_calib.pose.pose.position.y = p_CinI(1);
    odometry_calib.pose.pose.position.z = p_CinI(2);
    odometry_calib.pose.pose.orientation.x = q_ItoC(0);
    odometry_calib.pose.pose.orientation.y = q_ItoC(1);
    odometry_calib.pose.pose.orientation.z = q_ItoC(2);
    odometry_calib.pose.pose.orientation.w = q_ItoC(3);
    pub_keyframe_extrinsic.publish(odometry_calib);    

    sensor_msgs::CameraInfo cameraparams;
    cameraparams.header = header;
    cameraparams.header.frame_id = "imu";
    cameraparams.distortion_model = "plumb_bob";
    vector<double> vD(4);
    nh.getParam("cam0/distortion_coeffs", vD);
    cameraparams.D = {vD[0], vD[1], vD[2], vD[3]};
    vector<double> vK(4);
    nh.getParam("cam0/intrinsics", vK);
    double fx = vK[0];
    double fy = vK[1];
    double cx = vK[2];
    double cy = vK[3];    
    cameraparams.K = {fx, 0., cx, 0., fy, cy, 0., 0., 1.};
    pub_keyframe_intrinsics.publish(cameraparams);

    //======================================================
    // PUBLISH HISTORICAL POSE ESTIMATE
    nav_msgs::Odometry odometry_pose;
    odometry_pose.header = header;
    odometry_pose.header.frame_id = "global";
    Eigen::Quaterniond q_wi;
    Eigen::Vector3d t_wi;
    {
        Eigen::Quaterniond q_wc;
        Eigen::Vector3d t_wc;
        q_wc.x() = stateinG(0);
        q_wc.y() = stateinG(1);
        q_wc.z() = stateinG(2);
        q_wc.w() = stateinG(3);
        t_wc = stateinG.tail(3);

        Eigen::Matrix3d r_ci = T_ci.linear();
        Eigen::Vector3d t_ci = T_ci.translation();

        q_wi = q_wc * Eigen::Quaterniond(r_ci);
        t_wi = q_wc.toRotationMatrix() * t_ci + t_wc;
    }
    odometry_pose.pose.pose.position.x = t_wi(0);
    odometry_pose.pose.pose.position.y = t_wi(1);
    odometry_pose.pose.pose.position.z = t_wi(2);
    odometry_pose.pose.pose.orientation.x = q_wi.x();
    odometry_pose.pose.pose.orientation.y = q_wi.y();
    odometry_pose.pose.pose.orientation.z = q_wi.z();
    odometry_pose.pose.pose.orientation.w = q_wi.w();
    pub_keyframe_pose.publish(odometry_pose);   

#if WITH_LOG
    SPDLOG_LOGGER_DEBUG(logger, "[orcvio] hist_feat_timestamps size: {}", hist_feat_timestamps.size());
#endif 

    // Construct the message
    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = header;
    point_cloud.header.frame_id = "global";
    for(const auto &feattimes : hist_feat_timestamps) {
        StateIDType state_id = cam_state_margin.first;

        // Skip if this feature has no extraction in the "zero" camera
        if(feattimes.second.find(state_id)==feattimes.second.end()) // 0
            continue;
 
#if WITH_LOG
        // TODO: why these values are all the same?
        SPDLOG_LOGGER_DEBUG(logger, "[orcvio loop], {}, {}, {}",  
            *feattimes.second.at(state_id).begin(),
            *(feattimes.second.at(state_id).end()-1),
            hist_last_marginalized_time);          
#endif 

        // Skip if this feature does not have measurement at this time
        auto iter = std::find(feattimes.second.at(state_id).begin(), feattimes.second.at(state_id).end(), hist_last_marginalized_time);
        if(iter==feattimes.second.at(state_id).end())
            continue;

        // Get this feature information
        size_t featid = feattimes.first;
        size_t index = (size_t)std::distance(feattimes.second.at(state_id).begin(), iter);
        Eigen::Vector4d uv = hist_feat_uvs.at(featid).at(state_id).at(index);
        Eigen::Vector4d uv_n = hist_feat_uvs_norm.at(featid).at(state_id).at(index);
        Eigen::Vector3d pFinG = hist_feat_posinG.at(featid);

        // Push back 3d point
        geometry_msgs::Point32 p;
        p.x = pFinG(0);
        p.y = pFinG(1);
        p.z = pFinG(2);
        point_cloud.points.emplace_back(p);

        // Push back the norm, raw, and feature id
        sensor_msgs::ChannelFloat32 p_2d;
        p_2d.values.emplace_back(uv_n(0));
        p_2d.values.emplace_back(uv_n(1));
        p_2d.values.emplace_back(uv(0));
        p_2d.values.emplace_back(uv(1));
        p_2d.values.emplace_back(featid);
        point_cloud.channels.emplace_back(p_2d);
    }
    pub_keyframe_point.publish(point_cloud);     
}
#endif

void OrcVio::publish(const ros::Time& time) {
    // Convert the IMU frame to the body frame.
    const IMUState &imu_state = state_server.imu_state;

    Eigen::Isometry3d T_i_w = Eigen::Isometry3d::Identity();
    T_i_w.linear() = imu_state.orientation;
    T_i_w.translation() = imu_state.position;

    Eigen::Isometry3d T_b_w = IMUState::T_imu_body * T_i_w * IMUState::T_imu_body.inverse();
    Eigen::Vector3d body_velocity = IMUState::T_imu_body.linear() * imu_state.velocity;

    // save the pose to txt for trajectory evaluation 
    // ============================
    Eigen::Matrix3d m3_r = T_b_w.rotation();
    Eigen::Vector3d v3_t = T_b_w.translation();
    Eigen::Quaterniond q4_r(m3_r);

    // TUM format
    // timestamp tx ty tz qx qy qz qw
    pose_outfile << std::fixed << std::setprecision(3) << time.toSec();
    pose_outfile << " "
                  << v3_t[0] << " " << v3_t[1] << " " << v3_t[2] << " "
                  << q4_r.x() << " " << q4_r.y() << " " << q4_r.z() << " " << q4_r.w() << std::endl;
#if WITH_LOG
    SPDLOG_LOGGER_DEBUG(logger, "tx {0:.4f} ty {0:.4f} tz {0:.4f} qx {0:.4f} qy {0:.4f} qz {0:.4f} qw {0:.4f}", 
        v3_t[0], v3_t[1], v3_t[2],
        q4_r.x(), q4_r.y(), q4_r.z(), q4_r.w());
#endif 

    // Publish tf
    if (publish_tf) {
        tf::Transform T_b_w_tf;
        tf::transformEigenToTF(T_b_w, T_b_w_tf);
        tf_pub.sendTransform(tf::StampedTransform(T_b_w_tf, time, fixed_frame_id, child_frame_id));
    }

    // Publish the odometry
    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = time;
    odom_msg.header.frame_id = fixed_frame_id;
    odom_msg.child_frame_id  = child_frame_id;

    tf::poseEigenToMsg(T_b_w, odom_msg.pose.pose);
    tf::vectorEigenToMsg(body_velocity, odom_msg.twist.twist.linear);

    // Convert the covariance.
    Matrix3d P_oo = state_server.state_cov.block<3, 3>(0, 0);
    Matrix3d P_op = state_server.state_cov.block<3, 3>(0, 6);
    Matrix3d P_po = state_server.state_cov.block<3, 3>(6, 0);
    Matrix3d P_pp = state_server.state_cov.block<3, 3>(6, 6);
    Matrix<double, 6, 6> P_imu_pose = Matrix<double, 6, 6>::Zero();
    P_imu_pose << P_pp, P_po, P_op, P_oo;

    Matrix<double, 6, 6> H_pose = Matrix<double, 6, 6>::Zero();
    H_pose.block<3, 3>(0, 0) = IMUState::T_imu_body.linear();
    H_pose.block<3, 3>(3, 3) = IMUState::T_imu_body.linear();
    Matrix<double, 6, 6> P_body_pose = H_pose * P_imu_pose * H_pose.transpose();

    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            odom_msg.pose.covariance[6 * i + j] = P_body_pose(i, j);

    // Construct the covariance for the velocity.
    Matrix3d P_imu_vel = state_server.state_cov.block<3, 3>(3, 3);
    Matrix3d H_vel = IMUState::T_imu_body.linear();
    Matrix3d P_body_vel = H_vel * P_imu_vel * H_vel.transpose();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            odom_msg.twist.covariance[i * 6 + j] = P_body_vel(i, j);

    odom_pub.publish(odom_msg);

#if WITH_LC
    // for loop closure
    geometry_msgs::PoseWithCovarianceStamped poseIinM;
    poseIinM.header.stamp = time;
    poseIinM.header.frame_id = fixed_frame_id;
    poseIinM.pose = odom_msg.pose;
    pub_poseimu.publish(poseIinM);
#endif    

    // Publish Pose Stamped 
    // ============================
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = time;
    pose_stamped.header.frame_id = fixed_frame_id;
    pose_stamped.pose = odom_msg.pose.pose;

    posestamped_pub.publish(pose_stamped);

    // Publish the path
    // ============================
    path_msg.header.stamp = time;
    path_msg.header.frame_id = fixed_frame_id;
    path_msg.poses.push_back(pose_stamped);
    path_pub.publish(path_msg);

    // Publish the 3D positions of the features that has been initialized.
    boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > feature_msg_ptr(new pcl::PointCloud<pcl::PointXYZ>());
    feature_msg_ptr->header.frame_id = fixed_frame_id;
    feature_msg_ptr->height = 1;
    for (const auto &item : map_server) {
        const auto &feature = item.second;
        if (feature.is_initialized) {
            Vector3d feature_position = IMUState::T_imu_body.linear() * feature.position;
            feature_msg_ptr->points.push_back(pcl::PointXYZ(feature_position(0), feature_position(1), feature_position(2)));
        }
    }
    feature_msg_ptr->width = feature_msg_ptr->points.size();

    feature_pub.publish(feature_msg_ptr);

    return;
}



} // namespace orcvio
