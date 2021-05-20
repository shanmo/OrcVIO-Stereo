/*
 * COPYRIGHT AND PERMISSION NOTICE
 * UCSD Software ORCVIO
 * Copyright (C) 2021 
 * All rights reserved.
 */

#pragma once

#include <map>
#include <set>
#include <vector>
#include <string>
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <tf/transform_broadcaster.h>
#include <std_srvs/Trigger.h>

#include <boost/shared_ptr.hpp>

#if WITH_LOG
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE // Must: define SPDLOG_ACTIVE_LEVEL before `#include "spdlog/spdlog.h"`
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h" // support for basic file logging
#include "spdlog/fmt/ostr.h"
#endif

#include "state.h"
#include "feature.h"
#include "orcvio/CameraMeasurement.h"
#include "initializer/FlexibleInitializer.h"

namespace orcvio {
    /**
     * @brief OrcVio Implements the algorithm in
     *    Anatasios I. Mourikis, and Stergios I. Roumeliotis,
     *    "A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation",
     *    http://www.ee.ucr.edu/~mourikis/tech_reports/TR_MSCKF.pdf
     */
    class OrcVio {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Constructor
        OrcVio(ros::NodeHandle &pnh);

        // Disable copy and assign constructor
        OrcVio(const OrcVio &) = delete;

        OrcVio operator=(const OrcVio &) = delete;

        // Destructor
        ~OrcVio() noexcept; 

        /**
         * @brief initialize Initialize the VIO.
         */
        bool initialize();

        /**
         * @brief reset Resets the VIO to initial status.
         */
        void reset();

        typedef std::shared_ptr<OrcVio> Ptr;
        typedef std::shared_ptr<const OrcVio> ConstPtr;

    private:
        /**
         * @brief StateServer Store one IMU states and several camera states for constructing measurement model.
         */
        struct StateServer {
            IMUState imu_state;
            CamStateServer cam_states;

            // first estimated imu state for calculate Jacobian
            IMUState imu_state_FEJ;

            // State covariance matrix
            Eigen::MatrixXd state_cov;
            Eigen::Matrix<double, 21, 21> continuous_noise_cov;
        };


        /**
         * @brief loadParameters Load parameters from the parameter server.
         */
        bool loadParameters();

        /**
         * @brief createRosIO Create ros publisher and subscirbers.
         */
        bool createRosIO() noexcept;

        /**
         * @brief imuCallback Callback function for the imu message.
         * @details save inertial measurements to imu_msg_buffer
         * @param msg IMU msg.
         */
        void imuCallback(const sensor_msgs::ImuConstPtr &msg) noexcept;

        /**
         * @brief featureCallback Callback function for feature measurements.
         * @details use features to update the state 
         * @param msg Stereo feature measurements.
         */
        void featureCallback(const CameraMeasurementConstPtr &msg);

        /**
         * @brief publish Publish the results of VIO.
         * @param time The time stamp of output msgs.
         */
        void publish(const ros::Time &time);

        /**
         * @biref resetCallback
         *    Callback function for the reset service.
         *    Note that this is NOT anytime-reset.
         *    This function should only be called before the sensor suite starts moving.
         *    e.g. while the robot is still on the ground.
         */
        bool resetCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);

        /**
         * @brief Filter related functions Propogate the state
         * @param time_bound
         */
        void batchImuProcessing(const double &time_bound);

        /**
         * @brief propagate the state and covariance 
         * @param time
         * @param m_gyro
         * @param m_acc
         */
        void processModel(const double &time, const Eigen::Vector3d &m_gyro, const Eigen::Vector3d &m_acc);

        // predict new state 
        void predictNewState(const double &dt, const Eigen::Vector3d &gyro, const Eigen::Vector3d &acc) noexcept;

        // Measurement update
        void stateAugmentation(const double &time);

        /**
         * @brief Add new observations for existing features or new features in the map server.
         * @param msg
         */
        void addFeatureObservations(const CameraMeasurementConstPtr &msg) noexcept;

        void nullspace_project_inplace(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::VectorXd &res);

        void measurement_compress_inplace(Eigen::MatrixXd &H_x, Eigen::VectorXd &res) noexcept;

        // This function is used to compute the measurement Jacobian for a single feature observed at a single camera frame.
        void measurementJacobian(const StateIDType &cam_state_id, const FeatureIDType &feature_id,
                                 Eigen::Matrix<double, 4, 6> &H_x, 
                                 Eigen::Matrix<double, 4, 6> &H_e,
                                 Eigen::Matrix<double, 4, 3> &H_f, 
                                 Eigen::Vector4d &r);

        // This function computes the Jacobian of all measurements viewed in the given camera states of this feature.
        void featureJacobian(const FeatureIDType &feature_id, const std::vector<StateIDType> &cam_state_ids,
                             Eigen::MatrixXd &H_x, Eigen::VectorXd &r);

        /**
         * @details do update when 
         *           1）features are lost in removeLostFeatures
         *           2）need to marginalize camera state in pruneCamStateBuffer
         * @param H
         * @param r
         */
        void measurementUpdate(const Eigen::MatrixXd &H, const Eigen::VectorXd &r);

        bool gatingTest(const Eigen::MatrixXd &H, const Eigen::VectorXd &r, const int &dof) noexcept;

        void removeLostFeatures();

        void findRedundantCamStates(std::vector<StateIDType> &rm_cam_state_ids) noexcept;

        void pruneCamStateBuffer();

        // Reset the system online if the uncertainty is too large.
        void onlineReset() noexcept;

        // Chi squared test table.
        static std::map<int, double> chi_squared_test_table;

        // State vector
        StateServer state_server;
        // Maximum number of camera states
        int max_cam_state_size;

        // Features used
        MapServer map_server;

        // IMU data buffer
        // This is buffer is used to handle the unsynchronization or
        // transfer delay between IMU and Image messages.
        std::vector<sensor_msgs::Imu> imu_msg_buffer;

        // Indicate if the gravity vector is set.
        bool is_gravity_set;

        // whether to use first estimate jacobian 
        bool use_FEJ_flag; 

        // Indicate if the received image is the first one. The
        // system will start after receiving the first image.
        bool is_first_img;

        // The position uncertainty threshold is used to determine
        // when to reset the system online. Otherwise, the ever-
        // increaseing uncertainty will make the estimation unstable.
        // Note this online reset will be some dead-reckoning.
        // Set this threshold to nonpositive to disable online reset.
        double position_std_threshold;

        // Tracking rate
        double tracking_rate;

        // Threshold for determine keyframes
        double translation_threshold;
        double rotation_threshold;
        double tracking_rate_threshold;

        // Ros node handle
        ros::NodeHandle nh;

        // Subscribers and publishers
        ros::Subscriber imu_sub;
        ros::Subscriber feature_sub;
        ros::Publisher odom_pub;
        ros::Publisher posestamped_pub;
        ros::Publisher path_pub;
        ros::Publisher feature_pub;
        tf::TransformBroadcaster tf_pub;
        ros::ServiceServer reset_srv;

        nav_msgs::Path path_msg;

        // Frame id
        std::string fixed_frame_id;
        std::string child_frame_id;

        // Whether to publish tf or not.
        bool publish_tf;

        // for initialization 
        boost::shared_ptr<FlexibleInitializer> flexInitPtr;
        // flag for first useful image features
        bool bFirstFeatures;

        // Debugging variables and functions
        void mocapOdomCallback(const nav_msgs::OdometryConstPtr &msg);

        ros::Subscriber mocap_odom_sub;
        ros::Publisher mocap_odom_pub;
        geometry_msgs::TransformStamped raw_mocap_odom_msg;
        Eigen::Isometry3d mocap_initial_frame;

        bool use_mono_flag; 

        // for saving trajectory and log 
        std::string output_dir_traj, output_dir_log;
        std::ofstream pose_outfile;
#if WITH_LOG
        std::shared_ptr<spdlog::logger> logger;
#endif         

#if WITH_LC
    private:
        std::unordered_map<size_t, FeatureLcPtr> featuresLCDB;
        std::pair<StateIDType, CAMState> cam_state_margin;
        bool is_start_loop = false;

        void update_feature(Feature feat) noexcept;

        void update_keyframe_historical_information(const std::vector<FeatureLcPtr> &features);

        void publish_keyframe_information();

        ros::Publisher pub_poseimu, pub_keyframe_pose, pub_keyframe_point, pub_keyframe_extrinsic, pub_keyframe_intrinsics;

        // Historical information of the filter (last marg time, historical states, features seen from all frames)
        double hist_last_marginalized_time = -1;
        std::map<double,Eigen::Matrix<double,7,1>> hist_stateinG;
        std::unordered_map<size_t, Eigen::Vector3d> hist_feat_posinG;
        std::unordered_map<size_t, std::unordered_map<size_t, std::vector<Eigen::Vector4d>>> hist_feat_uvs;
        std::unordered_map<size_t, std::unordered_map<size_t, std::vector<Eigen::Vector4d>>> hist_feat_uvs_norm;
        std::unordered_map<size_t, std::unordered_map<size_t, std::vector<double>>> hist_feat_timestamps;
#endif        
    };

    typedef OrcVio::Ptr OrcVioPtr;
    typedef OrcVio::ConstPtr OrcVioConstPtr;

} // namespace orcvio


