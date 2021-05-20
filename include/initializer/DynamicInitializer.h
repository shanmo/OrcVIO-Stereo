#pragma once 

#include <map>
#include <list>
#include <vector>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <sensor_msgs/Imu.h>
#include <boost/shared_ptr.hpp>

#include "orcvio/feature.h"
#include "orcvio/state.h"
#include "initializer/initial_alignment.h"
#include "initializer/initial_sfm.h"
#include "initializer/solve_5pts.h"
#include "initializer/math_utils.h"
#include "initializer/feature_manager.h"

namespace {
    const double imu_rate = 200; 
}

namespace orcvio {

class DynamicInitializer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor.
    DynamicInitializer() = delete;
    DynamicInitializer(
        const double& acc_n_, const double& acc_w_, const double& gyr_n_, 
        const double& gyr_w_, const Eigen::Matrix3d& R_c2b, 
        const Eigen::Vector3d& t_bc_b) : 
        bInit(false), state_time(0.0), curr_time(-1), 
        first_imu(false), frame_count(0), acc_n(acc_n_), acc_w(acc_w_), 
        gyr_n(gyr_n_), gyr_w(gyr_w_), initial_timestamp(0.0),
        RIC(R_c2b), TIC(t_bc_b), lower_time_bound(0.0) {
        
        imu_img_timeTh = 1 / (2 * imu_rate); 

        // this is not estimated in stereo orcvio 
        td = 0; 

        gyro_bias = Eigen::Vector3d::Zero();
        acc_bias = Eigen::Vector3d::Zero();
        position = Eigen::Vector3d::Zero();
        velocity = Eigen::Vector3d::Zero();
        
        orientation = Eigen::Matrix3d::Identity();

        for (int i = 0; i < WINDOW_SIZE + 1; i++)   
        {
            Rs[i].setIdentity();
            Ps[i].setZero();
            Vs[i].setZero();
            Bas[i].setZero();
            Bgs[i].setZero();
            dt_buf[i].clear();
            linear_acceleration_buf[i].clear();
            angular_velocity_buf[i].clear();
        }

        g = Eigen::Vector3d::Zero();

        // Initialize feature manager
        f_manager.clearState();
        // f_manager.init(Rs);
        f_manager.setRic(R_c2b);

        Times.resize(WINDOW_SIZE + 1);
    }

    // Destructor.
    ~DynamicInitializer(){};

    // Interface for trying to initialize.
    bool tryDynInit(const std::vector<sensor_msgs::Imu>& imu_msg_buffer,
        CameraMeasurementConstPtr img_msg);

    // Assign the initial state if initialized successfully.
    void assignInitialState(std::vector<sensor_msgs::Imu>& imu_msg_buffer, IMUState& imu_state);

    // If initialized.
    bool ifInitialized() {
        return bInit;
    }

private:

    // Time lower bound for used imu data.
    double lower_time_bound;

    // Threshold for deciding which imu is at the same time with a img frame
    double imu_img_timeTh;

    // Flag indicating if initialized.
    bool bInit;

    // Error bewteen timestamp of imu and img.
    double td;

    // Error between img and its nearest imu msg.
    double ddt;

    // Relative rotation between camera and imu.
    Eigen::Matrix3d RIC;
    Eigen::Vector3d TIC;

    // Initialize results.
    double state_time;
    Eigen::Vector3d gyro_bias;
    Eigen::Vector3d acc_bias;

    Eigen::Matrix3d orientation;

    Eigen::Vector3d position;
    Eigen::Vector3d velocity;

    // Flag for declare the first imu data.
    bool first_imu;

    // Imu data for initialize every imu preintegration base.
    Eigen::Vector3d acc_0, gyr_0;

    // Frame counter in sliding window.
    int frame_count;

    // Current imu time.
    double curr_time;

    // Imu noise param.
    double acc_n, acc_w;
    double gyr_n, gyr_w;

    // IMU preintegration between keyframes.
    boost::shared_ptr<IntegrationBase> pre_integrations[(WINDOW_SIZE + 1)];

    // Temporal buff for imu preintegration between ordinary frames.
    boost::shared_ptr<IntegrationBase> tmp_pre_integration;

    // Store the information of ordinary frames
    std::map<double, ImageFrame> all_image_frame;

    // Bias of gyro and accelerometer of imu corresponding to every keyframe.
    Eigen::Vector3d Bas[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Bgs[(WINDOW_SIZE + 1)];

    // Every member of this vector store the dt between every adjacent imu 
    // between two keyframes in sliding window.
    std::vector<double> dt_buf[(WINDOW_SIZE + 1)];

    // Every member of this two vectors store all imu measurements 
    // between two keyframes in sliding window.
    std::vector<Eigen::Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    std::vector<Eigen::Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    // Gravity under reference camera frame.
    Eigen::Vector3d g;

    // Feature manager.
    FeatureManager f_manager;

    // State of body frame under reference frame.
    Eigen::Matrix3d Rs[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Ps[(WINDOW_SIZE + 1)];
    Eigen::Vector3d Vs[(WINDOW_SIZE + 1)];

    // Flags for marginalization.
    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };
    MarginalizationFlag  marginalization_flag;

    // Timestamps of sliding window.
    std::vector<double> Times;

    // Initial timestamp
    double initial_timestamp;

    // For solving relative motion.
    MotionEstimator m_estimator;

private:

    // Process every imu frame before the img.
    void processIMU(const sensor_msgs::Imu& imu_msg);

    // Process img frame.
    void processImage(CameraMeasurementConstPtr img_msg);

    // Check if the condition is fit to conduct the vio initialization, and conduct it while suitable.
    bool initialStructure();

    // Try to recover relative pose.
    bool relativePose(Eigen::Matrix3d &relative_R, Eigen::Vector3d &relative_T, int &l);

    // Align the visual sfm with imu preintegration.
    bool visualInitialAlign();

    // Slide the window.
    void slideWindow();
};

}