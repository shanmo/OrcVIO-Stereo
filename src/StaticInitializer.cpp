#include "initializer/StaticInitializer.h"

#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>

using namespace std;
using namespace Eigen;

namespace orcvio {

bool StaticInitializer::tryIncInit(const std::vector<sensor_msgs::Imu>& imu_msg_buffer,
    CameraMeasurementConstPtr img_msg) {
        
    // return false if this is the 1st image for inclinometer-initializer
    if (0 == staticImgCounter) {
        staticImgCounter++;
        init_features.clear();
        // add features to init_features
        for (const auto& feature : img_msg->features)
            init_features[feature.id] = Vector2d(feature.u0, feature.v0);
        // assign the lower time bound as the timestamp of first img
        lower_time_bound = img_msg->header.stamp.toSec();
        return false;
    }

    // calculate feature distance of matched features between prev and curr images
    InitFeatures curr_features;
    list<double> feature_dis;
    for (const auto& feature : img_msg->features) {
    curr_features[feature.id] = Vector2d(feature.u0, feature.v0);
        if (init_features.find(feature.id) != init_features.end()) {
            Vector2d vec2d_c(feature.u0, feature.v0);
            Vector2d vec2d_p = init_features[feature.id];
            feature_dis.emplace_back((vec2d_c-vec2d_p).norm());
        }
    }
    // return false if number of matched features is small
    if (feature_dis.empty() || feature_dis.size() < 20) {  
        staticImgCounter = 0;
        return false;
    }
    // ignore outliers rudely
    // ignore 20 features that have max. distance 
    feature_dis.sort();
    auto itr = feature_dis.end();
    for (int i = 0; i < 19; i++)  
        itr--;
    double maxDis = *itr;
    // classified as static image if maxDis is smaller than threshold, otherwise reset image counter
    if (maxDis < max_feature_dis) {
        staticImgCounter++;
        init_features.swap(curr_features);
        if (staticImgCounter < static_Num)  // return false if number of consecitive static images does not reach @static_Num
            return false;
        } 
    else 
    {
        //    printf("inclinometer-initializer failed at No.%d static image.",staticImgCounter+1);
        staticImgCounter = 0;
        return false;
    }

    /* reach here means staticImgCounter is equal to static_Num */

    // initialize rotation and gyro bias by imu data between the 1st and the No.static_Num image
    // set take_off_stamp as time of the No.static_Num image
    // set initial imu_state as the state of No.static_Num image
    // erase imu data with timestamp earlier than the No.static_Num image
    initializeGravityAndBias(img_msg->header.stamp.toSec(), imu_msg_buffer);

    bInit = true;

    return true;
}


void StaticInitializer::initializeGravityAndBias(const double& time_bound, 
    const std::vector<sensor_msgs::Imu>& imu_msg_buffer) noexcept {

    // Initialize gravity and gyro bias.
    // take the average acceleration and angular velocity  
    Vector3d sum_angular_vel = Vector3d::Zero();
    Vector3d sum_linear_acc  = Vector3d::Zero();

    int usefulImuSize = 0;
    double last_imu_time;
    std::for_each(imu_msg_buffer.begin(), imu_msg_buffer.end(),
        [&] (const auto& imu_msg) {
            
            double imu_time = imu_msg.header.stamp.toSec();
            if (imu_time >= lower_time_bound && imu_time <= time_bound)
            {
                Vector3d angular_vel = Vector3d::Zero();
                Vector3d linear_acc  = Vector3d::Zero();

                tf::vectorMsgToEigen(imu_msg.angular_velocity, angular_vel);
                tf::vectorMsgToEigen(imu_msg.linear_acceleration, linear_acc);

                sum_angular_vel += angular_vel;
                sum_linear_acc  += linear_acc;

                usefulImuSize++;
                last_imu_time = imu_time;
            }
        });

    // average angular velocity as bias 
    gyro_bias = sum_angular_vel / usefulImuSize;

    // avearge acceleration as gravity 
    // This is the gravity in the IMU frame
    Vector3d gravity_imu = sum_linear_acc / usefulImuSize;

    // Initialize the initial orientation, so that the estimation is consistent with the inertial frame.
    // norm of average acceleration to be norm of gravity 
    double gravity_norm = gravity_imu.norm(); 

    // gravity in world frame 
    IMUState::gravity = Vector3d(0.0, 0.0, -gravity_norm); 

    // obtain the transformation between world frame gravity and IMU frame gravity 
    // world frame is NED or forward right down, so -IMUState::gravity is used 
    // q0_i_w is from IMU to world frame 
    Quaterniond q0_i_w = Quaterniond::FromTwoVectors(gravity_imu, -IMUState::gravity);
    orientation = q0_i_w.toRotationMatrix();

    // Set other state and timestamp
    state_time = last_imu_time;
    position = Vector3d(0.0, 0.0, 0.0);
    velocity = Vector3d(0.0, 0.0, 0.0);
    acc_bias = Vector3d(0.0, 0.0, 0.0);

    ROS_INFO("Inclinometer-initializer completed by using %d imu data !!!\n\n", usefulImuSize);

    return;
}


void StaticInitializer::assignInitialState(std::vector<sensor_msgs::Imu>& imu_msg_buffer, IMUState& imu_state) {
    
    if (!bInit) {
        printf("Cannot assign initial state before initialization !!!\n");
        return;
    }

    // Remove used imu data
    int usefulImuSize = 0;
    for (const auto& imu_msg : imu_msg_buffer) {
        double imu_time = imu_msg.header.stamp.toSec();
        if (imu_time > state_time) break;
        usefulImuSize++;
    }
    if (usefulImuSize >= imu_msg_buffer.size())
        usefulImuSize--;

    // Earse used imu data
    imu_msg_buffer.erase(imu_msg_buffer.begin(),
        imu_msg_buffer.begin() + usefulImuSize);

    // Set initial state
    imu_state.time = state_time;
    imu_state.gyro_bias = gyro_bias;
    imu_state.acc_bias = acc_bias;
    imu_state.orientation = orientation;
    imu_state.position = position;
    imu_state.velocity = velocity;

    return;
}


}