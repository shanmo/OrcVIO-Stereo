#pragma once 

#include <map>
#include <list>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <sensor_msgs/Imu.h>

#include "orcvio/state.h"
#include "orcvio/feature.h"
#include "orcvio/CameraMeasurement.h"

namespace orcvio {

class StaticInitializer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    StaticInitializer() :
    max_feature_dis(2e-3),
    bInit(false),
    lower_time_bound(0.0)
    {        
        staticImgCounter = 0;
        init_features.clear();
        gyro_bias = Eigen::Vector3d::Zero();
        acc_bias = Eigen::Vector3d::Zero();
        position = Eigen::Vector3d::Zero();
        velocity = Eigen::Vector3d::Zero();

        orientation = Eigen::Matrix3d::Identity();

        // FIXEME: load from launch file 
        static_Num = 20; 
    }

    // Destructor
    ~StaticInitializer(){}

    // Interface for trying to initialize
    bool tryIncInit(const std::vector<sensor_msgs::Imu>& imu_msg_buffer,
        CameraMeasurementConstPtr img_msg);

    // Assign the initial state if initialized successfully
    void assignInitialState(std::vector<sensor_msgs::Imu>& imu_msg_buffer, IMUState& imu_state);

    // If initialized
    bool ifInitialized() {
        return bInit;
    }

private:

    // Time lower bound for used imu data.
    double lower_time_bound;

    // Maximum feature distance allowed bewteen static images
    double max_feature_dis;

    // Number of consecutive image for trigger static initializer
    unsigned int static_Num;

    // Defined type for initialization
    typedef std::map<FeatureIDType, Eigen::Vector2d, std::less<int>,
        Eigen::aligned_allocator<
        std::pair<const FeatureIDType, Eigen::Vector2d> > > InitFeatures;
    InitFeatures init_features;

    // Counter for static images that will be used in inclinometer-initializer
    unsigned int staticImgCounter;

    // Initialize results
    double state_time;
    Eigen::Vector3d gyro_bias;
    Eigen::Vector3d acc_bias;

    // Eigen::Vector4d orientation;
    Eigen::Matrix3d orientation;
    
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;

    // Flag indicating if initialized
    bool bInit;

    /**
    * @brief initializegravityAndBias
    *    Initialize the IMU bias and initial orientation based on the first few IMU readings.
    * @details take the average of first N frames, average acceleration norm is gravity norm, 
    *          average angular velocity is gyro bias 
    *          IMU needs to be static for the first N frames 
    */
    void initializeGravityAndBias(const double& time_bound,
        const std::vector<sensor_msgs::Imu>& imu_msg_buffer) noexcept;
};

} // namespace orcvio