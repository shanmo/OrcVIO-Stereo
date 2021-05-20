#pragma once 

#include <iostream>
#include <boost/shared_ptr.hpp>

#include <sensor_msgs/Imu.h>

#include "initializer/StaticInitializer.h"
#include "initializer/DynamicInitializer.h"

namespace orcvio {

class FlexibleInitializer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    FlexibleInitializer() = delete;
    FlexibleInitializer(const double& acc_n_, const double& acc_w_, const double& gyr_n_, 
        const double& gyr_w_, const Eigen::Matrix3d& R_c2b, 
        const Eigen::Vector3d& t_bc_b) {
        
        staticInitPtr.reset(new StaticInitializer());
        dynamicInitPtr.reset(new DynamicInitializer(acc_n_, acc_w_, gyr_n_,
            gyr_w_, R_c2b, t_bc_b));

    }

    // Destructor
    ~FlexibleInitializer(){}

    // Interface for trying to initialize
    bool tryIncInit(std::vector<sensor_msgs::Imu>& imu_msg_buffer,
        CameraMeasurementConstPtr img_msg, IMUState& imu_state);

private:

    // Inclinometer-initializer
    boost::shared_ptr<StaticInitializer> staticInitPtr;
    // Dynamic initializer
    boost::shared_ptr<DynamicInitializer> dynamicInitPtr;


};

}