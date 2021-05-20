#include "initializer/FlexibleInitializer.h"

namespace orcvio {

bool FlexibleInitializer::tryIncInit(std::vector<sensor_msgs::Imu>& imu_msg_buffer,
        CameraMeasurementConstPtr img_msg, IMUState& imu_state) {

    if(staticInitPtr->tryIncInit(imu_msg_buffer, img_msg)) {
        staticInitPtr->assignInitialState(imu_msg_buffer, imu_state);
        return true;
    }  
    else if (dynamicInitPtr->tryDynInit(imu_msg_buffer, img_msg)) {
        dynamicInitPtr->assignInitialState(imu_msg_buffer, imu_state);
        return true;
    }

    return false;
}


}
