#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>

#include "orcvio/euroc_gt.h"

using namespace orcvio;

int main(int argc, char **argv) {

    // Create ros node
    ros::init(argc, argv, "publish_gt_path");
    ros::NodeHandle nh("~");

    // Get parameters to subscribe
    std::string path_gt, topic_est;
    nh.getParam("path_gt", path_gt);
    nh.getParam("topic_pose_est", topic_est);

    // Debug
    ROS_INFO("Done reading config values");
    ROS_INFO(" - path for gt pose = %s", path_gt.c_str());
    ROS_INFO(" - topic for estimated pose = %s", topic_est.c_str());

    EuRoCPub euroc_pub(nh, path_gt, topic_est);

    // Done!
    ros::spin();
    return EXIT_SUCCESS;

}