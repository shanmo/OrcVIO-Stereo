#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>

#include <Eigen/Eigen>
#include <Eigen/Geometry>

#include "orcvio/euroc_gt.h"

namespace orcvio 
{

    // this subscribes to the estimated pose
    void EuRoCPub::gt_odom_path_cb(const nav_msgs::Odometry::ConstPtr &est_odom_ptr){

        double timestamp = est_odom_ptr->header.stamp.toSec();
        // Our groundtruth state
        Eigen::Matrix<double,17,1> state_gt;

        // Check that we have the timestamp in our GT file [time(sec),q_GtoI,p_IinG,v_IinG,b_gyro,b_accel]
        if(!DatasetReader::get_gt_state(timestamp, state_gt, gt_states)) {
            return;
        }

        Eigen::Vector4d q_gt;
        Eigen::Vector3d t_gt;

        q_gt << state_gt(1,0),state_gt(2,0),state_gt(3,0),state_gt(4,0);
        t_gt << state_gt(5,0),state_gt(6,0),state_gt(7,0);

        Eigen::Quaterniond q_in = Eigen::Quaterniond(q_gt);

        if (first_pose_flag)
        {
            t0 = t_gt;
            // convert to rotation matrix
            R0 = q_in.normalized().toRotationMatrix();

            // align the estimated pose and groundtruth pose 
            t0 = slamTgt.linear() * t0 + slamTgt.translation(); 
            R0 = slamTgt.linear() * R0;

            first_pose_flag = false;
        }

        Eigen::Vector3d t1, t1_new;
        t1 = t_gt;

        // convert to rotation matrix
        Eigen::Matrix3d R1, R_new; 
        R1 = q_in.normalized().toRotationMatrix();

        // align the estimated pose and groundtruth pose 
        t1_new = slamTgt.linear() * t1 + slamTgt.translation(); 
        // initial position is 0 
        t1_new = t1_new - t0; 
        // NOTE, rotation is not identity since orcvio initializes using gravity 
        R_new = slamTgt.linear() * R1;

        // convert to quaternion
        Eigen::Quaterniond q1_normalized = Eigen::Quaterniond(R_new);
        // normalize the quaternion
        q1_normalized = q1_normalized.normalized();

        geometry_msgs::PoseStamped cur_pose;
        cur_pose.header = est_odom_ptr->header;
        cur_pose.header.frame_id = "/global";

        cur_pose.pose.position.x = t1_new(0,0);
        cur_pose.pose.position.y = t1_new(1,0);
        cur_pose.pose.position.z = t1_new(2,0);

        cur_pose.pose.orientation.x = q1_normalized.x(); 
        cur_pose.pose.orientation.y = q1_normalized.y(); 
        cur_pose.pose.orientation.z = q1_normalized.z(); 
        cur_pose.pose.orientation.w = q1_normalized.w(); 

        path.header = cur_pose.header;
        path.header.frame_id = "/global";
        path.poses.push_back(cur_pose);

        pub_gt_path.publish(path);

        // save the pose to txt for trajectory evaluation 
        // ============================
        // TUM format
        // timestamp tx ty tz qx qy qz qw
        fStateToSave << std::fixed << std::setprecision(3) << cur_pose.header.stamp.toSec();
        fStateToSave << " "
            << cur_pose.pose.position.x << " " << cur_pose.pose.position.y << " " << cur_pose.pose.position.z << " "
            << cur_pose.pose.orientation.x << " " << cur_pose.pose.orientation.y << " " << cur_pose.pose.orientation.z << " " << cur_pose.pose.orientation.w << std::endl; 
                

        return;

    }

}