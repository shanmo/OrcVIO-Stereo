#pragma once

#include <string>
#include <fstream>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <Eigen/Eigen>
#include <Eigen/Geometry>

#include "dataset_reader.h"

// ref https://gist.github.com/kartikmohta/67cafc968ba5146bc6dcaf721e61b914

namespace orcvio  
{

    /**
     * @brief This class takes in published euroc gt and converts it to path for rviz 
     */
    class EuRoCPub {

        public:

        /**
         * @brief Default constructor 
         */
        EuRoCPub(ros::NodeHandle& nh, std::string path_gt, std::string topic_est) {
            
            DatasetReader::load_gt_file(path_gt, gt_states);
            pub_gt_path = nh.advertise<nav_msgs::Path>("/orcvio/gt_path", 2);
            sub_est_pose = nh.subscribe(topic_est, 9999, &EuRoCPub::gt_odom_path_cb, this);

            nh.param<std::string>("output_dir_traj", output_dir_traj, "./cache/");
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
            fStateToSave.open((output_dir_traj+"stamped_groundtruth.txt").c_str(), std::ofstream::trunc);

            first_pose_flag = true;

            std::vector<double> matrix_gtTslam;
            std::vector<double> matrix_gtTslam_default = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
            nh.param<std::vector<double>>("gtTslam", matrix_gtTslam, matrix_gtTslam_default);
            int i = 0;
            gtTslam.matrix() << matrix_gtTslam.at(0), matrix_gtTslam.at(1), matrix_gtTslam.at(2), matrix_gtTslam.at(3),
                matrix_gtTslam.at(4), matrix_gtTslam.at(5), matrix_gtTslam.at(6), matrix_gtTslam.at(7),
                matrix_gtTslam.at(8), matrix_gtTslam.at(9), matrix_gtTslam.at(10), matrix_gtTslam.at(11),
                matrix_gtTslam.at(12), matrix_gtTslam.at(13), matrix_gtTslam.at(14), matrix_gtTslam.at(15);

            slamTgt = gtTslam.inverse();

        };

        /**
         * @brief Default desctructor  
         */
        ~EuRoCPub()
        {
            fStateToSave.close();
        }

        /**
         * @brief callback function to convert odometry to path 
         * @param a pointer to the gt odometry 
         */
        void gt_odom_path_cb(const nav_msgs::Odometry::ConstPtr &est_odom_ptr);

        nav_msgs::Path path;
        
        Eigen::Matrix3d R0;
        Eigen::Vector3d t0;

        ros::Publisher pub_gt_path;
        // Our groundtruth states
        std::map<double, Eigen::Matrix<double,17,1>> gt_states;
        ros::Subscriber sub_est_pose;

        std::string output_dir_traj;
        std::ofstream fStateToSave;

        bool first_pose_flag;

        // for aligning gt and orcvio 
        Eigen::Isometry3d gtTslam, slamTgt;
    };

}

