#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <map>

#include <boost/shared_ptr.hpp>

#include "orcvio/feature.h"
#include "orcvio/CameraMeasurement.h"
#include "initializer/ImuPreintegration.h"
#include "initializer/feature_manager.h"

namespace orcvio {

class ImageFrame
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        ImageFrame(){};
        ImageFrame(CameraMeasurementConstPtr _points, const double& td):is_key_frame{false}
        {
            t = _points->header.stamp.toSec() + td;
            for (const auto& pt : _points->features)
            {
                // only x, y values are used 
                double x = pt.u0;
                double y = pt.v0;
                double z = 1;
                Eigen::Vector3d xyz;
                xyz << x, y, z;
                points[pt.id] = xyz;
            }
        };
        // points are used in initialStructure
        // only x, y values are used 
        std::map<int, Eigen::Vector3d> points;
        double t;
        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        boost::shared_ptr<IntegrationBase> pre_integration;
        bool is_key_frame;
};

bool VisualIMUAlignment(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d* Bgs, Eigen::Vector3d &g, Eigen::VectorXd &x, const Eigen::Vector3d& TIC);

}


