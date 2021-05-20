#pragma once 

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>

#include <Eigen/Dense>

#include "orcvio/FeatureMeasurement.h"
#include "orcvio/CameraMeasurement.h"

namespace {

  const int WINDOW_SIZE = 10;
  const double MIN_PARALLAX = 10/460;

}

namespace orcvio {

class FeaturePerFrame
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    FeaturePerFrame(const FeatureMeasurement& _point, double td)
    {
        point.x() = _point.u0;
        point.y() = _point.v0;
        point.z() = 1;
        cur_td = td;
    }
    double cur_td;
    Eigen::Vector3d point;
    double z;
    bool is_used;
    double parallax;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    double dep_gradient;
};

class FeaturePerId
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    const int feature_id;
    int start_frame;
    std::vector<FeaturePerFrame> feature_per_frame;

    int used_num;
    bool is_outlier;
    bool is_margin;
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Eigen::Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame();
};

class FeatureManager
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FeatureManager(){};

    void setRic(const Eigen::Matrix3d& _ric);

    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(int frame_count, CameraMeasurementConstPtr image, double td);
    
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void setDepth(const Eigen::VectorXd &x);
    void removeFailures();
    void clearDepth(const Eigen::VectorXd &x);
    Eigen::VectorXd getDepthVector();
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    std::list<FeaturePerId> feature;
    int last_track_num;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    Eigen::Matrix3d ric;
};

}

