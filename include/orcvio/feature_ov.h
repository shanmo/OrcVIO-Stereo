/*
 * COPYRIGHT AND PERMISSION NOTICE
 * UCSD Software ORCVIO
 * Copyright (C) 2021 
 * All rights reserved.
 */

#pragma once

#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>
#include <memory>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include "state.h"

namespace {
    /// Max condition number of linear triangulation matrix accept triangulated features
    const int max_cond_number = 10000;
    /// Minimum distance to accept triangulated features
    const double min_dist = 0.10;
    /// Minimum distance to accept triangulated features
    const double max_dist = 60;
    /// Max baseline ratio to accept triangulated features
    double max_baseline = 40;
}

namespace orcvio {

#if WITH_LC
    /**
     * @brief Feature for loop closure
     * 
     */
    class FeatureLC {
    public:
        /// Unique ID of this feature
        size_t featid;

        /// UV coordinates that this feature has been seen from (mapped by camera ID)
        std::unordered_map<size_t, std::vector<Eigen::Vector4d>> uvs;

        /// UV normalized coordinates that this feature has been seen from (mapped by camera ID)
        std::unordered_map<size_t, std::vector<Eigen::Vector4d>> uvs_norm;

        /// Timestamps of each UV measurement (mapped by camera ID)
        std::unordered_map<size_t, std::vector<double>> timestamps;

        /// Triangulated position of this feature, in the global frame
        Eigen::Vector3d p_FinG;
    };
    using FeatureLcPtr = std::shared_ptr<FeatureLC>;
    using FeatureLcConstPtr = std::shared_ptr<const FeatureLC>;
#endif    

/**
* @brief Feature Salient part of an image. Please refer
*    to the Appendix of "A Multi-State Constraint Kalman
*    Filter for Vision-aided Inertial Navigation" for how
*    the 3d position of a feature is initialized.
*/
struct Feature {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef long long int FeatureIDType;

    /**
    * @brief OptimizationConfig Configuration parameters for 3d feature position optimization.
    */
    struct OptimizationConfig {
        double translation_threshold;
        double huber_epsilon;
        double estimation_precision;
        double initial_damping;
        int outer_loop_max_iteration;
        int inner_loop_max_iteration;

        OptimizationConfig() :
                translation_threshold(0.2),
                huber_epsilon(0.01),
                estimation_precision(5e-7),
                initial_damping(1e-3),
                outer_loop_max_iteration(10),
                inner_loop_max_iteration(10) {
            return;
        }
    };

    // Constructors for the class.
    Feature() : id(0), position(Eigen::Vector3d::Zero()), is_initialized(false) {}

    Feature(const FeatureIDType &new_id) : id(new_id), position(Eigen::Vector3d::Zero()), is_initialized(false) {}

    /**
    * @brief cost Compute the cost of the camera observations
    * @param T_c0_c1 A rigid body transformation takes a vector in c0 frame to ci frame.
    * @param x The current estimation.
    * @param z The ith measurement of the feature j in ci frame.
    * @return e The cost of this observation.
    */
    void cost(const Eigen::Isometry3d &T_c0_ci,
                const Eigen::Vector3d &x, 
                const Eigen::Vector2d &z, 
                double &e) const noexcept;

    /**
    * @brief jacobian Compute the Jacobian of the camera observation wrt the feature position 
    * @param T_c0_c1 A rigid body transformation takes a vector in c0 frame to ci frame.
    * @param x The current estimation.
    * @param z The actual measurement of the feature in ci frame.
    * @return J The computed Jacobian.
    * @return r The computed residual.
    * @return w Weight induced by huber kernel.
    */
    void jacobian(const Eigen::Isometry3d &T_c0_ci,
                    const Eigen::Vector3d &x, 
                    const Eigen::Vector2d &z,
                    Eigen::Matrix<double, 2, 3> &J, 
                    Eigen::Vector2d &r, double &w) const noexcept;

    /**
    * @brief triangulate features using openvins method, ref https://docs.openvins.com/update-featinit.html 
    * @param cam_poses: pose of cameras  
    * @param measurements: feature observations
    * @param p: Computed feature position in first camera frame
    * @return boolean flag whether the initialization is successful 
    */
    bool generateInitialGuess(
            const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d> > &cam_poses,
            const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > &measurements,
            Eigen::Vector3d &p) const noexcept;

    /**
    * @brief checkMotion
    *    Check the input camera poses to ensure there is enough translation to triangulate the feature positon.
    * @param cam_states : input camera poses.
    * @return True if the translation between the input camera poses is sufficient.
    */
    bool checkMotion(const CamStateServer &cam_states) const noexcept;

    /**
    * @brief InitializePosition Intialize the feature position based on all current available measurements.
    * @param cam_states: A map containing the camera poses with its ID as the associated key value.
    * @return The computed 3d position is used to set the position member variable. Note the resulted position is in world frame.
    * @return True if the estimated 3d position of the feature is valid.
    */
    bool initializePosition(const CamStateServer &cam_states) noexcept;


    // An unique identifier for the feature.
    // In case of long time running, the variable
    // type of id is set to FeatureIDType in order
    // to avoid duplication.
    FeatureIDType id;

    // id for next feature
    static FeatureIDType next_id;

    // Store the observations of the features in the
    // state_id(key)-image_coordinates(value) manner.
    std::map<StateIDType, Eigen::Vector4d, std::less<StateIDType>,
            Eigen::aligned_allocator<std::pair<const StateIDType, Eigen::Vector4d> > > observations;

#if WITH_LC
    // pixel coordinates for loop closure
    std::map<StateIDType, Eigen::Vector4d, std::less<StateIDType>,
            Eigen::aligned_allocator<std::pair<const StateIDType, Eigen::Vector4d> > > observations_uvs;   

    // double timestamp;
    std::map<StateIDType, double> timestamp;
#endif        

    // 3d postion of the feature in the world frame.
    Eigen::Vector3d position;

    // A indicator to show if the 3d postion of the feature has been initialized or not.
    bool is_initialized;

    // Noise for a normalized feature measurement.
    static double observation_noise;

    // Optimization configuration for solving the 3d position.
    static OptimizationConfig optimization_config;
};

typedef Feature::FeatureIDType FeatureIDType;
typedef std::map<FeatureIDType, Feature, std::less<int>,
        Eigen::aligned_allocator<std::pair<const FeatureIDType, Feature> > > MapServer;

} // namespace orcvio

