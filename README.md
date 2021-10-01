# Readme

This repo is used in OrcVIO [project website](http://me-llamo-sean.cf/orcvio_githubpage/), [Journal version](https://arxiv.org/abs/2007.15107)

If you find this repo useful, kindly cite our publications 

```
@inproceedings{shan2020orcvio,
  title={OrcVIO: Object residual constrained Visual-Inertial Odometry},
  author={Shan, Mo and Feng, Qiaojun and Atanasov, Nikolay},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={5104--5111},
  year={2020},
  organization={IEEE}
}  
```

# OrcVIO 

* Tested on Ubuntu 18.04 with ROS Melodic 
* The core algorithm depends on `Eigen`, `Boost`, `Suitesparse`, `OpenCV`, `Sophus`, `ceres`, `spdlog`
* logging is using [spdlog](https://github.com/gabime/spdlog), follow [this link](https://github.com/gabime/spdlog/issues/1366#issuecomment-567748773) and [this link](https://github.com/gabime/spdlog/issues/1405#issuecomment-603859234) to install it 

# Build 

```
mkdir -p stereo_orcvio_ws/src
cd stereo_orcvio_ws/src
git clone https://github.com/shanmo/OrcVIO-Stereo.git
cd ../
catkin_make --pkg orcvio --cmake-args -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash 
```

# Docker 

- This docker images supports OpenVINS, VINS fusion, OrcVIO, and OS is Ubuntu 18.04, with ROS Melodic, and VNC
```
docker pull moshan/open_orcvio_cpp_docker:cpu
```

# Run 

- EuRoC dataset 
```sh 
roslaunch orcvio orcvio_euroc.launch
rosbag play V1_01_easy.bag or rosbag play MH_01_easy.bag
```
- to visualize the groundtruth path for EuRoC, need to specify the csv file path in the launch file 

- loop closure follows OpenVINS + [ov_secondary](https://github.com/rpng/ov_secondary), this is enabled/disabled by `add_definitions(-w -DWITH_LC=1) # close warning` in cmake, to run it 
1. download and build **ov_secondary**, run
  ```sh
  roslaunch loop_fusion posegraph.launch
  ```
2. run OrcVIO with EoRoC MH_01_easy
  ```sh
  roslaunch orcvio orcvio_euroc.launch 
  rosbag play MH_01_easy.bag
  ```

# evaluation 

### ATE on EuRoC

- Stereo OrcVIO is evaluated on EuRoC, and the results show that our method is comparable with SOTA. 
- Note that the start time are all 0s. 

|Dataset         |  VINS  |  S-MSCKF  |ORB SLAM  |   SVO2  |   Stereo OrcVIO |
|:-:|:-:|:-:|:-:|:-:|:-:|
|Sensor          | Mono+IMU|Stereo+IMU|Stereo   |Stereo   |Stereo+IMU|
|MH_01_easy      |0.156025 |   x      |**0.037896** |0.111732 |0.231|
|MH_02_easy      |0.178418 | 0.152133 |**0.044086** |    x    |0.416|
|MH_03_medium    |0.194874 | 0.289593 |**0.040688** |0.360784 |0.279|
|MH_04_difficult |0.346300 | 0.210353 |**0.088795** |2.891935 |0.320|
|MH_05_difficult |0.302346 | 0.293128 |**0.067401** |1.199866 |0.453|
|V1_01_easy      |0.088925 | 0.070955 |0.087481 |0.061025 |**0.056**|
|V1_02_medium    |0.110438 | 0.126732 |0.079843 |**0.081536**|0.168|
|V1_03_difficult |**0.187195**| 0.203363 |0.284315 |0.248401 |0.203|
|V2_01_easy      |0.086263 |**0.065962**|0.077287 |0.076514 |0.073|
|V2_02_medium    |0.157444 | 0.175961 |**0.117782**|0.204471 |0.208|
|V2_03_difficult |**0.277569** |   x      |  x      |   x     | x      |

### Efficiency 

We evaluated the speed of Stereo OrcVIO with Xavier NX and Realsense D435i, and here are the result. It shows that Stereo OrcVIO can run in real time.   

| Freq. (Hz) | Detection  | VIO only  | VIO + mapping  |
|:-:|:-:|:-:|:-:|
| YOLO4  | 2.1  | 25.8  | 22.7  |
| YOLO3  | 4.9  |  25.8 | 21.6  |
| YOLO3-tiny  | 26.2  | 25.8  | 17.8  |

# ROS Nodes

### `image_processor` node

**Subscribed Topics**

* `imu` (`sensor_msgs/Imu`): IMU messages is used for compensating rotation in feature tracking, and 2-point RANSAC.

* `cam[x]_image` (`sensor_msgs/Image`): Synchronized stereo images.

**Published Topics**

* `features` (`orcvio/CameraMeasurement`): Records the feature measurements on the current stereo image pair.

* `tracking_info` (`orcvio/TrackingInfo`): Records the feature tracking status for debugging purpose.

* `debug_stereo_img` (`sensor_msgs::Image`): Draw current features on the stereo images for debugging purpose. Note that this debugging image is only generated upon subscription.

### `vio` node

**Subscribed Topics**

* `imu` (`sensor_msgs/Imu`): IMU measurements.

* `features` (`orcvio/CameraMeasurement`): Stereo feature measurements from the `image_processor` node.

**Published Topics**

* `odom` (`nav_msgs/Odometry`): Odometry of the IMU frame including a proper covariance.

* `feature_point_cloud` (`sensor_msgs/PointCloud2`): Shows current features in the map which is used for estimation.

# File structure 

```
.
├── assets
├── cache
├── cmake
│   └── FindSuiteSparse.cmake
├── CMakeLists.txt
├── config
│   ├── camchain-imucam-dcist-acl.yaml
│   ├── camchain-imucam-dcist-erl.yaml
│   ├── camchain-imucam-dcist-mit.yaml
│   ├── camchain-imucam-erl.yaml
│   ├── camchain-imucam-euroc-noextrinsics.yaml
│   ├── camchain-imucam-euroc.yaml
│   ├── camchain-imucam-fla.yaml
│   └── euroc_config
│       ├── MH_01_easy.yaml
│       └── V1_01_easy.yaml
├── include
│   ├── initializer
│   │   ├── DynamicInitializer.h
│   │   ├── feature_manager.h
│   │   ├── FlexibleInitializer.h
│   │   ├── ImuPreintegration.h
│   │   ├── initial_alignment.h
│   │   ├── initial_sfm.h
│   │   ├── math_utils.h
│   │   ├── solve_5pts.h
│   │   └── StaticInitializer.h
│   └── orcvio
│       ├── dataset_reader.h
│       ├── euroc_gt.h
│       ├── feature.h
│       ├── image_processor.h
│       ├── image_processor_nodelet.h
│       ├── orcvio.h
│       ├── orcvio_nodelet.h
│       ├── state.h
│       └── utils.h
├── launch
│   ├── image_processor_dcist.launch
│   ├── image_processor_euroc.launch
│   ├── image_processor_fla.launch
│   ├── orcvio_dcist.launch
│   ├── orcvio_euroc_eval.launch
│   ├── orcvio_euroc.launch
│   ├── orcvio_fla.launch
│   └── reset.launch
├── LICENSE.txt
├── msg
│   ├── CameraMeasurement.msg
│   ├── FeatureMeasurement.msg
│   └── TrackingInfo.msg
├── nodelets.xml
├── package.xml
├── README.md
├── rviz
│   ├── rviz_dcist_config.rviz
│   ├── rviz_euroc_config.rviz
│   └── rviz_fla_config.rviz
├── scripts
│   └── batch_run_euroc.py
├── src
│   ├── DynamicInitializer.cpp
│   ├── euroc_gt.cpp
│   ├── feature_manager.cpp
│   ├── FlexibleInitializer.cpp
│   ├── image_processor.cpp
│   ├── image_processor_nodelet.cpp
│   ├── initial_alignment.cpp
│   ├── initial_sfm.cpp
│   ├── orcvio.cpp
│   ├── orcvio_nodelet.cpp
│   ├── publish_euroc_gt.cpp
│   ├── solve_5pts.cpp
│   ├── StaticInitializer.cpp
│   └── utils.cpp
└── test
    ├── feature_initialization_test.cpp
    └── math_utils_test.cpp
```

# Acknowledgement 

Some parts of the repo are from the following 

- https://github.com/KumarRobotics/msckf_vio
- https://github.com/cggos/msckf_cg
- https://github.com/PetWorm/LARVIO
- https://github.com/rpng/open_vins
- https://github.com/symao/open_vins
- https://github.com/symao/vio_evaluation
- https://github.com/HKUST-Aerial-Robotics/VINS-Mono
