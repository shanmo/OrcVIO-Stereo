cmake_minimum_required(VERSION 2.8.12) 
project(orcvio)

IF(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_compile_options(-std=c++14)

##################
## flags  start ##
##################

add_definitions(-w -DWITH_LC=1) # close warning
# add_definitions(-w -DWITH_LC=0) # close warning

add_definitions(-w -DWITH_LIFETIME_STATISTICS=0)

# for logging 
add_definitions(-w -DWITH_LOG=0)
# add_definitions(-w -DWITH_LOG=1)
find_package(spdlog)
# if spdlog is not installed, then no log 
# ref https://stackoverflow.com/a/10227196/8519188
IF(NOT spdlog_FOUND)
  MESSAGE(STATUS "Could not find spdlog")
  add_definitions(-w -DWITH_LOG=0)
ELSE()
  MESSAGE(STATUS "Found spdlog")
ENDIF()

##################
## flags end    ##
##################

# Modify cmake module path if new .cmake files are required
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  tf
  nav_msgs
  sensor_msgs
  geometry_msgs
  eigen_conversions
  tf_conversions
  random_numbers
  message_generation
  nodelet
  image_transport
  cv_bridge
  message_filters
  pcl_conversions
  pcl_ros
  std_srvs
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(SuiteSparse REQUIRED)
find_package(Sophus REQUIRED)
find_package(Ceres REQUIRED)

##################
## ROS messages ##
##################
add_message_files(
  FILES

  FeatureMeasurement.msg
  CameraMeasurement.msg
  TrackingInfo.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES orcvio image_processor
  CATKIN_DEPENDS
    roscpp std_msgs tf nav_msgs sensor_msgs geometry_msgs
    eigen_conversions tf_conversions random_numbers message_runtime
    image_transport cv_bridge message_filters pcl_conversions
    pcl_ros std_srvs
  DEPENDS Boost EIGEN3 OpenCV SUITESPARSE
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${SUITESPARSE_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

# OrcVio
add_library(orcvio
  src/orcvio.cpp
  src/utils.cpp
)
add_dependencies(orcvio
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(orcvio
  ${catkin_LIBRARIES}
  ${SUITESPARSE_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
)

IF(spdlog_FOUND)
target_link_libraries(orcvio
  spdlog::spdlog
)
ENDIF()

# Initializer
# -- Static Initializer
add_library(staticInitializer
    src/StaticInitializer.cpp
)
target_link_libraries(staticInitializer
    ${SUITESPARSE_LIBRARIES}
)
# -- Dynamic Initializer
add_library(dynamicInitializer
    src/DynamicInitializer.cpp
    src/feature_manager.cpp
    src/initial_alignment.cpp
    src/initial_sfm.cpp
    src/solve_5pts.cpp
)
target_link_libraries(dynamicInitializer
    ${SUITESPARSE_LIBRARIES}
    ${CERES_LIBRARIES}
    ${OpenCV_LIBRARIES}
)
# -- Flexible Initializer
add_library(flexibleInitializer
    src/FlexibleInitializer.cpp
)
target_link_libraries(flexibleInitializer
    staticInitializer
    dynamicInitializer
)
target_link_libraries(orcvio
    flexibleInitializer
)

# OrcVio nodelet
add_library(orcvio_nodelet
  src/orcvio_nodelet.cpp
)
add_dependencies(orcvio_nodelet
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(orcvio_nodelet
  orcvio
  ${catkin_LIBRARIES}
)
IF(spdlog_FOUND)
target_link_libraries(orcvio_nodelet
  spdlog::spdlog
)
ENDIF()


# Image processor
add_library(image_processor
  src/image_processor.cpp
  src/utils.cpp
)
add_dependencies(image_processor
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(image_processor
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
IF(spdlog_FOUND)
target_link_libraries(image_processor
  spdlog::spdlog
)
ENDIF()

# Image processor nodelet
add_library(image_processor_nodelet
  src/image_processor_nodelet.cpp
)
add_dependencies(image_processor_nodelet
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(image_processor_nodelet
  image_processor
  ${catkin_LIBRARIES}
)
IF(spdlog_FOUND)
target_link_libraries(image_processor_nodelet
  spdlog::spdlog
)
ENDIF()

# euroc gt path publisher
add_executable(publish_euroc_gt src/publish_euroc_gt.cpp src/euroc_gt.cpp)
target_link_libraries(publish_euroc_gt 
        ${catkin_LIBRARIES} 
        )
IF(spdlog_FOUND)
target_link_libraries(publish_euroc_gt
  spdlog::spdlog
)
ENDIF()


#############
## Install ##
#############

install(TARGETS
  orcvio orcvio_nodelet image_processor image_processor_nodelet 
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  PATTERN "*_nodelet.h" EXCLUDE
)

install(FILES nodelets.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############

IF(CATKIN_ENABLE_TESTING)
  # Feature initialization test
  catkin_add_gtest(test_feature_init
    test/feature_initialization_test.cpp
  )
  add_dependencies(test_feature_init
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS}
  )
  target_link_libraries(test_feature_init
    ${catkin_LIBRARIES}
  )
  IF(spdlog_FOUND)
  target_link_libraries(test_feature_init
    spdlog::spdlog
  )
  ENDIF()

  # Math utils test
  catkin_add_gtest(test_math_utils
    test/math_utils_test.cpp
  )
ENDIF()