/*
 * COPYRIGHT AND PERMISSION NOTICE
 * UCSD Software ORCVIO
 * Copyright (C) 2021 
 * All rights reserved.
 */

#include <orcvio/image_processor_nodelet.h>

namespace orcvio {

void ImageProcessorNodelet::onInit() {
    img_processor_ptr.reset(new ImageProcessor(getPrivateNodeHandle()));
    if (!img_processor_ptr->initialize()) {
        ROS_ERROR("Cannot initialize Image Processor...");
        return;
    }
    return;
}

PLUGINLIB_EXPORT_CLASS(orcvio::ImageProcessorNodelet, nodelet::Nodelet);

}  // end namespace orcvio
