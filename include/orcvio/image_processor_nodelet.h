/*
 * COPYRIGHT AND PERMISSION NOTICE
 * UCSD Software ORCVIO
 * Copyright (C) 2021 
 * All rights reserved.
 */

#pragma once

#include <orcvio/image_processor.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

namespace orcvio {
class ImageProcessorNodelet : public nodelet::Nodelet {
   public:
    ImageProcessorNodelet() { return; }
    ~ImageProcessorNodelet() { return; }

   private:
    virtual void onInit();
    ImageProcessorPtr img_processor_ptr;
};
}  // end namespace orcvio


