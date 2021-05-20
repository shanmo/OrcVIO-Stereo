/*
 * COPYRIGHT AND PERMISSION NOTICE
 * UCSD Software ORCVIO
 * Copyright (C) 2021 
 * All rights reserved.
 */

#pragma once

#include <orcvio/orcvio.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

namespace orcvio {
class OrcVioNodelet : public nodelet::Nodelet {
   public:
    OrcVioNodelet() { return; }
    ~OrcVioNodelet() { return; }

   private:
    virtual void onInit();
    OrcVioPtr orcvio_ptr;
};
}  // end namespace orcvio


