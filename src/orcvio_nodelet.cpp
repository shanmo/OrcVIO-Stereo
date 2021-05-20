/*
 * COPYRIGHT AND PERMISSION NOTICE
 * UCSD Software ORCVIO
 * Copyright (C) 2021 
 * All rights reserved.
 */

#include <orcvio/orcvio_nodelet.h>

namespace orcvio {
    
void OrcVioNodelet::onInit() {
    orcvio_ptr.reset(new OrcVio(getPrivateNodeHandle()));
    if (!orcvio_ptr->initialize()) {
        ROS_ERROR("Cannot initialize OrcVIO...");
        return;
    }
    return;
}

PLUGINLIB_EXPORT_CLASS(orcvio::OrcVioNodelet, nodelet::Nodelet);

}  // end namespace orcvio
