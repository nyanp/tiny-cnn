/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "gtest/gtest.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn::activation;

#include "test_activation_layer.h"
#include "test_average_pooling_layer.h"
#include "test_network.h"
// TODO(yida): fix broken test
//#include "test_average_unpooling_layer.h"
#include "test_batch_norm_layer.h"
#include "test_concat_layer.h"
#include "test_convolutional_layer.h"
#include "test_core.h"
#include "test_deconvolutional_layer.h"
#include "test_dropout_layer.h"
#include "test_fully_connected_layer.h"
#include "test_global_average_pooling_layer.h"
#include "test_large_thread_count.h"
#include "test_lrn_layer.h"
#include "test_max_pooling_layer.h"
#include "test_models.h"
#include "test_node.h"
#include "test_nodes.h"
#include "test_power_layer.h"
//#include "test_quantization.h"
//#include "test_quantized_convolutional_layer.h"
//#include "test_quantized_deconvolutional_layer.h"
#include "test_slice_layer.h"
#include "test_target_cost.h"
#include "test_tensor.h"
#include "test_optimizers.h"

#ifndef CNN_NO_SERIALIZATION
#include "test_serialization.h"
#endif  // CNN_NO_SERIALIZATION

#ifdef CNN_USE_GEMMLOWP
#include "test_quantized_fully_connected_layer.h"
#endif  // CNN_USE_GEMMLOOP

#ifdef CNN_USE_CAFFE_CONVERTER
#include "test_caffe_converter.h"
#endif  // CNN_USE_CAFFE_CONVERTER

#ifdef DNN_USE_IMAGE_API
#include "test_image.h"
#endif  // DNN_USE_IMAGE_API

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
