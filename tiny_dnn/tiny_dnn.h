/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/config.h"
#include "tiny_dnn/network.h"
#include "tiny_dnn/nodes.h"

#include "tiny_dnn/core/framework/tensor.h"

#include "tiny_dnn/core/framework/device.h"
#include "tiny_dnn/core/framework/program_manager.h"

#include "tiny_dnn/layers/arithmetic_layer.h"
#include "tiny_dnn/layers/average_pooling_layer.h"
#include "tiny_dnn/layers/average_unpooling_layer.h"
#include "tiny_dnn/layers/batch_normalization_layer.h"
#include "tiny_dnn/layers/concat_layer.h"
#include "tiny_dnn/layers/convolutional_layer.h"
#include "tiny_dnn/layers/deconvolutional_layer.h"
#include "tiny_dnn/layers/dropout_layer.h"
#include "tiny_dnn/layers/fully_connected_layer.h"
#include "tiny_dnn/layers/global_average_pooling_layer.h"
#include "tiny_dnn/layers/input_layer.h"
#include "tiny_dnn/layers/linear_layer.h"
#include "tiny_dnn/layers/lrn_layer.h"
#include "tiny_dnn/layers/max_pooling_layer.h"
#include "tiny_dnn/layers/max_unpooling_layer.h"
#include "tiny_dnn/layers/power_layer.h"
//#include "tiny_dnn/layers/quantized_convolutional_layer.h"
//#include "tiny_dnn/layers/quantized_deconvolutional_layer.h"
#include "tiny_dnn/layers/recurrent_cell_layer.h"
#include "tiny_dnn/layers/slice_layer.h"

#include "tiny_dnn/activations/elu_layer.h"
#include "tiny_dnn/activations/leaky_relu_layer.h"
#include "tiny_dnn/activations/relu_layer.h"
#include "tiny_dnn/activations/selu_layer.h"
#include "tiny_dnn/activations/sigmoid_layer.h"
#include "tiny_dnn/activations/softmax_layer.h"
#include "tiny_dnn/activations/softplus_layer.h"
#include "tiny_dnn/activations/softsign_layer.h"
#include "tiny_dnn/activations/tanh_layer.h"
#include "tiny_dnn/activations/tanh_p1m2_layer.h"

#ifdef CNN_USE_GEMMLOWP
#include "tiny_dnn/layers/quantized_fully_connected_layer.h"
#endif  // CNN_USE_GEMMLOWP

#include "tiny_dnn/lossfunctions/loss_function.h"
#include "tiny_dnn/optimizers/optimizer.h"

#include "tiny_dnn/util/deform.h"
#include "tiny_dnn/util/graph_visualizer.h"
#include "tiny_dnn/util/product.h"
#include "tiny_dnn/util/weight_init.h"

#include "tiny_dnn/io/cifar10_parser.h"
#include "tiny_dnn/io/display.h"
#include "tiny_dnn/io/layer_factory.h"
#include "tiny_dnn/io/mnist_parser.h"

#ifdef DNN_USE_IMAGE_API
#include "tiny_dnn/util/image.h"
#endif  // DNN_USE_IMAGE_API

#ifndef CNN_NO_SERIALIZATION
#include "tiny_dnn/util/deserialization_helper.h"
#include "tiny_dnn/util/serialization_helper.h"
// to allow upcasting
CEREAL_REGISTER_TYPE(tiny_dnn::elu_layer)
CEREAL_REGISTER_TYPE(tiny_dnn::leaky_relu_layer)
CEREAL_REGISTER_TYPE(tiny_dnn::relu_layer)
CEREAL_REGISTER_TYPE(tiny_dnn::sigmoid_layer)
CEREAL_REGISTER_TYPE(tiny_dnn::softmax_layer)
CEREAL_REGISTER_TYPE(tiny_dnn::softplus_layer)
CEREAL_REGISTER_TYPE(tiny_dnn::softsign_layer)
CEREAL_REGISTER_TYPE(tiny_dnn::tanh_layer)
CEREAL_REGISTER_TYPE(tiny_dnn::tanh_p1m2_layer)
#endif  // CNN_NO_SERIALIZATION

// shortcut version of layer names
namespace tiny_dnn {
namespace layers {

using conv = tiny_dnn::convolutional_layer;

// using q_conv = tiny_dnn::quantized_convolutional_layer;

using max_pool = tiny_dnn::max_pooling_layer;

using ave_pool = tiny_dnn::average_pooling_layer;

using fc = tiny_dnn::fully_connected_layer;

using dense = tiny_dnn::fully_connected_layer;

using recurrent_cell = tiny_dnn::recurrent_cell_layer;

#ifdef CNN_USE_GEMMLOWP
using q_fc = tiny_dnn::quantized_fully_connected_layer;
#endif

using add = tiny_dnn::elementwise_add_layer;

using dropout = tiny_dnn::dropout_layer;

using input = tiny_dnn::input_layer;

using linear = linear_layer;

using lrn = tiny_dnn::lrn_layer;

using concat = tiny_dnn::concat_layer;

using deconv = tiny_dnn::deconvolutional_layer;

using max_unpool = tiny_dnn::max_unpooling_layer;

using ave_unpool = tiny_dnn::average_unpooling_layer;

}  // namespace layers

namespace activation {

using sigmoid = tiny_dnn::sigmoid_layer;

using tanh = tiny_dnn::tanh_layer;

using relu = tiny_dnn::relu_layer;

using rectified_linear = tiny_dnn::relu_layer;

using softmax = tiny_dnn::softmax_layer;

using leaky_relu = tiny_dnn::leaky_relu_layer;

using elu = tiny_dnn::elu_layer;

using selu = tiny_dnn::selu_layer;

using tanh_p1m2 = tiny_dnn::tanh_p1m2_layer;

using softplus = tiny_dnn::softplus_layer;

using softsign = tiny_dnn::softsign_layer;

}  // namespace activation

#include "tiny_dnn/models/alexnet.h"

using batch_norm = tiny_dnn::batch_normalization_layer;

using slice = tiny_dnn::slice_layer;

using power = tiny_dnn::power_layer;

}  // namespace tiny_dnn

#ifdef CNN_USE_CAFFE_CONVERTER
// experimental / require google protobuf
#include "tiny_dnn/io/caffe/layer_factory.h"
#endif
