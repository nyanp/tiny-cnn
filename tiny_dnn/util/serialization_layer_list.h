/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/layers/layers.h"

namespace tiny_dnn {

template <typename T>
void register_layers(T* h) {
  h->template register_layer<elementwise_add_layer>("elementwise_add");
  h->template register_layer<average_pooling_layer>("avepool");
  h->template register_layer<average_unpooling_layer>("aveunpool");
  h->template register_layer<batch_normalization_layer>("batchnorm");
  h->template register_layer<concat_layer>("concat");
  h->template register_layer<convolutional_layer>("conv");
  h->template register_layer<deconvolutional_layer>("deconv");
  h->template register_layer<dropout_layer>("dropout");
  h->template register_layer<fully_connected_layer>("fully_connected");
  h->template register_layer<global_average_pooling_layer>(
    "global_average_pooling");
  h->template register_layer<input_layer>("input");
  h->template register_layer<linear_layer>("linear");
  h->template register_layer<lrn_layer>("lrn");
  h->template register_layer<max_pooling_layer>("maxpool");
  h->template register_layer<max_unpooling_layer>("maxunpool");
  h->template register_layer<power_layer>("power");
  //h->template register_layer<quantized_convolutional_layer>("q_conv");
  //h->template register_layer<quantized_deconvolutional_layer>("q_deconv");
  //h->template register_layer<quantized_fully_connected_layer>("q_fully_connected");
  h->template register_layer<recurrent_cell_layer>("recurrent_cell");
  h->template register_layer<slice_layer>("slice");

  h->template register_layer<sigmoid_layer>("sigmoid");
  h->template register_layer<tanh_layer>("tanh");
  h->template register_layer<relu_layer>("relu");
  h->template register_layer<softmax_layer>("softmax");
  h->template register_layer<leaky_relu_layer>("leaky_relu");
  h->template register_layer<elu_layer>("elu");
  h->template register_layer<tanh_p1m2_layer>("tanh_scaled");
  h->template register_layer<softplus_layer>("softplus");
  h->template register_layer<softsign_layer>("softsign");
  h->template register_layer<selu_layer>("selu");
}

}  // namespace tiny_dnn
