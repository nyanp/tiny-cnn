/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <string>
#include "tiny_dnn/tiny_dnn.h"

#include "tiny_dnn/util/nn_error.h"
#include "tiny_dnn/util/parameter_init.h"

namespace tiny_dnn {
namespace models {

/**
 * VGG16 architecture for tiny-dnn model zoo.
 *
 * Reference:
 * Very Deep Convolutional Networks for Large-Scale Image Recognition
 * - (https://arxiv.org/abs/1409.1556)
 */
class vgg16 : public tiny_dnn::network<tiny_dnn::sequential> {
 public:
  explicit vgg16(const std::string &name = "vgg16", bool include_top = true)
    : tiny_dnn::network<tiny_dnn::sequential>(name) {
    using conv     = convolutional_layer;
    using relu     = relu_layer;
    using max_pool = max_pooling_layer;
    using dense    = fully_connected_layer;

    // Block 1
    *this << conv(224, 224, 3, 3, 3, 64, padding::same) << relu();
    *this << conv(224, 224, 3, 3, 64, 64, padding::same) << relu();
    *this << max_pool(224, 224, 64, 2) << relu();

    // Block 2
    *this << conv(112, 112, 3, 3, 64, 128, padding::same) << relu();
    *this << conv(112, 112, 3, 3, 128, 128, padding::same) << relu();
    *this << max_pool(112, 112, 128, 2) << relu();

    // Block 3
    *this << conv(56, 56, 3, 3, 128, 256, padding::same) << relu();
    *this << conv(56, 56, 3, 3, 256, 256, padding::same) << relu();
    *this << conv(56, 56, 3, 3, 256, 256, padding::same) << relu();
    *this << max_pool(56, 56, 256, 2) << relu();

    // Block 4
    *this << conv(28, 28, 3, 3, 256, 512, padding::same) << relu();
    *this << conv(28, 28, 3, 3, 512, 512, padding::same) << relu();
    *this << conv(28, 28, 3, 3, 512, 512, padding::same) << relu();
    *this << max_pool(28, 28, 512, 2) << relu();

    // Block 5
    *this << conv(14, 14, 3, 3, 512, 512, padding::same) << relu();
    *this << conv(14, 14, 3, 3, 512, 512, padding::same) << relu();
    *this << conv(14, 14, 3, 3, 512, 512, padding::same) << relu();
    *this << max_pool(14, 14, 512, 2) << relu();

    if (include_top) {
      *this << dense(7 * 7 * 512, 4096) << relu();
      *this << dense(4096, 4096) << relu();
      *this << dense(4096, 1000) << relu();
      *this << softmax();
    } else {
      // following the Network-in-Network architecture to use a Global
      // Average Pooling layer instead of Fully Connected Layers.
      *this << global_average_pooling_layer(1000, 1, 1);
    }

    if (load_pretrained) {
      throw nn_error("Pretrained weights loading support coming soon.");
    } else {
      // If not loading pretrained model then initialize as per strategy
      // mentioned in paper.
      this->weight_init(parameter_init::gaussian(0.01));
      this->bias_init(parameter_init::constant(0.0));
      this->init_parameters();
    }
  }
};

}  // namespace models
}  // namespace tiny_dnn
