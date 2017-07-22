/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "gtest/gtest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn::activation;

namespace tiny_dnn {

TEST(fully_connected, constructor_default) {
  fully_connected_layer l(4, 2);

  EXPECT_EQ(l.fan_in_size(), 4u);
  EXPECT_EQ(l.fan_out_size(), 2u);
  EXPECT_EQ(l.in_channels(), 3u);  // in, W and b

  EXPECT_EQ(l.params().has_bias, true);
  EXPECT_EQ(l.params().backend_type, core::backend_t::internal);
}

TEST(fully_connected, constructor_default_nobias) {
  fully_connected_layer l(4, 2, false);

  EXPECT_EQ(l.fan_in_size(), 4u);
  EXPECT_EQ(l.fan_out_size(), 2u);
  EXPECT_EQ(l.in_channels(), 2u);  // in and W

  EXPECT_EQ(l.params().has_bias, false);
  EXPECT_EQ(l.params().backend_type, core::backend_t::internal);
}

TEST(fully_connected, layer_params_default) {
  fully_connected_layer_params params;

  EXPECT_EQ(params.has_bias, true);
  EXPECT_EQ(params.backend_type, core::backend_t::internal);
}

TEST(fully_connected, constructor_params_default) {
  fully_connected_layer_params params;
  fully_connected_layer l(4, 2, params);

  EXPECT_EQ(l.fan_in_size(), 4u);
  EXPECT_EQ(l.fan_out_size(), 2u);
  EXPECT_EQ(l.in_channels(), 3u);  // in, W and b

  EXPECT_EQ(l.params().has_bias, true);
  EXPECT_EQ(l.params().backend_type, core::backend_t::internal);
}

TEST(fully_connected, constructor_params_nobias) {
  fully_connected_layer_params params;
  params.has_bias = false;

  fully_connected_layer l(4, 2, params);

  EXPECT_EQ(l.fan_in_size(), 4u);
  EXPECT_EQ(l.fan_out_size(), 2u);
  EXPECT_EQ(l.in_channels(), 2u);  // in and W

  EXPECT_EQ(l.params().has_bias, false);
  EXPECT_EQ(l.params().backend_type, core::backend_t::internal);
}

TEST(fully_connected, constructor_params_backend) {
  fully_connected_layer_params params;
  params.backend_type = core::backend_t::avx;

  fully_connected_layer l(4, 2, params);

  EXPECT_EQ(l.fan_in_size(), 4u);
  EXPECT_EQ(l.fan_out_size(), 2u);
  EXPECT_EQ(l.in_channels(), 3u);  // in, W and b

  EXPECT_EQ(l.params().has_bias, true);
  EXPECT_EQ(l.params().backend_type, core::backend_t::avx);
}

TEST(fully_connected, forward_internal) {
  fully_connected_layer_params params;
  params.backend_type = core::backend_t::internal;

  fully_connected_layer l(4, 2, params);

  l.weight_init(weight_init::constant(1.0));
  l.bias_init(weight_init::constant(0.5));

  vec_t in = {0, 1, 2, 3};
  std::vector<const tensor_t *> o;
  l.forward({{in}}, o);
  vec_t out          = (*o[0])[0];
  vec_t out_expected = {6.5, 6.5};  // 0+1+2+3+0.5

  for (size_t i = 0; i < out_expected.size(); i++) {
    EXPECT_FLOAT_EQ(out_expected[i], out[i]);
  }
}

TEST(fully_connected, forward_internal_nobias) {
  fully_connected_layer l(4, 2, false);

  l.weight_init(weight_init::constant(1.0));

  vec_t in = {0, 1, 2, 3};
  std::vector<const tensor_t *> o;
  l.forward({{in}}, o);
  vec_t out          = (*o[0])[0];
  vec_t out_expected = {6.0, 6.0};  // 0+1+2+3

  for (size_t i = 0; i < out_expected.size(); i++) {
    EXPECT_FLOAT_EQ(out_expected[i], out[i]);
  }
}

#ifdef CNN_USE_NNPACK
TEST(fully_connected, forward_nnp) {
  nnp_initialize();  // TODO: remove once initializer gets inside master

  fully_connected_layer_params params;
  params.backend_type = core::backend_t::nnpack;

  fully_connected_layer l(4, 2, params);

  l.weight_init(weight_init::constant(1.0));
  l.bias_init(weight_init::constant(0.5));

  vec_t in           = {0, 1, 2, 3};
  vec_t out          = l.forward({{in}})[0][0];
  vec_t out_expected = {6.5, 6.5};  // 0+1+2+3+0.5

  for (size_t i = 0; i < out_expected.size(); i++) {
    EXPECT_FLOAT_EQ(out_expected[i], out[i]);
  }
}
#endif

TEST(fully_connected, serialization) {
  fully_connected_layer l1(100, 100);
  fully_connected_layer l2(100, 100);

  l1.setup(true);
  l2.setup(true);

  serialization_test(l1, l2);
}

}  // namespace tiny_dnn
