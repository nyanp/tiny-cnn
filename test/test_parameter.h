/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <gtest/gtest.h>

#include "test/testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(parameter, init) {
  Parameter p(1, 1, 3, 3, parameter_type::weight, true);

  EXPECT_EQ(p.shape().width_, 3u);
  EXPECT_EQ(p.shape().height_, 3u);
  EXPECT_EQ(p.shape().depth_, 1u);
  EXPECT_EQ(p.size(), 9u);
  EXPECT_EQ(p.type(), parameter_type::weight);

  EXPECT_TRUE(p.is_trainable());
}

TEST(parameter, getter_setter) {
  Parameter p(1, 1, 1, 4, parameter_type::bias, false);
  Tensor<float_t> t{vec_t{1.0, 2.0, 3.0, 4.0}};

  p.set_data(t);
  Tensor<float_t> *pt = p.data();

  for (size_t i = 0; i < t.size(); i++) {
    EXPECT_EQ(pt->host_at(i), t.host_at(i));
  }
}

TEST(parameter, merge_grads) {
  Tensor<float_t> gradp{tensor_t{{1.0, 2.0}, {2.0, 1.0}, {-4.0, 5.0}}};
  Tensor<float_t> grad0{vec_t{0.0, 0.0}};

  Parameter p(1, 1, 1, 2, parameter_type::bias, false);
  p.set_grad(gradp);
  p.merge_grads(&grad0);

  Tensor<float_t> expected{vec_t{-1.0, 8.0}};

  for (size_t i = 0; i < p.size(); i++) {
    EXPECT_EQ(grad0.host_at(0, i), expected.host_at(0, i));
  }
}

TEST(parameter, layer_adder) {
  fully_connected_layer fc(3, 2);

  auto parameters = fc.parameters();

  // check whether they were added in proper order
  ASSERT_GE(parameters.size(), 2u);
  EXPECT_EQ(parameters[0]->type(), parameter_type::weight);
  EXPECT_EQ(parameters[1]->type(), parameter_type::bias);
}

/* todo (karandesai) : inspect failure on travis job
TEST(parameter, xavier_init) {
  set_random_seed(42);

  parameter_init::xavier xavier;
  Parameter parameter(1, 1, 1, 10, parameter_type::weight);
  parameter.initialize(xavier);

  vec_t out_result = parameter.data()->toVec();

  // clang-format off
  vec_t out_expected = {-0.434, 1.027, 1.561, -1.096,  0.803,
                         0.968, 0.341, 0.335, -1.191, -0.187};
  // clang-format on

  for (size_t i = 0; i < out_result.size(); i++) {
    EXPECT_NEAR(out_result[i], out_expected[i], 1E-2);
  }
}
*/

TEST(parameter, constant_init) {
  parameter_init::constant constant(4);
  Parameter parameter(1, 1, 1, 5, parameter_type::weight);
  parameter.initialize(constant);

  vec_t out_result   = parameter.data()->toVec();
  vec_t out_expected = {4, 4, 4, 4, 4};

  for (size_t i = 0; i < out_result.size(); i++) {
    EXPECT_NEAR(out_result[i], out_expected[i], 1E-2);
  }
}

// todo (karandesai) : test getters and setters on fc layer

}  // namespace tiny_dnn
