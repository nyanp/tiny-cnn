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

namespace tiny_dnn {

TEST(parameter, init) {
  parameter p(3, 3, 1, 1, parameter_type::weight, true);

  ASSERT_EQ(p.shape().width_, 3u);
  ASSERT_EQ(p.shape().height_, 3u);
  ASSERT_EQ(p.shape().depth_, 1u);
  ASSERT_EQ(p.size(), 9u);
  ASSERT_EQ(p.type(), parameter_type::weight);

  ASSERT_TRUE(p.is_trainable());
}

TEST(parameter, getter_setter) {
  parameter p(4, 1, 1, 1, parameter_type::bias, false);
  Tensor<float_t> t{{1.0, 2.0, 3.0, 4.0}};

  p.set_data(t);
  Tensor<float_t> *pt = p.data();

  for (unsigned int i = 0; i < t.size(); i++) {
    ASSERT_EQ(pt->host_at(i), t.host_at(i));
  }
}

TEST(parameter, merge_grads) {
  Tensor<float_t> grad0{{{1.0, 2.0}, {2.0, 1.0}}};
  Tensor<float_t> gradp{{{2.0, 4.0}, {4.0, 2.0}}};

  parameter p(2, 1, 1, 1, parameter_type::bias, false);
  p.set_grad(gradp);
  p.merge_grads(&grad0);

  Tensor<float_t> expected{{{6.0, 6.0}, {6.0, 6.0}}};

  for (unsigned int i = 0; i < p.size(); i++) {
    ASSERT_EQ(grad0.host_at(0, i), expected.host_at(0, i));
    ASSERT_EQ(grad0.host_at(1, i), expected.host_at(1, i));
  }
}

}  // namespace tiny_dnn
