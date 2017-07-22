/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <string>

#include "gtest/gtest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(max_pool, read_write) {
  max_pooling_layer l1(100, 100, 5, 2);
  max_pooling_layer l2(100, 100, 5, 2);

  l1.init_weight();
  l2.init_weight();

  serialization_test(l1, l2);
}

TEST(max_pool, forward) {
  max_pooling_layer l(4, 4, 1, 2);
  vec_t in = {0, 1, 2, 3, 8, 7, 5, 6, 4, 3, 1, 2, 0, -1, -2, -3};

  vec_t expected = {8, 6, 4, 2};

  std::vector<const tensor_t*> out;
  l.forward({{in}}, out);
  vec_t res = (*out[0])[0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}

TEST(max_pool, setup_internal) {
  max_pooling_layer l(4, 4, 1, 2, 2, core::backend_t::internal);

  EXPECT_EQ(l.parallelize(), true);         // if layer can be parallelized
  EXPECT_EQ(l.in_channels(), size_t(1));    // num of input tensors
  EXPECT_EQ(l.out_channels(), size_t(1));   // num of output tensors
  EXPECT_EQ(l.in_data_size(), size_t(16));  // size of input tensors
  EXPECT_EQ(l.out_data_size(), size_t(4));  // size of output tensors
  EXPECT_EQ(l.in_data_shape().size(), size_t(1));   // num of inputs shapes
  EXPECT_EQ(l.out_data_shape().size(), size_t(1));  // num of output shapes
  EXPECT_EQ(l.weights().size(), size_t(0));         // the wieghts vector size
  EXPECT_EQ(l.weights_grads().size(), size_t(0));   // the wieghts vector size
  EXPECT_EQ(l.inputs().size(), size_t(1));          // num of input edges
  EXPECT_EQ(l.outputs().size(), size_t(1));         // num of outpus edges
  EXPECT_EQ(l.in_types().size(), size_t(1));        // num of input data types
  EXPECT_EQ(l.out_types().size(), size_t(1));       // num of output data types
  EXPECT_EQ(l.fan_in_size(), size_t(4));   // num of incoming connections
  EXPECT_EQ(l.fan_out_size(), size_t(1));  // num of outgoing connections
  EXPECT_STREQ(l.layer_type().c_str(), "max-pool");  // string with layer type
}

TEST(max_pool, forward_stride_internal) {
  max_pooling_layer l(4, 4, 1, 2, 2, core::backend_t::internal);

  // clang-format off
    vec_t in = {
        0, 1, 2, 3,
        8, 7, 5, 6,
        4, 3, 1, 2,
        0,-1,-2,-3
    };

    vec_t expected = {
        8, 6,
        4, 2
    };
  // clang-format on

  std::vector<const tensor_t*> out;
  l.forward({{in}}, out);
  vec_t res = (*out[0])[0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}

TEST(max_pool, forward_padding_same) {
  max_pooling_layer l(4, 4, 1, 2, 2, 1, 1, padding::same,
                      core::backend_t::internal);

  // clang-format off
    vec_t in = {
        0, 1, 2, 3,
        8, 7, 5, 6,
        4, 3, 1, 2,
        0,-1,-2,-3
    };

    vec_t expected = {
        8, 7, 6, 6,
        8, 7, 6, 6,
        4, 3, 2, 2,
        0,-1,-2,-3
    };
  // clang-format on

  std::vector<const tensor_t*> out;
  l.forward({{in}}, out);
  vec_t res = (*out[0])[0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}

TEST(max_pool, forward_stride_x) {
  max_pooling_layer l(4, 4, 1, 2, 1, 2, 1, padding::valid);
  // clang-format off
    vec_t in = {
        0, 1, 2, 3,
        8, 7, 5, 6,
        4, 3, 1, 2,
        0,-1,-2,-3
    };

    vec_t expected = {
        1, 3,
        8, 6,
        4, 2,
        0, -2
    };
  // clang-format on

  EXPECT_EQ(l.out_shape()[0].width_, 2u);
  EXPECT_EQ(l.out_shape()[0].height_, 4u);

  std::vector<const tensor_t*> out;
  l.forward({{in}}, out);
  vec_t res = (*out[0])[0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}

TEST(max_pool, forward_stride_y) {
  max_pooling_layer l(4, 4, 1, 1, 2, 1, 2, padding::valid);
  // clang-format off
    vec_t in = {
        0, 1, 2, 3,
        8, 7, 5, 6,
        4, 3, 1, 2,
        0,-1,-2,-3
    };

    vec_t expected = {
        8, 7, 5, 6,
        4, 3, 1, 2
    };
  // clang-format on

  EXPECT_EQ(l.out_shape()[0].width_, 4u);
  EXPECT_EQ(l.out_shape()[0].height_, 2u);

  std::vector<const tensor_t*> out;
  l.forward({{in}}, out);
  vec_t res = (*out[0])[0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}

#ifdef CNN_USE_NNPACK
TEST(max_pool, forward_stride_nnp) {
  nnp_initialize();
  max_pooling_layer l(4, 4, 1, 2, 2, core::backend_t::nnpack);
  // clang-format off
    vec_t in = {
        0, 1, 2, 3,
        8, 7, 5, 6,
        4, 3, 1, 2,
        0,-1,-2,-3
    };

    vec_t expected = {
        8, 6,
        4, 2
    };
  // clang-format on

  vec_t res = l.forward({{in}})[0][0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}
// add for pool size !=2 and stride !=2

TEST(max_pool, forward_stride_nnp_not_2x2) {
  nnp_initialize();
  max_pooling_layer l(4, 4, 1, 3, 1, core::backend_t::nnpack);
  // clang-format off
    vec_t in = {
        0, 1, 2, 3,
        8, 7, 5, 6,
        4, 3, 1, 2,
        0,-1,-2,-3
    };

    vec_t expected = {
        8, 7,
        8, 7
    };
  // clang-format on

  vec_t res = l.forward({{in}})[0][0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}
#endif

TEST(max_pool, forward_stride) {
  max_pooling_layer l(4, 4, 1, 2, 1);
  // clang-format off
    vec_t in = {
        0, 1, 2, 3,
        8, 7, 5, 6,
        4, 3, 1, 2,
        0,-1,-2,-3
    };

    vec_t expected = {
        8, 7, 6,
        8, 7, 6,
        4, 3, 2
    };
  // clang-format on

  std::vector<const tensor_t*> out;
  l.forward({{in}}, out);
  vec_t res = (*out[0])[0];

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_FLOAT_EQ(expected[i], res[i]);
  }
}

TEST(max_pool, backward) {
  max_pooling_layer l(4, 4, 1, 2);
  // clang-format off
    vec_t in = {
        0, 1, 2, 3,
        8, 7, 5, 6,
        4, 3, 1, 2,
        0,-1,-2,-3
    };

    vec_t out_grad = {
        1, 2,
        3, 4
    };

    vec_t in_grad_expected = {
        0, 0, 0, 0,
        1, 0, 0, 2,
        3, 0, 0, 4,
        0, 0, 0, 0
    };
  // clang-format on

  std::vector<const tensor_t*> out;
  l.forward({{in}}, out);
  vec_t in_grad = l.backward(std::vector<tensor_t>{{out_grad}})[0][0];

  for (size_t i = 0; i < in_grad.size(); i++) {
    EXPECT_FLOAT_EQ(in_grad_expected[i], in_grad[i]);
  }
}

#ifndef CNN_NO_SERIALIZATION
TEST(max_pool, serialization) {
  max_pooling_layer src(4, 4, 1, 2);

  std::string str = layer_to_json(src);

  auto dst = json_to_layer(str);

  EXPECT_EQ(src.in_shape()[0], dst->in_shape()[0]);
  EXPECT_EQ(src.out_shape()[0], dst->out_shape()[0]);

  // clang-format off
    vec_t in = {
        9, 4, 8, 8,
        0, 7, 3, 0,
        4, 8, 1, 7,
        0, 3,-2, 9
    };
  // clang-format on

  std::vector<const tensor_t*> out1, out2;
  src.forward({{in}}, out1);
  dst->forward({{in}}, out2);
  vec_t res1 = (*out1[0])[0];
  vec_t res2 = (*out2[0])[0];

  for (size_t i = 0; i < res1.size(); i++) {
    EXPECT_FLOAT_EQ(res1[i], res2[i]);
  }
}

TEST(max_pool, serialization_stride) {
  max_pooling_layer src(4, 4, 1, 2, 1, 1, 2, padding::valid);

  std::string str = layer_to_json(src);

  auto dst = json_to_layer(str);

  EXPECT_EQ(src.in_shape()[0], dst->in_shape()[0]);
  EXPECT_EQ(src.out_shape()[0], dst->out_shape()[0]);

  // clang-format off
    vec_t in = {
        9, 4, 8, 8,
        0, 7, 3, 0,
        4, 8, 1, 7,
        0, 3,-2, 9
    };
  // clang-format on

  std::vector<const tensor_t*> out1, out2;
  src.forward({{in}}, out1);
  dst->forward({{in}}, out2);
  vec_t res1 = (*out1[0])[0];
  vec_t res2 = (*out2[0])[0];

  for (size_t i = 0; i < res1.size(); i++) {
    EXPECT_FLOAT_EQ(res1[i], res2[i]);
  }
}

TEST(max_pool, serialization_padding) {
  max_pooling_layer src(4, 4, 1, 2, 2, 1, 1, padding::same);

  std::string str = layer_to_json(src);

  auto dst = json_to_layer(str);

  EXPECT_EQ(src.in_shape()[0], dst->in_shape()[0]);
  EXPECT_EQ(src.out_shape()[0], dst->out_shape()[0]);

  // clang-format off
    vec_t in = {
        9, 4, 8, 8,
        0, 7, 3, 0,
        4, 8, 1, 7,
        0, 3,-2, 9
    };
  // clang-format on

  std::vector<const tensor_t*> out1, out2;
  src.forward({{in}}, out1);
  dst->forward({{in}}, out2);
  vec_t res1 = (*out1[0])[0];
  vec_t res2 = (*out2[0])[0];

  for (size_t i = 0; i < res1.size(); i++) {
    EXPECT_FLOAT_EQ(res1[i], res2[i]);
  }
}
#endif

}  // namespace tiny_dnn
