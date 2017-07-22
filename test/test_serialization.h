/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <cstdio>

#include "gtest/gtest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn::activation;

namespace tiny_dnn {

inline void check_sequential_network_model_serialization(
  const network<sequential>& net) {
  auto path = unique_path();
  network<sequential> net2;
  net.save(path, content_type::model);
  net2.load(path, content_type::model);
  ASSERT_EQ(net.layer_size(), net2.layer_size());
  for (size_t i = 0; i < net.layer_size(); i++) {
    ASSERT_EQ(net[i]->in_shape(), net2[i]->in_shape());
    ASSERT_EQ(net[i]->out_shape(), net2[i]->out_shape());
    ASSERT_EQ(net[i]->layer_type(), net2[i]->layer_type());
  }
  std::remove(path.c_str());
}

TEST(serialization, serialize_avepool) {
  network<sequential> net;

  std::string json = R"(  
    {
        "nodes": [
            {
                "type": "avepool",
                "in_size": {
                    "width": 10,
                    "height": 10,
                    "depth": 3
                },
                "pool_size_x": 2,
                "pool_size_y": 2,
                "stride_x": 2,
                "stride_y": 2,
                "pad_type": 0
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "ave-pool");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(10, 10, 3));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(5, 5, 3));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_aveunpool) {
  network<sequential> net;

  std::string json = R"(  
    {
        "nodes": [
            {
                "type": "aveunpool",
                "in_size": {
                    "width": 10,
                    "height": 10,
                    "depth": 3
                },
                "pool_size": 2,
                "stride": 2
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "ave-unpool");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(10, 10, 3));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(20, 20, 3));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_batchnorm) {
  network<sequential> net;

  std::string json = R"(  
    {
        "nodes": [
            {
                "type": "batchnorm",
                "in_spatial_size": 3,
                "in_channels": 2,
                "epsilon": 0.001,
                "momentum": 0.8,
                "phase": 0,
                "mean": [
                    0,
                    0
                ],
                "variance": [
                    0,
                    0
                ]
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "batch-norm");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(3, 1, 2));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(3, 1, 2));
  EXPECT_FLOAT_EQ(net.at<batch_normalization_layer>(0).epsilon(), 0.001f);
  EXPECT_FLOAT_EQ(net.at<batch_normalization_layer>(0).momentum(), 0.8f);
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_concat) {
  network<sequential> net;

  std::string json = R"(  
    {
        "nodes": [
            {
                "type": "concat",
                "in_size": [
                    {
                        "width": 2,
                        "height": 1,
                        "depth": 1
                    },
                    {
                        "width": 2,
                        "height": 1,
                        "depth": 2
                    },
                    {
                        "width": 2,
                        "height": 1,
                        "depth": 3
                    }
                ]
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "concat");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(2, 1, 1));
  EXPECT_EQ(net[0]->in_shape()[1], shape3d(2, 1, 2));
  EXPECT_EQ(net[0]->in_shape()[2], shape3d(2, 1, 3));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(2, 1, 6));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_conv) {
  network<sequential> net;

  std::string json = R"(
    {
        "nodes": [
            {
                "type": "conv",
                "in_size" : {
                    "width": 20,
                    "height" : 20,
                    "depth" : 10
                },
                "window_width" : 5,
                "window_height" : 5,
                "out_channels" : 5,
                "connection_table" : {
                    "rows": 0,
                    "cols" : 0,
                    "connection" : "all"
                },
                "pad_type" : 1,
                "has_bias" : true,
                "w_stride" : 2,
                "h_stride" : 2
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "conv");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(20, 20, 10));
  EXPECT_EQ(net[0]->in_shape()[1], shape3d(5, 5, 10 * 5));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(10, 10, 5));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_deconv) {
  network<sequential> net;

  std::string json = R"(
    {
        "nodes": [
            {
                "type": "deconv",
                "in_size" : {
                    "width": 20,
                    "height" : 20,
                    "depth" : 10
                },
                "window_width" : 5,
                "window_height" : 5,
                "out_channels" : 5,
                "connection_table" : {
                    "rows": 0,
                    "cols" : 0,
                    "connection" : "all"
                },
                "pad_type" : 1,
                "has_bias" : true,
                "w_stride" : 2,
                "h_stride" : 2
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "deconv");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(20, 20, 10));
  EXPECT_EQ(net[0]->in_shape()[1], shape3d(5, 5, 10 * 5));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(40, 40, 5));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_dropout) {
  network<sequential> net;

  std::string json = R"(  
    {
        "nodes": [
            {
                "type": "dropout",
                "in_size": 3,
                "dropout_rate": 0.5,
                "phase": 1
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "dropout");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(3, 1, 1));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(3, 1, 1));
  EXPECT_FLOAT_EQ(net.at<dropout_layer>(0).dropout_rate(), 0.5f);
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_recurrent_cell) {
  network<sequential> net;

  std::string json = R"(
      {
          "nodes": [
              {
                  "type": "recurrent_cell",
                  "in_size": 100,
                  "out_size": 20,
                  "has_bias": true,
                  "value0" : {
                      "type": "tanh",
                      "in_size": {
                          "width": 0,
                          "height": 0,
                          "depth": 0
                      }
                  }
              }
          ]
      }
      )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "recurrent-cell");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(100, 1, 1));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(20, 1, 1));
}

TEST(serialization, serialize_fully) {
  network<sequential> net;

  std::string json = R"(  
    {
        "nodes": [
            {
                "type": "fully_connected",
                "in_size": 100,
                "out_size": 20,
                "has_bias": false
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "fully-connected");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(100, 1, 1));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(20, 1, 1));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_global_average_pooling) {
  network<sequential> net;

  std::string json = R"(
    {
        "nodes": [
            {
                "type": "global_average_pooling",
                "in_shape": {
                    "width": 5,
                    "height": 4,
                    "depth": 6
                }
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "global-ave-pool");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(5, 4, 6));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(6, 1, 1));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_input) {
  network<sequential> net;

  std::string json = R"(
    {
        "nodes": [
            {
                "type": "input",
                "shape": {
                    "width": 5,
                    "height": 4,
                    "depth": 6
                }
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "input");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(5, 4, 6));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(5, 4, 6));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_lrn) {
  network<sequential> net;

  std::string json = R"(  
    {
        "nodes": [
            {
                "type": "lrn",
                "in_shape": {
                    "width": 5,
                    "height": 4,
                    "depth": 6
                },
                "size": 3,
                "alpha": 0.5,
                "beta": 2.5,
                "region": 1
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "lrn");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(5, 4, 6));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(5, 4, 6));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_maxpool) {
  network<sequential> net;

  std::string json = R"(  
    {
        "nodes": [
            {
                "type": "maxpool",
                "in_size": {
                    "width": 10,
                    "height": 10,
                    "depth": 3
                },
                "pool_size_x": 2,
                "pool_size_y": 2,
                "stride_x": 1,
                "stride_y": 1,
                "pad_type": 1
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "max-pool");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(10, 10, 3));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(10, 10, 3));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_power) {
  network<sequential> net;

  std::string json = R"(  
    {
        "nodes": [
            {
                "type": "power",
                "in_size": {
                    "width": 3,
                    "height": 2,
                    "depth": 1
                },
                "factor": 0.5,
                "scale": 0.2
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "power");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(3, 2, 1));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(3, 2, 1));
  EXPECT_FLOAT_EQ(net.at<power_layer>(0).factor(), 0.5f);
  EXPECT_FLOAT_EQ(net.at<power_layer>(0).scale(), 0.2f);
  check_sequential_network_model_serialization(net);
}

/*
TEST(serialization, serialize_q_conv) {
  network<sequential> net;

  std::string json = R"(
    {
        "nodes": [
            {
                "type": "q_conv",
                "in_size" : {
                    "width": 20,
                    "height" : 20,
                    "depth" : 10
                },
                "window_width" : 5,
                "window_height" : 5,
                "out_channels" : 5,
                "connection_table" : {
                    "rows": 0,
                    "cols" : 0,
                    "connection" : "all"
                },
                "pad_type" : 1,
                "has_bias" : true,
                "w_stride" : 2,
                "h_stride" : 2
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "q_conv");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(20, 20, 10));
  EXPECT_EQ(net[0]->in_shape()[1], shape3d(5, 5, 10 * 5));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(10, 10, 5));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_q_deconv) {
  network<sequential> net;

  std::string json = R"(
    {
        "nodes": [
            {
                "type": "q_deconv",
                "in_size" : {
                    "width": 20,
                    "height" : 20,
                    "depth" : 10
                },
                "window_width" : 5,
                "window_height" : 5,
                "out_channels" : 5,
                "connection_table" : {
                    "rows": 0,
                    "cols" : 0,
                    "connection" : "all"
                },
                "pad_type" : 1,
                "has_bias" : true,
                "w_stride" : 2,
                "h_stride" : 2
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "q_deconv");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(20, 20, 10));
  EXPECT_EQ(net[0]->in_shape()[1], shape3d(5, 5, 10 * 5));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(40, 40, 5));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_q_fully) {
  network<sequential> net;

  std::string json = R"(  
    {
        "nodes": [
            {
                "type": "q_fully_connected",
                "in_size": 100,
                "out_size": 20,
                "has_bias": false
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "q_fully-connected");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(100, 1, 1));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(20, 1, 1));
  check_sequential_network_model_serialization(net);
}
*/

TEST(serialization, serialize_slice) {
  std::string json;
  network<sequential> net;
  slice_layer* sl;

  json = R"(  
    {
        "nodes": [
            {
                "type": "slice",
                "in_size": {
                    "width": 3,
                    "height": 2,
                    "depth": 1
                },
                "slice_type": 0,
                "num_outputs": 2
            }
        ]
    }
    )";

  net.from_json(json);

  sl = reinterpret_cast<slice_layer*>(net[0]);
  EXPECT_EQ(sl->layer_type(), "slice");
  EXPECT_EQ(sl->get_slice_type(), slice_type::slice_samples);
  EXPECT_EQ(sl->in_shape()[0], shape3d(3, 2, 1));
  EXPECT_EQ(sl->out_shape().size(), 2);
  EXPECT_EQ(sl->out_shape()[0], shape3d(3, 2, 1));
  EXPECT_EQ(sl->out_shape()[1], shape3d(3, 2, 1));
  check_sequential_network_model_serialization(net);

  json = R"(  
    {
        "nodes": [
            {
                "type": "slice",
                "in_size": {
                    "width": 3,
                    "height": 2,
                    "depth": 4
                },
                "slice_type": 1,
                "num_outputs": 2
            }
        ]
    }
    )";

  net.from_json(json);

  sl = reinterpret_cast<slice_layer*>(net[0]);
  EXPECT_EQ(sl->layer_type(), "slice");
  EXPECT_EQ(sl->in_shape()[0], shape3d(3, 2, 4));
  EXPECT_EQ(sl->get_slice_type(), slice_type::slice_channels);
  EXPECT_EQ(sl->out_shape().size(), 2);
  EXPECT_EQ(sl->out_shape()[0], shape3d(3, 2, 2));
  EXPECT_EQ(sl->out_shape()[1], shape3d(3, 2, 2));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_sigmoid) {
  network<sequential> net;

  std::string json = R"(
    {
        "nodes": [
            {
                "type": "sigmoid",
                "in_size" : {
                    "width": 3,
                    "height" : 2,
                    "depth" : 1
                }
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "sigmoid-activation");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(3, 2, 1));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(3, 2, 1));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_tanh) {
  network<sequential> net;

  std::string json = R"(
    {
        "nodes": [
            {
                "type": "tanh",
                "in_size" : {
                    "width": 20,
                    "height" : 20,
                    "depth" : 10
                }
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "tanh-activation");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(20, 20, 10));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(20, 20, 10));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_relu) {
  network<sequential> net;

  std::string json = R"(
    {
        "nodes": [
            {
                "type": "relu",
                "in_size" : {
                    "width": 1,
                    "height" : 128,
                    "depth" : 1
                }
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "relu-activation");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(1, 128, 1));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(1, 128, 1));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_softmax) {
  network<sequential> net;

  std::string json = R"(
    {
        "nodes": [
            {
                "type": "softmax",
                "in_size" : {
                    "width": 1000,
                    "height" : 1,
                    "depth" : 1
                }
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "softmax-activation");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(1000, 1, 1));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(1000, 1, 1));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_leaky_relu) {
  network<sequential> net;

  std::string json = R"(
    {
        "nodes": [
            {
                "type": "leaky_relu",
                "in_size" : {
                    "width": 256,
                    "height" : 256,
                    "depth" : 1
                },
                "epsilon": 0.1
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "leaky-relu-activation");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(256, 256, 1));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(256, 256, 1));
  EXPECT_FLOAT_EQ(net.at<leaky_relu_layer>(0).epsilon_value(), float_t(0.1));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_selu) {
  network<sequential> net;

  std::string json = R"(
    {
        "nodes": [
            {
                "type": "selu",
                "in_size" : {
                    "width": 256,
                    "height" : 256,
                    "depth" : 1
                },
                "lambda": 1.05070,
                "alpha" : 1.67326
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "selu-activation");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(256, 256, 1));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(256, 256, 1));
  EXPECT_FLOAT_EQ(net.at<selu_layer>(0).lambda_value(), float_t(1.05070));
  EXPECT_FLOAT_EQ(net.at<selu_layer>(0).alpha_value(), float_t(1.67326));
}

TEST(serialization, serialize_elu) {
  network<sequential> net;

  std::string json = R"(
    {
        "nodes": [
            {
                "type": "elu",
                "in_size" : {
                    "width": 10,
                    "height" : 10,
                    "depth" : 3
                }
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "elu-activation");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(10, 10, 3));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(10, 10, 3));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_tanh_p1m2) {
  network<sequential> net;

  std::string json = R"(
    {
        "nodes": [
            {
                "type": "tanh_scaled",
                "in_size" : {
                    "width": 10,
                    "height" : 10,
                    "depth" : 3
                }
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "tanh-scaled-activation");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(10, 10, 3));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(10, 10, 3));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_softplus) {
  network<sequential> net;

  std::string json = R"(
    {
        "nodes": [
            {
                "type": "softplus",
                "in_size" : {
                    "width": 5,
                    "height" : 10,
                    "depth" : 1
                },
                "beta": 1,
                "threshold": 20
            }
        ]
    }
    )";

  net.from_json(json);

  EXPECT_EQ(net[0]->layer_type(), "softplus-activation");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(5, 10, 1));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(5, 10, 1));
  EXPECT_FLOAT_EQ(net.at<softplus_layer>(0).beta_value(), float_t(1));
  EXPECT_FLOAT_EQ(net.at<softplus_layer>(0).threshold_value(), float_t(20));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, serialize_softsign) {
  network<sequential> net;

  std::string json = R"(
    {
        "nodes": [
            {
                "type": "softsign",
                "in_size" : {
                    "width": 10,
                    "height" : 10,
                    "depth" : 3
                }
            }
        ]
    }
    )";

  net.from_json(json);
  EXPECT_EQ(net[0]->layer_type(), "softsign-activation");
  EXPECT_EQ(net[0]->in_shape()[0], shape3d(10, 10, 3));
  EXPECT_EQ(net[0]->out_shape()[0], shape3d(10, 10, 3));
  check_sequential_network_model_serialization(net);
}

TEST(serialization, sequential_to_json) {
  network<sequential> net1, net2;

  net1 << fully_connected_layer(10, 100) << tanh_layer()
       << dropout_layer(100, 0.3f, net_phase::test)
       << fully_connected_layer(100, 9) << softmax()
       << recurrent_cell_layer(9, 9, false, new elu_layer)
       << convolutional_layer(3, 3, 3, 1, 1) << tanh_layer();

  auto json = net1.to_json();

  net2.from_json(json);

  EXPECT_EQ(net1.in_data_size(), net2.in_data_size());
  EXPECT_EQ(net1.layer_size(), net2.layer_size());

  for (size_t i = 0; i < net1.layer_size(); i++) {
    EXPECT_EQ(net1[i]->in_shape(), net2[i]->in_shape());
    EXPECT_EQ(net1[i]->layer_type(), net2[i]->layer_type());
  }

  EXPECT_FLOAT_EQ(net1.at<dropout_layer>(2).dropout_rate(),
                  net2.at<dropout_layer>(2).dropout_rate());
  check_sequential_network_model_serialization(net1);
}

TEST(serialization, sequential_model) {
  network<sequential> net1, net2;

  net1 << fully_connected_layer(10, 16) << tanh_layer()
       << average_pooling_layer(4, 4, 1, 2) << relu()
       << power_layer(shape3d(2, 2, 1), 0.5f);

  net1.init_weight();

  check_sequential_network_model_serialization(net1);
}

TEST(serialization, sequential_weights) {
  network<sequential> net1, net2;
  vec_t data = {1, 2, 3, 4, 5, 6};

  net1 << fully_connected_layer(6, 6) << sigmoid()
       << fully_connected_layer(6, 4) << tanh_layer()
       << fully_connected_layer(4, 2) << relu() << fully_connected_layer(2, 2)
       << softmax();

  net1.init_weight();
  net1.set_netphase(net_phase::test);

  auto path = unique_path();
  net1.save(path, content_type::weights_and_model);

  net2.load(path, content_type::weights_and_model);

  auto res1 = net1.predict(data);
  auto res2 = net2.predict(data);

  EXPECT_TRUE(net1.has_same_weights(net2, 1e-3f));

  for (int i = 0; i < 2; i++) {
    EXPECT_FLOAT_EQ(res1[i], res2[i]);
  }
  std::remove(path.c_str());
}

TEST(serialization, sequential_weights2) {
  network<sequential> net1, net2;
  vec_t data = {1, 2, 3, 4, 5, 0};

  net1 << batch_normalization_layer(3, 2, 0.01f, 0.99f, net_phase::train)
       << linear_layer(3 * 2, 2.0f, 0.5f) << elu()
       << power_layer(shape3d(3, 2, 1), 2.0, 1.5) << leaky_relu();

  net1.init_weight();
  net1.at<batch_normalization_layer>(0).update_immidiately(true);
  net1.predict(data);
  net1.set_netphase(net_phase::test);

  auto path = unique_path();
  net1.save(path, content_type::weights_and_model);

  net2.load(path, content_type::weights_and_model);

  auto res1 = net1.predict(data);
  auto res2 = net2.predict(data);

  EXPECT_TRUE(net1.has_same_weights(net2, 1e-3f));

  for (int i = 0; i < 6; i++) {
    EXPECT_FLOAT_EQ(res1[i], res2[i]);
  }
  std::remove(path.c_str());
}

TEST(serialization, graph_model_and_weights) {
  network<graph> net1, net2;
  vec_t in = {1, 2, 3};

  fully_connected_layer f1(3, 4);
  tanh_layer a1(4);
  slice_layer s1(shape3d(2, 1, 2), slice_type::slice_channels, 2);
  fully_connected_layer f2(2, 2);
  softmax_layer a2(2);
  fully_connected_layer f3(2, 2);
  elu_layer a3(2);
  elementwise_add_layer c4(2, 2);

  f1 << a1 << s1;
  f2 << a2;
  f3 << a3;
  s1 << (f2, f3) << c4;

  construct_graph(net1, {&f1}, {&c4});

  net1.init_weight();
  auto res1 = net1.predict(in);

  auto path = unique_path();

  net1.save(path, content_type::weights_and_model);

  net2.load(path, content_type::weights_and_model);

  auto res2 = net2.predict(in);

  EXPECT_FLOAT_EQ(res1[0], res2[0]);
  EXPECT_FLOAT_EQ(res1[1], res2[1]);
  std::remove(path.c_str());
}

}  // namespace tiny_dnn
