/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <gtest/gtest.h>
#include <string>

#include "test/testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(serialization_hdf, load_parameter) {
  std::string xor_hdf_path;
  resolve_path("testdata/xor.h5", xor_hdf_path);

  size_t in_dim(2), out_dim(4);
  Parameter dense_1_W(1, 1, out_dim, in_dim, parameter_type::weight);
  Parameter dense_1_b(1, 1, 1, out_dim, parameter_type::bias);
  dense_1_W.load(xor_hdf_path, "dense_1/dense_1/kernel:0");
  dense_1_b.load(xor_hdf_path, "dense_1/dense_1/bias:0");

  // clang-format off
  Tensor<> expected_W{vec_t{ 1.7683, -0.6835, -0.4390, -1.6895,
                            -1.7678, -0.2821, -0.2773,  1.6886}};
  // clang-format on

  Tensor<> expected_b{vec_t{0., 0., 0., -0.0012}};

  for (size_t i = 0; i < expected_W.size(); i++) {
    EXPECT_NEAR(expected_W.host_at(i), *(dense_1_W.data_at(i)), 1E-3);
  }
  for (size_t i = 0; i < expected_b.size(); i++) {
    EXPECT_NEAR(expected_b.host_at(i), *(dense_1_b.data_at(i)), 1E-3);
  }
}

TEST(serialization_hdf, load_layer) {
  std::string xor_hdf_path;
  resolve_path("testdata/xor.h5", xor_hdf_path);

  size_t in_dim(2), out_dim(4);
  fully_connected_layer fc1(in_dim, out_dim);
  fully_connected_layer fc2(in_dim, out_dim);

  // load fc1 directly, fc2 parameter wise
  fc1.load(xor_hdf_path, "dense_1");
  fc2.weights_at()[0]->load(xor_hdf_path, "dense_1/dense_1/kernel:0");
  fc2.bias_at()[0]->load(xor_hdf_path, "dense_1/dense_1/bias:0");

  EXPECT_TRUE(fc1.has_same_parameters(fc2, 1E-5));
}

TEST(serialization_hdf, load_network) {
  std::string xor_hdf_path;
  resolve_path("testdata/xor.h5", xor_hdf_path);

  network<sequential> net1;
  network<sequential> net2;
  net1 << fully_connected_layer(2, 4, true) << relu()
       << fully_connected_layer(4, 1, true) << sigmoid();

  net2 << fully_connected_layer(2, 4, true) << relu()
       << fully_connected_layer(4, 1, true) << sigmoid();

  // load net1 directly, net2 layer wise
  net1.load(xor_hdf_path, content_type::weights, file_format::hdf);

  net2[0]->load(xor_hdf_path, "dense_1");
  net2[2]->load(xor_hdf_path, "dense_2");

  EXPECT_TRUE(net1.has_same_parameters(net2, 1E-5));
}

TEST(serialization_hdf, load_network_and_predict) {
  std::string xor_hdf_path;
  resolve_path("testdata/xor.h5", xor_hdf_path);

  network<sequential> net;
  net << fully_connected_layer(2, 4, true) << relu()
      << fully_connected_layer(4, 1, true) << sigmoid();

  // load net1 directly, net2 layer wise
  net.load(xor_hdf_path, content_type::weights, file_format::hdf);

  EXPECT_NEAR(net.predict({0, 0})[0], 0.179693, 1E-5);  // near 0
  EXPECT_NEAR(net.predict({1, 1})[0], 0.179809, 1E-5);  // near 0
  EXPECT_NEAR(net.predict({0, 1})[0], 0.926772, 1E-5);  // near 1
  EXPECT_NEAR(net.predict({1, 0})[0], 0.913209, 1E-5);  // near 1
}

}  // namespace tiny_dnn
