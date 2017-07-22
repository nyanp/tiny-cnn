/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include <memory>
#include "gtest/gtest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

class test_fc_layer : public fully_connected_layer {
 public:
  using base = fully_connected_layer;

  test_fc_layer() : base(10, 10) { ++counter(); }

  test_fc_layer(const test_fc_layer &fc) : base(10, 10) {
    CNN_UNREFERENCED_PARAMETER(fc);
    ++counter();
  }

  virtual ~test_fc_layer() { --counter(); }

  test_fc_layer(test_fc_layer &&r) : base(std::move(r)) { ++counter(); }

  static int &counter() {
    static int i = 0;
    return i;
  }
};

TEST(network, construct_sequential_by_local_variables) {
  {
    network<sequential> net;
    test_fc_layer fc1, fc2;
    net << fc1 << fc2;
    ASSERT_EQ(test_fc_layer::counter(), 2);
  }
  ASSERT_EQ(test_fc_layer::counter(), 0);
}

TEST(network, construct_sequential_by_temporary_variables) {
  {
    network<sequential> net;
    net << test_fc_layer() << test_fc_layer();
    ASSERT_EQ(test_fc_layer::counter(), 2);
  }
  ASSERT_EQ(test_fc_layer::counter(), 0);
}

TEST(network, construct_sequential_by_shared_ptr) {
  {
    network<sequential> net;
    auto fc1 = std::make_shared<test_fc_layer>();
    auto fc2 = std::make_shared<test_fc_layer>();
    net << fc1 << fc2;
    ASSERT_EQ(test_fc_layer::counter(), 2);
  }
  ASSERT_EQ(test_fc_layer::counter(), 0);
}

TEST(network, construct_multi_by_local_variables) {
  network<sequential> net;
  conv conv1(32, 32, 5, 1, 6, padding::same);
  tanh_layer tanh1(32, 32, 6);
  conv conv2(32, 32, 7, 6, 12, padding::same);
  sigmoid_layer sgm2(32, 32, 12);
  max_pool pool1(32, 32, 12, 2);
  relu_layer relu1(16, 16, 12);
  lrn_layer lrn(16, 16, 4, 12);
  dropout dp(16 * 16 * 12, 0.5);
  fc full(16 * 16 * 12, 1);
  softmax_layer sft(1);

  net << conv1 << tanh1 << conv2 << sgm2 << pool1 << relu1 << lrn << dp << full
      << sft;
}

TEST(network, construct_multi_by_temporary_variables) {
  network<sequential> net;
  net << conv(32, 32, 5, 1, 6, padding::same) << tanh()
      << conv(32, 32, 7, 6, 12, padding::same) << sigmoid()
      << max_pool(32, 32, 12, 2) << relu() << lrn_layer(16, 16, 4, 12)
      << dropout(16 * 16 * 12, 0.5) << fc(16 * 16 * 12, 1) << softmax();
}

TEST(network, in_dim) {
  network<sequential> net;
  convolutional_layer c1(32, 32, 5, 3, 6, padding::same);
  max_pooling_layer p1(32, 32, 6, 2);
  net << c1 << p1;

  EXPECT_EQ(c1.in_data_size(), net.in_data_size());
}

TEST(network, out_dim) {
  network<sequential> net;
  convolutional_layer c1(32, 32, 5, 3, 6, padding::same);
  max_pooling_layer p1(32, 32, 6, 2);
  net << c1 << p1;

  EXPECT_EQ(p1.out_data_size(), net.out_data_size());
}

TEST(network, name) {
  network<sequential> net1;
  network<sequential> net2("foo");

  EXPECT_EQ(net1.name(), "");
  EXPECT_EQ(net2.name(), "foo");
}

TEST(network, add) {
  network<sequential> net;
  net << convolutional_layer(32, 32, 5, 3, 6, padding::same);

  EXPECT_EQ(net.depth(), 1u);
}

TEST(network, manual_init) {
  // initializing weights directly
  network<sequential> net;
  net << convolutional_layer(3, 3, 3, 1, 1)
      << fully_connected_layer(1, 2, false) << softmax();

  adagrad opt;

  vec_t *c1_w = net[0]->weights()[0];
  vec_t *c1_b = net[0]->weights()[1];
  vec_t *f1_w = net[1]->weights()[0];

  EXPECT_EQ(c1_w->size(), 9u);
  EXPECT_EQ(c1_b->size(), 1u);
  EXPECT_EQ(f1_w->size(), 2u);

  *c1_w = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  *c1_b = {1};
  *f1_w = {1, 2};

  // check if the training and predicting works
  // https://github.com/tiny-dnn/tiny-dnn/issues/330
  net.predict({1, 1, 1, 1, 1, 1, 1, 1, 1});

  net.train<mse, adagrad>(opt, tensor_t{{1, 1, 1, 1, 1, 1, 1, 1, 1}},
                          tensor_t{{1, 2}}, 1, 1);
}

// TODO(nyanp): check out values again since the routine it's a bit sensitive

/*TEST(network, multi_out) {
    network<graph> net;
    adam optimizer;
    size_t tnum = 600;

    std::vector<vec_t> data;
    std::vector<std::vector<vec_t>> out;

    auto in = std::make_shared<input_layer>(shape3d(2,1,1));
    auto hidden = std::make_shared<layers::fc>(2,4);
    auto out1 = std::make_shared<layers::fc>(4,2);
    auto out2 = std::make_shared<layers::fc>(4,2);

    in << hidden << out1;
    hidden << out2;

    for (size_t i = 0; i < tnum; i++) {
        bool in[2] = { bernoulli(0.5), bernoulli(0.5) };
        // label_t expected = (in[0] ^ in[1]) ? 1 : 0;

        data.push_back({ static_cast<float_t>(in[0]),
                         static_cast<float_t>(in[1]) });

        out.emplace_back(std::vector<vec_t>{
            { static_cast<float_t>(in[0] && in[1]),
              static_cast<float_t>(in[0] || in[1]) }, // 1st output train and/or
function
            { static_cast<float_t>(in[0] ^  in[1]),
              static_cast<float_t>(in[0] == in[1]) } // 2nd output train xor/eq
function
        });
    }

    // construct single input, dual output network
    construct_graph(net, {in}, {out1,out2});

    optimizer.alpha *= 10;

    //net.fit<mse>(optimizer, data, out, 10, 10, nop, [&](){ std::cout <<
net.get_loss<mse>(data,out) << std::endl;});
    net.fit<mse>(optimizer, data, out, 10, 10);

    for (size_t i = 0; i < tnum; i++) {
        bool in[2] = { bernoulli(0.5), bernoulli(0.5) };
        std::vector<vec_t> actual = net.predict(
            std::vector<vec_t>{{ static_cast<float_t>(in[0]),
                                 static_cast<float_t>(in[1]) }});
        vec_t actual_out1 = actual[0];
        vec_t actual_out2 = actual[1];

        EXPECT_NEAR(actual_out1[0], in[0] && in[1], 0.1);
        EXPECT_NEAR(actual_out1[1], in[0] || in[1], 0.1);
        EXPECT_NEAR(actual_out2[0], in[0] ^  in[1], 0.1);
        EXPECT_NEAR(actual_out2[1], in[0] == in[1], 0.1);
    }
}*/

TEST(network, train_predict) {
  // train xor function
  network<sequential> net;
  adagrad optimizer;

  std::vector<vec_t> data;
  std::vector<label_t> label;
  size_t tnum = 300;

  optimizer.alpha *= 10;

  for (size_t i = 0; i < tnum; i++) {
    bool in[2] = {bernoulli(0.5), bernoulli(0.5)};
    data.push_back({static_cast<float_t>(in[0]), static_cast<float_t>(in[1])});
    label.push_back((in[0] ^ in[1]) ? 1 : 0);
  }

  net << fully_connected_layer(2, 10) << tanh() << fully_connected_layer(10, 2)
      << tanh();

  net.train<mse>(optimizer, data, label, 10, 10);

  std::vector<tensor_t> parallel_input(tnum);
  std::vector<tensor_t> expected_parallel_output(tnum);

  for (size_t i = 0; i < tnum; i++) {
    const bool in[2]       = {bernoulli(0.5), bernoulli(0.5)};
    const label_t expected = (in[0] ^ in[1]) ? 1 : 0;
    const vec_t input      = {static_cast<float_t>(in[0]),
                         static_cast<float_t>(in[1])};
    const label_t actual = net.predict_label(input);
    EXPECT_EQ(expected, actual);

    const auto actual_vec = net.predict(input);
    EXPECT_EQ(expected == 1, actual_vec[1] > actual_vec[0]);

    parallel_input[i]           = tensor_t{input};
    expected_parallel_output[i] = tensor_t{actual_vec};
  }

  // test predicting multiple samples in parallel
  const auto actual_parallel_output = net.predict(parallel_input);
  for (size_t i = 0; i < tnum; i++) {
    EXPECT_NEAR(expected_parallel_output[i][0][0],
                actual_parallel_output[i][0][0], 1e-10);
    EXPECT_NEAR(expected_parallel_output[i][0][1],
                actual_parallel_output[i][0][1], 1e-10);
  }
}

TEST(network, train_predict_different_batches) {
  auto batch_sizes = {2, 7, 11, 16};
  size_t data_size = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 1,
                                     std::multiplies<int>());
  for (auto &batch_sz : batch_sizes) {
    // train xor function
    network<sequential> net;
    adagrad optimizer;

    std::vector<vec_t> data;
    std::vector<label_t> label;

    optimizer.alpha *= 10;

    for (size_t i = 0; i < data_size; i++) {
      bool in[2] = {bernoulli(0.5), bernoulli(0.5)};
      data.push_back(
        {static_cast<float_t>(in[0]), static_cast<float_t>(in[1])});
      label.push_back((in[0] ^ in[1]) ? 1 : 0);
    }

    net << fully_connected_layer(2, 10) << tanh()
        << fully_connected_layer(10, 2) << tanh();

    net.train<mse>(optimizer, data, label, batch_sz, 10);

    std::vector<tensor_t> parallel_input(data_size);
    std::vector<tensor_t> expected_parallel_output(data_size);

    for (size_t i = 0; i < data_size; i++) {
      const bool in[2]       = {bernoulli(0.5), bernoulli(0.5)};
      const label_t expected = (in[0] ^ in[1]) ? 1 : 0;
      const vec_t input      = {static_cast<float_t>(in[0]),
                           static_cast<float_t>(in[1])};
      const label_t actual = net.predict_label(input);
      EXPECT_EQ(expected, actual);

      const auto actual_vec = net.predict(input);
      EXPECT_EQ(expected == 1, actual_vec[1] > actual_vec[0]);

      parallel_input[i]           = tensor_t{input};
      expected_parallel_output[i] = tensor_t{actual_vec};
    }

    // test predicting multiple samples in parallel
    const auto actual_parallel_output = net.predict(parallel_input);
    for (size_t i = 0; i < data_size; i++) {
      EXPECT_NEAR(expected_parallel_output[i][0][0],
                  actual_parallel_output[i][0][0], 1e-10);
      EXPECT_NEAR(expected_parallel_output[i][0][1],
                  actual_parallel_output[i][0][1], 1e-10);
    }
  }
}

TEST(network, set_netphase) {
  // TODO: add unit-test for public api
}

TEST(network, test) {
  network<sequential> net;
  fully_connected_layer fc(30, 1);
  int data_num = 300;

  net << fc;
  net.weight_init(weight_init::constant(1.0));
  net.init_weight();

  std::vector<vec_t> in, expected;

  for (int i = 0; i < data_num; i++) {
    vec_t v(30);
    uniform_rand(v.begin(), v.end(), -1.0, 1.0);
    float_t sum = std::accumulate(v.begin(), v.end(), (float_t)0.0);

    in.emplace_back(v);
    expected.emplace_back(vec_t{sum});
  }

  auto out = net.test(in);
  for (int i = 0; i < data_num; i++) {
    for (size_t j = 0; j < out[i].size(); j++)
      EXPECT_FLOAT_EQ(out[i][j], expected[i][0]);
  }
}

TEST(network, get_loss) {
  // TODO: add unit-test for public api
}

TEST(network, at) {
  network<sequential> net;
  convolutional_layer c1(32, 32, 5, 3, 6, padding::same);
  average_pooling_layer p1(32, 32, 6, 2);

  net << c1 << p1;
  net.init_weight();

  // auto& c = net.at<convolutional_layer>(0);
  // auto& p = net.at<average_pooling_layer>(1);
}

TEST(network, bracket_operator) {
  network<sequential> net;

  net << convolutional_layer(32, 32, 5, 3, 6, padding::same)
      << average_pooling_layer(32, 32, 6, 2);

  EXPECT_EQ(net[0]->layer_type(), "conv");
  EXPECT_EQ(net[1]->layer_type(), "ave-pool");
}

TEST(network, weight_init) {
  network<sequential> net;

  net << convolutional_layer(32, 32, 5, 3, 6, padding::same)
      << average_pooling_layer(32, 32, 6, 2);

  // change all layers at once
  net.weight_init(weight_init::constant(2.0));
  net.init_weight();

  vec_t &w1 = *net[0]->weights()[0];
  vec_t &w2 = *net[1]->weights()[0];

  for (size_t i = 0; i < w1.size(); i++) EXPECT_NEAR(w1[i], 2.0, 1e-10);

  for (size_t i = 0; i < w2.size(); i++) EXPECT_NEAR(w2[i], 2.0, 1e-10);
}

TEST(network, weight_init_per_layer) {
  network<sequential> net;

  net << convolutional_layer(32, 32, 5, 3, 6, padding::same)
      << average_pooling_layer(32, 32, 6, 2);

  // change specific layer
  net[0]->weight_init(weight_init::constant(2.0));
  net[1]->weight_init(weight_init::constant(1.0));
  net.init_weight();

  vec_t &w1 = *net[0]->weights()[0];
  vec_t &w2 = *net[1]->weights()[0];

  for (size_t i = 0; i < w1.size(); i++) EXPECT_NEAR(w1[i], 2.0, 1e-10);

  for (size_t i = 0; i < w2.size(); i++) EXPECT_NEAR(w2[i], 1.0, 1e-10);
}

TEST(network, bias_init) {
  network<sequential> net;

  net << convolutional_layer(32, 32, 5, 3, 6, padding::same)
      << average_pooling_layer(32, 32, 6, 2);

  net.bias_init(weight_init::constant(2.0));
  net.init_weight();

  vec_t &w1 = *net[0]->weights()[1];
  vec_t &w2 = *net[1]->weights()[1];

  for (size_t i = 0; i < w1.size(); i++) EXPECT_NEAR(w1[i], 2.0, 1e-10);

  for (size_t i = 0; i < w2.size(); i++) EXPECT_NEAR(w2[i], 2.0, 1e-10);
}

TEST(network, bias_init_per_layer) {
  network<sequential> net;

  net << convolutional_layer(32, 32, 5, 3, 6, padding::same)
      << average_pooling_layer(32, 32, 6, 2);

  net[0]->bias_init(weight_init::constant(2.0));
  net[1]->bias_init(weight_init::constant(1.0));
  net.init_weight();

  vec_t &w1 = *net[0]->weights()[1];
  vec_t &w2 = *net[1]->weights()[1];

  for (size_t i = 0; i < w1.size(); i++) EXPECT_NEAR(w1[i], 2.0, 1e-10);

  for (size_t i = 0; i < w2.size(); i++) EXPECT_NEAR(w2[i], 1.0, 1e-10);
}

TEST(network, gradient_check) {  // sigmoid - cross-entropy
  using loss_func  = cross_entropy;
  using activation = sigmoid;
  using network    = network<sequential>;

  network nn;
  nn << fully_connected_layer(10, 14 * 14 * 3) << activation()
     << convolutional_layer(14, 14, 5, 3, 6) << activation()
     << average_pooling_layer(10, 10, 6, 2) << activation()
     << fully_connected_layer(5 * 5 * 6, 3) << activation();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<loss_func>(
    test_data.first, test_data.second, epsilon<float_t>(), GRAD_CHECK_RANDOM));
}

TEST(network, gradient_check2) {  // tan_h - mse
  using loss_func  = mse;
  using network    = network<sequential>;
  using activation = tanh;

  network nn;
  nn << fully_connected_layer(10, 14 * 14 * 3) << activation()
     << convolutional_layer(14, 14, 5, 3, 6) << activation()
     << average_pooling_layer(10, 10, 6, 2) << activation()
     << fully_connected_layer(5 * 5 * 6, 3) << activation();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<loss_func>(
    test_data.first, test_data.second, epsilon<float_t>(), GRAD_CHECK_RANDOM));
}

TEST(network, gradient_check3) {  // mixture - mse
  using loss_func = mse;
  using network   = network<sequential>;

  network nn;
  nn << fully_connected_layer(10, 14 * 14 * 3) << tanh()
     << convolutional_layer(14, 14, 5, 3, 6) << sigmoid()
     << average_pooling_layer(10, 10, 6, 2) << relu()
     << fully_connected_layer(5 * 5 * 6, 3);

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<loss_func>(
    test_data.first, test_data.second, epsilon<float_t>(), GRAD_CHECK_RANDOM));
}

TEST(network, gradient_check4) {  // sigmoid - cross-entropy
  using loss_func  = cross_entropy;
  using activation = sigmoid;
  using network    = network<sequential>;

  network nn;
  nn << fully_connected_layer(10, 14 * 14 * 3) << activation()
     << convolutional_layer(14, 14, 5, 3, 6) << activation()
     << average_pooling_layer(10, 10, 6, 2) << activation()
     << fully_connected_layer(5 * 5 * 6, 3) << activation();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<loss_func>(
    test_data.first, test_data.second, epsilon<float_t>(), GRAD_CHECK_RANDOM));
}

TEST(network, gradient_check5) {  // softmax - cross-entropy
  using loss_func  = cross_entropy;
  using network    = network<sequential>;
  using activation = softmax;

  network nn;
  nn << fully_connected_layer(10, 14 * 14 * 3) << activation()
     << convolutional_layer(14, 14, 5, 3, 6) << activation()
     << average_pooling_layer(10, 10, 6, 2) << activation()
     << fully_connected_layer(5 * 5 * 6, 3) << activation();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           1e-1f, GRAD_CHECK_RANDOM));
}

TEST(network, gradient_check6) {  // sigmoid - cross-entropy
  using loss_func  = cross_entropy;
  using network    = network<sequential>;
  using activation = sigmoid;

  network nn;
  nn << fully_connected_layer(3, 201) << activation()
     << fully_connected_layer(201, 2) << activation();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(network, gradient_check7) {  // leaky-relu - mse
  using loss_func  = mse;
  using activation = leaky_relu;

  auto nn = make_mlp<activation>({3, 201, 2});

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  // We need to use larger threshold here, because
  // relu/elu/leaky-relu has non-smooth region of the form
  //
  // @todo improve gradient-checker to ignore non-smooth region like
  //       caffe's GradientChecker (they have kink/kink-range parameter to
  //       handle it)
  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           10 * epsilon<float_t>(),
                                           GRAD_CHECK_ALL));
}

TEST(network, gradient_check8) {  // elu - mse
  using loss_func  = mse;
  using activation = elu;

  auto nn = make_mlp<activation>({3, 201, 2});

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();

  // We need to use larger threshold here, because
  // relu/elu/leaky-relu has non-smooth region of the form
  //
  // @todo improve gradient-checker to ignore non-smooth region like
  //       caffe's GradientChecker (they have kink/kink-range parameter to
  //       handle it)
  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           10 * epsilon<float_t>(),
                                           GRAD_CHECK_ALL));
}

TEST(network, gradient_check9) {  // tanh_p1m2 - mse
  using loss_func  = mse;
  using activation = tanh_p1m2;

  auto nn = make_mlp<activation>({3, 201, 2});

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(network, gradient_check10) {  // softplus - mse
  using loss_func  = mse;
  using activation = softplus;

  auto nn = make_mlp<activation>({3, 201, 2});

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(network, gradient_check11) {  // softsign - mse
  using loss_func  = mse;
  using activation = softsign;

  auto nn = make_mlp<activation>({3, 201, 2});

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<loss_func>(test_data.first, test_data.second,
                                           epsilon<float_t>(), GRAD_CHECK_ALL));
}

#ifndef CNN_NO_SERIALIZATION

TEST(network, read_write) {
  using loss_func = mse;
  using network   = network<sequential>;

  network n1, n2;

  n1 << convolutional_layer(32, 32, 5, 1, 6) << tanh()
     << average_pooling_layer(28, 28, 6, 2)
     << convolutional_layer(14, 14, 5, 6, 16) << tanh()
     << average_pooling_layer(10, 10, 16, 2)
     << convolutional_layer(5, 5, 5, 16, 120) << tanh()
     << fully_connected_layer(120, 10) << softmax();

  n2 << convolutional_layer(32, 32, 5, 1, 6) << tanh()
     << average_pooling_layer(28, 28, 6, 2)
     << convolutional_layer(14, 14, 5, 6, 16) << tanh()
     << average_pooling_layer(10, 10, 16, 2)
     << convolutional_layer(5, 5, 5, 16, 120) << tanh()
     << fully_connected_layer(120, 10) << softmax();

  n1.init_weight();
  n2.init_weight();

  std::vector<vec_t> t;
  std::vector<label_t> l;
  t.push_back(vec_t(32 * 32, 0.0));
  l.push_back(3);
  adagrad optimizer;
  n1.train<loss_func>(optimizer, t, l, 1, 1);

  network_serialization_test(n1, n2);

  vec_t in(32 * 32, 0.0);

  auto res1 = n1.predict(in);
  auto res2 = n2.predict(in);

  ASSERT_TRUE(n1.has_same_weights(n2, epsilon<float_t>()));

  for (int i = 0; i < 10; i++) {
    tiny_dnn::float_t eps = std::abs(res1[i]) * 1e-5f;
    ASSERT_NEAR(res1[i], res2[i], eps);
  }
}

#endif  // #ifndef CNN_NO_SERIALIZATION

TEST(network, trainable) {
  auto net = make_mlp<sigmoid>({2, 3, 2, 1});  // fc(2,3) - fc(3,2) - fc(2,1)

  // trainable=false, or "freeze" 2nd layer fc(3,2)
  net[2]->set_trainable(false);

  vec_t w0 = {0, 1, 2, 3, 4, 5};
  vec_t w2 = {6, 7, 8, 9, 8, 7};
  vec_t w4 = {6, 5};

  *net[0]->weights()[0] = {0, 1, 2, 3, 4, 5};
  *net[2]->weights()[0] = {6, 7, 8, 9, 8, 7};
  *net[4]->weights()[0] = {6, 5};

  adam a;

  net.init_weight();

  auto w0_standby = *net[0]->weights()[0];
  auto w2_standby = *net[2]->weights()[0];
  auto w4_standby = *net[4]->weights()[0];

  EXPECT_NE(w0, w0_standby);
  EXPECT_EQ(w2, w2_standby);
  EXPECT_NE(w4, w4_standby);

  std::vector<vec_t> data{{1, 0}, {0, 2}};
  std::vector<vec_t> out{{2}, {1}};

  net.fit<mse>(a, data, out, 1, 1);

  auto w0_after_update = *net[0]->weights()[0];
  auto w2_after_update = *net[2]->weights()[0];
  auto w4_after_update = *net[4]->weights()[0];

  EXPECT_NE(w0, w0_after_update);
  EXPECT_EQ(w2, w2_after_update);
  EXPECT_NE(w4, w4_after_update);
}

TEST(network, fully_connected_train) {
  network<sequential> nn;
  adagrad optimizer;

  nn << fully_connected_layer(3, 2) << sigmoid();

  vec_t a(3), t(2), a2(3), t2(2);

  // clang-format off
  a[0] = 3.0f; a[1] = 0.0f; a[2] = -1.0f;
  t[0] = 0.3f; t[1] = 0.7f;

  a2[0] = 0.2f; a2[1] = 0.5f; a2[2] = 4.0f;
  t2[0] = 0.5f; t2[1] = 0.1f;
  // clang-format on

  std::vector<vec_t> data, train;

  for (int i = 0; i < 100; i++) {
    data.push_back(a);
    data.push_back(a2);
    train.push_back(t);
    train.push_back(t2);
  }
  optimizer.alpha = 0.1f;
  nn.train<mse>(optimizer, data, train, 1, 10);

  vec_t predicted = nn.predict(a);

  EXPECT_NEAR(predicted[0], t[0], 1E-5);
  EXPECT_NEAR(predicted[1], t[1], 1E-5);

  predicted = nn.predict(a2);

  EXPECT_NEAR(predicted[0], t2[0], 1E-5);
  EXPECT_NEAR(predicted[1], t2[1], 1E-5);
}

TEST(network, fully_connected_train_different_batches) {
  auto batch_sizes = {2, 7, 10, 12};
  size_t data_size = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 1,
                                     std::multiplies<int>());
  for (auto &batch_sz : batch_sizes) {
    network<sequential> nn;
    adagrad optimizer;

    nn << fully_connected_layer(3, 2) << sigmoid();

    vec_t a(3), t(2), a2(3), t2(2);

    // clang-format off
    a[0] = 3.0f; a[1] = 0.0f; a[2] = -1.0f;
    t[0] = 0.3f; t[1] = 0.7f;

    a2[0] = 0.2f; a2[1] = 0.5f; a2[2] = 4.0f;
    t2[0] = 0.5f; t2[1] = 0.1f;
    // clang-format on

    std::vector<vec_t> data, train;

    for (size_t i = 0; i < data_size; i++) {
      data.push_back(a);
      data.push_back(a2);
      train.push_back(t);
      train.push_back(t2);
    }
    optimizer.alpha = 0.1f;
    nn.train<mse>(optimizer, data, train, batch_sz, 10);

    vec_t predicted = nn.predict(a);

    EXPECT_NEAR(predicted[0], t[0], 1E-5);
    EXPECT_NEAR(predicted[1], t[1], 1E-5);

    predicted = nn.predict(a2);

    EXPECT_NEAR(predicted[0], t2[0], 1E-5);
    EXPECT_NEAR(predicted[1], t2[1], 1E-5);
  }
}

TEST(network, fully_connected_train2) {
  network<sequential> nn;
  gradient_descent optimizer;

  nn << fully_connected_layer(4, 6) << tanh() << fully_connected_layer(6, 3)
     << tanh();

  vec_t a(4, 0.0), t(3, 0.0), a2(4, 0.0), t2(3, 0.0);

  // clang-format off
    a[0] = 3.0f; a[1] = 1.0f; a[2] = -1.0f; a[3] = 4.0f;
    t[0] = 0.3f; t[1] = 0.7f; t[2] = 0.3f;

    a2[0] = 1.0f; a2[1] = 0.0f; a2[2] = 4.0f; a2[3] = 2.0f;
    t2[0] = 0.6f; t2[1] = 0.0f; t2[2] = 0.1f;
  // clang-format on

  std::vector<vec_t> data, train;

  for (int i = 0; i < 100; i++) {
    data.push_back(a);
    data.push_back(a2);
    train.push_back(t);
    train.push_back(t2);
  }
  optimizer.alpha = 0.1f;
  nn.train<mse>(optimizer, data, train, 1, 10);

  vec_t predicted = nn.predict(a);

  EXPECT_NEAR(predicted[0], t[0], 1E-4);
  EXPECT_NEAR(predicted[1], t[1], 1E-4);

  predicted = nn.predict(a2);

  EXPECT_NEAR(predicted[0], t2[0], 1E-4);
  EXPECT_NEAR(predicted[1], t2[1], 1E-4);
}

TEST(network, fully_connected_gradient_check) {
  network<sequential> nn;
  nn << fully_connected_layer(50, 10) << tanh();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

}  // namespace tiny_dnn
