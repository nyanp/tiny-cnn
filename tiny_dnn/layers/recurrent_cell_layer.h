/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "tiny_dnn/activations/activation_layer.h"
#include "tiny_dnn/activations/tanh_layer.h"
#include "tiny_dnn/core/kernels/recurrent_cell_grad_op.h"
#include "tiny_dnn/core/kernels/recurrent_cell_op.h"
#include "tiny_dnn/layers/layer.h"

namespace tiny_dnn {

/**
 * RNN layer.
 *
 * y(t-1)    y(t)   > h(t) = tanh(b + W*h(t-1) + U*x(t))
 *   ^        ^     > y(t) = c + V*h(t)
 *   |V+c     | V+c
 * h(t-1) -> h(t)
 *   ^ +b W   ^ +b
 *   |U       |U
 * x(t-1)    x(t)
 *
 **/
class recurrent_cell_layer : public layer {
 public:
  using layer::parallelize_;

  inline std::vector<vector_type> recurrent_order(bool has_bias) {
    if (has_bias) {
      return {vector_type::data,    // input vector
              vector_type::aux,     // input state (h(t-1))
              vector_type::weight,  // input weights (U)
              vector_type::weight,  // transition weights (W)
              vector_type::weight,  // output weights (V)
              vector_type::bias,    // transition bias
              vector_type::bias};
    } else {
      return {vector_type::data,     // input vector
              vector_type::aux,      // input state (h(t-1))
              vector_type::weight,   // input weights (U)
              vector_type::weight,   // transition weights (W)
              vector_type::weight};  // output weights (V)
    }
  }

  /**
   * @param in_dim [in] number of elements of the input
   * @param out_dim [in] number of elements of the output
   * @param has_bias [in] whether to include additional bias to the layer
   * @param activation [in] activation function to be used internally
   **/
  recurrent_cell_layer(size_t in_dim,
                       size_t out_dim,
                       bool has_bias                = true,
                       activation_layer *activation = new tanh_layer,
                       backend_t backend_type       = core::default_backend())
    : layer(recurrent_order(has_bias),  // output bias
            {vector_type::data,         // output vector
             vector_type::aux}) {
    set_params(in_dim, out_dim, has_bias, activation);
    init_backend(backend_type);
    layer::set_backend_type(backend_type);
  }

  // move constructor
  recurrent_cell_layer(recurrent_cell_layer &&other)
    : layer(std::move(other)),
      params_(std::move(other.params_)),
      kernel_fwd_(std::move(other.kernel_fwd_)),
      kernel_back_(std::move(other.kernel_back_)) {
    init_backend(std::move(other.backend_type()));
  }

  size_t fan_in_size(size_t i) const override { return in_shape()[i].width_; }

  size_t fan_out_size(size_t i) const override { return in_shape()[i].height_; }

  std::vector<index3d<size_t>> in_shape() const override {
    if (params_.has_bias_) {
      return {index3d<size_t>(params_.in_size_, 1, 1),   // x
              index3d<size_t>(params_.out_size_, 1, 1),  // h(t-1)
              index3d<size_t>(params_.in_size_, params_.out_size_, 1),   // U
              index3d<size_t>(params_.out_size_, params_.out_size_, 1),  // W
              index3d<size_t>(params_.out_size_, params_.out_size_, 1),  // V
              index3d<size_t>(params_.out_size_, 1, 1),                  // b
              index3d<size_t>(params_.out_size_, 1, 1)};                 // c
    } else {
      return {index3d<size_t>(params_.in_size_, 1, 1),   // x
              index3d<size_t>(params_.out_size_, 1, 1),  // h(t-1)
              index3d<size_t>(params_.in_size_, params_.out_size_, 1),    // U
              index3d<size_t>(params_.out_size_, params_.out_size_, 1),   // W
              index3d<size_t>(params_.out_size_, params_.out_size_, 1)};  // V
    }
  }

  std::vector<index3d<size_t>> out_shape() const override {
    return {index3d<size_t>(params_.out_size_, 1, 1),
            index3d<size_t>(params_.out_size_, 1, 1)};  // h(t)
  }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    // forward fully connected op context
    fwd_ctx_.set_in_out(in_data, out_data);
    fwd_ctx_.setParallelize(layer::parallelize());
    fwd_ctx_.setBackendType(layer::backend_type());

    // launch recurrent kernel
    kernel_fwd_->compute(fwd_ctx_);
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    // backward fully connected op context
    bwd_ctx_.set_in_out(in_data, out_data, out_grad, in_grad);
    bwd_ctx_.setParallelize(layer::parallelize());
    bwd_ctx_.setBackendType(layer::backend_type());

    // launch recurrent kernel
    kernel_back_->compute(bwd_ctx_);
  }

  void set_activation(std::shared_ptr<activation_layer> activation) {
    params_.activation_ = activation;
  }

  std::string layer_type() const override { return "recurrent-cell"; }

  friend struct serialization_buddy;

 protected:
  void set_params(const size_t in_size,
                  const size_t out_size,
                  bool has_bias,
                  activation_layer *activation) {
    params_.in_size_    = in_size;
    params_.out_size_   = out_size;
    params_.has_bias_   = has_bias;
    params_.activation_ = std::shared_ptr<activation_layer>(activation);
  }

  void init_backend(backend_t backend_type) {
    CNN_UNREFERENCED_PARAMETER(backend_type);
    core::OpKernelConstruction ctx =
      core::OpKernelConstruction(layer::device(), &params_);
    kernel_fwd_.reset(new RecurrentCellOp(ctx));
    kernel_back_.reset(new RecurrentCellGradOp(ctx));
  }

 private:
  /* The layer parameters */
  recurrent_cell_params params_;

  /* forward op context */
  OpKernelContext fwd_ctx_;

  /* backward op context */
  OpKernelContext bwd_ctx_;

  /* Forward and backward ops */
  std::shared_ptr<core::OpKernel> kernel_fwd_;
  std::shared_ptr<core::OpKernel> kernel_back_;
};

}  // namespace tiny_dnn
