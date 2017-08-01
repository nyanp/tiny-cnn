/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "tiny_dnn/layers/layer.h"

#include "tiny_dnn/core/kernels/fully_connected_grad_op.h"
#include "tiny_dnn/core/kernels/fully_connected_op.h"

namespace tiny_dnn {

struct layer_params {
  bool parallellze = true;

  backend_t backend_type = core::default_engine();
};

struct fully_connected_layer_params : public layer_params {
  bool bias = true;
};

/**
 * compute fully-connected(matmul) operation
 **/
class fully_connected_layer : public layer {
 public:
  /**
   * @param in_features [in] number of elements of the input
   * @param out_features [in] number of elements of the output
   * @param bias [in] whether to include additional bias to the layer
   **/
  fully_connected_layer(size_t in_features,
                        size_t out_features,
                        bool bias              = true,
                        backend_t backend_type = core::default_engine())
    : layer({vector_type::data}, {vector_type::data}) {
    layer::add_parameter(1, 1, out_features, in_features,
                         parameter_type::weight, true);
    if (bias) {
      layer::add_parameter(1, 1, 1, out_features, parameter_type::bias, true);
    }
    set_params(in_features, out_features, bias);
    init_backend(backend_type);
    layer::set_backend_type(backend_type);
  }

  /**
    * @param in_features [in] number of elements of the input
    * @param out_features [in] number of elements of the output
    * @param bias [in] whether to include additional bias to the layer
    **/
  fully_connected_layer(size_t in_features,
                        size_t out_features,
                        fully_connected_layer_params params)
    : layer({vector_type::data}, {vector_type::data}) {
    set_params(in_features, out_features, params.bias);
    init_backend(params.backend_type);
    layer::set_backend_type(params.backend_type);
  }

  // move constructor
  fully_connected_layer(fully_connected_layer &&other)
    : layer(std::move(other)),
      params_(std::move(other.params_)),
      kernel_fwd_(std::move(other.kernel_fwd_)),
      kernel_back_(std::move(other.kernel_back_)) {
    init_backend(std::move(other.engine()));
  }

  size_t fan_in_size() const override { return params_.in_size_; }

  size_t fan_out_size() const override { return params_.out_size_; }

  std::vector<shape3d> in_shape() const override {
    return {shape3d(params_.in_size_, 1, 1)};
  }

  std::vector<shape3d> out_shape() const override {
    return {shape3d(params_.out_size_, 1, 1)};
  }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    // forward fully connected op context
    fwd_ctx_.set_in_out(in_data, out_data);
    fwd_ctx_.setParallelize(layer::parallelize());
    fwd_ctx_.setEngine(layer::engine());
    fwd_ctx_.setParameters(parameters());

    // launch fully connected kernel
    kernel_fwd_->compute(fwd_ctx_);
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    // backward fully connected op context
    bwd_ctx_.set_in_out(in_data, out_data, out_grad, in_grad);
    bwd_ctx_.setParallelize(layer::parallelize());
    bwd_ctx_.setEngine(layer::engine());
    bwd_ctx_.setParameters(parameters());

    // launch fully connected kernel
    kernel_back_->compute(bwd_ctx_);
  }

  std::string layer_type() const override { return "fully-connected"; }

  friend struct serialization_buddy;

 protected:
  void set_params(const size_t in_size, const size_t out_size, bool has_bias) {
    params_.in_size_  = in_size;
    params_.out_size_ = out_size;
    params_.has_bias_ = has_bias;
  }

  void init_backend(backend_t backend_type) {
    core::OpKernelConstruction ctx =
      core::OpKernelConstruction(layer::device(), &params_);

    if (backend_type == backend_t::internal || backend_type == backend_t::avx ||
        backend_type == backend_t::nnpack) {
      kernel_fwd_.reset(new FullyConnectedOp(ctx));
      kernel_back_.reset(new FullyConnectedGradOp(ctx));
    } else {
      throw nn_error("Not supported engine: " + to_string(backend_type));
    }
  }

 private:
  /* The layer parameters */
  fully_params params_;

  /* forward op context */
  OpKernelContext fwd_ctx_;

  /* backward op context */
  OpKernelContext bwd_ctx_;

  /* Forward and backward ops */
  std::shared_ptr<core::OpKernel> kernel_fwd_;
  std::shared_ptr<core::OpKernel> kernel_back_;
};

}  // namespace tiny_dnn
