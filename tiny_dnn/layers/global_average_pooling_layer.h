/*
    Copyright (c) 2015, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/util/util.h"

#include "tiny_dnn/core/kernels/global_avepool_grad_op.h"
#include "tiny_dnn/core/kernels/global_avepool_op.h"

namespace tiny_dnn {

/**
 * applies channel-wise global average pooling to spatial data.
 **/
class global_average_pooling_layer : public layer {
 public:
  global_average_pooling_layer(const shape3d &in_shape,
                               backend_t backend_type = core::default_backend())
    : global_average_pooling_layer(
        in_shape.width_, in_shape.height_, in_shape.depth_, backend_type) {}

  /**
   * @param in_width     [in] width of input image
   * @param in_height    [in] height of input image
   * @param in_channels  [in] the number of input image channels (depth)
  **/
  global_average_pooling_layer(size_t in_width,
                               size_t in_height,
                               size_t in_channels,
                               backend_t backend_type = core::default_backend())
    : layer({vector_type::data}, {vector_type::data}) {
    set_global_avepool_params(shape3d(in_width, in_height, in_channels),
                              shape3d(in_channels, 1, 1));

    init_backend(backend_type);
  }

  // move constructor
  global_average_pooling_layer(global_average_pooling_layer &&other)  // NOLINT
    : layer(std::move(other)), params_(std::move(other.params_)) {
    init_backend(std::move(layer::backend_type()));
  }

  size_t fan_in_size() const override {
    return params_.in.width_ * params_.in.height_;
  }

  size_t fan_out_size() const override { return 1; }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    fwd_ctx_.set_in_out(in_data, out_data);
    fwd_ctx_.setParallelize(layer::parallelize());
    fwd_ctx_.setBackendType(layer::backend_type());

    kernel_fwd_->compute(fwd_ctx_);
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    bwd_ctx_.set_in_out(in_data, out_data, out_grad, in_grad);
    bwd_ctx_.setParallelize(layer::parallelize());
    bwd_ctx_.setBackendType(layer::backend_type());

    kernel_back_->compute(bwd_ctx_);
  }

  std::vector<index3d<size_t>> in_shape() const override {
    return {params_.in};
  }

  std::vector<index3d<size_t>> out_shape() const override {
    return {params_.out};
  }

  std::string layer_type() const override {
    return std::string("global-ave-pool");
  }

  std::pair<size_t, size_t> pool_size() const {
    return std::make_pair(params_.in.width_, params_.in.height_);
  }

  friend struct serialization_buddy;

 private:
  global_avepool_params params_;

  /* forward op context */
  OpKernelContext fwd_ctx_;

  /* backward op context */
  OpKernelContext bwd_ctx_;

  /* Forward and backward ops */
  std::shared_ptr<core::OpKernel> kernel_fwd_;
  std::shared_ptr<core::OpKernel> kernel_back_;

  void init_backend(backend_t backend_type) {
    core::OpKernelConstruction ctx =
      core::OpKernelConstruction(layer::device(), &params_);

    layer::set_backend_type(backend_type);
    if (backend_type == core::backend_t::avx) {
#ifndef CNN_USE_AVX
      nn_warn(
        "tiny-dnn has not been compiled with AVX support, "
        "fallback to internal backend for global avepool layer.\n");
      layer::set_backend_type(core::backend_t::internal);
#endif
    }
    kernel_fwd_.reset(new GlobalAvePoolOp(ctx));
    kernel_back_.reset(new GlobalAvePoolGradOp(ctx));
    return;
  }

  void set_global_avepool_params(const shape3d &in, const shape3d &out) {
    params_.in  = in;
    params_.out = out;
  }
};

}  // namespace tiny_dnn
