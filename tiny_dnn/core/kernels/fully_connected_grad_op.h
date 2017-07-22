/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"

#include "tiny_dnn/core/kernels/fully_connected_op_avx.h"
#include "tiny_dnn/core/kernels/fully_connected_op_internal.h"

namespace tiny_dnn {

class FullyConnectedGradOp : public core::OpKernel {
 public:
  explicit FullyConnectedGradOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->fully();

    // incoming/outcoming data
    const tensor_t &prev_out = context.input(0);
    const tensor_t &W        = context.input(1);
    tensor_t &dW             = context.input_grad(1);
    tensor_t *db         = params.has_bias_ ? &context.input_grad(2) : nullptr;
    tensor_t &prev_delta = context.input_grad(0);
    tensor_t &curr_delta = context.output_grad(0);
    tensor_t dummy;  // need lvalue for non-const reference

    // initialize outputs
    fill_tensor(prev_delta, float_t{0});

    // call the algorithm depending on the selected engine type

    const core::backend_t engine = context.engine();

    if (engine == core::backend_t::internal) {
      kernels::fully_connected_op_internal(
        prev_out, W[0], dW, params.has_bias_ ? *db : dummy, curr_delta,
        prev_delta, params, params.has_bias_, context.parallelize());
    } else if (engine == core::backend_t::avx) {
      kernels::fully_connected_op_avx(
        prev_out, W[0], dW, params.has_bias_ ? *db : dummy, curr_delta,
        prev_delta, params, params.has_bias_, context.parallelize());
    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
  }
};

}  // namespace tiny_dnn
