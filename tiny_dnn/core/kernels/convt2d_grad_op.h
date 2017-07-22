/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"

#include "tiny_dnn/core/kernels/tiny_deconv2d_back_kernel.h"

namespace tiny_dnn {

class Convt2dGradOp : public core::OpKernel {
 public:
  explicit Convt2dGradOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->deconv();

    // incoming/outcoming data
    const tensor_t &prev_out = context.input(0);
    const tensor_t &W        = context.input(1);
    tensor_t &dW             = context.input_grad(1);
    tensor_t &db             = context.input_grad(2);
    tensor_t &prev_delta     = context.input_grad(0);
    tensor_t &curr_delta     = context.output_grad(0);

    // initalize outputs
    fill_tensor(prev_delta, float_t{0});

    // call convolution algorithm depending
    // on the selected engine type

    const core::backend_t backend = context.backend_type();

    if (backend == core::backend_t::internal) {
			core::kernels::tiny_deconv2d_back_kernel(params, prev_out, W[0], dW, db,
curr_delta, &prev_delta);
    } else {
      throw nn_error("Not supported backend: " + to_string(backend));
    }
  }
};

}  // namespace tiny_dnn
