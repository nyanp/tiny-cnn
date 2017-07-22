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
#include "tiny_dnn/core/kernels/fully_connected_op_nnpack.h"

namespace tiny_dnn {

class FullyConnectedOp : public core::OpKernel {
 public:
  explicit FullyConnectedOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->fully();

    // incomimg/outcoming data
    const tensor_t &in_data = context.input(0);
    const tensor_t &W       = context.input(1);
    const tensor_t *bias    = params.has_bias_ ? &context.input(2) : nullptr;
    tensor_t &out_data      = context.output(0);

    // initialize outputs
    fill_tensor(out_data, float_t{0});

    // call the algorithm depending  on the selected engine type

    const core::backend_t engine = context.engine();

    if (engine == core::backend_t::internal) {
      kernels::fully_connected_op_internal(
        in_data, W[0], params.has_bias_ ? (*bias)[0] : vec_t(), out_data,
        params, params.has_bias_, context.parallelize());
    } else if (engine == core::backend_t::nnpack) {
      kernels::fully_connected_op_nnpack(
        in_data, W[0], params.has_bias_ ? (*bias)[0] : vec_t(), out_data,
        params, context.parallelize());
    } else if (engine == core::backend_t::avx) {
      kernels::fully_connected_op_avx(in_data, W[0],
                                      params.has_bias_ ? (*bias)[0] : vec_t(),
                                      out_data, params, params.has_bias_, context.parallelize());
    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
  }
};

}  // namespace tiny_dnn
