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

    // TODO(Randl): Remove once layers forward and backward by themself.
    const Tensor<float_t> prev_out(context.input(0));
    Tensor<float_t> prev_delta(context.input_grad(0));
    Tensor<float_t> curr_delta(context.output_grad(0));

    const Tensor<float_t> weights(*(context.ith_parameter(0)->data()));
    const Tensor<float_t> bias(params.has_bias_
                                 ? *(context.ith_parameter(1)->data())
                                 : Tensor<float_t>());

    Tensor<float_t> weights_grads(*(context.ith_parameter(0)->grad()));
    Tensor<float_t> bias_grads(params.has_bias_
                                 ? *(context.ith_parameter(1)->grad())
                                 : Tensor<float_t>());

    // initialize outputs
    prev_delta.fill(float_t{0});

    // call the algorithm depending on the selected engine type

    const core::backend_t engine = context.engine();

    if (engine == core::backend_t::internal) {
      kernels::fully_connected_op_internal(prev_out, weights, weights_grads,
                                           bias_grads, curr_delta, prev_delta,
                                           context.parallelize());
    } else if (engine == core::backend_t::avx) {
      kernels::fully_connected_op_avx(prev_out, weights, weights_grads,
                                      bias_grads, curr_delta, prev_delta,
                                      context.parallelize());
    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
    context.ith_parameter(0)->set_grad(weights_grads);
    context.ith_parameter(1)->set_grad(bias_grads);
    context.input_grad(0) = prev_delta.toTensor();
  }
};

}  // namespace tiny_dnn
