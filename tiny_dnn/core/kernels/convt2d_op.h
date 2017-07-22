/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"

#include "tiny_dnn/core/kernels/tiny_deconv2d_kernel.h"

namespace tiny_dnn {

class Convt2dOp : public core::OpKernel {
 public:
  explicit Convt2dOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->deconv();

    // incomimg/outcoming data
    const tensor_t &in_data = context.input(0);
    const tensor_t &W       = context.input(1);
    const tensor_t &bias    = context.input(2);
    tensor_t &out_data      = context.output(0);

    // initialize outputs
    fill_tensor(out_data, float_t{0});

    // call convolution algorithm depending
    // on the selected engine type

    const core::backend_t backend = context.backend_type();

    if (backend == core::backend_t::internal || backend == core::backend_t::avx) {
      core::kernels::tiny_deconv2d_kernel(params, in_data, W[0], bias[0],
                                          out_data, context.parallelize());
    } else {
      throw nn_error("Not supported backend: " + to_string(backend));
    }
  }
};

}  // namespace tiny_dnn
