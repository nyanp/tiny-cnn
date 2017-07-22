/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"
#include "tiny_dnn/core/kernels/recurrent_cell_op_internal.h"

namespace tiny_dnn {

class RecurrentCellOp : public core::OpKernel {
 public:
  explicit RecurrentCellOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->recurrent_cell();

    // incomimg/outcoming data
    const tensor_t &in_data = context.input(0);
    const tensor_t &prev_h  = context.input(1);
    const tensor_t &U       = context.input(2);
    const tensor_t &W       = context.input(3);
    const tensor_t &V       = context.input(4);
    const tensor_t *bias    = params.has_bias_ ? &context.input(5) : nullptr;
    const tensor_t *c       = params.has_bias_ ? &context.input(6) : nullptr;
    tensor_t &out_data      = context.output(0);
    tensor_t &next_h        = context.output(1);

    // initialize outputs
    fill_tensor(out_data, float_t{0});
    fill_tensor(next_h, float_t{0});

    // call the algorithm depending  on the selected engine type

    const core::backend_t backend = context.backend_type();

    if (backend == core::backend_t::internal || backend == core::backend_t::avx) {
      kernels::recurrent_cell_op_internal(
        in_data, prev_h, U[0], W[0], V[0],
        params.has_bias_ ? (*bias)[0] : vec_t(),
        params.has_bias_ ? (*c)[0] : vec_t(), out_data, next_h, params,
        context.parallelize());
    } else {
      throw nn_error("Not supported backend: " + to_string(backend));
    }
  }
};

}  // namespace tiny_dnn
