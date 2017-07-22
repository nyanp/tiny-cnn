/*
    Copyright (c) 2017, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"
#include "tiny_dnn/core/kernels/global_avepool_op_avx.h"
#include "tiny_dnn/core/kernels/global_avepool_op_internal.h"

namespace tiny_dnn {

class GlobalAvePoolOp : public core::OpKernel {
 public:
  explicit GlobalAvePoolOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto &params = OpKernel::params_->global_avepool();

    // incomimg / outcoming data
    const tensor_t &in_data = context.input(0);
    tensor_t &out_data      = context.output(0);

    // initialize outputs
    fill_tensor(out_data, float_t{0});

    const core::backend_t backend = context.backend_type();

    if (backend == core::backend_t::avx) {
#ifdef CNN_USE_AVX
      kernels::global_avepool_op_avx(in_data, out_data, params,
                                     context.parallelize());
#endif
    } else {
      kernels::global_avepool_op_internal(in_data, out_data, params,
                                          context.parallelize());
    }
  }
};

}  // namespace tiny_dnn
