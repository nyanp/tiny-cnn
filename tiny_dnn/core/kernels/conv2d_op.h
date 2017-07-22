/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"

#include "tiny_dnn/core/kernels/conv2d_op_avx.h"
#include "tiny_dnn/core/kernels/conv2d_op_internal.h"
#include "tiny_dnn/core/kernels/conv2d_op_nnpack.h"

namespace tiny_dnn {

class Conv2dOp : public core::OpKernel {
 public:
  explicit Conv2dOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->conv();

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

    if (backend == core::backend_t::internal) {
      kernels::conv2d_op_internal(in_data, W[0], bias[0], out_data, params,
                                  context.parallelize());
    } else if (backend == core::backend_t::nnpack) {
      kernels::conv2d_op_nnpack(in_data, W[0], bias[0], out_data, params);
    } else if (backend == core::backend_t::avx) {
      kernels::conv2d_op_avx(in_data, W[0], bias[0], out_data, params,
                             context.parallelize());
#if 0
		} else if (engine == core::backend_t::internal_quantization) {
			kernels::tiny_quantized_conv2d_kernel(*params_c_, *in[i], W, bias, out[i], layer_->parallelize());
		} else if (engine == core::backend_t::internal_eficient_quantization) {
			kernels::tiny_quantized_conv2d_kernel(*params_c_, *in[i], W, bias, in_r[i], W_r, b_r, out[i], out_r[i], layer_->parallelize());
#endif
    } else {
      throw nn_error("Not supported backend: " + to_string(backend));
    }
  }
};

}  // namespace tiny_dnn
