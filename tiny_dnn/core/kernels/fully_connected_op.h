/*
    COPYRIGHT

    All contributions by Taiga Nomi
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    All other contributions:
    Copyright (c) 2013-2016, the respective contributors.
    All rights reserved.

    Each contributor holds copyright over their respective contributions.
    The project versioning (Git) records all such contribution source information.

    LICENSE

    The BSD 3-Clause License


    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of tiny-dnn nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"

#include "tiny_dnn/core/kernels/fully_connected_op_avx.h"
#include "tiny_dnn/core/kernels/fully_connected_op_internal.h"
#include "tiny_dnn/core/kernels/fully_connected_op_nnpack.h"

namespace tiny_dnn {

class FullyConnectedOp : public core::OpKernel {
 public:
    explicit FullyConnectedOp(const core::OpKernelConstruction& context)
        : core::OpKernel(context) {}

    void compute(const core::OpKernelContext& context) override {
        auto params = OpKernel::params_->fully();

        // incomimg/outcoming data 
        const tensor_t& in_data = context.input(0);
        const tensor_t&       W = context.input(1);
        const tensor_t*    bias = params.has_bias_ ? &context.input(2) : nullptr;
        tensor_t&      out_data = context.output(1);

        // initialize outputs
        fill_tensor(out_data, float_t(0));

        // call the algorithm depending  on the selected engine type

        const core::backend_t engine = context.engine();

	// tensor_t data converted to Tensor class
	// NOTE: this hack is temporal
	// TODO: Remove once layers forward and backward by themself.
	const Tensor<float_t> in_data_t(in_data.size(), params.in_size_, 1, 1,
			                in_data);
	const Tensor<float_t> weights_t(W.size(), W[0].size(), 1, 1, W);
	const Tensor<float_t> bias_t(params.has_bias_ ? bias->size() : 0,
			             params.has_bias_ ? (*bias)[0].size() : 0,
				     params.has_bias_ ? 1 : 0,
				     params.has_bias_ ? 1 : 0,
				     params.has_bias_ ? *bias : tensor_t());
        Tensor<float_t> out_data_t(out_data.size(), params.out_size_, 1, 1,
			           out_data);

        if (engine == core::backend_t::internal) {
            kernels::fully_connected_op_internal(
                in_data_t,
                weights_t,
                bias_t,
                &out_data_t,
                context.parallelize());

                // convert Tensor class to tensor_t
		// NOTE: this hack is temporal
		// TODO: Remove once layers forward and backward by themself.
                out_data = out_data_t.toTensor();
        }
        else if (engine == core::backend_t::nnpack) {
            kernels::fully_connected_op_nnpack(
                in_data,
                W[0],
                params.has_bias_ ? (*bias)[0] : vec_t(),
                out_data,
                params,
                context.parallelize());
        }
        else if (engine == core::backend_t::avx) {
            kernels::fully_connected_op_avx(
                in_data,
                W[0],
                params.has_bias_ ? (*bias)[0] : vec_t(),
                out_data,
                params,
                context.parallelize());
        }
        else {
            throw nn_error("Not supported engine: " + to_string(engine));
        }
    }
};

}  // namespace tiny_dnn
