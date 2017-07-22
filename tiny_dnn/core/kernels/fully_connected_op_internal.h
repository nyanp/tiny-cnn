/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/fully_params.h"

namespace tiny_dnn {
namespace kernels {

inline void fully_connected_op_internal(const tensor_t &in_data,
                                        const vec_t &W,
                                        const vec_t &bias,
                                        tensor_t &out_data,
                                        const fully_params &params,
                                        const bool has_bias,
                                        const bool layer_parallelize) {
  for_i(layer_parallelize, in_data.size(), [&](size_t sample) {
    const vec_t &in = in_data[sample];
    vec_t &out      = out_data[sample];

    for (size_t i = 0; i < params.out_size_; i++) {
      out[i] = float_t{0};
      for (size_t c = 0; c < params.in_size_; c++) {
        out[i] += W[c * params.out_size_ + i] * in[c];
      }

      if (has_bias) {
        out[i] += bias[i];
      }
    }
  });
}

inline void fully_connected_op_internal(const tensor_t &prev_out,
                                        const vec_t &W,
                                        tensor_t &dW,
                                        tensor_t &db,
                                        tensor_t &curr_delta,
                                        tensor_t &prev_delta,
                                        const fully_params &params,
                                        const bool has_bias,
                                        const bool layer_parallelize) {
  for (size_t sample = 0; sample < prev_out.size(); sample++) {
    for (size_t c = 0; c < params.in_size_; c++) {
      // propagate delta to previous layer
      // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
      prev_delta[sample][c] += vectorize::dot(
        &curr_delta[sample][0], &W[c * params.out_size_], params.out_size_);
    }

    for_(layer_parallelize, 0, params.out_size_, [&](const blocked_range &r) {
      // accumulate weight-step using delta
      // dW[c * out_size + i] += current_delta[i] * prev_out[c]
      for (size_t c = 0; c < params.in_size_; c++) {
        vectorize::muladd(&curr_delta[sample][r.begin()], prev_out[sample][c],
                          r.end() - r.begin(),
                          &dW[sample][c * params.out_size_ + r.begin()]);
      }

      if (has_bias) {
        // vec_t& db = *in_grad[2];
        for (size_t i = r.begin(); i < r.end(); i++) {
          db[sample][i] += curr_delta[sample][i];
        }
      }
    });
  }
}

}  // namespace kernels
}  // namespace tiny_dnn
