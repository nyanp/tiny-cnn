// Copyright (c) 2013-2016, Taiga Nomi. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#pragma once

namespace tiny_dnn {
namespace kernels {

inline void conv2d_op_custom(const tensor_t& in_data, const vec_t& W,
                             const vec_t& bias, tensor_t& out_data,
                             const core::conv_params& params,
                             const bool parallelize) {
  for_i(parallelize, in_data.size(), [&](int sample) {
    const vec_t& in = in_data[sample];
    vec_t& a = out_data[sample];

    for (cnn_size_t o = 0; o < params.out.depth_; o++) {
      for (cnn_size_t inc = 0; inc < params.in.depth_; inc++) {
        if (!params.tbl.is_connected(o, inc)) continue;

        cnn_size_t idx = 0;
        idx = params.in.depth_ * o + inc;
        idx = params.weight.get_index(0, 0, idx);
        const float_t* pw = &W[idx];

        idx = params.in_padded.get_index(0, 0, inc);
        const float_t* pi = &in[idx];

        idx = params.out.get_index(0, 0, o);
        float_t* pa = &a[idx];

        for (cnn_size_t y = 0; y < params.out.height_; y++) {
          for (cnn_size_t x = 0; x < params.out.width_; x++) {
            const float_t* ppw = pw;
            const float_t* ppi =
                pi + params.in_padded.width_ * (y * params.h_stride) +
                x * params.w_stride;
            float_t sum = float_t(0);

            // should be optimized for small kernel(3x3,5x5)
            for (cnn_size_t wy = 0; wy < params.weight.height_;
                 wy++) {  // NOLINT
              for (cnn_size_t wx = 0; wx < params.weight.width_;
                   wx++) {  // NOLINT
                idx = wy * params.in_padded.width_ + wx;
                sum += *ppw++ * ppi[idx];
              }
            }
            pa[y * params.out.width_ + x] += sum;
          }
        }
      }

      if (params.has_bias) {
        float_t* pa = &a[params.out.get_index(0, 0, o)];
        float_t* paa = pa + params.out.width_ * params.out.height_;
        std::for_each(pa, paa, [&](float_t& f) { f += bias[o]; });
      }
    }
  });
}

/******************************************************************/

inline void conv2d_op_custom(const tensor_t& prev_out, const vec_t& W,
                             tensor_t& dW, tensor_t& db, tensor_t& curr_delta,
                             tensor_t& prev_delta,
                             const core::conv_params& params,
                             const bool parallelize) {
  for_i(parallelize, prev_out.size(), [&](int sample) {
    // propagate delta to previous layer
    for (cnn_size_t inc = 0; inc < params.in.depth_; inc++) {
      for (cnn_size_t outc = 0; outc < params.out.depth_; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        cnn_size_t idx = 0;
        idx = params.in.depth_ * outc + inc;
        idx = params.weight.get_index(0, 0, idx);
        const float_t* pw = &W[idx];

        idx = params.out.get_index(0, 0, outc);
        const float_t* pdelta_src = &curr_delta[sample][idx];

        idx = params.in_padded.get_index(0, 0, inc);
        // float_t *pdelta_dst = &(*prev_delta)[sample][idx];
        float_t* pdelta_dst = &prev_delta[sample][idx];

        for (cnn_size_t y = 0; y < params.out.height_; y++) {
          for (cnn_size_t x = 0; x < params.out.width_; x++) {
            const float_t* ppw = pw;

            idx = y * params.out.width_ + x;
            const float_t ppdelta_src = pdelta_src[idx];

            float_t* ppdelta_dst =
                pdelta_dst + y * params.h_stride * params.in_padded.width_ +
                x * params.w_stride;

            for (cnn_size_t wy = 0; wy < params.weight.height_;
                 wy++) {  // NOLINT
              for (cnn_size_t wx = 0; wx < params.weight.width_;
                   wx++) {  // NOLINT
                idx = wy * params.in_padded.width_ + wx;
                ppdelta_dst[idx] += *ppw++ * ppdelta_src;
              }
            }
          }
        }
      }
    }

    // accumulate dw
    for (cnn_size_t inc = 0; inc < params.in.depth_; inc++) {
      for (cnn_size_t outc = 0; outc < params.out.depth_; outc++) {
        if (!params.tbl.is_connected(outc, inc)) continue;

        for (cnn_size_t wy = 0; wy < params.weight.height_; wy++) {
          for (cnn_size_t wx = 0; wx < params.weight.width_; wx++) {
            float_t dst = float_t(0);

            cnn_size_t idx = 0;
            idx = params.in_padded.get_index(wx, wy, inc);
            const float_t* prevo = &prev_out[sample][idx];

            idx = params.out.get_index(0, 0, outc);
            const float_t* delta = &curr_delta[sample][idx];

            if (params.w_stride > 1) {
              for (cnn_size_t y = 0; y < params.out.height_; y++) {
                cnn_size_t prevo_idx =
                    y * params.in_padded.width_ * params.h_stride;
                cnn_size_t delta_idx = y * params.out.width_;

                for (cnn_size_t x = 0; x < params.out.width_; x++) {
                  dst += prevo[prevo_idx + x * params.w_stride] *
                         delta[delta_idx + x];
                }
              }
            } else {
              for (cnn_size_t y = 0; y < params.out.height_; y++) {
                dst += vectorize::dot(
                    prevo + y * params.in_padded.width_ * params.h_stride,
                    delta + y * params.out.width_, params.out.width_);
              }
            }

            idx = params.in.depth_ * outc + inc;
            dW[sample][params.weight.get_index(wx, wy, idx)] += dst;
          }
        }
      }
    }

    // accumulate db
    if (params.has_bias) {
      for (cnn_size_t outc = 0; outc < params.out.depth_; outc++) {
        cnn_size_t idx = params.out.get_index(0, 0, outc);
        const float_t* delta = &curr_delta[sample][idx];
        const float_t* deltaa = delta + params.out.width_ * params.out.height_;
        db[sample][outc] += std::accumulate(delta, deltaa, float_t(0));
      }
    }
  });
}

}  // namespace kernels
}  // namespace tiny_dnn
