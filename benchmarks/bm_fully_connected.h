/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "benchmark/benchmark.h"
#include "tiny_dnn/tiny_dnn.h"

namespace bm = benchmark;

const size_t BM_CONST_IN_SIZE     = 64;
const size_t BM_CONST_OUT_SIZE    = 64;
const size_t BM_CONST_SAMPLE_SIZE = 1;

namespace tiny_dnn {
namespace benchmarks {

/**
 * Benchmark fixture for Fully Connected Op kernel forward on any backend.
 * Used for both single threaded as well as parallelized op kernel.
 *
 * Performs uniform random initialization of all tensors depending on the
 * provided size.
 */
class BM_FullyConnectedOp : public bm::Fixture {
 public:
  void SetUp(const bm::State& state) {
    in_size  = state.range(0);
    out_size = state.range(1);
    samples  = state.range(2);

    in_data.reshape({samples, in_size});
    out_data.reshape({samples, out_size});
    weights.reshape({in_size * out_size});
    bias.reshape({out_size});

    uniform_rand(in_data.host_begin(), in_data.host_end(), -1, 1);
    uniform_rand(weights.host_begin(), weights.host_end(), -1, 1);
    uniform_rand(bias.host_begin(), bias.host_end(), -1, 1);
    uniform_rand(out_data.host_begin(), out_data.host_end(), -1, 1);
  }

  void TearDown(const bm::State&) {}

  size_t in_size, out_size, samples;
  Tensor<> in_data, out_data, weights, bias;
};

/**
 * Benchmark fixture for Fully Connected Op kernel backward on any backend.
 * Similar as the forward fixture.
 */
class BM_FullyConnectedGradOp : public bm::Fixture {
 public:
  void SetUp(const bm::State& state) {
    in_size  = state.range(0);
    out_size = state.range(1);
    samples  = state.range(2);

    prev_out.reshape({samples, in_size});
    weights.reshape({in_size * out_size});
    weights_grads.reshape({samples, in_size * out_size});
    bias_grads.reshape({samples, out_size});
    curr_delta.reshape({samples, out_size});
    prev_delta.reshape({samples, in_size});

    uniform_rand(prev_out.host_begin(), prev_out.host_end(), -1, 1);
    uniform_rand(weights.host_begin(), weights.host_end(), -1, 1);
    uniform_rand(weights_grads.host_begin(), weights_grads.host_end(), -1, 1);
    uniform_rand(bias_grads.host_begin(), bias_grads.host_end(), -1, 1);
    uniform_rand(curr_delta.host_begin(), curr_delta.host_end(), -1, 1);
    uniform_rand(prev_delta.host_begin(), prev_delta.host_end(), -1, 1);
  }

  void TearDown(const bm::State&) {}

  size_t in_size, out_size, samples;
  Tensor<> prev_out, curr_delta, prev_delta;
  Tensor<> weights, weights_grads, bias_grads;
};

/** Keeping same number of samples in single threaded benchmarks. */
void args_with_same_samples(bm::internal::Benchmark* b) {
  b->Args({16, 4, BM_CONST_SAMPLE_SIZE})
    ->Args({64, 4, BM_CONST_SAMPLE_SIZE})
    ->Args({256, 4, BM_CONST_SAMPLE_SIZE})
    ->Args({256, 16, BM_CONST_SAMPLE_SIZE})
    ->Args({256, 64, BM_CONST_SAMPLE_SIZE})
    ->Args({256, 256, BM_CONST_SAMPLE_SIZE});
}

/** Keeping same input output size in multi threaded benchmarks. */
void args_with_same_in_out(bm::internal::Benchmark* b) {
  b->Args({BM_CONST_IN_SIZE, BM_CONST_OUT_SIZE, 1})
    ->Args({BM_CONST_IN_SIZE, BM_CONST_OUT_SIZE, 2})
    ->Args({BM_CONST_IN_SIZE, BM_CONST_OUT_SIZE, 4})
    ->Args({BM_CONST_IN_SIZE, BM_CONST_OUT_SIZE, 8})
    ->Args({BM_CONST_IN_SIZE, BM_CONST_OUT_SIZE, 16})
    ->Args({BM_CONST_IN_SIZE, BM_CONST_OUT_SIZE, 32});
}

BENCHMARK_DEFINE_F(BM_FullyConnectedOp, internal)(bm::State& state) {
  while (state.KeepRunning()) {
    kernels::fully_connected_op_internal(in_data, weights, bias, out_data,
                                         false);
  }
}

BENCHMARK_DEFINE_F(BM_FullyConnectedOp, internal_parallel)(bm::State& state) {
  while (state.KeepRunning()) {
    kernels::fully_connected_op_internal(in_data, weights, bias, out_data,
                                         true);
  }
}

BENCHMARK_DEFINE_F(BM_FullyConnectedGradOp, internal)(bm::State& state) {
  while (state.KeepRunning()) {
    kernels::fully_connected_op_internal(prev_out, weights, weights_grads,
                                         bias_grads, curr_delta, prev_delta,
                                         false);
  }
}

BENCHMARK_DEFINE_F(BM_FullyConnectedGradOp, internal_parallel)
(bm::State& state) {
  while (state.KeepRunning()) {
    kernels::fully_connected_op_internal(prev_out, weights, weights_grads,
                                         bias_grads, curr_delta, prev_delta,
                                         true);
  }
}

#ifdef CNN_USE_AVX
BENCHMARK_DEFINE_F(BM_FullyConnectedOp, avx)(bm::State& state) {
  while (state.KeepRunning()) {
    kernels::fully_connected_op_avx(in_data, weights, bias, out_data, false);
  }
}

BENCHMARK_DEFINE_F(BM_FullyConnectedOp, avx_parallel)(bm::State& state) {
  while (state.KeepRunning()) {
    kernels::fully_connected_op_avx(in_data, weights, bias, out_data, true);
  }
}

BENCHMARK_DEFINE_F(BM_FullyConnectedGradOp, avx)(bm::State& state) {
  while (state.KeepRunning()) {
    kernels::fully_connected_op_avx(prev_out, weights, weights_grads,
                                    bias_grads, curr_delta, prev_delta, false);
  }
}

BENCHMARK_DEFINE_F(BM_FullyConnectedGradOp, avx_parallel)(bm::State& state) {
  while (state.KeepRunning()) {
    kernels::fully_connected_op_avx(prev_out, weights, weights_grads,
                                    bias_grads, curr_delta, prev_delta, true);
  }
}
#endif

BENCHMARK_REGISTER_F(BM_FullyConnectedOp, internal)
  ->Apply(args_with_same_samples);

BENCHMARK_REGISTER_F(BM_FullyConnectedOp, internal_parallel)
  ->Apply(args_with_same_in_out);

BENCHMARK_REGISTER_F(BM_FullyConnectedGradOp, internal)
  ->Apply(args_with_same_samples);

BENCHMARK_REGISTER_F(BM_FullyConnectedGradOp, internal_parallel)
  ->Apply(args_with_same_in_out);

#ifdef CNN_USE_AVX
BENCHMARK_REGISTER_F(BM_FullyConnectedOp, avx)->Apply(args_with_same_samples);

BENCHMARK_REGISTER_F(BM_FullyConnectedOp, avx_parallel)
  ->Apply(args_with_same_in_out);

BENCHMARK_REGISTER_F(BM_FullyConnectedGradOp, avx)
  ->Apply(args_with_same_samples);

BENCHMARK_REGISTER_F(BM_FullyConnectedGradOp, avx_parallel)
  ->Apply(args_with_same_in_out);
#endif

}  // namespace benchmarks
}  // namespace tiny_dnn
