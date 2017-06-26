/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/tensor.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

enum class parameter_type : int8_t { weight = 0x0001, bias = 0x0002 };

class parameter {
 public:
  parameter(serial_size_t width,
            serial_size_t height,
            serial_size_t depth,
            serial_size_t n_fmaps,
            parameter_type type,
            bool trainable = true)
    : type_(type),
      shape_(width, height, depth),
      n_fmaps_(n_fmaps),
      trainable_(trainable),
      data_({size()}),
      grad_({1, size()}) {}

  shape3d shape() { return shape_; }

  size_t size() { return shape_.size() * n_fmaps_; }

  parameter_type type() { return type_; }

  void set_dims(serial_size_t width,
                serial_size_t height,
                serial_size_t depth,
                serial_size_t n_fmaps) {
    shape_.width_  = width;
    shape_.height_ = height;
    shape_.depth_  = depth;
    n_fmaps_       = n_fmaps;
    data_.reshape({size()});
    grad_.reshape({grad_.shape()[0], size()});
  }

  bool is_trainable() { return trainable_; }

  void set_trainable() { trainable_ = true; }

  void freeze_trainable() { trainable_ = false; }

  Tensor<float_t> *data() { return &data_; }

  const Tensor<float_t> *data() const { return &data_; }

  void set_data(const Tensor<float_t> &data) { data_ = data; }

  void set_data(const Tensor<float_t> *data) { data_ = *data; }

  Tensor<float_t> *grad() { return &grad_; }

  const Tensor<float_t> *grad() const { return &grad_; }

  void set_grad(const Tensor<float_t> &grad) { grad_ = grad; }

  void set_grad(const Tensor<float_t> *grad) { grad_ = *grad; }

  void resize_grad(size_t sample_count) {
    grad_.reshape({sample_count, size()});
  }

  void merge_grads(Tensor<float_t> *dst) {
    const auto &grad_head = grad_[0];
    size_t sz             = grad_head.size();
    dst->reshape({sz});
    //    float_t *pdst = &(*dst)[0];
    // dst = grad_[0]
    std::copy(grad_head.host_begin(), grad_head.host_end(), dst->host_begin());
    // @todo consider adding parallelism
    for (size_t sample = 1, sample_count = grad_.shape()[0];
         sample < sample_count; ++sample) {
      // dst += grad_[sample]
      vectorize::reduce<float_t>(&grad_.host_at(sample, 0), sz,
                                 &dst->host_at(0));
    }
  }

  void clear_grads() { grad_.fill(float_t{0}); }

  float_t *data_at(size_t i) { return &data_.host_at(i); }

  float_t *grad_at(size_t sample, size_t i) {
    return &grad_.host_at(sample, i);
  }

 private:
  parameter_type type_;
  shape3d shape_;
  size_t n_fmaps_;
  bool trainable_;

  Tensor<float_t> data_;
  Tensor<float_t> grad_;
};
}  // namespace tiny_dnn
