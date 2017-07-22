/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/device.fwd.h"
#include "tiny_dnn/core/params/conv_params.h"

namespace tiny_dnn {
namespace core {

class OpKernel;  // delared below

class OpKernelConstruction {
 public:
  explicit OpKernelConstruction() {}
  explicit OpKernelConstruction(Device *device, Params *params)
    : device_(device), params_(params) {}

  // Returns the device raw pointer
  Device *device() const { return device_; }

  // Returns the device raw pointer
  Params *params() const { return params_; }

 private:
  Device *device_ = nullptr;
  Params *params_ = nullptr;
};

class OpKernelContext {
 public:
  struct OpParams {
    // the op kernel being computed.
    OpKernel *op_kernel_ptr = nullptr;

    // the device on which the kernel is running.
    Device *device_ptr = nullptr;

    // the layer on which kernel is runnning
    layer *layer_ptr_ = nullptr;

    // the operation params
    Params *params_ptr_ = nullptr;

    // parallelize operation
    bool parallelize = false;

    backend_t backend_type = default_backend();
  };

  OpKernelContext()
    : in_data_(nullptr),
      out_data_(nullptr),
      out_grad_(nullptr),
      in_grad_(nullptr) {
    op_params_ = std::unique_ptr<OpParams>(new OpParams());
  }

  void set_in_out(const std::vector<tensor_t *> &in_data,
                  std::vector<tensor_t *> &out_data) {
    in_data_  = const_cast<std::vector<tensor_t *> *>(&in_data);
    out_data_ = &out_data;
  }

  void set_in_out(const std::vector<tensor_t *> &in_data,
                  const std::vector<tensor_t *> &out_data,
                  std::vector<tensor_t *> &out_grad,
                  std::vector<tensor_t *> &in_grad) {
    in_data_  = const_cast<std::vector<tensor_t *> *>(&in_data);
    out_data_ = const_cast<std::vector<tensor_t *> *>(&out_data);
    out_grad_ = &out_grad;
    in_grad_  = &in_grad;
  }

  tensor_t &input(const int idx) { return *(*in_data_)[idx]; }
  const tensor_t &input(const int idx) const { return *(*in_data_)[idx]; }

  tensor_t &output(const int idx) { return *(*out_data_)[idx]; }
  const tensor_t &output(const int idx) const { return *(*out_data_)[idx]; }

  tensor_t &input_grad(const int idx) { return *(*in_grad_)[idx]; }
  const tensor_t &input_grad(const int idx) const { return *(*in_grad_)[idx]; }

  tensor_t &output_grad(const int idx) { return *(*out_grad_)[idx]; }
  const tensor_t &output_grad(const int idx) const {
    return *(*out_grad_)[idx];
  }

  void setParams(Params *params) { op_params_->params_ptr_ = params; }

  Params *params() const { return op_params_->params_ptr_; }

  void setParallelize(const bool parallelize) {
    op_params_->parallelize = parallelize;
  }

  bool parallelize() const { return op_params_->parallelize; }

  void setDevice(Device *device) { op_params_->device_ptr = device; }

  Device *device() const { return op_params_->device_ptr; }

  void setLayer(layer *layer) { op_params_->layer_ptr_ = layer; }

  layer *Layer() const { return op_params_->layer_ptr_; }

  backend_t backend_type() const { return op_params_->backend_type; }

  void setBackendType(const backend_t backend_type) { op_params_->backend_type = backend_type; }

 private:
  std::vector<tensor_t *> *in_data_;
  std::vector<tensor_t *> *out_data_;
  std::vector<tensor_t *> *out_grad_;
  std::vector<tensor_t *> *in_grad_;

  std::unique_ptr<OpParams> op_params_;
};

class OpKernel {
 public:
  explicit OpKernel() {}
  explicit OpKernel(const OpKernelConstruction &context)
    : device_(context.device()), params_(context.params()) {}

  virtual ~OpKernel() {}

  virtual void compute(OpKernelContext &context) = 0;

 protected:
  Device *device_ = nullptr;
  Params *params_ = nullptr;
};

}  // namespace core
}  // namespace tiny_dnn
