#pragma once

#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

class activation_layer : public layer {
 public:
  /**
   * @param in_shape [in] shape of input tensor
   */
  activation_layer(const shape3d &in_shape)
    : layer({vector_type::data}, {vector_type::data}), in_shape_(in_shape) {}

  std::vector<shape3d> in_shape() const override { return {in_shape_}; }

  std::vector<shape3d> out_shape() const override { return {in_shape_}; }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) {
    const tensor_t &x = *in_data[0];
    tensor_t &y       = *out_data[0];

    for (serial_size_t i = 0; i < x.size(); i++) {
      forward_activation(x[i], y[i]);
    }
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    tensor_t &dx       = *in_grad[0];
    const tensor_t &dy = *out_grad[0];
    const tensor_t &x  = *in_data[0];
    const tensor_t &y  = *out_data[0];

    for (serial_size_t i = 0; i < x.size(); i++) {
      backward_activation(x[i], y[i], dx[i], dy[i]);
    }
  }

  virtual std::string layer_type() const = 0;

  virtual void forward_activation(const vec_t &x, vec_t &y) = 0;

  virtual void backward_activation(const vec_t x,
                                   const vec_t &y,
                                   vec_t &dx,
                                   const vec_t &dy) = 0;

 private:
  shape3d in_shape_;
};
}  // namespace tiny_dnn
