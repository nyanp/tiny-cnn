/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "tiny_dnn/core/kernels/conv2d_grad_op.h"
#include "tiny_dnn/core/kernels/conv2d_op.h"
#include "tiny_dnn/core/kernels/conv2d_op_libdnn.h"
#include "tiny_dnn/core/kernels/conv2d_op_opencl.h"

#include "tiny_dnn/util/util.h"

#ifdef DNN_USE_IMAGE_API
#include "tiny_dnn/util/image.h"
#endif  // DNN_USE_IMAGE_API

using namespace tiny_dnn::core;

namespace tiny_dnn {

/**
 * 2D convolution layer
 *
 * take input as two-dimensional *image* and applying filtering operation.
 **/
class convolutional_layer : public layer {
 public:
  /**
   * constructing convolutional layer
   *
   * @param in_width     [in] input image width
   * @param in_height    [in] input image height
   * @param window_size  [in] window(kernel) size of convolution
   * @param in_channels  [in] input image channels (grayscale=1, rgb=3)
   * @param out_channels [in] output image channels
   * @param padding      [in] rounding strategy
   *                          - valid: use valid pixels of input only.
   *```output-size = (in-width - window_width + 1) *
   *(in-height - window_height + 1) * out_channels```
   *                          - same: add zero-padding to keep same
   *width/height.
   *```output-size = in-width * in-height *
   *out_channels```
   * @param has_bias     [in] whether to add a bias vector to the filter
   *outputs
   * @param w_stride     [in] specify the horizontal interval at which to apply
   *the filters to the input
   * @param h_stride     [in] specify the vertical interval at which to apply
   *the
   *filters to the input
   * @param backend_type [in] specify backend engine you use
   **/
  convolutional_layer(size_t in_width,
                      size_t in_height,
                      size_t window_size,
                      size_t in_channels,
                      size_t out_channels,
                      padding pad_type       = padding::valid,
                      bool has_bias          = true,
                      size_t w_stride        = 1,
                      size_t h_stride        = 1,
                      backend_t backend_type = core::default_backend())
    : convolutional_layer(in_width,
                          in_height,
                          window_size,
                          window_size,
                          in_channels,
                          out_channels,
                          connection_table(),
                          pad_type,
                          has_bias,
                          w_stride,
                          h_stride,
                          backend_type) {}

  /**
   * constructing convolutional layer
   *
   * @param in_width      [in] input image width
   * @param in_height     [in] input image height
   * @param window_width  [in] window_width(kernel) size of convolution
   * @param window_height [in] window_height(kernel) size of convolution
   * @param in_channels   [in] input image channels (grayscale=1, rgb=3)
   * @param out_channels  [in] output image channels
   * @param padding       [in] rounding strategy
   *                           - valid: use valid pixels of input only.
   *```output-size = (in-width - window_width + 1) *
   *(in-height - window_height + 1) * out_channels```
   *                           - same: add zero-padding to keep same
   *width/height. ```output-size = in-width * in-height
   ** out_channels```
   * @param has_bias      [in] whether to add a bias vector to the filter
   *outputs
   * @param w_stride      [in] specify the horizontal interval at which to
   *apply
   *the filters to the input
   * @param h_stride      [in] specify the vertical interval at which to apply
   *the filters to the input
   * @param backend_type  [in] specify backend engine you use
   **/
  convolutional_layer(size_t in_width,
                      size_t in_height,
                      size_t window_width,
                      size_t window_height,
                      size_t in_channels,
                      size_t out_channels,
                      padding pad_type       = padding::valid,
                      bool has_bias          = true,
                      size_t w_stride        = 1,
                      size_t h_stride        = 1,
                      backend_t backend_type = core::default_backend())
    : convolutional_layer(in_width,
                          in_height,
                          window_width,
                          window_height,
                          in_channels,
                          out_channels,
                          connection_table(),
                          pad_type,
                          has_bias,
                          w_stride,
                          h_stride,
                          backend_type) {}

  /**
   * constructing convolutional layer
   *
   * @param in_width         [in] input image width
   * @param in_height        [in] input image height
   * @param window_size      [in] window(kernel) size of convolution
   * @param in_channels      [in] input image channels (grayscale=1, rgb=3)
   * @param out_channels     [in] output image channels
   * @param connection_table [in] definition of connections between in-channels
   *and out-channels
   * @param pad_type         [in] rounding strategy
   *                              - valid: use valid pixels of input only.
   *```output-size = (in-width - window_width + 1)
   ** (in-height - window_height + 1) * out_channels```
   *                              - same: add zero-padding to keep same
   *width/height. ```output-size = in-width *
   *in-height * out_channels```
   * @param has_bias         [in] whether to add a bias vector to the filter
   *outputs
   * @param w_stride         [in] specify the horizontal interval at which to
   *apply the filters to the input
   * @param h_stride         [in] specify the vertical interval at which to
   *apply
   *the filters to the input
   * @param backend_type [in] specify backend engine you use
   **/
  convolutional_layer(size_t in_width,
                      size_t in_height,
                      size_t window_size,
                      size_t in_channels,
                      size_t out_channels,
                      const connection_table &connection_table,
                      padding pad_type       = padding::valid,
                      bool has_bias          = true,
                      size_t w_stride        = 1,
                      size_t h_stride        = 1,
                      backend_t backend_type = core::default_backend())
    : convolutional_layer(in_width,
                          in_height,
                          window_size,
                          window_size,
                          in_channels,
                          out_channels,
                          connection_table,
                          pad_type,
                          has_bias,
                          w_stride,
                          h_stride,
                          backend_type) {}

  /**
  * constructing convolutional layer
  *
  * @param in_width         [in] input image width
  * @param in_height        [in] input image height
  * @param window_width     [in] window_width(kernel) size of convolution
  * @param window_height    [in] window_height(kernel) size of convolution
  * @param in_channels      [in] input image channels (grayscale=1, rgb=3)
  * @param out_channels     [in] output image channels
  * @param connection_table [in] definition of connections between in-channels
  *and out-channels
  * @param pad_type         [in] rounding strategy
  *                              - valid: use valid pixels of input only.
  *```output-size = (in-width - window_width + 1)
  ** (in-height - window_height + 1) * out_channels```
  *                              - same: add zero-padding to keep same
  *width/height. ```output-size = in-width *
  *in-height * out_channels```
  * @param has_bias         [in] whether to add a bias vector to the filter
  *outputs
  * @param w_stride         [in] specify the horizontal interval at which to
  *apply the filters to the input
  * @param h_stride         [in] specify the vertical interval at which to
  *apply
  *the filters to the input
  * @param backend_type [in] specify backend engine you use
  **/
  convolutional_layer(size_t in_width,
                      size_t in_height,
                      size_t window_width,
                      size_t window_height,
                      size_t in_channels,
                      size_t out_channels,
                      const connection_table &connection_table,
                      padding pad_type       = padding::valid,
                      bool has_bias          = true,
                      size_t w_stride        = 1,
                      size_t h_stride        = 1,
                      backend_t backend_type = core::default_backend())
    : layer(std_input_order(has_bias), {vector_type::data}) {
    conv_set_params(shape3d(in_width, in_height, in_channels), window_width,
                    window_height, out_channels, pad_type, has_bias, w_stride,
                    h_stride, connection_table);
    init_backend(backend_type);
    layer::set_backend_type(backend_type);
  }

  // move constructor
  convolutional_layer(convolutional_layer &&other)  // NOLINT
    : layer(std::move(other)),
      params_(std::move(other.params_)),
      padding_op_(std::move(other.padding_op_)),
      kernel_fwd_(std::move(other.kernel_fwd_)),
      kernel_back_(std::move(other.kernel_back_)),
      cws_(std::move(other.cws_)) {
    init_backend(std::move(other.backend_type()));
  }

  ///< number of incoming connections for each output unit
  size_t fan_in_size() const override {
    return params_.weight.width_ * params_.weight.height_ * params_.in.depth_;
  }

  ///< number of outgoing connections for each input unit
  size_t fan_out_size() const override {
    return (params_.weight.width_ / params_.w_stride) *
           (params_.weight.height_ / params_.h_stride) * params_.out.depth_;
  }

  /**
   * @param in_data      input vectors of this layer (data, weight, bias)
   * @param out_data     output vectors
   **/
  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    // apply padding to the input tensor
    padding_op_.copy_and_pad_input(*in_data[0], cws_.prev_out_padded_);

    fwd_in_data_.resize(in_data.size());
    std::copy(in_data.begin(), in_data.end(), fwd_in_data_.begin());
    fwd_in_data_[0] = in_data_padded(in_data);

    // forward convolutional op context
    fwd_ctx_.set_in_out(fwd_in_data_, out_data);
    fwd_ctx_.setParallelize(layer::parallelize());
    fwd_ctx_.setBackendType(layer::backend_type());

    // launch convolutional kernel
    kernel_fwd_->compute(fwd_ctx_);
  }

  /**
   * return delta of previous layer (delta=\frac{dE}{da}, a=wx in
   *fully-connected layer)
   * @param in_data      input vectors (same vectors as forward_propagation)
   * @param out_data     output vectors (same vectors as forward_propagation)
   * @param out_grad     gradient of output vectors (i-th vector correspond
   *with
   *out_data[i])
   * @param in_grad      gradient of input vectors (i-th vector correspond
   *with
   *in_data[i])
   **/
  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    bwd_in_data_.resize(in_data.size());
    std::copy(in_data.begin(), in_data.end(), bwd_in_data_.begin());
    bwd_in_data_[0] = in_data_padded(in_data);

    bwd_in_grad_.resize(in_grad.size());
    std::copy(in_grad.begin(), in_grad.end(), bwd_in_grad_.begin());
    if (params_.pad_type == padding::same) {
      bwd_in_grad_[0] = &cws_.prev_delta_padded_;
    }

    bwd_ctx_.set_in_out(bwd_in_data_, out_data, out_grad, bwd_in_grad_);
    bwd_ctx_.setParams(&params_);
    bwd_ctx_.setParallelize(layer::parallelize());
    bwd_ctx_.setBackendType(layer::backend_type());

    // launch convolutional kernel
    kernel_back_->compute(bwd_ctx_);

    // unpad deltas
    padding_op_.copy_and_unpad_delta(cws_.prev_delta_padded_, *in_grad[0]);
  }

  void set_sample_count(size_t sample_count) override {
    layer::set_sample_count(sample_count);
    cws_.prev_delta_padded_.resize(sample_count,
                                   vec_t(params_.in_padded.size(), float_t(0)));
  }

  std::vector<index3d<size_t>> in_shape() const override {
    if (params_.has_bias) {
      return {params_.in, params_.weight,
              index3d<size_t>(1, 1, params_.out.depth_)};
    } else {
      return {params_.in, params_.weight};
    }
  }

  std::vector<index3d<size_t>> out_shape() const override {
    return {params_.out};
  }

  std::string layer_type() const override { return std::string("conv"); }

  // TODO(edgar): check this
  std::string kernel_file() const override {
    return std::string(
      "../tiny_cnn/core/kernels/cl_kernels/conv_layer_spatial.cl");
  }

  // TODO(edgar): is it really needed?
  std::string kernel_header() const override {
    std::stringstream ss;
    ss << "#define MULTI\n";
    ss << "#define KERNEL_H " << params_.weight.height_ << "\n";
    ss << "#define KERNEL_W " << params_.weight.width_ << "\n";
    ss << "#define CHANNELS " << params_.weight.depth_ << "\n";
    ss << "#define STRIDE_H " << params_.h_stride << "\n";
    ss << "#define STRIDE_W " << params_.w_stride << "\n";
    ss << "#define APPLY_BIAS " << params_.has_bias << "\n";
    ss << "#define OUTPUT_Z " << params_.out.depth_ << "\n";
    // TODO(edgar): REVISE THIS
    ss << "#define ZPAR " << params_.out.depth_ << "\n";
    return ss.str();
  }

#ifdef DNN_USE_IMAGE_API
  image<> weight_to_image() const {
    image<> img;
    const size_t border_width = 1;
    const auto pitch          = params_.weight.width_ + border_width;
    const auto width          = params_.out.depth_ * pitch + border_width;
    const auto height         = params_.in.depth_ * pitch + border_width;
    const image<>::intensity_t bg_color = 255;
    const vec_t &W                      = *this->weights()[0];

    img.resize(width, height);
    img.fill(bg_color);

    auto minmax = std::minmax_element(W.begin(), W.end());

    for (size_t r = 0; r < params_.in.depth_; ++r) {
      for (size_t c = 0; c < params_.out.depth_; ++c) {
        if (!params_.tbl.is_connected(c, r)) continue;

        const auto top  = r * pitch + border_width;
        const auto left = c * pitch + border_width;

        size_t idx = 0;

        for (size_t y = 0; y < params_.weight.height_; ++y) {
          for (size_t x = 0; x < params_.weight.width_; ++x) {
            idx             = c * params_.in.depth_ + r;
            idx             = params_.weight.get_index(x, y, idx);
            const float_t w = W[idx];

            img.at(left + x, top + y) = static_cast<image<>::intensity_t>(
              rescale(w, *minmax.first, *minmax.second, 0, 255));
          }
        }
      }
    }
    return img;
  }
#endif  // DNN_USE_IMAGE_API

  friend struct serialization_buddy;

 private:
  tensor_t *in_data_padded(const std::vector<tensor_t *> &in) {
    return (params_.pad_type == padding::valid) ? in[0]
                                                : &cws_.prev_out_padded_;
  }

  void conv_set_params(const shape3d &in,
                       size_t w_width,
                       size_t w_height,
                       size_t outc,
                       padding ptype,
                       bool has_bias,
                       size_t w_stride,
                       size_t h_stride,
                       const connection_table &tbl = connection_table()) {
    params_.in = in;
    params_.in_padded =
      shape3d(in_length(in.width_, w_width, ptype),
              in_length(in.height_, w_height, ptype), in.depth_);
    params_.out =
      shape3d(conv_out_length(in.width_, w_width, w_stride, ptype),
              conv_out_length(in.height_, w_height, h_stride, ptype), outc);
    params_.weight   = shape3d(w_width, w_height, in.depth_ * outc);
    params_.has_bias = has_bias;
    params_.pad_type = ptype;
    params_.w_stride = w_stride;
    params_.h_stride = h_stride;
    params_.tbl      = tbl;

    // init padding buffer
    if (params_.pad_type == padding::same) {
      cws_.prev_delta_padded_.resize(
        1, vec_t(params_.in_padded.size(), float_t(0)));
    }

    // set parameters to padding operation
    padding_op_ = Conv2dPadding(params_);
  }

  size_t in_length(size_t in_length,
                   size_t window_size,
                   padding pad_type) const {
    return pad_type == padding::same ? (in_length + window_size - 1)
                                     : in_length;
  }

  static size_t conv_out_dim(size_t in_width,
                             size_t in_height,
                             size_t window_size,
                             size_t w_stride,
                             size_t h_stride,
                             padding pad_type) {
    return conv_out_length(in_width, window_size, w_stride, pad_type) *
           conv_out_length(in_height, window_size, h_stride, pad_type);
  }

  size_t conv_out_dim(size_t in_width,
                      size_t in_height,
                      size_t window_width,
                      size_t window_height,
                      size_t w_stride,
                      size_t h_stride,
                      padding pad_type) const {
    return conv_out_length(in_width, window_width, w_stride, pad_type) *
           conv_out_length(in_height, window_height, h_stride, pad_type);
  }

  void createOp() override { init_backend(layer::backend_type()); }

  void init_backend(const backend_t backend_type) {
    core::OpKernelConstruction ctx =
      core::OpKernelConstruction(layer::device(), &params_);

    if (backend_type == backend_t::internal ||
        backend_type == backend_t::nnpack || backend_type == backend_t::avx) {
      kernel_fwd_.reset(new Conv2dOp(ctx));
      kernel_back_.reset(new Conv2dGradOp(ctx));
      return;
    } else if (backend_type == backend_t::opencl) {
      throw nn_error("Not implemented engine: " + to_string(backend_type));
      /*kernel_fwd_.reset(new Conv2dOpenCLForwardOp(ctx));
      kernel_back_.reset(new Conv2dOpenCLBackwardOp(ctx));
      return;*/
    } else if (backend_type == backend_t::libdnn) {
      if (layer::device() == nullptr) return;
      kernel_fwd_.reset(new Conv2dLibDNNForwardOp(ctx));
      kernel_back_.reset(new Conv2dLibDNNBackwardOp(ctx));
      return;
    } else {
      throw nn_error("Not supported engine: " + to_string(backend_type));
    }
  }

 private:
  /* The convolution parameters */
  conv_params params_;

  /* Padding operation */
  Conv2dPadding padding_op_;

  /* forward op context */
  OpKernelContext fwd_ctx_;

  /* backward op context */
  OpKernelContext bwd_ctx_;

  /* Forward and backward ops */
  std::shared_ptr<core::OpKernel> kernel_fwd_;
  std::shared_ptr<core::OpKernel> kernel_back_;

  std::vector<tensor_t *> fwd_in_data_;
  std::vector<tensor_t *> bwd_in_data_;
  std::vector<tensor_t *> bwd_in_grad_;

  /* Buffer to store padded data */
  struct conv_layer_worker_specific_storage {
    tensor_t prev_out_padded_;
    tensor_t prev_delta_padded_;
  } cws_;
};

}  // namespace tiny_dnn
