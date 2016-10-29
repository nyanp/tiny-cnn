/*
    Copyright (c) 2015, Taiga Nomi
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <string>
#include <vector>
#include <algorithm>

#include "tiny_dnn/core/backend_tiny.h"
#include "tiny_dnn/core/backend_nnp.h"
#include "tiny_dnn/core/backend_dnn.h"
#ifdef CNN_USE_AVX
#include "tiny_dnn/core/backend_avx.h"
#endif

#include "tiny_dnn/util/util.h"
#include "tiny_dnn/util/image.h"
#include "tiny_dnn/activations/activation_function.h"

namespace tiny_dnn {

/**
 * applies max-pooing operaton to the spatial data
 **/
template <typename Activation = activation::identity>
class max_pooling_layer : public feedforward_layer<Activation> {
 public:
    CNN_USE_LAYER_MEMBERS;
    typedef feedforward_layer<Activation> Base;

    /**
     * @param in_width     [in] width of input image
     * @param in_height    [in] height of input image
     * @param in_channels  [in] the number of input image channels(depth)
     * @param pooling_size [in] factor by which to downscale
     **/
    max_pooling_layer(cnn_size_t     in_width,
                      cnn_size_t     in_height,
                      cnn_size_t     in_channels,
                      cnn_size_t     pooling_size,
                      backend_t      backend_type = core::default_engine(),
                      backend_params b_params = backend_params())
            : max_pooling_layer(in_width, in_height, in_channels, pooling_size, pooling_size, backend_type, b_params) {
    }

    max_pooling_layer(const shape3d& in_shape,
                      cnn_size_t     pooling_size,
                      cnn_size_t     stride,
                      backend_t      backend_type = core::default_engine(),
                      backend_params b_params = backend_params())
        : max_pooling_layer(in_shape.width_, in_shape.height_, in_shape.depth_, pooling_size, stride, backend_type, b_params) {
    }

    max_pooling_layer(cnn_size_t     in_width,
                      cnn_size_t     in_height,
                      cnn_size_t     in_channels,
                      cnn_size_t     pooling_size,
                      cnn_size_t     stride,
                      backend_t      backend_type = core::default_engine(),
                      backend_params b_params = backend_params())
        : max_pooling_layer(in_width, in_height, in_channels, pooling_size, pooling_size, stride, stride, padding::valid, backend_type, b_params) {
    }

    /**
     * @param in_width     [in] width of input image
     * @param in_height    [in] height of input image
     * @param in_channels  [in] the number of input image channels(depth)
     * @param pooling_size [in] factor by which to downscale
     * @param stride       [in] interval at which to apply the filters to the input
    **/
    max_pooling_layer(cnn_size_t     in_width,
                      cnn_size_t     in_height,
                      cnn_size_t     in_channels,
                      cnn_size_t     pooling_size_x,
                      cnn_size_t     pooling_size_y,
                      cnn_size_t     stride_x,
                      cnn_size_t     stride_y,
                      core::padding  pad_type,
                      backend_t      backend_type = backend_t::tiny_dnn,
                      backend_params b_params = backend_params())
            : Base({ vector_type::data }) {
        set_maxpool_params(
            shape3d(in_width, in_height, in_channels),
            shape3d(conv_out_length(in_width, pooling_size_x, stride_x, pad_type),
                    conv_out_length(in_height, pooling_size_y, stride_y, pad_type),
                    in_channels),
            pooling_size_x, pooling_size_y, stride_x, stride_y, pad_type);

        init_connection();
        init_backend(backend_type);
        Base::set_backend_type(backend_type);
    }

    // move constructor
    max_pooling_layer(max_pooling_layer&& other)  // NOLINT
            : Base(std::move(other))
            , params_(std::move(other.params_))
            , out2in_(std::move(other.out2in_))
            , in2out_(std::move(other.in2out_))
            , max_pooling_layer_worker_storage_(
                std::move(other.max_pooling_layer_worker_storage_)) {
        init_connection();
        init_backend(std::move(Base::backend_type()));
    }

    size_t fan_in_size() const override {
        return out2in_[0].size();
    }

    size_t fan_out_size() const override {
        return 1;
    }

    void forward_propagation(const std::vector<tensor_t*>& in_data,
                             std::vector<tensor_t*>&       out_data) override {
        // launch maxpool kernel
        Base::backend_->maxpool(in_data, out_data);

        // activations
        this->forward_activation(*out_data[0], *out_data[1]);
    }

    void back_propagation(const std::vector<tensor_t*>& in_data,
                          const std::vector<tensor_t*>& out_data,
                          std::vector<tensor_t*>&       out_grad,
                          std::vector<tensor_t*>&       in_grad) override {
        // launch maxpool kernel
        Base::backend_->maxpool(in_data, out_data, out_grad, in_grad);
    }

    std::vector<index3d<cnn_size_t>>
    in_shape() const override { return { params_.in }; }

    std::vector<index3d<cnn_size_t>>
    out_shape() const override { return { params_.out, params_.out }; }

    std::string layer_type() const override {
        return std::string("max-pool");
    }

    std::string kernel_file() const override {
        return std::string("../tiny_cnn/core/kernels/cl_kernels/pooling.cl");
    }

    size_t pool_size() const { return params_.pool_size_; }

    void set_sample_count(cnn_size_t sample_count) override {
        Base::set_sample_count(sample_count);
        max_pooling_layer_worker_storage_.out2inmax_.resize(sample_count, std::vector<cnn_size_t>(params_.out.size()));
    }


    template <class Archive>
    static void load_and_construct(Archive & ar, cereal::construct<max_pooling_layer> & construct) {
        shape3d in;
        cnn_size_t stride_x, stride_y, pool_size_x, pool_size_y;
        core::padding pad_type;

        ar(cereal::make_nvp("in_size", in),
           cereal::make_nvp("pool_size_x", pool_size_x),
           cereal::make_nvp("pool_size_y", pool_size_y),
           cereal::make_nvp("stride_x", stride_x),
           cereal::make_nvp("stride_y", stride_y),
            cereal::make_nvp("pad_type", pad_type));
        construct(in.width_, in.height_, in.depth_, pool_size_x, pool_size_y, stride_x, stride_y, pad_type);
    }

    template <class Archive>
    void serialize(Archive & ar) {
        layer::serialize_prolog(ar);
        ar(cereal::make_nvp("in_size", params_.in),
            cereal::make_nvp("pool_size_x", params_.pool_size_x),
            cereal::make_nvp("pool_size_y", params_.pool_size_y),
            cereal::make_nvp("stride_x", params_.stride_x),
            cereal::make_nvp("stride_y", params_.stride_y),
            cereal::make_nvp("pad_type", params_.pad_type));
    }

private:
    maxpool_params params_;

    /* mapping out => in (1:N) */
    std::vector<std::vector<cnn_size_t> > out2in_;
    /* mapping in => out (N:1) */
    std::vector<cnn_size_t> in2out_;

    max_pooling_layer_worker_specific_storage
    max_pooling_layer_worker_storage_;

    void connect_kernel(cnn_size_t pooling_size_x,
                        cnn_size_t pooling_size_y,
                        cnn_size_t outx,
                        cnn_size_t outy,
                        cnn_size_t c) {
        cnn_size_t dxmax = static_cast<cnn_size_t>(
            std::min(static_cast<size_t>(pooling_size_x),
                     params_.in.width_ - outx * params_.stride_x));

        cnn_size_t dymax = static_cast<cnn_size_t>(
            std::min(static_cast<size_t>(pooling_size_y),
                     params_.in.height_ - outy * params_.stride_y));

        for (cnn_size_t dy = 0; dy < dymax; dy++) {
            for (cnn_size_t dx = 0; dx < dxmax; dx++) {
                cnn_size_t in_index = params_.in.get_index(
                    static_cast<cnn_size_t>(outx * params_.stride_x + dx),
                    static_cast<cnn_size_t>(outy * params_.stride_y + dy), c);
                cnn_size_t out_index = params_.out.get_index(outx, outy, c);

                if (in_index >= in2out_.size()) {
                    throw nn_error("index overflow");
                }
                if (out_index >= out2in_.size()) {
                    throw nn_error("index overflow");
                }
                in2out_[in_index] = out_index;
                out2in_[out_index].push_back(in_index);
            }
        }
    }

    void init_connection() {
        in2out_.resize(params_.in.size());
        out2in_.resize(params_.out.size());

        for (cnn_size_t c = 0; c < params_.in.depth_; ++c) {
            for (cnn_size_t y = 0; y < params_.out.height_; ++y) {
                for (cnn_size_t x = 0; x < params_.out.width_; ++x) {
                    connect_kernel(params_.pool_size_x,
                                   params_.pool_size_y,
                                   x, y, c);
                }
            }
        }
    }

    void init_backend(backend_t backend_type) {
        std::shared_ptr<core::backend> backend = nullptr;

        // allocate new backend
        if (backend_type == backend_t::tiny_dnn) {
            backend = std::make_shared<core::tiny_backend>(
                &out2in_,
                &in2out_,
                [this](const tensor_t& p_delta,
                       const tensor_t& out, tensor_t& c_delta) {
                    return Base::backward_activation(p_delta, out, c_delta);
                },
                &max_pooling_layer_worker_storage_);
        } else if (backend_type == backend_t::nnpack) {
            backend = std::make_shared<core::nnp_backend>(&params_);
        } else if (backend_type == backend_t::libdnn) {
            backend = std::make_shared<core::dnn_backend>();
#ifdef CNN_USE_AVX
        } else if (backend_type == backend_t::avx) {
            backend = std::make_shared<core::avx_backend>(
                &out2in_,
                &in2out_,
                [this](const tensor_t& p_delta,
                       const tensor_t& out, tensor_t& c_delta) {
                    return Base::backward_activation(p_delta, out, c_delta);
                },
                &max_pooling_layer_worker_storage_);
#endif
        } else {
            throw nn_error("Not supported backend type.");
        }

        if (backend) {
            Base::set_backend(backend);
            Base::backend_->set_layer(this);
        } else {
            throw nn_error("Could not allocate the backend.");
        }
    }

    void set_maxpool_params(const shape3d& in,
                            const shape3d& out,
                            cnn_size_t pooling_size_x,
                            cnn_size_t pooling_size_y,
                            cnn_size_t stride_x,
                            cnn_size_t stride_y,
                            core::padding pad_type) {
        params_.in        = in;
        params_.out       = out;
        params_.pool_size_x = pooling_size_x;
        params_.pool_size_y = pooling_size_y;
        params_.stride_x    = stride_x;
        params_.stride_y    = stride_y;
        params_.pad_type    = pad_type;
    }
};

}  // namespace tiny_dnn

