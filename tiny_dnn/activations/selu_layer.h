/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "tiny_dnn/activations/activation_layer.h"
#include "tiny_dnn/layers/layer.h"

namespace tiny_dnn{

	class selu_layer : public activation_layer{
	public:
		
		selu_layer(const float_t lambda=1.05070,const float_t alpha=1.67326)
		: selu_layer(shape3d(0,0,0),lambda,alpha){}

		selu_layer(serial_size_t in_dim,const float_t lambda=1.05070,const float_t alpha=1.67326)
		: selu_layer(shape3d(in_dim,1,1),lambda,alpha){}

		selu_layer(serial_size_t in_width,
				   serial_size_t in_height,
				   serial_size_t in_channels,
				   const float_t lambda=1.05070,
				   const float_t alpha=1.67326)
		: selu_layer(shape3d(in_width,in_height,in_channels),lambda,alpha){}

		selu_layer(const shape3d &in_shape,const float_t lambda=1.05070,const float_t alpha=1.67326)
		: activation_layer(in_shape),lambda_(lambda),alpha_(alpha){}

		selu_layer(const layer &prev_layer,const float_t lambda=1.05070,const float_t alpha=1.67326)
		: activation_layer(prev_layer),lambda_(lambda),alpha_(alpha){}


		std::string layer_type() const override {
			return "selu-activation";
		}

		float_t lambda_value(){return lambda_;}

		float_t alpha_value(){return alpha_;}

		void forward_activation(const vec_t &x,vec_t &y) override{
			for (serial_size_t j = 0; j < x.size(); j++) {
				y[j] = lambda_ * (x[j] > float_t(0) ? x[j] : alpha_*(std::exp(x[j])-float_t(1)) )
			}
		}

		void backward_activation(const vec_t &x,
								 const vec_t &y,
								 vec_t &dx,
								 const vec_t &dy) override{
			// dx = dy * (gradient of selu)
			for(serial_size_t j=0; j<x.size();j++){
				dx[j] = dy[j] * (y[j] > float_t(0) ? lambda_ : (y[j] + lambda_*alpha_))

			}
		}

		std::pair<float_t, float_t> scale() const override {
    		return std::make_pair(float_t(0.1), float_t(0.9));
  		}

  		float_t lambda_;
  		float_t alpha_;

  		friend struct serialization_buddy;


	};
	
} //namespace tiny_dnn