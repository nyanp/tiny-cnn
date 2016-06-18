/*
    Copyright (c) 2013, Taiga Nomi
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
#include "tiny_cnn/util/util.h"
#include <unordered_map>

namespace tiny_cnn {

/**
 * base class of optimizer
 * usesHessian : true if an optimizer uses hessian (2nd order derivative of loss function)
 **/
struct optimizer {
    optimizer() = default;
    optimizer(const optimizer &) = default;
#ifndef CNN_DEFAULT_MOVE_CONSTRUCTOR_UNAVAILABLE
    optimizer(optimizer &&) = default;
#endif
    optimizer &operator =(const optimizer &) = default;
#ifndef CNN_DEFAULT_ASSIGNMENT_OPERATOR_UNAVAILABLE
    optimizer &operator =(optimizer &&) = default;
#endif
    virtual ~optimizer() = default;
    virtual void update(const vec_t& dW, vec_t &W) = 0;
    virtual void reset() {} // override to implement pre-learning action
};

// helper class to hold N values for each weight
template <int N>
struct stateful_optimizer : public optimizer {
    void reset() override {
        for (auto& e : E_) e.clear();
    }

protected:
    template <int Index>
    vec_t& get(const vec_t& key) {
        static_assert(Index < N, "index out of range");
        if (E_[Index][&key].empty())
            E_[Index][&key].resize(key.size(), float_t());
        return E_[Index][&key];
    }
    std::unordered_map<const vec_t*, vec_t> E_[N];
};

#if defined(CNN_USE_AVX)
inline __m256 madd(__m256 a, __m256 b, __m256 c) { return _mm256_add_ps(_mm256_mul_ps(a, b), c); }
inline __m128 madd(__m128 a, __m128 b, __m128 c) { return _mm_add_ps(_mm_mul_ps(a, b), c); }
inline __m128 madd_ss(__m128 a, __m128 b, __m128 c) { return _mm_add_ss(_mm_mul_ss(a, b), c); }

inline __m256d madd(__m256d a, __m256d b, __m256d c) { return _mm256_add_pd(_mm256_mul_pd(a, b), c); }
inline __m128d madd(__m128d a, __m128d b, __m128d c) { return _mm_add_pd(_mm_mul_pd(a, b), c); }
inline __m128d madd_sd(__m128d a, __m128d b, __m128d c) { return _mm_add_sd(_mm_mul_sd(a, b), c); }
#endif

/**
 * adaptive gradient method
 *
 * J Duchi, E Hazan and Y Singer,
 * Adaptive subgradient methods for online learning and stochastic optimization
 * The Journal of Machine Learning Research, pages 2121-2159, 2011.
 **/
struct adagrad : public stateful_optimizer<1> {
    adagrad() : alpha(float_t(0.01)), eps(float_t(1e-8)) {}

	inline void update_impl(size_t begin, size_t end, fvec_t& g, const fvec_t& dW, fvec_t &W)
	{
#ifdef CNN_USE_AVX
		size_t sz = end - begin;
		if (begin & 7) {
			size_t headLen = 8 - (begin & 7);
			size_t headEnd = std::min(begin+headLen, end);
			for (size_t i=begin; i<headEnd; ++i) {
				g[i] += dW[i] * dW[i];
				W[i] -= float(alpha * dW[i] / (std::sqrt(g[i]) + eps));
			}
			begin = headEnd;
			if (begin == end) {
				return;
			}
		}
		sz = end - begin;
		float* pg = &g[begin];
		const float* pdW = &dW[begin];
		float* pW = &W[begin];

		__m256 yalpha = _mm256_set1_ps(float(alpha));

		size_t nblocks = sz >> 4;
		for (cnn_size_t i=0; i<nblocks; ++i) {
			__m256 yg0 = _mm256_load_ps(pg);
			__m256 yg1 = _mm256_load_ps(pg+8);
			__m256 ydw0 = _mm256_load_ps(pdW);
			__m256 ydw1 = _mm256_load_ps(pdW+8);
			__m256 yW0 = _mm256_load_ps(pW);
			__m256 yW1 = _mm256_load_ps(pW+8);

			yg0 = madd(ydw0, ydw0, yg0);
			yg1 = madd(ydw1, ydw1, yg1);
			yW0 = _mm256_sub_ps(yW0, _mm256_mul_ps(yalpha, _mm256_mul_ps(ydw0, _mm256_rsqrt_ps(yg0))));
			yW1 = _mm256_sub_ps(yW1, _mm256_mul_ps(yalpha, _mm256_mul_ps(ydw1, _mm256_rsqrt_ps(yg1))));
			_mm256_store_ps(pg, yg0);
			_mm256_store_ps(pg+8, yg1);
			_mm256_store_ps(pW, yW0);
			_mm256_store_ps(pW+8, yW1);

			pg += 16;
			pdW += 16;
			pW += 16;
		}
		for (cnn_size_t i=begin+(nblocks<<4); i<end; ++i) {
			g[i] += dW[i] * dW[i];
			W[i] -= float(alpha * dW[i] / (std::sqrt(g[i]) + eps));
		}
#else
		for (size_t i=begin; i<end; ++i) {
			g[i] += dW[i] * dW[i];
			W[i] -= alpha * dW[i] / (std::sqrt(g[i]) + eps);
		}
#endif
	}

	inline void update_impl(size_t begin, size_t end, dvec_t& g, const dvec_t& dW, dvec_t &W)
	{
#if defined(CNN_USE_AVX)
		size_t sz = end - begin;
		if (begin & 3) {
			size_t headLen = 4 - (begin & 3);
			size_t headEnd = std::min(begin+headLen, end);
			for (size_t i=begin; i<headEnd; ++i) {
				g[i] += dW[i] * dW[i];
				W[i] -= alpha * dW[i] / (std::sqrt(g[i]) + eps);
			}
			begin = headEnd;
			if (begin == end) {
				return;
			}
		}
		sz = end - begin;
		double* pg = &g[begin];
		const double* pdW = &dW[begin];
		double* pW = &W[begin];

		__m256d yalpha = _mm256_set1_pd(alpha);

		size_t nblocks = sz >> 3;
		for (cnn_size_t i=0; i<nblocks; ++i) {
			__m256d yg0 = _mm256_load_pd(pg);
			__m256d yg1 = _mm256_load_pd(pg+4);
			__m256d ydw0 = _mm256_load_pd(pdW);
			__m256d ydw1 = _mm256_load_pd(pdW+4);
			__m256d yW0 = _mm256_load_pd(pW);
			__m256d yW1 = _mm256_load_pd(pW+4);

			yg0 = madd(ydw0, ydw0, yg0);
			yg1 = madd(ydw1, ydw1, yg1);
#if 1
			__m128 rsqrt_of_xg0 = _mm_rsqrt_ps(_mm256_cvtpd_ps(yg0));
			__m128 rsqrt_of_xg1 = _mm_rsqrt_ps(_mm256_cvtpd_ps(yg1));
			yW0 = _mm256_sub_pd(yW0, _mm256_mul_pd(yalpha, _mm256_mul_pd(ydw0, _mm256_cvtps_pd(rsqrt_of_xg0))));
			yW1 = _mm256_sub_pd(yW1, _mm256_mul_pd(yalpha, _mm256_mul_pd(ydw1, _mm256_cvtps_pd(rsqrt_of_xg1))));
#else
			yW0 = _mm256_sub_pd(yW0, _mm256_mul_pd(yalpha, _mm256_div_pd(ydw0, _mm256_sqrt_pd(yg0))));
			yW1 = _mm256_sub_pd(yW1, _mm256_mul_pd(yalpha, _mm256_div_pd(ydw1, _mm256_sqrt_pd(yg1))));
#endif
			_mm256_store_pd(pg, yg0);
			_mm256_store_pd(pg+4, yg1);
			_mm256_store_pd(pW, yW0);
			_mm256_store_pd(pW+4, yW1);

			pg += 8;
			pdW += 8;
			pW += 8;
		}
		for (cnn_size_t i=begin+(nblocks<<3); i<end; ++i) {
			g[i] += dW[i] * dW[i];
			W[i] -= alpha * dW[i] / (std::sqrt(g[i]) + eps);
		}
#else
		for (size_t i=begin; i<end; ++i) {
			g[i] += dW[i] * dW[i];
			W[i] -= alpha * dW[i] / (std::sqrt(g[i]) + eps);
		}
#endif
	}

    void update(const vec_t& dW, vec_t &W) {
        vec_t& g = get<0>(W);
#if 0
        for_i(static_cast<int>(W.size()), [&](int i) {
            g[i] += dW[i] * dW[i];
            W[i] -= alpha * dW[i] / (std::sqrt(g[i]) + eps);
        });
#else
		// compiler may be able to perform vectorization in this way
		int sz = static_cast<int>(W.size());
		int sz_per_thread = sz / std::thread::hardware_concurrency();
		if (sz < 64) {
			update_impl(0, sz, g, dW, W);
		}else {
			for_(true, 0, sz,
				[&](const blocked_range& r) {
					update_impl(r.begin(), r.end(), g, dW, W);
				}
			);
		}
#endif
	}

    float_t alpha; // learning rate
private:
    float_t eps;
};

/**
 * RMSprop
 *
 * T Tieleman, and G E Hinton,
 * Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning (2012)
 **/
struct RMSprop : public stateful_optimizer<1> {
    RMSprop() : alpha(float_t(0.0001)), mu(float_t(0.99)), eps(float_t(1e-8)) {}

    void update(const vec_t& dW, vec_t& W) {
        vec_t& g = get<0>(W);

        for_i(static_cast<int>(W.size()), [&](int i)
        {
            g[i] = mu * g[i] + (1 - mu) * dW[i] * dW[i];
            W[i] -= alpha * dW[i] / std::sqrt(g[i] + eps);
        });
    }

    float_t alpha; // learning rate
    float_t mu; // decay term
private:
    float_t eps; // constant value to avoid zero-division
};


/**
 * @brief [a new optimizer (2015)]
 * @details [see Adam: A Method for Stochastic Optimization (Algorithm 1)
 *               http://arxiv.org/abs/1412.6980]
 * 
 */
struct adam : public stateful_optimizer<2> {
    adam() : alpha(float_t(0.001)), b1(float_t(0.9)), b2(float_t(0.999)), b1_t(float_t(0.9)), b2_t(float_t(0.999)), eps(float_t(1e-8)) {}

    void update(const vec_t& dW, vec_t& W) {
        vec_t& mt = get<0>(W);
        vec_t& vt = get<1>(W);

        b1_t*=b1;b2_t*=b2;

        for_i(static_cast<int>(W.size()), [&](int i){
            mt[i] = b1 * mt[i] + (float_t(1) - b1) * dW[i];
            vt[i] = b2 * vt[i] + (float_t(1) - b2) * dW[i] * dW[i];

            W[i] -= alpha * ( mt[i]/(float_t(1) -b1_t) ) / std::sqrt( (vt[i]/(float_t(1)-b2_t)) + eps);
        });
    }

    float_t alpha; // learning rate
    float_t b1; // decay term
    float_t b2; // decay term
    float_t b1_t; // decay term power t
    float_t b2_t; // decay term power t   
private:
    float_t eps; // constant value to avoid zero-division
};



/**
 * SGD without momentum
 *
 * slightly faster than tiny_cnn::momentum
 **/
struct gradient_descent : public optimizer {
    gradient_descent() : alpha(float_t(0.01)), lambda(float_t(0)) {}

    void update(const vec_t& dW, vec_t& W) {
        for_i(static_cast<int>(W.size()), [&](int i){
            W[i] = W[i] - alpha * (dW[i] + lambda * W[i]);
        });
    }

    float_t alpha; // learning rate
    float_t lambda; // weight decay
};

/**
 * SGD with momentum
 *
 * B T Polyak,
 * Some methods of speeding up the convergence of iteration methods
 * USSR Computational Mathematics and Mathematical Physics, 4(5):1-17, 1964.
 **/
struct momentum : public stateful_optimizer<1> {
public:
    momentum() : alpha(float_t(0.01)), lambda(float_t(0)), mu(float_t(0.9)) {}

    void update(const vec_t& dW, vec_t& W) {
        vec_t& dWprev = get<0>(W);

        for_i(static_cast<int>(W.size()), [&](int i){
            float_t V = mu * dWprev[i] - alpha * (dW[i] + W[i] * lambda);
            W[i]      += V;
            dWprev[i] =  V;
        });
    }

    float_t alpha; // learning rate
    float_t lambda; // weight decay
    float_t mu; // momentum
};

} // namespace tiny_cnn
