/*
    Copyright (c) 2016, Goran Rauker
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

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
   FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
   THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include "cereal/archives/portable_binary.hpp"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

namespace tiny_dnn {
namespace models {

class prediction {
 public:
  prediction(const std::string &label, float confidence)
    : confidence(confidence), label(label) {}
  float confidence = -1;
  std::string label;
};

class archive {
 public:
  archive(const std::string &path_to_archive_file) {
    std::ifstream ifs(path_to_archive_file, std::ios::binary | std::ios::in);
    cereal::BinaryInputArchive bi(ifs);

    try {
      this->net.from_archive(bi);

      std::int32_t w, h, depth;
      image_type type;
      std::vector<float> data;

      bi(cereal::make_nvp("width", w), cereal::make_nvp("height", h),
         cereal::make_nvp("depth", depth), cereal::make_nvp("type", type));

      bi(cereal::make_nvp("data", data));

      this->mean = image<float>(shape3d(w, h, depth), type);
      this->mean.from_rgb(data.begin(), data.end());

      bi(cereal::make_nvp("labels", this->labels));

    } catch (const std::exception &e) {
      // TODO what should we do?
      std::cout << "archive::archive() : exception - " << e.what() << std::endl;
    }
  }

  archive(const network<sequential> &net,
          const image<float> &mean,
          const std::vector<std::string> &labels) {
    this->net    = net;
    this->mean   = mean;
    this->labels = labels;
  }

  std::vector<prediction> predict(const image<float> &image,
                                  int top_n_results = 5) {
    vec_t vec;
    preprocess(image, &vec);

    auto result = net.predict(vec);

    std::vector<prediction> predictions;
    std::vector<tiny_dnn::float_t> sorted(result.begin(), result.end());

    size_t top_n = top_n_results;
    std::partial_sort(sorted.begin(), sorted.begin() + top_n, sorted.end(),
                      std::greater<tiny_dnn::float_t>());

    for (size_t i = 0; i < top_n; i++) {
      size_t idx =
        distance(result.begin(), find(result.begin(), result.end(), sorted[i]));
      predictions.push_back(prediction(labels[idx], sorted[i]));
    }

    return predictions;
  }

  void save(const std::string &path_to_archive_file) const {
    std::ofstream ofs(path_to_archive_file, std::ios::binary | std::ios::out);
    cereal::BinaryOutputArchive bo(ofs);
    net.to_archive(bo);

    bo(cereal::make_nvp("width", (std::int32_t)mean.width()),
       cereal::make_nvp("height", (std::int32_t)mean.height()),
       cereal::make_nvp("depth", (std::int32_t)mean.depth()),
       cereal::make_nvp("type", mean.type()),
       cereal::make_nvp("data", mean.to_rgb<float>()),
       cereal::make_nvp("labels", labels));
  }

 protected:
  void preprocess(const image<float> &img, vec_t *dst) {
    image<float> resized = resize_image(img, net[0]->in_data_shape()[0].width_,
                                        net[0]->in_data_shape()[0].height_);

    image<> resized_uint8(resized);

    if (!mean.empty()) {
      image<float> normalized = subtract_scalar(resized, mean);
      *dst                    = normalized.to_vec();
    } else {
      *dst = resized.to_vec();
    }
  }

  network<sequential> net;
  image<float> mean;
  std::vector<std::string> labels;
};

}  // namespace models
}
