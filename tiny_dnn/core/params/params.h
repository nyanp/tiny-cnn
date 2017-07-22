/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

namespace tiny_dnn {
namespace core {

class conv_params;
class deconv_params;
class fully_params;
class maxpool_params;
class global_avepool_params;
class recurrent_cell_params;

/* Base class to model operation parameters */
class Params {
 public:
  Params() {}

  conv_params &conv();
  deconv_params &deconv();
  fully_params &fully();
  maxpool_params &maxpool();
  global_avepool_params &global_avepool();
  recurrent_cell_params &recurrent_cell();
};

}  // namespace core
}  // namespace tiny_dnn
