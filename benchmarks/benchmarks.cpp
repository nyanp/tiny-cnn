/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "benchmark/benchmark.h"
#include "tiny_dnn/tiny_dnn.h"

#include "bm_fully_connected.h"

// todo (karandesai, Randl): refactor needed after xtensor integration
//#include "bm_global_avepool.h"

using namespace tiny_dnn::benchmarks;

BENCHMARK_MAIN();
