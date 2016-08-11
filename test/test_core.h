/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
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
#include "picotest/picotest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;

namespace tiny_dnn {

TEST(core, device) {
    // CPU and GPU devices are instantiated

    Device my_cpu_device(device_t::CPU);
    Device my_gpu_device(device_t::CPU, 0, 0);
}

TEST(core, add_bad_device) {
    // A simple CPU device cannot register an op.
    // A warning is expected telling the user to use
    // more parameters when device is created.

    Device my_gpu_device(device_t::CPU);

    convolutional_layer<sigmoid> l(5, 5, 3, 1, 2,
        padding::valid, true, 1, 1, backend_t::OpenCL);

    my_gpu_device.registerOp(l);
}

TEST(core, add_bad_layer) {
    // A GPU device cannot register an op with non-OpenCL engine.
    // A warning is expected telling the user to redefine the op engine.
 
    Device my_gpu_device(device_t::CPU, 2, 0);

    convolutional_layer<sigmoid> l(5, 5, 3, 1, 2,
        padding::valid, true, 1, 1, backend_t::tiny_dnn);

    my_gpu_device.registerOp(l);
}

TEST(core, device_add_op) {
    // An Op with OpenCL engine is registeres to
    // a GPU device which will compile its program, and
    // will place it to the general register.

    Device my_gpu_device(device_t::GPU, 2, 0);

    convolutional_layer<sigmoid> l(5, 5, 3, 1, 2,
        padding::valid, true, 1, 1, backend_t::OpenCL);

    //max_pooling_layer<identity> l(4, 4, 1, 2, 2, core::backend_t::OpenCL);

    ASSERT_EQ(ProgramManager::getInstance().num_programs(), 0);

#if defined(USE_OPENCL) || defined(USE_CUDA)
    // first time op registration: OK
    my_gpu_device.registerOp(l);

    ASSERT_EQ(ProgramManager::getInstance().num_programs(), 1);

    // second time op registraion: we expect that Op it's not
    // registrated since it's already there.
    my_gpu_device.registerOp(l);

    ASSERT_EQ(ProgramManager::getInstance().num_programs(), 1);
#endif
}

TEST(core, ocl_conv) {

    Device my_gpu_device(device_t::GPU, 2, 0);

    convolutional_layer<sigmoid> l(5, 5, 3, 1, 2,
        padding::valid, true, 1, 1, backend_t::OpenCL);

    // first time op registration: OK
    my_gpu_device.registerOp(l);


    auto create_simple_tensor = [](size_t vector_size) {
        return tensor_t(1, vec_t(vector_size));
    };

    // create simple tensors that wrap the payload vectors of the correct size
    tensor_t in_tensor     = create_simple_tensor(25)
           , out_tensor    = create_simple_tensor(18)
           , a_tensor      = create_simple_tensor(18)
           , weight_tensor = create_simple_tensor(18)
           , bias_tensor   = create_simple_tensor(2);

    // short-hand references to the payload vectors
    vec_t &in     = in_tensor[0]
        , &out    = out_tensor[0]
        , &weight = weight_tensor[0];

    ASSERT_EQ(l.in_shape()[1].size(), 18); // weight

    uniform_rand(in.begin(), in.end(), -1.0, 1.0);

    std::vector<tensor_t*> in_data, out_data;
    in_data.push_back(&in_tensor);
    in_data.push_back(&weight_tensor);
    in_data.push_back(&bias_tensor);
    out_data.push_back(&out_tensor);
    out_data.push_back(&a_tensor);
    l.setup(false);
    {
        l.forward_propagation(in_data, out_data);

        for (auto o: out)
            EXPECT_DOUBLE_EQ(o, tiny_cnn::float_t(0.5));
    }

    weight[0] = 0.3;  weight[1] = 0.1; weight[2] = 0.2;
    weight[3] = 0.0;  weight[4] =-0.1; weight[5] =-0.1;
    weight[6] = 0.05; weight[7] =-0.2; weight[8] = 0.05;

    weight[9]  = 0.0; weight[10] =-0.1; weight[11] = 0.1;
    weight[12] = 0.1; weight[13] =-0.2; weight[14] = 0.3;
    weight[15] = 0.2; weight[16] =-0.3; weight[17] = 0.2;

    in[0] = 3;  in[1] = 2;  in[2] = 1;  in[3] = 5; in[4] = 2;
    in[5] = 3;  in[6] = 0;  in[7] = 2;  in[8] = 0; in[9] = 1;
    in[10] = 0; in[11] = 6; in[12] = 1; in[13] = 1; in[14] = 10;
    in[15] = 3; in[16] =-1; in[17] = 2; in[18] = 9; in[19] = 0;
    in[20] = 1; in[21] = 2; in[22] = 1; in[23] = 5; in[24] = 5;

    {
        l.forward_propagation(in_data, out_data);

        EXPECT_NEAR(0.4875026, out[0], 1E-5);
        EXPECT_NEAR(0.8388910, out[1], 1E-5);
        EXPECT_NEAR(0.8099984, out[2], 1E-5);
        EXPECT_NEAR(0.7407749, out[3], 1E-5);
        EXPECT_NEAR(0.5000000, out[4], 1E-5);
        EXPECT_NEAR(0.1192029, out[5], 1E-5);
        EXPECT_NEAR(0.5986877, out[6], 1E-5);
        EXPECT_NEAR(0.7595109, out[7], 1E-5);
        EXPECT_NEAR(0.6899745, out[8], 1E-5);
    }

}

} // namespace tiny-dnn
