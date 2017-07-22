/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "gtest/gtest.h"
#include "testhelper.h"

#include "tiny_dnn/tiny_dnn.h"

#if defined(USE_OPENCL) || defined(USE_CUDA)
#include "third_party/CLCudaAPI/clpp11.h"
#endif  // defined(USE_OPENCL) || defined(USE_CUDA)

namespace tiny_dnn {

#if defined(USE_OPENCL) || defined(USE_CUDA)
device_t device_type(size_t *platform, size_t *device) {
  // check which platforms are available
  auto platforms = CLCudaAPI::GetAllPlatforms();

  // if no platforms - return -1
  if (platforms.size() == 0) {
    return device_t::NONE;
  }

  std::array<std::string, 2> devices_order        = {"GPU", "CPU"};
  std::map<std::string, device_t> devices_t_order = {
    std::make_pair("GPU", device_t::GPU), std::make_pair("CPU", device_t::CPU)};
  for (auto d_type : devices_order)
    for (auto p = platforms.begin(); p != platforms.end(); ++p)
      for (size_t d = 0; d < p->NumDevices(); ++d) {
        auto dev = CLCudaAPI::Device(*p, d);
        if (dev.Type() == d_type) {
          *platform = p - platforms.begin();
          *device   = d;
          return devices_t_order[d_type];
        }
      }
  // no CPUs or GPUs
  return device_t::NONE;
}

#define TINY_DNN_GET_DEVICE_AND_PLATFORM \
  size_t cl_platform = 0, cl_device = 0; \
  device_t device = device_type(&cl_platform, &cl_device);
#else
#define TINY_DNN_GET_DEVICE_AND_PLATFORM \
  size_t cl_platform = 0, cl_device = 0; \
  device_t device = device_t::NONE;
#endif  // defined(USE_OPENCL) || defined(USE_CUDA)

/*
TEST(core, platforms_and_devices) {
    // Since Singleton has a general state,
    // in each test we reset program register
    ProgramManager::getInstance().reset();

    //check which platforms are available and which devices
    auto platforms = CLCudaAPI::GetAllPlatforms();
    EXPECT_LT(0, platforms.size());
    for (auto &p: platforms)  {
        EXPECT_LT(0, p.NumDevices());
        for (size_t d = 0; d < p.NumDevices(); ++d) {
            auto dev = CLCudaAPI::Device(p, d);
            std::cout << "Device " << d << " is " << dev.Type() << "\n";
        }
    }
}*/

TEST(core, device) {
  // Since Singleton has a general state,
  // in each test we reset program register
  ProgramManager::getInstance().reset();

  // CPU and GPU devices are instantiated
  Device my_cpu_device(device_t::CPU);

  TINY_DNN_GET_DEVICE_AND_PLATFORM;
  if (device != device_t::NONE) {
    Device my_gpu_device(device, cl_platform, cl_device);
  }
}

TEST(core, add_bad_device) {
  // A simple CPU device cannot register an op.
  // A warning is expected telling the user to use
  // more parameters when device is created.

  // Since Singleton has a general state,
  // in each test we reset program register
  ProgramManager::getInstance().reset();

  Device my_gpu_device(device_t::CPU);

  convolutional_layer l(5, 5, 3, 1, 2, padding::valid, true, 1, 1,
                        backend_t::libdnn);

  EXPECT_THROW(my_gpu_device.registerOp(l), nn_error);
}

TEST(core, add_bad_layer) {
  // A GPU device cannot register an op with non-OpenCL engine.
  // A warning is expected telling the user to redefine the op engine.

  // Since Singleton has a general state,
  // in each test we reset program register
  ProgramManager::getInstance().reset();

  TINY_DNN_GET_DEVICE_AND_PLATFORM;
  if (device != device_t::NONE) {
    Device my_gpu_device(device, cl_platform, cl_device);

    convolutional_layer l(5, 5, 3, 1, 2, padding::valid, true, 1, 1,
                          backend_t::internal);

    EXPECT_THROW(my_gpu_device.registerOp(l), nn_error);
  }
}

TEST(core, device_add_op) {
  // An Op with OpenCL engine is registered to
  // a GPU device which will compile its program, and
  // will place it to the general register.

  // Since Singleton has a general state,
  // in each test we reset program register
  ProgramManager::getInstance().reset();

  TINY_DNN_GET_DEVICE_AND_PLATFORM;
  if (device != device_t::NONE) {
    Device my_gpu_device(device, cl_platform, cl_device);

    convolutional_layer l(5, 5, 3, 1, 2, padding::valid, true, 1, 1,
                          backend_t::libdnn);

    // max_pooling_layer l(4, 4, 1, 2, 2,
    // core::backend_t::opencl);

    ASSERT_EQ(ProgramManager::getInstance().num_programs(), 0u);

#if defined(USE_OPENCL) || defined(USE_CUDA)
    // first time op registration: OK
    my_gpu_device.registerOp(l);

    ASSERT_EQ(ProgramManager::getInstance().num_programs(), 1u);

    // second time op registraion: we expect that Op it's not
    // registrated since it's already there.
    my_gpu_device.registerOp(l);

    ASSERT_EQ(ProgramManager::getInstance().num_programs(), 1u);
#endif
  }
}
TEST(core, ocl_conv) {
  // Since Singleton has a general state,
  // in each test we reset program register
  ProgramManager::getInstance().reset();

  TINY_DNN_GET_DEVICE_AND_PLATFORM;
  if (device != device_t::NONE) {
    Device my_gpu_device(device, cl_platform, cl_device);

    convolutional_layer l(5, 5, 3, 1, 2, padding::valid, true, 1, 1,
                          backend_t::libdnn);
    sigmoid_layer sgm(3, 3, 2);
    l << sgm;

    // first time op registration: OK
    my_gpu_device.registerOp(l);

    auto create_simple_tensor = [](size_t vector_size) {
      return tensor_t(1, vec_t(vector_size));
    };

    // create simple tensors that wrap the
    // payload vectors of the correct size
    tensor_t in_tensor     = create_simple_tensor(25),
             out_tensor    = create_simple_tensor(18),
             a_tensor      = create_simple_tensor(18),
             weight_tensor = create_simple_tensor(18),
             bias_tensor   = create_simple_tensor(2);

    // short-hand references to the payload vectors
    vec_t &in = in_tensor[0], &out = out_tensor[0], &weight = weight_tensor[0];

    ASSERT_EQ(l.in_shape()[1].size(), 18u);  // weight

    uniform_rand(in.begin(), in.end(), -1.0, 1.0);

    std::vector<tensor_t *> in_data, out_data;
    in_data.push_back(&in_tensor);
    in_data.push_back(&weight_tensor);
    in_data.push_back(&bias_tensor);
    out_data.push_back(&out_tensor);
    out_data.push_back(&a_tensor);
    l.setup(false);
    {
      l.forward_propagation(in_data, out_data);

      for (auto o : out) EXPECT_DOUBLE_EQ(o, tiny_dnn::float_t(0.5));
    }

    // clang-format off
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

    // clang-format on
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
}

}  // namespace tiny-dnn
