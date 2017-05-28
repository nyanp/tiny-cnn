/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include "tiny_dnn/core/framework/device.fwd.h"

#if defined(USE_OPENCL) || defined(USE_CUDA)
#ifdef USE_OPENCL
#include "third_party/CLCudaAPI/clpp11.h"
#else
#include "third_party/CLCudaAPI/cupp11.h"
#endif
#endif

namespace tiny_dnn {

/**
 * A tensor of the given dimension.
 *
 * U is type of the data stored
 * Storage is type of underlying storage
 */
template <typename U = float_t, typename Storage = xt::xarray<U>>
class Tensor {
  typedef U *UPtr;

 public:
  /**
   * Initializes an empty tensor.
   * @return
   */
  Tensor() { storage_ = xt::xarray<U>(); }

  /**
   * Constructor that assepts an initializer list of shape and create a
   * Tensor with that shape. For example, given shape = {2,3,4,5,6}, tensor
   * will be of size 2x3x4x5x6. Note: tensor isn't initialized by default
   * @param shape array containing N integers, sizes of dimensions
   * @return
   */
  explicit Tensor(std::vector<size_t> const &shape) {
    storage_ = xt::xarray<U>(shape);
  }

  /**
   * Constructor that assepts an initializer list of shape and create a
   * Tensor with that shape. For example, given shape = {2,3,4,5,6}, tensor
   * will be of size 2x3x4x5x6. Note: tensor isn't initialized by default
   * @param shape array containing N integers, sizes of dimensions
   * @return
   */
  explicit Tensor(std::initializer_list<size_t> const &shape) {
    storage_ = xt::xarray<U>(shape);
  }

  /**
   * Constructor that assepts an initializer list of shape and create a
   * Tensor with that shape and filling it with value. For example,
   * given shape = {2,3,4,5,6}, value = 0 tensor will be of size 2x3x4x5x6
   * filled with zeros
   * @param shape  shape array containing N integers, sizes of dimensions
   * @param value value to fill
   */
  explicit Tensor(std::initializer_list<size_t> const &shape, U value) {
    storage_ = xt::xarray<U>(shape);
    fill(value);
  }

  /**
   *
   * @param T
   * @return
   */
  Tensor<U> &operator=(const Tensor<U> &T) {
    storage_ = T.storage_;
    return *this;
  }

//~Tensor() = default;

// TODO(Randl): implement copy and move constructors
#if 0
    //TODO(Randl):deep copy
    Tensor(const Tensor&other) {
        other.fromDevice();
        shape_ = other.shape_;
        storage_pointer_ = other.storage_pointer_;
        data_is_on_host_ = true;
        data_dirty_ = true;
        //device_data_ is intentionally left uninitialized.
    }
    //TODO(Randl): Move constructors for Tensor and TensorStorage
    Tensor &operator = (const Tensor& other) {}

    Tensor(Tensor&& other) = default;        // move ctor
    Tensor &operator = (Tensor&&) = default; // move assign
#endif

  /**
   *
   * @return the tensor shape
   */
  const auto shape() const { return storage_.shape(); }

  /**
   *
   * @return the total number of elements in Tensor
   */
  const size_t size() const {
    auto shape = storage_.shape();
    return std::accumulate(shape.begin(), shape.end(), 1,
                           std::multiplies<size_t>());
  }

  /**
   * Access to indexes in tensor
   * @param args indexes in tensor
   * @return the value of a specified index in the tensor
   */
  template <typename... Args>
  U &host_at(const Args... args) {
    return storage_(args...);
  }

  /**
   * Constant access to indexes in tensor
   * @param args indexes in tensor
   * @return the value of a specified index in the tensor
   */
  template <typename... Args>
  U host_at(const Args... args) const {
    return storage_(args...);
  }

  /**
   *
   * @return Iterator pointing to the beginning of Tensor
   */
  auto host_begin() { return storage_.xbegin(); }

  const auto host_begin() const { return storage_.cxbegin(); }

  auto host_end() { return storage_.xend(); }

  const auto host_end() const { return storage_.cxend(); }

// TODO(Randl)
/*
const auto host_flatten() const {
  // fromDevice();
  return xt::broadcast(storage_, {size()});
}

auto host_data() {
  // fromDevice();
  return xt::broadcast(storage_, {size()});
}*/
// TODO(ּּRandl): should we enable this again?
#if 0
    U* mutable_host_data() {
        static_assert(!kConst, "Non-constant operation on constant Tensor");
        //fromDevice();
        //data_dirty_ = true;
        return storage_pointer_->data(offset);
    }
#endif

#if defined(USE_OPENCL) || defined(USE_CUDA)
  const void *device_data() const {
    storage_ptr_->toDevice();
    return (*storage_ptr_->device_data_)();
  }

  void *mutable_device_data() {
    static_assert(!kConst, "Non-constant operation on constant Tensor");
    storage_ptr_->toDevice();
    storage_ptr_->data_dirty_ = true;
    return (*storage_ptr_->device_data_)();
  }
#endif

  /**
   * Fill tensor with particular value
   * @param value
   * @return
   */
  Tensor &fill(U value) {
    // static_assert(!kConst, "Non-constant operation on constant Tensor");
    // data_is_on_host_ = true;
    // data_dirty_ = true;
    std::fill(storage_.xbegin(), storage_.xend(), value);
    return *this;
  }

  // TODO(Randl): checked version
  /**
   * Reshape tensor
   * @param sz
   */
  void reshape(const std::vector<size_t> &sz) { storage_.reshape(sz); }

  void resize(const std::vector<size_t> &sz) { storage_.reshape(sz); }

  // size_t size() const { return size_; }

  Tensor operator[](size_t index) { return Tensor(storage_[index]); }

  /**
   * Returns new Tensor which has shared storage with current, but different
   * shape
   *
   * Example:
   * @code
   * Tensor b({4,2});
   * a = b.subView({2,2,2};
   * @endcode
   * b is Tensor of shape 4x2 and a is Tensor of shape 2x2x2. Changing a(0,0,0)
   * will change b(0,0) too.
   * @param new_shape
   * @return
   */
  /*
 Tensor<U,xt::xbroadcast<Storage, std::vector<size_t>>>
 subView(std::initializer_list<size_t> const &new_shape) {
   auto res = Tensor<U,xt::xbroadcast<Storage,
 std::vector<size_t>>>(xt::broadcast(storage_, storage_.shape()));
   res.storage_.reshape(new_shape);
   return res;
 }*/

  // TODO(Randl) variadic version
  /**
   * Rutruns sub-Tensor of the current Tensor which shares storage with it.
   * @param dim1
   * @param dim2
   * @return
   */
  Tensor<U, xt::xview<Storage>> subView(
    std::initializer_list<size_t> const &dim1,
    std::initializer_list<size_t> const &dim2) {
    return Tensor<U, xt::xview<Storage>>(
      xt::view(storage_, xt::range(*(dim1.begin()), *(dim1.begin()) + 1),
               xt::range(*(dim2.begin()), *(dim2.begin()) + 1)));
    // return subview_impl(start, new_shape);
  }

  Tensor<U,
         xt::xview<Storage &,
                   xt::xrange<long unsigned int>,
                   xt::xrange<long unsigned int>,
                   xt::xrange<long unsigned int>,
                   xt::xrange<long unsigned int>>>
  subView(std::initializer_list<size_t> const &dim1,
          std::initializer_list<size_t> const &dim2,
          std::initializer_list<size_t> const &dim3,
          std::initializer_list<size_t> const &dim4) {
    using TensNew = Tensor<
      U, xt::xview<Storage &, xt::xrange<long unsigned int>,
                   xt::xrange<long unsigned int>, xt::xrange<long unsigned int>,
                   xt::xrange<long unsigned int>>>;
    // auto t = xt::view(storage_, xt::range(*(dim1.begin()), *(dim1.begin()) +
    // 1), xt::range(*(dim2.begin()), *(dim2.begin()) + 1),
    // xt::range(*(dim3.begin()), *(dim3.begin()) + 1),
    // xt::range(*(dim4.begin()), *(dim4.begin()) + 1));
    return TensNew(storage_, xt::range(*(dim1.begin()), *(dim1.begin()) + 1),
                   xt::range(*(dim2.begin()), *(dim2.begin()) + 1),
                   xt::range(*(dim3.begin()), *(dim3.begin()) + 1),
                   xt::range(*(dim4.begin()), *(dim4.begin()) + 1));

    // return subview_impl(start, new_shape);
  }

  /*
  // TODO: is needed?
  bool isSubView() const {
    return true; //std::is_same(Storage, xt::xview<U>);
  }
*/

  /**
   * Creates Tensor given the storage
   * @tparam T
   * @param storage
   */
  template <class T>
  explicit Tensor(T &storage,
                  xt::xrange<long unsigned int> r1,
                  xt::xrange<long unsigned int> r2,
                  xt::xrange<long unsigned int> r3,
                  xt::xrange<long unsigned int> r4)
    : storage_(xt::view(storage, r1, r2, r3, r4)) {}

 private:
  Storage storage_;
  // xt::xexpression<Storage> xepr_stotrage_;

  /**
   *
   * @tparam T
   * @param os
   * @param tensor
   * @return
   */
  template <class T>
  friend inline std::ostream &operator<<(std::ostream &os,
                                         const Tensor<T> &tensor);
};

}  // namespace tiny_dnn