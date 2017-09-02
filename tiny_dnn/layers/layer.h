/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tiny_dnn/core/backend.h"
#include "tiny_dnn/core/framework/device.fwd.h"
#include "tiny_dnn/core/framework/tensor.h"
#include "tiny_dnn/core/framework/tensor_utils.h"
#include "tiny_dnn/node.h"
#include "tiny_dnn/parameter.h"

#include "tiny_dnn/util/parallel_for.h"
#include "tiny_dnn/util/parameter_init.h"
#include "tiny_dnn/util/product.h"
#include "tiny_dnn/util/util.h"

#include "tiny_dnn/optimizers/optimizer.h"

#ifdef DNN_USE_IMAGE_API
#include "tiny_dnn/util/image.h"
#endif

#ifdef CNN_USE_HDF
#include <H5Cpp.h>
#endif

namespace tiny_dnn {

/**
 * base class of all kind of NN layers
 *
 * sub-class should override these methods:
 * - forward_propagation ... body of forward-pass calculation
 * - back_propagation    ... body of backward-pass calculation
 * - in_shape            ... specify input data shapes
 * - out_shape           ... specify output data shapes
 * - layer_type          ... name of layer
 **/
class layer : public node {
 public:
  friend void connection_mismatch(const layer &from, const layer &to);

  /**
   * @brief Default layer constructor that instantiates a N-input, M-output
   *layer
   *
   * @param in_type[N] type of input vector (data, weight, bias...)
   * @param out_type[M] type of output vector
   *
   **/
  layer(const std::vector<vector_type> &in_type,
        const std::vector<vector_type> &out_type)
    : node(in_type.size(), out_type.size()),
      parallelize_(true),
      in_channels_(in_type.size()),
      out_channels_(out_type.size()),
      in_type_(in_type),
      out_type_(out_type) {
    weight_init_     = std::make_shared<parameter_init::xavier>();
    bias_init_       = std::make_shared<parameter_init::constant>();
    parameters_diff_ = Tensor<float_t>();
  }

  layer(const layer &) = default;
  layer &operator=(const layer &) = default;

  layer(layer &&) = default;
  layer &operator=(layer &&) = default;

  void set_parallelize(bool parallelize) { parallelize_ = parallelize; }

  void set_backend(std::shared_ptr<core::backend> backend) {
    backend_ = backend;
  }

  void set_backend_type(core::backend_t backend_type) {
    backend_type_ = backend_type;
  }

  /////////////////////////////////////////////////////////////////////////
  // getter

  bool parallelize() const { return parallelize_; }

  // TODO(edgar): Deprecated: use the below method
  core::backend_t backend_type() const { return backend_->type(); }

  core::backend_t engine() const { return backend_type_; }

  virtual std::string kernel_file() const {
    return std::string("empty_kernel_str");
  }

  virtual std::string kernel_header() const { return std::string(); }

  virtual void createOp() {}

  void setDevice(const Device &device) {
    device_ptr_ = const_cast<Device *>(&device);
  }

  Device *device() const { return device_ptr_; }

  std::shared_ptr<core::backend> backend() { return backend_; }

  ///< number of incoming edges in this layer
  size_t in_channels() const { return in_channels_; }

  ///< number of outgoing edges in this layer
  size_t out_channels() const { return out_channels_; }

  size_t in_data_size() const {
    return sumif(in_shape(),
                 [&](size_t i) {  // NOLINT
                   return in_type_[i] == vector_type::data;
                 },
                 [](const shape3d &s) { return s.size(); });
  }

  size_t out_data_size() const {
    return sumif(out_shape(),
                 [&](size_t i) {  // NOLINT
                   return out_type_[i] == vector_type::data;
                 },
                 [](const shape3d &s) { return s.size(); });
  }

  std::vector<shape3d> in_data_shape() {
    return filter(in_shape(), [&](size_t i) {  // NOLINT
      return in_type_[i] == vector_type::data;
    });
  }

  std::vector<shape3d> out_data_shape() {
    return filter(out_shape(), [&](size_t i) {  // NOLINT
      return out_type_[i] == vector_type::data;
    });
  }

  ///! @deprecated use in_data_size() instead
  size_t in_size() const { return in_data_size(); }

  ///! @deprecated use out_data_size() instead
  size_t out_size() const { return out_data_size(); }

  std::vector<edgeptr_t> inputs() {
    std::vector<edgeptr_t> nodes(in_channels_);
    for (size_t i = 0; i < in_channels_; i++) {
      nodes[i] = ith_in_node(i);
    }
    return nodes;
  }

  std::vector<edgeptr_t> outputs() {
    std::vector<edgeptr_t> nodes(out_channels_);
    for (size_t i = 0; i < out_channels_; i++) {
      nodes[i] = ith_out_node(i);
    }
    return nodes;
  }

  std::vector<edgeptr_t> outputs() const {
    std::vector<edgeptr_t> nodes(out_channels_);
    for (size_t i = 0; i < out_channels_; i++) {
      nodes[i] = const_cast<layer *>(this)->ith_out_node(i);
    }
    return nodes;
  }

  void set_out_grads(const std::vector<const vec_t *> *grad, size_t cnt) {
    CNN_UNREFERENCED_PARAMETER(cnt);
    size_t n = 0;
    for (size_t i = 0; i < out_channels_; i++) {
      if (out_type_[i] != vector_type::data) continue;
      Tensor<> &dst_grad = *ith_out_node(i)->get_gradient();
      assert(n < cnt);
      const auto &src_grad = grad[n++];
      size_t sz            = src_grad.size();
      dst_grad.reshape({sz, src_grad[0]->size()});
      for (size_t j = 0; j < sz; ++j) {
        // TODO(Randl)
        std::copy(src_grad[j]->begin(), src_grad[j]->end(),
                  dst_grad.host_iter(j, 0));
      }
    }
  }

  void set_in_data(const std::vector<const vec_t *> *data, size_t cnt) {
    CNN_UNREFERENCED_PARAMETER(cnt);
    size_t n = 0;
    for (size_t i = 0; i < in_channels_; i++) {
      if (in_type_[i] != vector_type::data) continue;
      Tensor<> &dst_data = *ith_in_node(i)->get_data();
      size_t in_size     = ith_in_node(i)->shape().size();
      assert(n < cnt);
      const auto &src_data = data[n++];
      size_t sz            = src_data.size();
      dst_data.resize_axis(sz);

      CNN_UNREFERENCED_PARAMETER(in_size);

      for (size_t j = 0; j < sz; ++j) {
        assert(
          src_data[j]->size() ==
          in_size);  // checking if training data is consistent with layer shape
        // TODO(Randl)
        std::copy(src_data[j]->begin(), src_data[j]->end(),
                  dst_data.host_iter(j, 0));
      }
    }
  }

  void output(std::vector<const Tensor<> *> &out) const {
    out.clear();
    for (size_t i = 0; i < out_channels_; i++) {
      if (out_type_[i] == vector_type::data) {
        out.push_back(ith_out_node(i)->get_data());
      }
    }
  }

  std::vector<vector_type> in_types() const { return in_type_; }

  std::vector<vector_type> out_types() const { return out_type_; }

  void set_trainable(bool trainable) {
    for (auto &parameter : parameters_) {
      parameter->set_trainable(trainable);
    }
  }

  /**
   * return output value range
   * used only for calculating target value from label-id in final(output)
   *layer
   * override properly if the layer is intended to be used as output layer
   **/
  virtual std::pair<float_t, float_t> out_value_range() const {
    return {float_t{0.0}, float_t{1.0}};
  }

  /**
   * array of input shapes (width x height x depth)
   **/
  virtual std::vector<shape3d> in_shape() const = 0;

  /**
   * set input shape of a layer (only used internally while shape inferring)
   */
  virtual void set_in_shape(const shape3d &in_shape) {
    CNN_UNREFERENCED_PARAMETER(in_shape);
    throw nn_error(
      "Can't set shape. Shape inferring not applicable for this "
      "layer (yet).");
  }

  /**
   * array of output shapes (width x height x depth)
   **/
  virtual std::vector<shape3d> out_shape() const = 0;

  /**
   * name of layer, should be unique for each concrete class
   **/
  virtual std::string layer_type() const = 0;

  /**
   * number of incoming connections for each output unit
   * used only for weight/bias initialization methods which require fan-in
   *size
   *(e.g. xavier)
   * override if the layer has trainable weights, and scale of initialization
   *is
   *important
   **/
  virtual size_t fan_in_size() const { return in_shape()[0].width_; }
  // override to allow initialization of multiple size weight matrices.
  virtual size_t fan_in_size(size_t) const {
    return fan_in_size();  // fallback to single weight matrix.
  }

  /**
   * number of outgoing connections for each input unit
   * used only for weight/bias initialization methods which require fan-out
   *size
   *(e.g. xavier)
   * override if the layer has trainable weights, and scale of initialization
   *is
   *important
   **/
  virtual size_t fan_out_size() const { return out_shape()[0].width_; }
  // override to allow initialization of multiple size weight vectors.
  virtual size_t fan_out_size(size_t) const {
    return fan_out_size();  // fallback to single weight matrix
  }

  /////////////////////////////////////////////////////////////////////////
  /**
   * @name Parameter Init Methods
   * @{
   */
  template <typename WeightInit>
  layer &weight_init(const WeightInit &f) {
    weight_init_ = std::make_shared<WeightInit>(f);
    return *this;
  }

  template <typename BiasInit>
  layer &bias_init(const BiasInit &f) {
    bias_init_ = std::make_shared<BiasInit>(f);
    return *this;
  }

  template <typename WeightInit>
  layer &weight_init(std::shared_ptr<WeightInit> f) {
    weight_init_ = f;
    return *this;
  }

  template <typename BiasInit>
  layer &bias_init(std::shared_ptr<BiasInit> f) {
    bias_init_ = f;
    return *this;
  }
  /** @} */  // Parameter Init Methods

  /**
   * @name Parameter Getters and Setters
   * @{
   */

  /** @brief Add new parameter to this layer, to be called in constructor. */
  void add_parameter(size_t out_channels,
                     size_t in_channels,
                     size_t height,
                     size_t width,
                     parameter_type type,
                     bool trainable = true) {
    parameters_.push_back(std::make_shared<Parameter>(
      out_channels, in_channels, height, width, type, trainable));
  }

  /**
   * @brief Get pointers to parameters of this layer.
   *
   * This can be used during training phase when parameter update is
   * required.
   *
   * @param trainable_only flag to return only the trainable parameters
   * @return std::vector of pointers to parameters
   */
  Parameters parameters(bool trainable_only = false) {
    Parameters parameters;
    for (auto &parameter : parameters_) {
      if (!trainable_only || (parameter->is_trainable() && trainable_only)) {
        parameters.push_back(parameter.get());
      }
    }
    return parameters;
  }

  ConstParameters parameters(bool trainable_only = false) const {
    ConstParameters parameters;
    for (auto &parameter : parameters_) {
      if (!trainable_only || (parameter->is_trainable() && trainable_only)) {
        parameters.push_back(parameter.get());
      }
    }
    return parameters;
  }

  /** To be generally used internally by kernels and optimizers. */
  Parameter &parameter_at(size_t i) { return *parameters_[i]; }

  /** To be generally used internally by kernels and optimizers. */
  const Parameter &parameter_at(size_t i) const { return *parameters_[i]; }

  /** Convenience method to access and manipulate weights. */
  Parameters weights_at() {
    Parameters weights;
    for (auto &parameter : parameters_) {
      if (parameter->type() == parameter_type::weight) {
        weights.push_back(parameter.get());
      }
    }
    return weights;
  }

  /** Convenience method to access and analyze particular weights. */
  ConstParameters weights_at() const {
    ConstParameters weights;
    for (auto &parameter : parameters_) {
      if (parameter->type() == parameter_type::weight) {
        weights.push_back(parameter.get());
      }
    }
    return weights;
  }

  /** Convenience method to access and manipulate particular bias. */
  Parameters bias_at() {
    Parameters biases;
    for (auto &parameter : parameters_) {
      if (parameter->type() == parameter_type::bias) {
        biases.push_back(parameter.get());
      }
    }
    return biases;
  }

  /** Convenience method to access and analyze particular biases. */
  ConstParameters bias_at() const {
    ConstParameters biases;
    for (auto &parameter : parameters_) {
      if (parameter->type() == parameter_type::bias) {
        biases.push_back(parameter.get());
      }
    }
    return biases;
  }

  /** @} */  // Parameter Getters and Setters

  virtual void save(
    std::ostream &os,
    const int precision = std::numeric_limits<float_t>::digits10 + 2
    /*by default, we want there to be enough precision*/) const {  // NOLINT
    os << std::setprecision(precision);
    for (auto &parameter : parameters_) {
      parameter->save(os, precision);
    }
  }

  virtual void load(
    std::istream &is,
    const int precision = std::numeric_limits<float_t>::digits10 + 2
    /*by default, we want there to be enough precision*/) {  // NOLINT
    is >> std::setprecision(precision);
    for (auto &parameter : parameters_) {
      parameter->load(is, precision);
    }
  }

#ifdef CNN_USE_HDF
  void load(const std::string &file_path, const std::string &layer_name) {
    // this id can be seen as H5::File object
    hid_t file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    // it will have multiple H5::Group objects, each representing a layer
    hid_t layer_id = H5Gopen1(file_id, layer_name.c_str());

    // each H5::Group shall have one H5::Attribute named 'weight_names'
    hid_t parameter_names_id = H5Aopen(layer_id, "weight_names", H5P_DEFAULT);

    // type is required to read names from H5::Attribute as vector of string
    hid_t type =
      H5Tget_native_type(H5Aget_type(parameter_names_id), H5T_DIR_ASCEND);

    // get the maximum character length of parameter name
    H5A_info_t a_info;
    H5Aget_info(parameter_names_id, &a_info);
    size_t max_parameter_name_length = a_info.data_size / parameters_.size();

    // string buffer to accumulate data from H5::Attribute
    char parameter_names_buf[1000];

    // read parameter names into vector of strings
    H5Aread(parameter_names_id, type, parameter_names_buf);
    std::vector<std::string> parameter_names;
    for (size_t i = 0; i < parameters_.size(); i++) {
      std::string parameter_name(
        &parameter_names_buf[i * max_parameter_name_length]);
      if (parameter_name.length() > max_parameter_name_length) {
        parameter_name = parameter_name.substr(0, max_parameter_name_length);
      }
      parameter_names.push_back(parameter_name);
    }
    H5Aclose(parameter_names_id);

    // parameter_names vector will usually have 1 or 2 members named
    // <layer_name>/kernel:0 and <layer_name>/bias:0
    for (size_t i = 0; i < parameter_names.size(); i++) {
      parameters_[i]->load(file_path, layer_name + "/" + parameter_names[i]);
    }
    H5Gclose(layer_id);
    H5Fclose(file_id);
  }
#endif

/////////////////////////////////////////////////////////////////////////
// visualize

///< visualize latest output of this layer
///< default implementation interpret output as 1d-vector,
///< so "visual" layer(like convolutional layer) should override this for better
/// visualization.
#ifdef DNN_USE_IMAGE_API
  virtual image<> output_to_image(size_t channel = 0) const {
    // TODO(Randl)
    const vec_t *output = &(*(outputs()[channel]->get_data())).toTensor()[0];
    return vec2image<unsigned char>(*output, out_shape()[channel]);
  }
#endif

  /////////////////////////////////////////////////////////////////////////
  // fprop/bprop

  /**
   * @param in_data  input vectors of this layer (data, weight, bias)
   * @param out_data output vectors
   **/
  virtual void forward_propagation(const std::vector<Tensor<> *> &in_data,
                                   std::vector<Tensor<> *> &out_data) = 0;

  /**
   * return delta of previous layer (delta=\frac{dE}{da}, a=wx in
   *fully-connected layer)
   * @param in_data  input vectors (same vectors as forward_propagation)
   * @param out_data output vectors (same vectors as forward_propagation)
   * @param out_grad gradient of output vectors (i-th vector correspond with
   *out_data[i])
   * @param in_grad  gradient of input vectors (i-th vector correspond with
   *in_data[i])
   **/
  virtual void back_propagation(const std::vector<Tensor<> *> &in_data,
                                const std::vector<Tensor<> *> &out_data,
                                std::vector<Tensor<> *> &out_grad,
                                std::vector<Tensor<> *> &in_grad) = 0;

  /**
   * return delta2 of previous layer (delta2=\frac{d^2E}{da^2}, diagonal of
   *hessian matrix)
   * it is never called if optimizer is hessian-free
   **/
  // virtual void back_propagation_2nd(const std::vector<vec_t>& delta_in) =
  // 0;

  // called afrer updating weight
  virtual void post_update() {}

  /**
   * notify changing context (train <=> test)
   **/
  virtual void set_context(net_phase ctx) { CNN_UNREFERENCED_PARAMETER(ctx); }

  // convenience wrapper for function below
  std::vector<const Tensor<> *> forward(
    const std::vector<Tensor<>> &input) {  // for test
    std::vector<const Tensor<> *> output;
    forward(input, output);
    return output;
  }

  /* @brief Performs layer forward operation given an input tensor and
   * returns the computed data in tensor form.
   *
   * @param input Vector of `Tensor<>` with incoming data.
   *
   * Internally, it first allocates data without resetting the weights,
   * forwards the input data to the computational graph, inside the
   * forward() method the data from the computational embedded to container
   * to finally be forwarded to the computational operation kernels.
   *
   * TODO: Probably there's an overhead of moving from/to the computational
   * graph. Will be this overhead reduced once we have the Tensor
   * class integrated?
   */
  void forward(const std::vector<Tensor<>> &input,
               std::vector<const Tensor<> *> &out) {  // for test
    // allocate data in the computational graph without resetting the weights.
    setup(false);

    std::vector<std::vector<const vec_t *>> input2;
    std::vector<std::vector<vec_t>> input2_st;  // TODO(Randl) temporary
    input2.resize(input.size());
    input2_st.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
      input2[i].resize(input[i].shape()[0]);
      input2_st[i].resize(input[i].shape()[0]);
      for (size_t j = 0; j < input[i].shape()[0]; ++j) {
        assert(input[i].shape().size() == 2);
        // TODO(Randl)
        input2_st[i][j] = input[i].lineToVec(j);
        input2[i][j]    = &input2_st[i][j];
      }
    }

    // the incoming data is forwarded to the computational graph.
    set_in_data(&input2[0], input2.size());
    // pick up the data from the computational graph and perform
    // computation.
    forward();
    // retrieve computed data and return values in form of 4D tensor.
    output(out);
  }

  std::vector<Tensor<>> backward(
    const std::vector<Tensor<>> &out_grads) {  // for test
    setup(false);

    std::vector<std::vector<const vec_t *>> grads2;
    std::vector<std::vector<vec_t>> grads2_st;  // TODO(Randl) temporary
    grads2.resize(out_grads.size());
    grads2_st.resize(out_grads.size());
    for (size_t i = 0; i < out_grads.size(); ++i) {
      grads2[i].resize(out_grads[i].shape()[0]);
      grads2_st[i].resize(out_grads[i].shape()[0]);
      for (size_t j = 0; j < out_grads[i].shape()[0]; ++j) {
        assert(out_grads[i].shape().size() == 2);
        // TODO(Randl)
        grads2_st[i][j] = out_grads[i].lineToVec(j);
        grads2[i][j]    = &grads2_st[i][j];
      }
    }

    set_out_grads(&grads2[0], grads2.size());
    backward();
    return map_<Tensor<>>(inputs(),
                          [](edgeptr_t e) { return *e->get_gradient(); });
  }

  /* @brief The purpose of this method is to forward the data from the
   * computational graph to the layer interface.
   *
   * This is one of the out of two core (forward/backward) methods that
   * retrieves the data allocated in the heap by the computational graph
   * and constructs the containers to handle the computation by batches.
   * Additionally, the sample count a.k.a number of batches is set.
   *
   * Note: in_data and out_data attempt to contain tensors. However, they
   * are not real tensors since Tensor<> have three dimensions instead of
   * four. For this reason they are embedded in to std::vector. Also note
   * that when std::vector<Tensor<>*> it's constructed we cannot assure
   * that data is contiguous.
   *
   * After Tensor class integration we should be able to avoid to have
   * in_data and out_data in vectors since Tensor class itself can handle
   * batches storage in one single vector with contiguous data.
   *
   */
  void forward() {
    // the computational graph
    fwd_in_data_.resize(in_channels_);
    fwd_out_data_.resize(out_channels_);

    // Organize input/output vectors from storage (computational graph).
    // Internally ith_in_node() will create a connection/edge in the
    // computational graph and will allocate memory in case that it's not
    // done yet.
    for (size_t i = 0; i < in_channels_; i++) {
      fwd_in_data_[i] = ith_in_node(i)->get_data();
    }

    // resize outs and stuff to have room for every input sample in the batch
    set_sample_count(fwd_in_data_[0]->shape()[0]);

    // Internally ith_out_node() will create a connection/edge to the
    // computational graph and will allocate memory in case that it's not
    // done yet. In addition, gradient vector are initialized to default
    // values.
    for (size_t i = 0; i < out_channels_; i++) {
      fwd_out_data_[i] = ith_out_node(i)->get_data();
      ith_out_node(i)->clear_grads();
    }

    // call the forward computation kernel/routine
    forward_propagation(fwd_in_data_, fwd_out_data_);
  }

  void backward() {
    bwd_in_data_.resize(in_channels_);
    bwd_in_grad_.resize(in_channels_);
    bwd_out_data_.resize(out_channels_);
    bwd_out_grad_.resize(out_channels_);

    // organize input/output vectors from storage
    for (size_t i = 0; i < in_channels_; i++) {
      const auto &nd  = ith_in_node(i);
      bwd_in_data_[i] = nd->get_data();
      bwd_in_grad_[i] = nd->get_gradient();
    }
    for (size_t i = 0; i < out_channels_; i++) {
      const auto &nd   = ith_out_node(i);
      bwd_out_data_[i] = nd->get_data();
      bwd_out_grad_[i] = nd->get_gradient();
    }

    back_propagation(bwd_in_data_, bwd_out_data_, bwd_out_grad_, bwd_in_grad_);
  }

  /* @brief Allocates data in the computational graph and reset weights if
   * it's needed or the data is not already initialized.
   *
   * @param reset_weight Boolean value to force to reset the weights.
   * Weights will be automatically reset if the are not initialized.
   *
   */
  void setup(bool reset_weight) {
    // The input shape (width x height x depth) must be equal to the number
    // of input channels a.k.a the number of incoming vectors or 'edges' in
    // the computational nomenclature. Same is applied to output shape and
    // numbers of output edges.
    if (in_shape().size() != in_channels_ ||
        out_shape().size() != out_channels_) {
      throw nn_error("Connection mismatch at setup layer");
    }

    // An 'edge' is created in the computational graph from the current
    // layer/node to each output node and allocates the needed memory.
    // The number of output nodes is determined by the layer interface.
    // In order to handle graph based networks, which a layer/node might
    // have multiple input/output connections, we need to check that the
    // connection edge does not already exists if we don't want duplicated
    // memory allocation.
    for (size_t i = 0; i < out_channels_; i++) {
      if (!next_[i]) {
        // connection edge doesn't exist, so we proceed to allocate the
        // necessary memory.
        next_[i] = std::make_shared<edge>(this, out_shape()[i], out_type_[i]);
      }
    }

    // reset the weights if necessary, or in case that the data is
    // still not initialized.
    for (auto &parameter : parameters_) {
      if (!parameter->initialized()) {
        init_parameters();
        return;
      }
    }
    if (reset_weight) {
      init_parameters();
    }
  }

  /** @brief Initializes the trainable parameters of this layer.
   *
   * In case that a layer/node is set to be not trainable, it does
   * nothing and returns a void. Otherwise, for each input connection
   * and depending of the data nature (weight or bias) calls their
   * pertinent initialization function and fill the vectors with the
   * data generated by the mentioned functions.
   *
   */
  void init_parameters() {
    // Fill parameter values with data generated by the initialization
    // function. The pointer to the data is obtained from the
    // computational graph and the methods fan_in_size() and fan_out_size()
    // return the number of incoming/outcoming connections for each
    // input/output unit.
    for (size_t i = 0; i < parameters_.size(); i++) {
      switch (parameters_[i]->type()) {
        // fill parameters of weight type
        case parameter_type::weight:
          parameters_[i]->initialize(*weight_init_, fan_in_size(i),
                                     fan_out_size(i));
          break;
        // fill vector of bias type
        case parameter_type::bias:
          parameters_[i]->initialize(*bias_init_, fan_in_size(i),
                                     fan_out_size(i));
          break;
        default: break;
      }
      parameters_[i]->set_initialized();
    }
  }

  void clear_grads() {
    for (size_t i = 0; i < in_type_.size(); i++) {
      ith_in_node(i)->clear_grads();
    }

    for (auto &parameter : parameters_) {
      parameter->clear_grads();
    }
  }

  void update_parameters(optimizer *optimizer_ptr, size_t batch_size) {
    float_t rcp_batch_size = float_t(1) / float_t(batch_size);
    auto &diff             = parameters_diff_;
    for (auto &parameter : parameters_) {
      if (!parameter->is_trainable()) {
        continue;
      }
      parameter->merge_grads(&diff);

      for (size_t j = 0; j < diff.size(); ++j) {
        diff.host_at(j) *= rcp_batch_size;
      }
      // parallelize only when target size is big enough to mitigate
      // thread spawning overhead.
      bool parallelize = (parameter->size() >= 512);

      Tensor<> &target_t = *parameter->data();
      optimizer_ptr->update(diff, target_t, parallelize);
      parameter->set_data(Tensor<>(target_t));
    }
    clear_grads();
    post_update();
  }

  bool has_same_parameters(const layer &rhs, float_t eps) const {
    ConstParameters lhs_parameters = parameters();
    ConstParameters rhs_parameters = rhs.parameters();
    if (lhs_parameters.size() != rhs_parameters.size()) return false;

    for (size_t i = 0; i < parameters_.size(); i++) {
      if (lhs_parameters[i]->size() != rhs_parameters[i]->size()) return false;

      const Tensor<> *lhs_data = lhs_parameters[i]->data();
      const Tensor<> *rhs_data = rhs_parameters[i]->data();
      for (size_t j = 0; j < lhs_parameters[i]->size(); j++) {
        if (std::abs(lhs_data->host_at(j) - rhs_data->host_at(j)) > eps)
          return false;
      }
    }
    return true;
  }

  virtual void set_sample_count(size_t sample_count) {
    // increase the size if necessary - but do not decrease
    auto resize = [sample_count](Tensor<> *tensor) {
      if (tensor->dim() < 1 || tensor->shape()[0] <= sample_count)
        tensor->resize_axis(sample_count);
    };

    for (size_t i = 0; i < in_channels_; i++) {
      resize(ith_in_node(i)->get_data());
      resize(ith_in_node(i)->get_gradient());
    }

    for (size_t i = 0; i < out_channels_; i++) {
      resize(ith_out_node(i)->get_data());
      resize(ith_out_node(i)->get_gradient());
    }

    for (auto &parameter : parameters_) {
      parameter->resize_grad(sample_count);
    }
  }

  /**
   * generate layer from cereal's Archive
   **/
  template <typename InputArchive>
  static std::shared_ptr<layer> load_layer(InputArchive &ia);

  template <typename OutputArchive>
  static void save_layer(OutputArchive &oa, const layer &l);

  template <class Archive>
  void serialize_prolog(Archive &ar);

 protected:
  /** Flag indicating whether the layer/node operations ara paralellized */
  bool parallelize_;
  /** The number of input vectors/edges */
  size_t in_channels_;
  /** The number of output vectors/edges */
  size_t out_channels_;
  /** Vector containing the type of data for inputs */
  std::vector<vector_type> in_type_;
  /** Vector containing the type of data for outputs */
  std::vector<vector_type> out_type_;
  /** The current backend type for operations */
  core::backend_t backend_type_;
  /** The backend instance (deprecated) */
  std::shared_ptr<core::backend> backend_;
  /** Pointer to the device on which the layer/node will run */
  Device *device_ptr_ = nullptr;
  /** Used in update_weight method. Kept as a member variable to reduce
   * frequent memory allocation */
  vec_t weights_diff_;
  /** Used in ``update_parameters`` method. Kept as a member variable
   * to reduce frequent memory allocation */
  Tensor<> parameters_diff_;

  template <typename T, typename Func>
  inline void for_i(T size, Func f, size_t grainsize = 100) {
    tiny_dnn::for_i(parallelize_, size, f, grainsize);
  }

  friend struct serialization_buddy;

 private:
  /** A vector of trainable and constant parameters. */
  std::vector<std::shared_ptr<Parameter>> parameters_;

  /** Pointer to the function for weights initialization */
  std::shared_ptr<parameter_init::function> weight_init_;
  /** Pointer to the function for biases initialization */
  std::shared_ptr<parameter_init::function> bias_init_;

  std::vector<Tensor<> *> fwd_in_data_;
  std::vector<Tensor<> *> fwd_out_data_;
  std::vector<Tensor<> *> bwd_in_data_;
  std::vector<Tensor<> *> bwd_in_grad_;
  std::vector<Tensor<> *> bwd_out_data_;
  std::vector<Tensor<> *> bwd_out_grad_;

  /* @brief Allocates the necessary edge memory in a specific
   * incoming connection.
   *
   * @param i The position to store the previous edge.
   *
   * Graphical explanation:
   *
   *     nullptr -- |edge| -- prev(i) ---- |layer|
   *               nullptr -- prev(i+1) -´
   */
  void alloc_input(size_t i) const {
    // the created incoming edge won't have a previous connection,
    // for this reason first parameter is a nullptr.
    prev_[i] = std::make_shared<edge>(nullptr, in_shape()[i], in_type_[i]);
  }

  /* @brief Allocates the necessary edge memory in a specific
   * outcoming connection.
   *
   * @param i The position to store the next edge.
   *
   * Graphical explanation:
   *
   *     |layer| -- next(i) ---- |edge|
   *             `- next(i+1) -- nullptr
   */
  void alloc_output(size_t i) const {
    // the created outcoming will have the current layer as the
    // previous node.
    next_[i] = std::make_shared<edge>(const_cast<layer *>(this), out_shape()[i],
                                      out_type_[i]);
  }

  /* @brief Creates an edge between the current node and one incoming
   * or previous node.
   *
   * @param i The position to store the previous edge.
   *
   * The method checks if the edge already exists, otherwise we create it
   * and the necessary memory it's allocated. The method returns the pointer
   * to the previous edge.
   */
  edgeptr_t ith_in_node(size_t i) {
    // in case that the  edge doesn't exist, we create it
    if (!prev_[i]) alloc_input(i);
    return prev()[i];
  }

  /* @brief Creates an edge between the current node and one outcoming
   * or next node.
   *
   * @param i The position to store the next edge.
   *
   * The method checks if the edge already exists, otherwise we create it
   * and the necessary memory it's allocated. The method returns the pointer
   * to the next edge.
   */
  edgeptr_t ith_out_node(size_t i) {
    // in case that the  edge doesn't exist, we create it
    if (!next_[i]) alloc_output(i);
    return next()[i];
  }
  edgeptr_t ith_out_node(size_t i) const { return next()[i]; }
};

inline void connect(layer *head,
                    layer *tail,
                    size_t head_index = 0,
                    size_t tail_index = 0) {
  auto out_shape = head->out_shape()[head_index];
  auto in_shape  = tail->in_shape()[tail_index];

  head->setup(false);

  // todo (karandesai) enable shape inferring for all layers
  // currently only possible for activation layers.
  if (in_shape.size() == 0) {
    tail->set_in_shape(out_shape);
    in_shape = out_shape;
  }

  if (out_shape.size() != in_shape.size()) {
    connection_mismatch(*head, *tail);
  }

  if (!head->next_[head_index]) {
    throw nn_error("output edge must not be null");
  }

  tail->prev_[tail_index] = head->next_[head_index];
  tail->prev_[tail_index]->add_next_node(tail);
}

inline layer &operator<<(layer &lhs, layer &rhs) {
  connect(&lhs, &rhs);
  return rhs;
}

template <typename Char, typename CharTraits>
std::basic_ostream<Char, CharTraits> &operator<<(
  std::basic_ostream<Char, CharTraits> &os, const layer &v) {
  v.save(os);
  return os;
}

template <typename Char, typename CharTraits>
std::basic_istream<Char, CharTraits> &operator>>(
  std::basic_istream<Char, CharTraits> &os, layer &v) {
  v.load(os);
  return os;
}

// error message functions

inline void connection_mismatch(const layer &from, const layer &to) {
  std::ostringstream os;

  os << std::endl;
  os << "output size of Nth layer must be equal to input of (N+1)th layer\n";

  os << "layerN:   " << std::setw(12) << from.layer_type()
     << " in:" << from.in_data_size() << "(" << from.in_shape() << "), "
     << "out:" << from.out_data_size() << "(" << from.out_shape() << ")\n";

  os << "layerN+1: " << std::setw(12) << to.layer_type()
     << " in:" << to.in_data_size() << "(" << to.in_shape() << "), "
     << "out:" << to.out_data_size() << "(" << to.out_shape() << ")\n";

  os << from.out_data_size() << " != " << to.in_data_size() << std::endl;
  std::string detail_info = os.str();

  throw nn_error("layer dimension mismatch!" + detail_info);
}

inline void data_mismatch(const layer &layer, const vec_t &data) {
  std::ostringstream os;

  os << std::endl;
  os << "data dimension:    " << data.size() << "\n";
  os << "network dimension: " << layer.in_data_size() << "("
     << layer.layer_type() << ":" << layer.in_shape() << ")\n";

  std::string detail_info = os.str();

  throw nn_error("input dimension mismatch!" + detail_info);
}

inline void pooling_size_mismatch(size_t in_width,
                                  size_t in_height,
                                  size_t pooling_size_x,
                                  size_t pooling_size_y) {
  std::ostringstream os;

  os << std::endl;
  os << "WxH:" << in_width << "x" << in_height << std::endl;
  os << "pooling-size:" << pooling_size_x << "x" << pooling_size_y << std::endl;

  std::string detail_info = os.str();

  throw nn_error("width/height not multiple of pooling size" + detail_info);
}

template <typename T, typename U>
void graph_traverse(layer *root_node, T &&node_callback, U &&edge_callback) {
  std::unordered_set<layer *> visited;
  std::queue<layer *> S;

  S.push(root_node);

  while (!S.empty()) {
    layer *curr = S.front();
    S.pop();
    visited.insert(curr);

    node_callback(*curr);

    auto edges = curr->next();
    for (auto e : edges) {
      if (e != nullptr) edge_callback(*e);
    }

    auto prev = curr->prev_nodes();
    for (auto p : prev) {
      // TODO(nyanp): refactoring
      // which type of refactoring do you have in mind for that?
      layer *l = dynamic_cast<layer *>(p);
      if (visited.find(l) == visited.end()) {
        S.push(l);
      }
    }

    auto next = curr->next_nodes();
    for (auto n : next) {
      // TODO(nyanp): refactoring
      // which type of refactoring do you have in mind for that?
      layer *l = dynamic_cast<layer *>(n);
      if (visited.find(l) == visited.end()) {
        S.push(l);
      }
    }
  }
}

}  // namespace tiny_dnn
