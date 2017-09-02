/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef CNN_NO_SERIALIZATION
#include <cereal/types/tuple.hpp>
#include <cereal/types/utility.hpp>

#include <H5Cpp.h>
#endif

#include "tiny_dnn/layers/layer.h"
#include "tiny_dnn/optimizers/optimizer.h"
#include "tiny_dnn/util/util.h"

namespace cereal {

template <typename Archive>
void save(Archive &ar, const std::vector<tiny_dnn::layer *> &v) {
#ifndef CNN_NO_SERIALIZATION
  ar(cereal::make_size_tag(static_cast<cereal::size_type>(v.size())));
  for (auto n : v) {
    tiny_dnn::layer::save_layer(ar, *n);
  }
#else
  throw tiny_dnn::nn_error("tiny-dnn was not built with Serialization support");
#endif  // CNN_NO_SERIALIZATION
}

template <typename Archive>
void load(Archive &ar, std::vector<std::shared_ptr<tiny_dnn::layer>> &v) {
#ifndef CNN_NO_SERIALIZATION
  cereal::size_type size;
  ar(cereal::make_size_tag(size));

  for (size_t i = 0; i < size; i++) {
    v.emplace_back(tiny_dnn::layer::load_layer(ar));
  }
#else
  throw tiny_dnn::nn_error("tiny-dnn was not built with Serialization support");
#endif  // CNN_NO_SERIALIZATION
}

}  // namespace cereal

namespace tiny_dnn {

/** basic class of various network types (sequential, multi-in/multi-out).
 *
 * this class holds list of pointer of Node, and provides entry point of
 * forward / backward operations.
 * Node is a computational unit of tiny-dnn (for example, convolution).
 * Currently 2 kinds of implementation are available: sequential and graph.
 *
 * Nodes can accept lvalue, rvalue and shared_ptr forms of node.
 * If given type is rvalue or shared_ptr, nodes create shared_ptr<node> to keep
 * given node alive. If given type is lvalue, tiny-dnn holds raw-pointer only
 * (to avoid double-free).
 *
 *     sequential s;
 *     s.add(fc(100, 200));                          // rvalue, moved into nodes
 *
 *     s.add(std::make_shared<fc>(200, 100));        // shared_ptr, shared by
 *nodes
 *
 *     fc out(100, 10);
 *     softmax sft();
 *     s.add(out);                                   // lvalue, hold raw-pointer
 *only
 *
 **/
class nodes {
 public:
  typedef std::vector<layer *>::iterator iterator;
  typedef std::vector<layer *>::const_iterator const_iterator;

  /**
   * propagate gradient
   * @param first        : gradient of cost function(dE/dy)
   * @param worker_index : id of worker-task
   **/
  virtual void backward(const std::vector<tensor_t> &first) = 0;

  /**
   * @param first input  : data vectors
   * @param worker_index : id of worker-task
   **/
  virtual std::vector<tensor_t> forward(
    const std::vector<tensor_t> &first) = 0;  // NOLINT

  /**
   * update parameters and clear all gradients
   **/
  virtual void update_parameters(optimizer *opt, int batch_size) {
    for (auto l : nodes_) {
      l->update_parameters(opt, batch_size);
    }
  }

  /**
   * setup all weights, must be called before forward/backward
   **/
  virtual void setup(bool reset_weight) {
    for (auto l : nodes_) {
      l->setup(reset_weight);
    }
  }

  void clear_grads() {
    for (auto l : nodes_) {
      l->clear_grads();
    }
  }

  size_t size() const { return nodes_.size(); }
  iterator begin() { return nodes_.begin(); }
  iterator end() { return nodes_.end(); }
  const_iterator begin() const { return nodes_.begin(); }
  const_iterator end() const { return nodes_.end(); }
  layer *operator[](size_t index) { return nodes_[index]; }
  const layer *operator[](size_t index) const { return nodes_[index]; }
  size_t in_data_size() const { return nodes_.front()->in_data_size(); }
  size_t out_data_size() const { return nodes_.back()->out_data_size(); }

  template <typename T>
  const T &at(size_t index) const {
    const T *v = dynamic_cast<const T *>(nodes_[index]);
    if (v) return *v;
    throw nn_error("failed to cast");
  }

  template <typename T>
  T &at(size_t index) {
    T *v = dynamic_cast<T *>(nodes_[index]);
    if (v) return *v;
    throw nn_error("failed to cast");
  }

  // @todo: multiple output
  virtual float_t target_value_min(int out_channel = 0) const {
    CNN_UNREFERENCED_PARAMETER(out_channel);
    return nodes_.back()->out_value_range().first;
  }

  virtual float_t target_value_max(int out_channel = 0) const {
    CNN_UNREFERENCED_PARAMETER(out_channel);
    return nodes_.back()->out_value_range().second;
  }

  void save(std::ostream &os) const {  // NOLINT
    for (auto &l : nodes_) {
      l->save(os);
    }
  }

  void load(std::istream &is) {  // NOLINT
    setup(false);
    for (auto &l : nodes_) {
      l->load(is);
    }
  }

  void label2vec(const label_t *t, size_t num, std::vector<vec_t> &vec) const {
    size_t outdim = out_data_size();

    vec.reserve(num);
    for (size_t i = 0; i < num; i++) {
      assert(t[i] < outdim);
      vec.emplace_back(outdim, target_value_min());
      vec.back()[t[i]] = target_value_max();
    }
  }

  void label2vec(const std::vector<label_t> &labels,
                 std::vector<vec_t> &vec) const {
    return label2vec(&labels[0], labels.size(), vec);
  }

  template <typename OutputArchive>
  void save_model(OutputArchive &oa) const;

  template <typename InputArchive>
  void load_model(InputArchive &ia);

  template <typename OutputArchive>
  void save_weights(OutputArchive &oa) const {
    for (auto n : nodes_) {
      oa(*n);
    }
  }

  template <typename InputArchive>
  void load_weights(InputArchive &ia) {
    for (auto n : nodes_) {
      ia(*n);
    }
  }

#ifndef CNN_NO_SERIALIZATION
  void load_weights(const std::string &filename,
                    const std::vector<std::string> &layer_names) {
    size_t layer_names_idx = 0;
    for (auto n : nodes_) {
      if (n->parameters().size() > 0) {
        n->load(filename, layer_names[layer_names_idx]);
        layer_names_idx++;
      }
    }
  }
#endif

 protected:
  template <typename T>
  void push_back(T &&node) {
    push_back_impl(
      std::forward<T>(node),
      typename std::is_rvalue_reference<decltype(node)>::type());  // NOLINT
  }

  template <typename T>
  void push_back(std::shared_ptr<T> node) {
    own_nodes_.push_back(node);
    nodes_.push_back(own_nodes_.back().get());
  }

  // transform indexing so that it's more suitable for per-layer operations
  // input:  [sample][channel][feature]
  // output: [channel][sample][feature]
  void reorder_for_layerwise_processing(
    const std::vector<tensor_t> &input,
    std::vector<std::vector<const vec_t *>> &output) {
    size_t sample_count  = input.size();
    size_t channel_count = input[0].size();

    output.resize(channel_count);
    for (size_t i = 0; i < channel_count; ++i) {
      output[i].resize(sample_count);
    }

    for (size_t sample = 0; sample < sample_count; ++sample) {
      assert(input[sample].size() == channel_count);
      for (size_t channel = 0; channel < channel_count; ++channel) {
        output[channel][sample] = &input[sample][channel];
      }
    }
  }

  template <typename T>
  void push_back_impl(T &&node, std::true_type) {  // is_rvalue_reference
    own_nodes_.push_back(
      std::make_shared<typename std::remove_reference<T>::type>(
        std::forward<T>(node)));
    nodes_.push_back(own_nodes_.back().get());
  }

  template <typename T>
  void push_back_impl(T &&node, std::false_type) {
    nodes_.push_back(&node);
  }

  /* Nodes which this class has ownership */
  std::vector<std::shared_ptr<layer>> own_nodes_;
  /* List of all nodes which includes own_nodes */
  std::vector<layer *> nodes_;
};

/**
 * single-input, single-output feedforward network
 **/
class sequential : public nodes {
 public:
  void backward(const std::vector<tensor_t> &first) override {
    std::vector<std::vector<const vec_t *>> reordered_grad;
    reorder_for_layerwise_processing(first, reordered_grad);
    assert(reordered_grad.size() == 1);

    nodes_.back()->set_out_grads(&reordered_grad[0], 1);

    for (auto l = nodes_.rbegin(); l != nodes_.rend(); l++) {
      (*l)->backward();
    }
  }

  std::vector<tensor_t> forward(const std::vector<tensor_t> &first) override {
    std::vector<std::vector<const vec_t *>> reordered_data;
    reorder_for_layerwise_processing(first, reordered_data);
    assert(reordered_data.size() == 1);

    nodes_.front()->set_in_data(&reordered_data[0], 1);

    for (auto l : nodes_) {
      l->forward();
    }

    std::vector<const Tensor<> *> out;
    nodes_.back()->output(out);

    return normalize_out(out);
  }

  template <typename T>
  void add(T &&layer) {
    push_back(std::forward<T>(layer));

    if (nodes_.size() != 1) {
      auto head = nodes_[nodes_.size() - 2];
      auto tail = nodes_[nodes_.size() - 1];
      connect(head, tail, 0, 0);
      auto out = head->outputs();
      auto in  = tail->inputs();
    }
    check_connectivity();
  }

  void check_connectivity() {
    for (size_t i = 0; i < nodes_.size() - 1; i++) {
      auto out = nodes_[i]->outputs();
      auto in  = nodes_[i + 1]->inputs();

      if (out[0] != in[0]) {
        throw nn_error("");
      }
    }
  }

  template <typename InputArchive>
  void load_connections(InputArchive &ia) {
    CNN_UNREFERENCED_PARAMETER(ia);
    for (size_t i = 0; i < nodes_.size() - 1; i++) {
      auto head = nodes_[i];
      auto tail = nodes_[i + 1];
      connect(head, tail, 0, 0);
    }
  }

  template <typename OutputArchive>
  void save_connections(OutputArchive &) const {}

 private:
  friend class nodes;

  std::vector<tensor_t> normalize_out(
    const std::vector<const Tensor<> *> &out) {
    // normalize indexing back to [sample][layer][feature]
    std::vector<tensor_t> normalized_output;
    const size_t sample_count = out[0]->shape()[0];
    normalized_output.resize(sample_count, tensor_t(1));

    for (size_t sample = 0; sample < sample_count; ++sample) {
      normalized_output[sample][0] = out[0]->lineToVec(sample);
    }

    return normalized_output;
  }
};

/**
 * generic graph network
 * @todo not implemented
 **/
class graph : public nodes {
 public:
  void backward(const std::vector<tensor_t> &out_grad) override {
    size_t output_channel_count = out_grad[0].size();

    if (output_channel_count != output_layers_.size()) {
      throw nn_error("input size mismatch");
    }

    std::vector<std::vector<const vec_t *>> reordered_grad;
    reorder_for_layerwise_processing(out_grad, reordered_grad);
    assert(reordered_grad.size() == output_channel_count);

    for (size_t i = 0; i < output_channel_count; i++) {
      output_layers_[i]->set_out_grads(&reordered_grad[i], 1);
    }

    for (auto l = nodes_.rbegin(); l != nodes_.rend(); l++) {
      (*l)->backward();
    }
  }

  std::vector<tensor_t> forward(const std::vector<tensor_t> &in_data) override {
    size_t input_data_channel_count = in_data[0].size();

    if (input_data_channel_count != input_layers_.size()) {
      throw nn_error("input size mismatch");
    }

    std::vector<std::vector<const vec_t *>> reordered_data;
    reorder_for_layerwise_processing(in_data, reordered_data);
    assert(reordered_data.size() == input_data_channel_count);

    for (size_t channel_index = 0; channel_index < input_data_channel_count;
         channel_index++) {
      input_layers_[channel_index]->set_in_data(&reordered_data[channel_index],
                                                1);
    }

    for (auto l : nodes_) {
      l->forward();
    }
    return merge_outs();
  }

  void construct(const std::vector<layer *> &input,
                 const std::vector<layer *> &output) {
    std::vector<layer *> sorted;
    std::vector<node *> input_nodes(input.begin(), input.end());
    std::unordered_map<node *, std::vector<uint8_t>> removed_edge;

    // topological-sorting
    while (!input_nodes.empty()) {
      sorted.push_back(dynamic_cast<layer *>(input_nodes.back()));
      input_nodes.pop_back();

      layer *curr              = sorted.back();
      std::vector<node *> next = curr->next_nodes();

      for (size_t i = 0; i < next.size(); i++) {
        if (!next[i]) continue;
        // remove edge between next[i] and current
        if (removed_edge.find(next[i]) == removed_edge.end()) {
          removed_edge[next[i]] =
            std::vector<uint8_t>(next[i]->prev_nodes().size(), 0);
        }

        std::vector<uint8_t> &removed = removed_edge[next[i]];
        removed[find_index(next[i]->prev_nodes(), curr)] = 1;

        if (std::all_of(removed.begin(), removed.end(),
                        [](uint8_t x) { return x == 1; })) {
          input_nodes.push_back(next[i]);
        }
      }
    }

    for (auto &n : sorted) {
      nodes_.push_back(n);
    }

    input_layers_  = input;
    output_layers_ = output;

    setup(false);
  }

 private:
  friend class nodes;

  struct _graph_connection {
    void add_connection(size_t head,
                        size_t tail,
                        size_t head_index,
                        size_t tail_index) {
      if (!is_connected(head, tail, head_index, tail_index)) {
        connections.emplace_back(head, tail, head_index, tail_index);
      }
    }

    bool is_connected(size_t head,
                      size_t tail,
                      size_t head_index,
                      size_t tail_index) const {
      return std::find(connections.begin(), connections.end(),
                       std::make_tuple(head, tail, head_index, tail_index)) !=
             connections.end();
    }

    template <typename Archive>
    void serialize(Archive &ar) {
#ifndef CNN_NO_SERIALIZATION
      ar(CEREAL_NVP(connections), CEREAL_NVP(in_nodes), CEREAL_NVP(out_nodes));
#else
      throw nn_error("TinyDNN was not built with Serialization support");
#endif  // CNN_NO_SERIALIZATION
    }

    std::vector<std::tuple<size_t, size_t, size_t, size_t>> connections;
    std::vector<size_t> in_nodes, out_nodes;
  };

  template <typename OutputArchive>
  void save_connections(OutputArchive &oa) const {
#ifndef CNN_NO_SERIALIZATION
    _graph_connection gc;
    std::unordered_map<node *, size_t> node2id;
    size_t idx = 0;

    for (auto n : nodes_) {
      node2id[n] = idx++;
    }
    for (auto l : input_layers_) {
      gc.in_nodes.push_back(node2id[l]);
    }
    for (auto l : output_layers_) {
      gc.out_nodes.push_back(node2id[l]);
    }

    for (auto l : input_layers_) {
      graph_traverse(l, [=](layer &l) { CNN_UNREFERENCED_PARAMETER(l); },
                     [&](edge &e) {
                       auto next         = e.next();
                       size_t head_index = e.prev()->next_port(e);

                       for (auto n : next) {
                         size_t tail_index = n->prev_port(e);
                         gc.add_connection(node2id[e.prev()], node2id[n],
                                           head_index, tail_index);
                       }
                     });
    }

    oa(cereal::make_nvp("graph", gc));
#else
    throw nn_error("TinyDNN was not built with Serialization support");
#endif  // CNN_NO_SERIALIZATION
  }

  template <typename InputArchive>
  void load_connections(InputArchive &ia) {
#ifndef CNN_NO_SERIALIZATION
    _graph_connection gc;
    ia(cereal::make_nvp("graph", gc));

    for (auto c : gc.connections) {
      size_t head, tail, head_index, tail_index;
      std::tie(head, tail, head_index, tail_index) = c;
      connect(nodes_[head], nodes_[tail], head_index, tail_index);
    }
    for (auto in : gc.in_nodes) {
      input_layers_.push_back(nodes_[in]);
    }
    for (auto out : gc.out_nodes) {
      output_layers_.push_back(nodes_[out]);
    }
#else
    throw nn_error("TinyDNN was not built with Serialization support");
#endif  // CNN_NO_SERIALIZATION
  }

  // normalize indexing back to [sample][layer][feature]
  std::vector<tensor_t> merge_outs() {
    std::vector<tensor_t> merged;
    std::vector<const Tensor<> *> out;
    size_t output_channel_count = output_layers_.size();
    for (size_t output_channel = 0; output_channel < output_channel_count;
         ++output_channel) {
      output_layers_[output_channel]->output(out);

      size_t sample_count = out[0]->shape()[0];
      if (output_channel == 0) {
        assert(merged.empty());
        merged.resize(sample_count, tensor_t(output_channel_count));
      }

      assert(merged.size() == sample_count);

      for (size_t sample = 0; sample < sample_count; ++sample) {
        merged[sample][output_channel] =
          out[0]->subView(TensorSingleIndex(sample), TensorAll()).toVec();
      }
    }
    return merged;
  }

  size_t find_index(const std::vector<node *> &nodes, layer *target) {
    for (size_t i = 0; i < nodes.size(); i++) {
      if (nodes[i] == static_cast<node *>(&*target)) return i;
    }
    throw nn_error("invalid connection");
  }
  std::vector<layer *> input_layers_;
  std::vector<layer *> output_layers_;
};

template <typename OutputArchive>
void nodes::save_model(OutputArchive &oa) const {
#ifndef CNN_NO_SERIALIZATION
  oa(cereal::make_nvp("nodes", nodes_));

  if (typeid(*this) == typeid(sequential)) {
    dynamic_cast<const sequential *>(this)->save_connections(oa);
  } else {
    dynamic_cast<const graph *>(this)->save_connections(oa);
  }
#else
  throw nn_error("TinyDNN was not built with Serialization support");
#endif  // CNN_NO_SERIALIZATION
}

template <typename InputArchive>
void nodes::load_model(InputArchive &ia) {
#ifndef CNN_NO_SERIALIZATION
  own_nodes_.clear();
  nodes_.clear();

  ia(cereal::make_nvp("nodes", own_nodes_));

  for (auto &n : own_nodes_) {
    nodes_.push_back(&*n);
  }

  if (typeid(*this) == typeid(sequential)) {
    dynamic_cast<sequential *>(this)->load_connections(ia);
  } else {
    dynamic_cast<graph *>(this)->load_connections(ia);
  }
#else
  throw nn_error("TinyDNN was not built with Serialization support");
#endif  // CNN_NO_SERIALIZATION
}

}  // namespace tiny_dnn
