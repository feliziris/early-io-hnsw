/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace faiss {

/** One-query recurrent state for a streaming HNSW LSTM model. */
class HNSWOnnxQuery {
   public:
    ~HNSWOnnxQuery();
    HNSWOnnxQuery(HNSWOnnxQuery&&) noexcept;
    HNSWOnnxQuery& operator=(HNSWOnnxQuery&&) noexcept;

    HNSWOnnxQuery(const HNSWOnnxQuery&) = delete;
    HNSWOnnxQuery& operator=(const HNSWOnnxQuery&) = delete;

    /** Run one timestep. Features are rank-major [k, 6]. */
    std::vector<float> predict(const float* features, size_t feature_count);

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    explicit HNSWOnnxQuery(std::unique_ptr<Impl> impl);
    friend class HNSWOnnxModel;
};

/** Thread-safe ONNX Runtime session shared by concurrent HNSW queries. */
class HNSWOnnxModel {
   public:
    struct Impl;

    /** provider is "cpu" or "cuda" ("gpu" is accepted as a CUDA alias). */
    explicit HNSWOnnxModel(
            const std::string& model_path,
            const std::string& provider = "cpu",
            int device_id = 0);
    ~HNSWOnnxModel();

    HNSWOnnxModel(HNSWOnnxModel&&) noexcept;
    HNSWOnnxModel& operator=(HNSWOnnxModel&&) noexcept;

    HNSWOnnxModel(const HNSWOnnxModel&) = delete;
    HNSWOnnxModel& operator=(const HNSWOnnxModel&) = delete;

    int topk() const;
    int feature_count() const;
    int num_layers() const;
    int hidden_size() const;
    const std::string& model_path() const;
    const std::string& provider() const;
    int device_id() const;

    std::unique_ptr<HNSWOnnxQuery> create_query() const;

    /** True only when Faiss was built with FAISS_ENABLE_ONNX_RUNTIME=ON. */
    static bool runtime_enabled();

   private:
    std::unique_ptr<Impl> impl_;
};

} // namespace faiss
