/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/HNSWOnnxInference.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <faiss/impl/FaissAssert.h>

#ifdef FAISS_ENABLE_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace faiss {

#ifdef FAISS_ENABLE_ONNX_RUNTIME

namespace {

Ort::Env& onnx_environment() {
    static Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "faiss-hnsw-lstm");
    return environment;
}

Ort::SessionOptions make_session_options() {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return options;
}

std::vector<int64_t> tensor_shape(const Ort::Session& session, size_t input) {
    return session.GetInputTypeInfo(input)
            .GetTensorTypeAndShapeInfo()
            .GetShape();
}

void require_shape(
        const std::vector<int64_t>& shape,
        size_t rank,
        const char* name) {
    FAISS_THROW_IF_NOT_FMT(
            shape.size() == rank,
            "ONNX input %s must have rank %zu, got %zu",
            name,
            rank,
            shape.size());
    for (int64_t dimension : shape) {
        FAISS_THROW_IF_NOT_FMT(
                dimension > 0,
                "ONNX input %s must use fixed positive dimensions",
                name);
    }
}

} // namespace

struct HNSWOnnxModel::Impl {
    std::string path;
    // ONNX Runtime permits concurrent Run calls on one session, but its C++
    // wrapper exposes Run as non-const.
    mutable Ort::Session session;
    Ort::MemoryInfo memory_info;
    std::array<int64_t, 3> feature_shape{};
    std::array<int64_t, 3> state_shape{};
    int topk = 0;
    int feature_count = 0;
    int num_layers = 0;
    int hidden_size = 0;

    explicit Impl(const std::string& model_path)
            : path(model_path),
              session(
                      onnx_environment(),
                      model_path.c_str(),
                      make_session_options()),
              memory_info(Ort::MemoryInfo::CreateCpu(
                      OrtArenaAllocator,
                      OrtMemTypeDefault)) {
        FAISS_THROW_IF_NOT_FMT(
                session.GetInputCount() == 3,
                "HNSW ONNX model must have 3 inputs, got %zu",
                session.GetInputCount());
        FAISS_THROW_IF_NOT_FMT(
                session.GetOutputCount() == 3,
                "HNSW ONNX model must have 3 outputs, got %zu",
                session.GetOutputCount());

        const std::vector<int64_t> features = tensor_shape(session, 0);
        const std::vector<int64_t> hidden = tensor_shape(session, 1);
        const std::vector<int64_t> cell = tensor_shape(session, 2);
        require_shape(features, 3, "features");
        require_shape(hidden, 3, "hidden_in");
        require_shape(cell, 3, "cell_in");
        FAISS_THROW_IF_NOT_MSG(
                hidden == cell,
                "ONNX hidden_in and cell_in shapes must be identical");
        FAISS_THROW_IF_NOT_MSG(
                features[0] == 1 && hidden[1] == 1,
                "HNSW ONNX inference currently requires batch size 1");

        std::copy(features.begin(), features.end(), feature_shape.begin());
        std::copy(hidden.begin(), hidden.end(), state_shape.begin());
        topk = static_cast<int>(features[1]);
        feature_count = static_cast<int>(features[2]);
        num_layers = static_cast<int>(hidden[0]);
        hidden_size = static_cast<int>(hidden[2]);
        FAISS_THROW_IF_NOT_MSG(
                feature_count == 6,
                "HNSW ONNX model must accept exactly 6 features per rank");
    }
};

struct HNSWOnnxQuery::Impl {
    const HNSWOnnxModel::Impl& model;
    std::vector<float> hidden;
    std::vector<float> cell;

    explicit Impl(const HNSWOnnxModel::Impl& model_in)
            : model(model_in),
              hidden(
                      static_cast<size_t>(model.num_layers) *
                              model.hidden_size,
                      0.0f),
              cell(hidden.size(), 0.0f) {}
};

HNSWOnnxModel::HNSWOnnxModel(const std::string& model_path)
        : impl_(std::make_unique<Impl>(model_path)) {}

HNSWOnnxModel::~HNSWOnnxModel() = default;
HNSWOnnxModel::HNSWOnnxModel(HNSWOnnxModel&&) noexcept = default;
HNSWOnnxModel& HNSWOnnxModel::operator=(HNSWOnnxModel&&) noexcept = default;

int HNSWOnnxModel::topk() const {
    return impl_->topk;
}

int HNSWOnnxModel::feature_count() const {
    return impl_->feature_count;
}

int HNSWOnnxModel::num_layers() const {
    return impl_->num_layers;
}

int HNSWOnnxModel::hidden_size() const {
    return impl_->hidden_size;
}

const std::string& HNSWOnnxModel::model_path() const {
    return impl_->path;
}

std::unique_ptr<HNSWOnnxQuery> HNSWOnnxModel::create_query() const {
    return std::unique_ptr<HNSWOnnxQuery>(
            new HNSWOnnxQuery(std::make_unique<HNSWOnnxQuery::Impl>(*impl_)));
}

bool HNSWOnnxModel::runtime_enabled() {
    return true;
}

HNSWOnnxQuery::HNSWOnnxQuery(std::unique_ptr<Impl> impl)
        : impl_(std::move(impl)) {}
HNSWOnnxQuery::~HNSWOnnxQuery() = default;
HNSWOnnxQuery::HNSWOnnxQuery(HNSWOnnxQuery&&) noexcept = default;
HNSWOnnxQuery& HNSWOnnxQuery::operator=(HNSWOnnxQuery&&) noexcept = default;

std::vector<float> HNSWOnnxQuery::predict(
        const float* features,
        size_t feature_count) {
    const HNSWOnnxModel::Impl& model = impl_->model;
    const size_t expected_features =
            static_cast<size_t>(model.topk) * model.feature_count;
    FAISS_THROW_IF_NOT_FMT(
            feature_count == expected_features,
            "HNSW ONNX timestep has %zu feature values, expected %zu",
            feature_count,
            expected_features);

    Ort::Value feature_tensor = Ort::Value::CreateTensor<float>(
            model.memory_info,
            const_cast<float*>(features),
            feature_count,
            model.feature_shape.data(),
            model.feature_shape.size());
    Ort::Value hidden_tensor = Ort::Value::CreateTensor<float>(
            model.memory_info,
            impl_->hidden.data(),
            impl_->hidden.size(),
            model.state_shape.data(),
            model.state_shape.size());
    Ort::Value cell_tensor = Ort::Value::CreateTensor<float>(
            model.memory_info,
            impl_->cell.data(),
            impl_->cell.size(),
            model.state_shape.data(),
            model.state_shape.size());

    std::array<Ort::Value, 3> inputs = {
            std::move(feature_tensor),
            std::move(hidden_tensor),
            std::move(cell_tensor)};
    constexpr std::array<const char*, 3> input_names = {
            "features", "hidden_in", "cell_in"};
    constexpr std::array<const char*, 3> output_names = {
            "probabilities", "hidden_out", "cell_out"};
    std::vector<Ort::Value> outputs = model.session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            inputs.data(),
            inputs.size(),
            output_names.data(),
            output_names.size());

    const size_t probability_count = outputs[0]
                                             .GetTensorTypeAndShapeInfo()
                                             .GetElementCount();
    const size_t state_count = outputs[1]
                                       .GetTensorTypeAndShapeInfo()
                                       .GetElementCount();
    FAISS_THROW_IF_NOT_MSG(
            probability_count == static_cast<size_t>(model.topk),
            "ONNX probability output size does not match top-k");
    FAISS_THROW_IF_NOT_MSG(
            state_count == impl_->hidden.size() &&
                    outputs[2]
                                    .GetTensorTypeAndShapeInfo()
                                    .GetElementCount() == state_count,
            "ONNX recurrent output state size mismatch");

    const float* probability_data = outputs[0].GetTensorData<float>();
    const float* hidden_data = outputs[1].GetTensorData<float>();
    const float* cell_data = outputs[2].GetTensorData<float>();
    std::vector<float> probabilities(
            probability_data, probability_data + probability_count);
    std::copy(hidden_data, hidden_data + state_count, impl_->hidden.begin());
    std::copy(cell_data, cell_data + state_count, impl_->cell.begin());
    return probabilities;
}

#else

struct HNSWOnnxModel::Impl {};
struct HNSWOnnxQuery::Impl {};

HNSWOnnxModel::HNSWOnnxModel(const std::string&) {
    FAISS_THROW_MSG(
            "Faiss was built without ONNX Runtime; configure with "
            "FAISS_ENABLE_ONNX_RUNTIME=ON");
}
HNSWOnnxModel::~HNSWOnnxModel() = default;
HNSWOnnxModel::HNSWOnnxModel(HNSWOnnxModel&&) noexcept = default;
HNSWOnnxModel& HNSWOnnxModel::operator=(HNSWOnnxModel&&) noexcept = default;
int HNSWOnnxModel::topk() const {
    return 0;
}
int HNSWOnnxModel::feature_count() const {
    return 0;
}
int HNSWOnnxModel::num_layers() const {
    return 0;
}
int HNSWOnnxModel::hidden_size() const {
    return 0;
}
const std::string& HNSWOnnxModel::model_path() const {
    static const std::string empty;
    return empty;
}
std::unique_ptr<HNSWOnnxQuery> HNSWOnnxModel::create_query() const {
    FAISS_THROW_MSG("Faiss was built without ONNX Runtime");
}
bool HNSWOnnxModel::runtime_enabled() {
    return false;
}

HNSWOnnxQuery::HNSWOnnxQuery(std::unique_ptr<Impl> impl)
        : impl_(std::move(impl)) {}
HNSWOnnxQuery::~HNSWOnnxQuery() = default;
HNSWOnnxQuery::HNSWOnnxQuery(HNSWOnnxQuery&&) noexcept = default;
HNSWOnnxQuery& HNSWOnnxQuery::operator=(HNSWOnnxQuery&&) noexcept = default;
std::vector<float> HNSWOnnxQuery::predict(const float*, size_t) {
    FAISS_THROW_MSG("Faiss was built without ONNX Runtime");
}

#endif

} // namespace faiss
