/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/HNSWOnnxInference.h>
#include <faiss/impl/HNSWSearchProfile.h>
#include <chrono>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <faiss/impl/FaissAssert.h>

#ifdef FAISS_ENABLE_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif
#ifdef FAISS_ENABLE_HNSW_CUDA_PIPELINE
#include <cuda_runtime_api.h>
#endif

namespace faiss {

#ifdef FAISS_ENABLE_ONNX_RUNTIME

namespace {

#ifdef FAISS_ENABLE_HNSW_CUDA_PIPELINE
void check_cuda(cudaError_t status, const char* operation) {
    FAISS_THROW_IF_NOT_FMT(
            status == cudaSuccess, "%s: %s", operation,
            cudaGetErrorString(status));
}

struct HNSWCudaStream {
    cudaStream_t stream = nullptr;
    explicit HNSWCudaStream(int device) {
        check_cuda(cudaSetDevice(device), "cudaSetDevice");
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags");
    }
    ~HNSWCudaStream() {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
};

struct HNSWCudaHostBuffer {
    float* data = nullptr;
    explicit HNSWCudaHostBuffer(size_t elements) {
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&data),
                                  elements * sizeof(float)), "cudaMallocHost");
    }
    ~HNSWCudaHostBuffer() { cudaFreeHost(data); }
};

struct HNSWCudaEvent {
    cudaEvent_t event = nullptr;
    HNSWCudaEvent() {
        check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                   "cudaEventCreateWithFlags");
    }
    ~HNSWCudaEvent() { cudaEventDestroy(event); }
};
#endif

Ort::Env& onnx_environment() {
    static Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "faiss-hnsw-lstm");
    return environment;
}

bool contains_provider(
        const std::vector<std::string>& providers,
        const std::string& provider) {
    return std::find(providers.begin(), providers.end(), provider) !=
            providers.end();
}

std::string normalize_provider(std::string provider) {
    std::transform(
            provider.begin(),
            provider.end(),
            provider.begin(),
            [](unsigned char character) { return std::tolower(character); });
    if (provider == "gpu") {
        provider = "cuda";
    }
    FAISS_THROW_IF_NOT_FMT(
            provider == "cpu" || provider == "cuda",
            "ONNX provider must be cpu, cuda, or gpu; got %s",
            provider.c_str());
    return provider;
}

Ort::SessionOptions make_session_options(
        const std::string& provider,
        int device_id,
        void* compute_stream = nullptr) {
    FAISS_THROW_IF_NOT_FMT(
            device_id >= 0,
            "ONNX device ID must be non-negative, got %d",
            device_id);
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (provider == "cuda") {
        const std::vector<std::string> available = Ort::GetAvailableProviders();
        FAISS_THROW_IF_NOT_MSG(
                contains_provider(available, "CUDAExecutionProvider"),
                "CUDAExecutionProvider is not available in this ONNX Runtime");
        Ort::CUDAProviderOptions cuda_options;
        cuda_options.Update({
                {"device_id", std::to_string(device_id)},
                {"do_copy_in_default_stream", "1"},
        });
        if (compute_stream) {
            cuda_options.UpdateWithValue("user_compute_stream", compute_stream);
        }
        options.AppendExecutionProvider_CUDA_V2(*cuda_options);
        // A requested GPU run must not silently move unsupported model nodes
        // to CPU. Host/device copy nodes inserted by ORT remain permitted.
        options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    }
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
    std::string provider;
    int device_id = 0;
    size_t pipeline_depth = 0;
#ifdef FAISS_ENABLE_HNSW_CUDA_PIPELINE
    // Session is destroyed before its user stream. All query jobs must finish
    // before the caller destroys the model.
    std::unique_ptr<HNSWCudaStream> pipeline_stream;
    mutable std::mutex pipeline_mutex;
    std::unique_ptr<Ort::Allocator> pipeline_pinned_allocator;
#endif
    // ONNX Runtime permits concurrent Run calls on one session, but its C++
    // wrapper exposes Run as non-const.
    mutable Ort::Session session;
    Ort::MemoryInfo memory_info;
    std::unique_ptr<Ort::MemoryInfo> gpu_memory_info;
    std::array<int64_t, 3> feature_shape{};
    std::array<int64_t, 3> state_shape{};
    int topk = 0;
    int feature_count = 0;
    int num_layers = 0;
    int hidden_size = 0;

    explicit Impl(
            const std::string& model_path,
            const std::string& provider_name,
            int requested_device_id,
            size_t requested_depth)
            : path(model_path),
              provider(normalize_provider(provider_name)),
              device_id(requested_device_id),
              pipeline_depth(requested_depth),
#ifdef FAISS_ENABLE_HNSW_CUDA_PIPELINE
              pipeline_stream(requested_depth
                      ? std::make_unique<HNSWCudaStream>(requested_device_id)
                      : nullptr),
#endif
              session(
                      onnx_environment(),
                      model_path.c_str(),
                      make_session_options(provider, device_id
#ifdef FAISS_ENABLE_HNSW_CUDA_PIPELINE
                              , pipeline_stream ? pipeline_stream->stream : nullptr
#endif
                              )),
              memory_info(Ort::MemoryInfo::CreateCpu(
                      OrtArenaAllocator,
                      OrtMemTypeDefault)) {
        FAISS_THROW_IF_NOT_MSG(
                !pipeline_depth || provider == "cuda",
                "GPU pipeline requires the cuda provider");
        FAISS_THROW_IF_NOT_MSG(pipeline_depth <= 1024,
                              "GPU pipeline depth must not exceed 1024");
#ifndef FAISS_ENABLE_HNSW_CUDA_PIPELINE
        FAISS_THROW_IF_NOT_MSG(!pipeline_depth,
                              "Rebuild with FAISS_ENABLE_HNSW_CUDA_PIPELINE=ON");
#endif
        if (provider == "cuda") {
            gpu_memory_info = std::make_unique<Ort::MemoryInfo>(
                    "Cuda",
                    OrtDeviceAllocator,
                    device_id,
                    OrtMemTypeDefault);
#ifdef FAISS_ENABLE_HNSW_CUDA_PIPELINE
            if (pipeline_stream) {
                Ort::MemoryInfo pinned_info("CudaPinned", OrtDeviceAllocator,
                                            device_id, OrtMemTypeCPUOutput);
                pipeline_pinned_allocator = std::make_unique<Ort::Allocator>(
                        session, pinned_info);
            }
#endif
        }
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

    ~Impl() {
#ifdef FAISS_ENABLE_HNSW_CUDA_PIPELINE
        if (pipeline_stream) {
            // ORT can enqueue CPU-buffer-release callbacks during session
            // teardown. Keep its pinned arena alive until those callbacks
            // complete, then free pinned memory on this CPU thread rather
            // than from a CUDA host callback (where cudaFreeHost is illegal).
            cudaSetDevice(device_id);
            cudaStreamSynchronize(pipeline_stream->stream);
            auto pinned_guard = std::move(pipeline_pinned_allocator);
            session = Ort::Session{nullptr};
            cudaStreamSynchronize(pipeline_stream->stream);
            pinned_guard.reset();
        }
#endif
    }
};

struct HNSWOnnxQuery::Impl {
    const HNSWOnnxModel::Impl& model;
    std::vector<float> hidden;
    std::vector<float> cell;
    std::unique_ptr<Ort::Allocator> gpu_allocator;
    std::vector<Ort::Value> hidden_gpu;
    std::vector<Ort::Value> cell_gpu;
    size_t current_gpu_state = 0;
    bool has_gpu_state = false;
#ifdef FAISS_ENABLE_HNSW_CUDA_PIPELINE
    struct PipelineSlot {
        HNSWCudaHostBuffer host_features;
        HNSWCudaHostBuffer host_probabilities;
        Ort::Value device_features;
        Ort::Value device_probabilities;
        Ort::IoBinding binding;
        HNSWCudaEvent done;

        PipelineSlot(const HNSWOnnxModel::Impl& model, Ort::Allocator& allocator)
                : host_features(model.topk * model.feature_count),
                  host_probabilities(model.topk),
                  device_features(Ort::Value::CreateTensor<float>(
                          allocator, model.feature_shape.data(), 3)),
                  device_probabilities(Ort::Value::CreateTensor<float>(
                          allocator, model.feature_shape.data(), 2)),
                  binding(model.session) {}
    };
    std::vector<std::unique_ptr<PipelineSlot>> pipeline_slots;
    size_t pipeline_submitted = 0;
    size_t pipeline_collected = 0;
#endif

    explicit Impl(const HNSWOnnxModel::Impl& model_in)
            : model(model_in),
              hidden(
                      static_cast<size_t>(model.num_layers) *
                              model.hidden_size,
                      0.0f),
              cell(hidden.size(), 0.0f) {
        if (model.provider == "cuda") {
#ifdef FAISS_ENABLE_HNSW_CUDA_PIPELINE
            check_cuda(cudaSetDevice(model.device_id), "cudaSetDevice");
#endif
            gpu_allocator = std::make_unique<Ort::Allocator>(
                    model.session, *model.gpu_memory_info);
            hidden_gpu.reserve(2);
            cell_gpu.reserve(2);
            for (size_t buffer = 0; buffer < 2; ++buffer) {
                hidden_gpu.emplace_back(Ort::Value::CreateTensor<float>(
                        *gpu_allocator,
                        model.state_shape.data(),
                        model.state_shape.size()));
                cell_gpu.emplace_back(Ort::Value::CreateTensor<float>(
                        *gpu_allocator,
                        model.state_shape.data(),
                        model.state_shape.size()));
            }
#ifdef FAISS_ENABLE_HNSW_CUDA_PIPELINE
            if (model.pipeline_depth) {
                for (size_t slot = 0; slot < model.pipeline_depth; ++slot) {
                    pipeline_slots.push_back(std::make_unique<PipelineSlot>(
                            model, *gpu_allocator));
                }
                std::lock_guard<std::mutex> lock(model.pipeline_mutex);
                check_cuda(cudaMemsetAsync(
                        hidden_gpu[0].GetTensorMutableData<float>(), 0,
                        hidden.size() * sizeof(float), model.pipeline_stream->stream),
                        "initialize hidden state");
                check_cuda(cudaMemsetAsync(
                        cell_gpu[0].GetTensorMutableData<float>(), 0,
                        cell.size() * sizeof(float), model.pipeline_stream->stream),
                        "initialize cell state");
                has_gpu_state = true;
            }
#endif
        }
    }

    ~Impl() {
#ifdef FAISS_ENABLE_HNSW_CUDA_PIPELINE
        // Also protects outstanding buffers on exceptions or early exit.
        if (model.pipeline_stream) {
            cudaSetDevice(model.device_id);
            cudaStreamSynchronize(model.pipeline_stream->stream);
        }
#endif
    }

};

HNSWOnnxModel::HNSWOnnxModel(
        const std::string& model_path,
        const std::string& provider,
        int device_id,
        size_t gpu_pipeline_depth)
        : impl_(std::make_unique<Impl>(model_path, provider, device_id,
                                      gpu_pipeline_depth)) {}

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

const std::string& HNSWOnnxModel::provider() const {
    return impl_->provider;
}

int HNSWOnnxModel::device_id() const {
    return impl_->device_id;
}

size_t HNSWOnnxModel::gpu_pipeline_depth() const {
    return impl_->pipeline_depth;
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

size_t HNSWOnnxQuery::gpu_pipeline_depth() const {
    return impl_->model.pipeline_depth;
}

size_t HNSWOnnxQuery::gpu_pending() const {
#ifdef FAISS_ENABLE_HNSW_CUDA_PIPELINE
    return impl_->pipeline_submitted - impl_->pipeline_collected;
#else
    return 0;
#endif
}

void HNSWOnnxQuery::submit_gpu(const float* features, size_t feature_count,
                             HNSWOnnxPredictProfile* profile) {
#ifdef FAISS_ENABLE_HNSW_CUDA_PIPELINE
    using Clock = std::chrono::steady_clock;
    const auto started = profile ? Clock::now() : Clock::time_point{};
    const auto& model = impl_->model;
    FAISS_THROW_IF_NOT_MSG(model.pipeline_depth, "CUDA pipeline is disabled");
    FAISS_THROW_IF_NOT_MSG(gpu_pending() < model.pipeline_depth,
                          "CUDA pipeline slots are full");
    FAISS_THROW_IF_NOT_MSG(feature_count ==
                          static_cast<size_t>(model.topk) * model.feature_count,
                          "CUDA pipeline feature count mismatch");
    check_cuda(cudaSetDevice(model.device_id), "cudaSetDevice");
    auto& slot = *impl_->pipeline_slots[
            impl_->pipeline_submitted % model.pipeline_depth];
    std::memcpy(slot.host_features.data, features, feature_count * sizeof(float));
    const size_t next = 1 - impl_->current_gpu_state;
    auto& binding = slot.binding;
    binding.BindInput("features", slot.device_features);
    binding.BindInput("hidden_in", impl_->hidden_gpu[impl_->current_gpu_state]);
    binding.BindInput("cell_in", impl_->cell_gpu[impl_->current_gpu_state]);
    binding.BindOutput("probabilities", slot.device_probabilities);
    binding.BindOutput("hidden_out", impl_->hidden_gpu[next]);
    binding.BindOutput("cell_out", impl_->cell_gpu[next]);
    Ort::RunOptions options;
    options.AddConfigEntry("disable_synchronize_execution_providers", "1");
    const auto prepared = profile ? Clock::now() : Clock::time_point{};
    Clock::time_point run_started, run_ended, copies_ended;
    {
        // One ordered stream ensures h/c from t are ready before t+1 reads
        // them. Serialize host submissions on a session shared by queries.
        std::lock_guard<std::mutex> lock(model.pipeline_mutex);
        const auto stream = model.pipeline_stream->stream;
        check_cuda(cudaMemcpyAsync(
                slot.device_features.GetTensorMutableData<float>(),
                slot.host_features.data, feature_count * sizeof(float),
                cudaMemcpyHostToDevice, stream), "copy features to GPU");
        if (profile) run_started = Clock::now();
        model.session.Run(options, binding);
        if (profile) run_ended = Clock::now();
        check_cuda(cudaMemcpyAsync(
                slot.host_probabilities.data,
                slot.device_probabilities.GetTensorData<float>(),
                model.topk * sizeof(float), cudaMemcpyDeviceToHost, stream),
                "copy probabilities to pinned CPU buffer");
        check_cuda(cudaEventRecord(slot.done.event, stream), "record completion");
        if (profile) copies_ended = Clock::now();
    }
    impl_->current_gpu_state = next;
    ++impl_->pipeline_submitted;
    if (profile) {
        const auto ended = Clock::now();
        const auto ms = [](auto begin, auto end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        profile->input_tensor_ms = ms(started, prepared);
        profile->session_run_ms = ms(run_started, run_ended);
        profile->transfer_enqueue_ms = ms(prepared, run_started) + ms(run_ended, copies_ended);
        profile->state_update_ms = ms(copies_ended, ended);
    }
#else
    FAISS_THROW_MSG("Rebuild with FAISS_ENABLE_HNSW_CUDA_PIPELINE=ON");
#endif
}

bool HNSWOnnxQuery::collect_gpu(std::vector<float>& probabilities, bool wait,
                              HNSWOnnxPredictProfile* profile) {
#ifdef FAISS_ENABLE_HNSW_CUDA_PIPELINE
    using Clock = std::chrono::steady_clock;
    const auto started = profile ? Clock::now() : Clock::time_point{};
    if (!gpu_pending()) return false;
    const auto& model = impl_->model;
    check_cuda(cudaSetDevice(model.device_id), "cudaSetDevice");
    auto& slot = *impl_->pipeline_slots[
            impl_->pipeline_collected % model.pipeline_depth];
    if (wait) {
        check_cuda(cudaEventSynchronize(slot.done.event), "wait for GPU result");
    } else {
        const auto status = cudaEventQuery(slot.done.event);
        if (status == cudaErrorNotReady) return false;
        check_cuda(status, "query GPU result");
    }
    const auto ready = profile ? Clock::now() : Clock::time_point{};
    probabilities.assign(slot.host_probabilities.data,
                         slot.host_probabilities.data + model.topk);
    ++impl_->pipeline_collected;
    if (profile) {
        const auto ended = Clock::now();
        profile->collect_wait_ms =
                std::chrono::duration<double, std::milli>(ready - started).count();
        profile->probability_copy_ms =
                std::chrono::duration<double, std::milli>(ended - ready).count();
    }
    return true;
#else
    FAISS_THROW_MSG("Rebuild with FAISS_ENABLE_HNSW_CUDA_PIPELINE=ON");
#endif
}

std::vector<float> HNSWOnnxQuery::predict(
        const float* features,
        size_t feature_count,
        HNSWOnnxPredictProfile* profile) {
    using Clock = std::chrono::steady_clock;
    const auto input_started = profile ? Clock::now() : Clock::time_point{};
    const HNSWOnnxModel::Impl& model = impl_->model;
    FAISS_THROW_IF_NOT_MSG(!model.pipeline_depth,
                          "Use submit_gpu()/collect_gpu() for a pipeline model");
    FAISS_THROW_IF_NOT_MSG(
            !profile || model.provider == "cpu",
            "Detailed predict() profiling currently supports the CPU provider");
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
    if (model.provider == "cuda") {
        std::vector<float> probabilities(static_cast<size_t>(model.topk));
        Ort::Value probability_tensor = Ort::Value::CreateTensor<float>(
                model.memory_info,
                probabilities.data(),
                probabilities.size(),
                model.feature_shape.data(),
                2);
        Ort::Value initial_hidden = Ort::Value::CreateTensor<float>(
                model.memory_info,
                impl_->hidden.data(),
                impl_->hidden.size(),
                model.state_shape.data(),
                model.state_shape.size());
        Ort::Value initial_cell = Ort::Value::CreateTensor<float>(
                model.memory_info,
                impl_->cell.data(),
                impl_->cell.size(),
                model.state_shape.data(),
                model.state_shape.size());

        const size_t next_state = impl_->has_gpu_state
                ? 1 - impl_->current_gpu_state
                : 0;
        Ort::IoBinding binding(model.session);
        binding.BindInput("features", feature_tensor);
        binding.BindInput(
                "hidden_in",
                impl_->has_gpu_state
                        ? impl_->hidden_gpu[impl_->current_gpu_state]
                        : initial_hidden);
        binding.BindInput(
                "cell_in",
                impl_->has_gpu_state
                        ? impl_->cell_gpu[impl_->current_gpu_state]
                        : initial_cell);
        binding.BindOutput("probabilities", probability_tensor);
        binding.BindOutput("hidden_out", impl_->hidden_gpu[next_state]);
        binding.BindOutput("cell_out", impl_->cell_gpu[next_state]);
        model.session.Run(Ort::RunOptions{nullptr}, binding);
        binding.SynchronizeOutputs();

        impl_->current_gpu_state = next_state;
        impl_->has_gpu_state = true;
        return probabilities;
    }

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
    const auto run_started = profile ? Clock::now() : Clock::time_point{};
    std::vector<Ort::Value> outputs = model.session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            inputs.data(),
            inputs.size(),
            output_names.data(),
            output_names.size());
    const auto run_ended = profile ? Clock::now() : Clock::time_point{};

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
    const auto metadata_ended = profile ? Clock::now() : Clock::time_point{};
    std::vector<float> probabilities(
            probability_data, probability_data + probability_count);
    const auto copy_ended = profile ? Clock::now() : Clock::time_point{};
    std::copy(hidden_data, hidden_data + state_count, impl_->hidden.begin());
    std::copy(cell_data, cell_data + state_count, impl_->cell.begin());
    if (profile) {
        const auto state_ended = Clock::now();
        const auto ms = [](auto begin, auto end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        profile->input_tensor_ms = ms(input_started, run_started);
        profile->session_run_ms = ms(run_started, run_ended);
        profile->output_metadata_ms = ms(run_ended, metadata_ended);
        profile->probability_copy_ms = ms(metadata_ended, copy_ended);
        profile->state_update_ms = ms(copy_ended, state_ended);
    }
    return probabilities;
}

#else

struct HNSWOnnxModel::Impl {};
struct HNSWOnnxQuery::Impl {};

HNSWOnnxModel::HNSWOnnxModel(
        const std::string&, const std::string&, int, size_t) {
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
const std::string& HNSWOnnxModel::provider() const {
    static const std::string empty;
    return empty;
}
int HNSWOnnxModel::device_id() const {
    return -1;
}
size_t HNSWOnnxModel::gpu_pipeline_depth() const { return 0; }
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
size_t HNSWOnnxQuery::gpu_pipeline_depth() const { return 0; }
size_t HNSWOnnxQuery::gpu_pending() const { return 0; }
void HNSWOnnxQuery::submit_gpu(const float*, size_t, HNSWOnnxPredictProfile*) {
    FAISS_THROW_MSG("Faiss was built without ONNX Runtime");
}
bool HNSWOnnxQuery::collect_gpu(std::vector<float>&, bool, HNSWOnnxPredictProfile*) {
    FAISS_THROW_MSG("Faiss was built without ONNX Runtime");
}
std::vector<float> HNSWOnnxQuery::predict(
        const float*, size_t, HNSWOnnxPredictProfile*) {
    FAISS_THROW_MSG("Faiss was built without ONNX Runtime");
}

#endif

} // namespace faiss
