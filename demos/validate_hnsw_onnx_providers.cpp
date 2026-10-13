/* Validate recurrent HNSW ONNX inference across the CPU and CUDA providers. */

#include <faiss/impl/HNSWOnnxInference.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> make_features(int topk, int feature_count, int timestep) {
    std::vector<float> features(
            static_cast<size_t>(topk) * feature_count);
    for (size_t index = 0; index < features.size(); ++index) {
        const float phase = static_cast<float>(
                (index + 1) * 0.017 + (timestep + 1) * 0.031);
        features[index] = 0.5f + 0.25f * std::sin(phase) +
                0.1f * std::cos(phase * 0.37f);
    }
    return features;
}

void warm_up(faiss::HNSWOnnxModel& model, int timesteps) {
    auto query = model.create_query();
    for (int timestep = 0; timestep < timesteps; ++timestep) {
        const std::vector<float> features = make_features(
                model.topk(), model.feature_count(), timestep);
        query->predict(features.data(), features.size());
    }
}

struct RunResult {
    std::vector<float> probabilities;
    double seconds = 0.0;
};

RunResult run(faiss::HNSWOnnxModel& model, int timesteps) {
    auto query = model.create_query();
    RunResult result;
    result.probabilities.reserve(
            static_cast<size_t>(timesteps) * model.topk());
    const auto started = std::chrono::steady_clock::now();
    for (int timestep = 0; timestep < timesteps; ++timestep) {
        const std::vector<float> features = make_features(
                model.topk(), model.feature_count(), timestep);
        std::vector<float> probabilities =
                query->predict(features.data(), features.size());
        result.probabilities.insert(
                result.probabilities.end(),
                probabilities.begin(),
                probabilities.end());
    }
    result.seconds = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - started)
                             .count();
    return result;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: " << argv[0]
                  << " MODEL.onnx [timesteps=500] [cuda-device-id=0]\n";
        return 2;
    }

    try {
        const int timesteps = argc >= 3 ? std::stoi(argv[2]) : 500;
        const int device_id = argc >= 4 ? std::stoi(argv[3]) : 0;
        if (timesteps <= 0 || device_id < 0) {
            throw std::runtime_error(
                    "timesteps must be positive and device ID non-negative");
        }

        std::cerr << "validation_stage=create_cpu_session\n";
        faiss::HNSWOnnxModel cpu(argv[1], "cpu", device_id);
        std::cerr << "validation_stage=create_cuda_session\n";
        faiss::HNSWOnnxModel cuda(argv[1], "cuda", device_id);
        std::cerr << "validation_stage=warmup_cpu\n";
        warm_up(cpu, 8);
        std::cerr << "validation_stage=warmup_cuda\n";
        warm_up(cuda, 8);

        std::cerr << "validation_stage=measure_cpu\n";
        const RunResult cpu_result = run(cpu, timesteps);
        std::cerr << "validation_stage=measure_cuda\n";
        const RunResult cuda_result = run(cuda, timesteps);
        if (cpu_result.probabilities.size() !=
            cuda_result.probabilities.size()) {
            throw std::runtime_error("CPU/CUDA output size mismatch");
        }

        float max_abs_error = 0.0f;
        double mean_abs_error = 0.0;
        for (size_t index = 0; index < cpu_result.probabilities.size();
             ++index) {
            const float error = std::abs(
                    cpu_result.probabilities[index] -
                    cuda_result.probabilities[index]);
            max_abs_error = std::max(max_abs_error, error);
            mean_abs_error += error;
        }
        mean_abs_error /= cpu_result.probabilities.size();
        constexpr float tolerance = 1e-4f;

        std::cout << std::fixed << std::setprecision(6)
                  << "model=" << argv[1] << '\n'
                  << "topk=" << cpu.topk() << '\n'
                  << "timesteps=" << timesteps << '\n'
                  << "cpu_provider=" << cpu.provider() << '\n'
                  << "cuda_provider=" << cuda.provider() << '\n'
                  << "cuda_device_id=" << cuda.device_id() << '\n'
                  << "cpu_total_ms=" << cpu_result.seconds * 1000.0 << '\n'
                  << "cuda_total_ms=" << cuda_result.seconds * 1000.0 << '\n'
                  << "cpu_mean_ms_per_timestep="
                  << cpu_result.seconds * 1000.0 / timesteps << '\n'
                  << "cuda_mean_ms_per_timestep="
                  << cuda_result.seconds * 1000.0 / timesteps << '\n'
                  << "mean_abs_error=" << mean_abs_error << '\n'
                  << "max_abs_error=" << max_abs_error << '\n'
                  << "tolerance=" << tolerance << '\n'
                  << "validation_passed=" << (max_abs_error <= tolerance)
                  << '\n';
        return max_abs_error <= tolerance ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
