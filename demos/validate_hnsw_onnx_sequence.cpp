/* Compare streaming CPU inference against reference float32 probabilities.
 * Usage: validate_hnsw_onnx_sequence MODEL.onnx FEATURES.f32 EXPECTED.f32
 * Features: [T, k, 6]; expected probabilities: [T, k], native float32.
 */
#include <faiss/impl/HNSWOnnxInference.h>
#include <faiss/impl/HNSWSearchProfile.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
std::vector<float> read_floats(const char* path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open reference file");
    const auto bytes = stream.tellg();
    if (bytes <= 0 || bytes % sizeof(float)) {
        throw std::runtime_error("invalid float32 reference file size");
    }
    std::vector<float> data(static_cast<size_t>(bytes) / sizeof(float));
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char*>(data.data()), bytes)) {
        throw std::runtime_error("cannot read reference file");
    }
    return data;
}
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: " << argv[0]
                  << " MODEL.onnx FEATURES.f32 EXPECTED.f32\n";
        return 2;
    }
    try {
        faiss::HNSWOnnxModel model(argv[1], "cpu");
        const auto features = read_floats(argv[2]);
        const auto expected = read_floats(argv[3]);
        const size_t width = model.topk() * model.feature_count();
        const size_t timesteps = features.size() / width;
        if (features.size() % width || expected.size() != timesteps * model.topk()) {
            throw std::runtime_error("reference shape mismatch");
        }
        float max_error = 0;
        // Both passes start from zero recurrent state. Also exercise profiling.
        for (int pass = 0; pass < 2; ++pass) {
            auto query = model.create_query();
            for (size_t t = 0; t < timesteps; ++t) {
                faiss::HNSWOnnxPredictProfile profile;
                const auto probabilities = query->predict(
                        features.data() + t * width, width,
                        pass ? &profile : nullptr);
                for (size_t rank = 0; rank < probabilities.size(); ++rank) {
                    const float reference = expected[t * model.topk() + rank];
                    const float error = std::abs(probabilities[rank] - reference);
                    if (!std::isfinite(reference) || !std::isfinite(error)) {
                        throw std::runtime_error("non-finite probability");
                    }
                    max_error = std::max(max_error, error);
                }
            }
        }
        std::cout << "topk=" << model.topk() << " timesteps=" << timesteps
                  << " max_abs_error=" << max_error << '\n';
        return max_error <= 1e-4f ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
