/* Validate real asynchronous CUDA submissions against CPU and sync CUDA. */
#include <faiss/impl/HNSWOnnxInference.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <faiss/impl/HNSWRawDataPrefetch.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
std::vector<float> features(const faiss::HNSWOnnxModel& model, size_t t) {
    std::vector<float> values(model.topk() * model.feature_count());
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.5f + 0.25f * std::sin((i + 1) * 0.017f + (t + 1) * 0.031f);
    }
    return values;
}
std::vector<float> sequential(faiss::HNSWOnnxModel& model, size_t steps) {
    auto query = model.create_query();
    std::vector<float> probabilities;
    for (size_t t = 0; t < steps; ++t) {
        const auto input = features(model, t);
        const auto output = query->predict(input.data(), input.size());
        probabilities.insert(probabilities.end(), output.begin(), output.end());
    }
    return probabilities;
}
float max_error(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::runtime_error("output length mismatch");
    float error = 0;
    for (size_t i = 0; i < a.size(); ++i) error = std::max(error, std::abs(a[i] - b[i]));
    return error;
}
}

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    if (argc != 2 && argc != 3 && argc != 7) {
        std::cerr << "usage: validate_hnsw_cuda_pipeline MODEL.onnx [timesteps=200] "
                     "[INDEX QUERY.fvecs SIDECAR RAW]\n";
        return 2;
    }
    try {
        const size_t steps = argc >= 3 ? std::stoull(argv[2]) : 200;
        if (!steps) throw std::runtime_error("timesteps must be positive");
        std::cerr << "stage=baseline_models\n";
        faiss::HNSWOnnxModel cpu(argv[1], "cpu");
        faiss::HNSWOnnxModel gpu(argv[1], "cuda");
        sequential(cpu, 8);
        sequential(gpu, 8);
        auto started = Clock::now();
        const auto expected = sequential(cpu, steps);
        const double cpu_ms = elapsed(started);
        started = Clock::now();
        const auto synchronous = sequential(gpu, steps);
        const double sync_ms = elapsed(started);
        std::cout << std::fixed << std::setprecision(6)
                  << "cpu_total_ms=" << cpu_ms << '\n'
                  << "sync_cuda_total_ms=" << sync_ms << '\n'
                  << "sync_cuda_cpu_max_error=" << max_error(expected, synchronous) << '\n';
        for (size_t depth : {size_t{1}, size_t{2}, size_t{8}}) {
            std::cerr << "stage=pipeline_model depth=" << depth << '\n';
            faiss::HNSWOnnxModel pipeline(argv[1], "cuda", 0, depth);
            for (int repeat = 0; repeat < 2; ++repeat) {
                std::cerr << "stage=pipeline_query depth=" << depth << " repeat=" << repeat << '\n';
                auto query = pipeline.create_query();
                std::vector<float> actual;
                size_t peak = 0, not_ready = 0;
                double submit_ms = 0, collect_ms = 0;
                started = Clock::now();
                auto collect = [&](bool wait) {
                    std::vector<float> output;
                    const auto begin = Clock::now();
                    const bool ready = query->collect_gpu(output, wait);
                    collect_ms += elapsed(begin);
                    if (ready) actual.insert(actual.end(), output.begin(), output.end());
                    return ready;
                };
                for (size_t t = 0; t < steps; ++t) {
                    if (query->gpu_pending() == depth && !collect(true)) {
                        throw std::runtime_error("missing pending result");
                    }
                    auto input = features(pipeline, t);
                    const auto begin = Clock::now();
                    query->submit_gpu(input.data(), input.size());
                    submit_ms += elapsed(begin);
                    // Caller memory can change immediately after submit.
                    std::fill(input.begin(), input.end(), -1000.0f);
                    peak = std::max(peak, query->gpu_pending());
                    // Repeat 0 eagerly consumes ready results. Repeat 1
                    // deliberately fills the bounded ring, exercising slot
                    // reuse even when this small GPU model finishes early.
                    if (repeat == 0 && !collect(false)) ++not_ready;
                }
                while (query->gpu_pending()) collect(true);
                const double total_ms = elapsed(started);
                const float error = max_error(expected, actual);
                if (error > 1e-4f) {
                    std::cerr << "max_error=" << error << " first_expected=" << expected[0]
                              << " first_actual=" << actual[0] << '\n';
                    throw std::runtime_error("CUDA pipeline predictions differ");
                }
                std::cout << "depth=" << depth << " repeat=" << repeat
                          << " timesteps=" << steps << " total_ms=" << total_ms
                          << " cpu_submit_ms=" << submit_ms
                          << " cpu_collect_ms=" << collect_ms
                          << " deferred_collection=" << (repeat == 1)
                          << " peak_pending=" << peak << " not_ready_polls=" << not_ready
                          << " async_overlap_observed=" << (peak > 1 && not_ready > 0)
                          << " max_error=" << error << '\n';
            }
            // Destruction must safely drain work even if the caller stops early.
            {
                auto query = pipeline.create_query();
                const auto input = features(pipeline, 0);
                query->submit_gpu(input.data(), input.size());
            }
        }
        if (argc == 7) {
            auto base = faiss::read_index_up(argv[3]);
            auto* index = dynamic_cast<faiss::IndexHNSW*>(base.get());
            if (!index) throw std::runtime_error("not an HNSW index");
            index->read_level0_avg_neighbor_distance(argv[5]);
            std::ifstream input(argv[4], std::ios::binary);
            int32_t d = 0;
            input.read(reinterpret_cast<char*>(&d), sizeof(d));
            if (!input || d != index->d) throw std::runtime_error("invalid query dimensions");
            std::vector<float> vector(d);
            input.read(reinterpret_cast<char*>(vector.data()), d * sizeof(float));
            if (!input) throw std::runtime_error("failed reading query");
            const int k = cpu.topk();
            std::vector<float> distances(k);
            std::vector<faiss::idx_t> labels(k);
            faiss::HNSWFeatureCollector collector;
            faiss::SearchParametersHNSW params;
            params.efSearch = 200;
            params.onnx_model = &cpu;
            params.feature_collector = &collector;
            index->search(1, vector.data(), k, distances.data(), labels.data(), &params);
            const auto reference = collector.queries[0];
            const auto reference_labels = labels;
            faiss::HNSWOnnxModel pipeline(argv[1], "cuda", 0, 8);
            faiss::HNSWRawDataPrefetcher raw(argv[6], 4096, 4, 4096, true);
            params.onnx_model = &pipeline;
            params.onnx_async_inference = true;
            params.raw_data_prefetcher = &raw;
            params.prefetch_enabled = true;
            index->search(1, vector.data(), k, distances.data(), labels.data(), &params);
            raw.drain();
            const auto& actual = collector.queries[0];
            if (labels != reference_labels || actual.timesteps.size() != reference.timesteps.size()) {
                throw std::runtime_error("HNSW traversal/results changed");
            }
            float error = 0;
            for (size_t t = 0; t < actual.timesteps.size(); ++t) {
                if (actual.timesteps[t].node_ids != reference.timesteps[t].node_ids) {
                    throw std::runtime_error("HNSW snapshots changed");
                }
                if (actual.timesteps[t].prediction_probabilities.empty()) {
                    continue; // Cancelled at traversal completion.
                }
                if (actual.timesteps[t].prefetch_decisions != reference.timesteps[t].prefetch_decisions) {
                    throw std::runtime_error("HNSW snapshot/prefetch decisions changed");
                }
                error = std::max(error, max_error(
                        actual.timesteps[t].prediction_probabilities,
                        reference.timesteps[t].prediction_probabilities));
            }
            if (error > 1e-4f || actual.gpu_pipeline_submitted > actual.timesteps.size()) {
                throw std::runtime_error("HNSW pipeline inference validation failed");
            }
            const auto stats = raw.summary();
            std::cout << "hnsw_timesteps=" << actual.timesteps.size()
                      << "\nhnsw_gpu_submitted=" << actual.gpu_pipeline_submitted
                      << "\nhnsw_peak_inflight=" << actual.gpu_pipeline_peak_inflight
                      << "\nhnsw_max_error=" << error
                      << "\nhnsw_prefetch_tp=" << stats.prefetch_tp
                      << "\nhnsw_prefetch_fp=" << stats.prefetch_fp
                      << "\nhnsw_prefetch_fn=" << stats.prefetch_fn
                      << "\nhnsw_search_ms=" << stats.search_latency.mean_us / 1000
                      << "\nhnsw_data_wait_ms=" << stats.data_ready_wait_latency.mean_us / 1000
                      << "\nhnsw_end_to_end_ms=" << stats.end_to_end_latency.mean_us / 1000 << '\n';
        }
        std::cout << "validation_passed=true\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
