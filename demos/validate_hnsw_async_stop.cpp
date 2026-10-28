/* Validate asynchronous inference cancellation and concurrent final payload I/O.
 * Usage: validate_hnsw_async_stop MODEL.onnx
 * Uses synthetic vectors and a temporary payload file; no dataset is required.
 */
#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSWOnnxInference.h>
#include <faiss/impl/HNSWRawDataPrefetch.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>
#include <unistd.h>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct Payload {
    char path[64] = "/tmp/faiss-async-stop-XXXXXX";
    Payload() {
        int fd = mkstemp(path);
        if (fd < 0) throw std::runtime_error("mkstemp failed");
        const int result = ftruncate(fd, 512 * 4096);
        close(fd);
        if (result != 0) throw std::runtime_error("ftruncate failed");
    }
    ~Payload() { std::remove(path); }
};
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    try {
        Payload payload;
        // Exercise a saturated queue while final-result accounting races with
        // speculative submission. Deduplication must include all final reads.
        for (int repetition = 0; repetition < 50; ++repetition) {
            faiss::HNSWRawDataPrefetcher raw(payload.path, 4096, 1, 1, false);
            auto query = raw.create_query(true);
            std::atomic<bool> go{false};
            std::thread producer([&] {
                while (!go.load()) std::this_thread::yield();
                for (int node = 0; node < 128; ++node) query->submit(node);
            });
            go.store(true);
            query->finish({0, 1, 2, 3});
            producer.join();
            query->submit(200); // Must be ignored after finalization.
            raw.drain();
            auto stats = raw.summary();
            require(stats.prefetch_tp + stats.prefetch_fn == 4, "missing final reads");
            require(stats.async_read_count == stats.prefetch_tp + stats.prefetch_fp + stats.prefetch_fn,
                    "duplicate or post-finalization reads");
        }
        faiss::HNSWOnnxModel model(argv[1], "cpu");
        const int k = model.topk();
        constexpr int d = 128, n = 512, nq = 4;
        omp_set_num_threads(1);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> distribution(0, 1);
        std::vector<float> vectors(n * d), queries(nq * d);
        for (auto& v : vectors) v = distribution(rng);
        for (auto& v : queries) v = distribution(rng);
        faiss::IndexHNSWFlat index(d, 16);
        index.add(n, vectors.data());
        index.compute_level0_avg_neighbor_distance();
        std::vector<float> reference_distances(nq * k), distances(nq * k);
        std::vector<faiss::idx_t> reference_ids(nq * k), ids(nq * k);
        faiss::SearchParametersHNSW params;
        params.efSearch = 200;
        index.search(nq, queries.data(), k, reference_distances.data(), reference_ids.data(), &params);
        faiss::HNSWFeatureCollector collector;
        params.onnx_model = &model;
        params.onnx_async_inference = true;
        params.feature_collector = &collector;
        index.search(nq, queries.data(), k, distances.data(), ids.data(), &params);
        const auto reference_traces = collector.queries;
        for (const auto& trace : reference_traces)
            for (const auto& step : trace.timesteps)
                require(step.prediction_probabilities.size() == static_cast<size_t>(k),
                        "trace-only inference must remain complete");
        // Test no predicted reads, all predicted reads, and normal decisions,
        // including concurrent OpenMP queries sharing one reader/session.
        for (int threads : {1, 4}) {
            omp_set_num_threads(threads);
            for (float threshold : {1.0f, 0.0f, 0.5f}) {
                faiss::HNSWRawDataPrefetcher raw(payload.path, 4096, 2, 1, false);
                params.raw_data_prefetcher = &raw;
                params.prefetch_enabled = true;
                params.onnx_threshold = threshold;
                index.search(nq, queries.data(), k, distances.data(), ids.data(), &params);
                raw.drain();
                require(ids == reference_ids && distances == reference_distances,
                        "final HNSW results changed");
                size_t predicted = 0, steps = 0;
                for (int q = 0; q < nq; ++q) {
                    const auto& trace = collector.queries[q];
                    require(trace.timesteps.size() == reference_traces[q].timesteps.size(),
                            "snapshot count changed");
                    for (size_t t = 0; t < trace.timesteps.size(); ++t) {
                        const auto& actual = trace.timesteps[t].prediction_probabilities;
                        const auto& expected = reference_traces[q].timesteps[t].prediction_probabilities;
                        if (!actual.empty()) {
                            ++predicted;
                            for (int r = 0; r < k; ++r)
                                require(std::abs(actual[r] - expected[r]) < 1e-4f,
                                        "query state changed after cancellation");
                        }
                        ++steps;
                    }
                }
                require(predicted < steps, "prediction backlog was not cancelled");
                const auto stats = raw.summary();
                require(stats.prefetch_tp + stats.prefetch_fn == nq * k, "final payload missing");
                require(stats.async_read_count == stats.prefetch_tp + stats.prefetch_fp + stats.prefetch_fn,
                        "duplicate payload request");
                std::cout << "threads=" << threads << " threshold=" << threshold
                          << " predicted=" << predicted << "/" << steps
                          << " tp=" << stats.prefetch_tp << " fn=" << stats.prefetch_fn << '\n';
            }
        }
        std::cout << "validation_passed=true\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
