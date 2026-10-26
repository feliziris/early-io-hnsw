/* Profile one real async HNSW query. All output times are milliseconds. */
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <faiss/impl/HNSWOnnxInference.h>
#include <faiss/impl/HNSWRawDataPrefetch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 7 || argc > 10) {
        std::cerr << "usage: profile_hnsw_single_query INDEX QUERY.fvecs SIDECAR "
                     "MODEL.onnx RAW_PAYLOAD OUTPUT.json [query-row=0] "
                     "[provider=cpu] [gpu-inflight=8]\n";
        return 2;
    }
    try {
        const size_t row = argc >= 8 ? std::stoull(argv[7]) : 0;
        const std::string provider = argc >= 9 ? argv[8] : "cpu";
        const size_t depth = argc >= 10 ? std::stoull(argv[9]) :
                (provider == "cuda" || provider == "gpu" ? 8 : 0);
        auto base = faiss::read_index_up(argv[1]);
        auto* index = dynamic_cast<faiss::IndexHNSW*>(base.get());
        if (!index) throw std::runtime_error("index is not HNSW");
        index->read_level0_avg_neighbor_distance(argv[3]);
        std::ifstream input(argv[2], std::ios::binary);
        int32_t d = 0;
        input.read(reinterpret_cast<char*>(&d), sizeof(d));
        if (!input || d != index->d) throw std::runtime_error("query dimension mismatch");
        input.seekg(row * (static_cast<size_t>(d) + 1) * sizeof(float));
        int32_t row_d = 0;
        input.read(reinterpret_cast<char*>(&row_d), sizeof(row_d));
        std::vector<float> query(d);
        input.read(reinterpret_cast<char*>(query.data()), d * sizeof(float));
        if (!input || row_d != d) throw std::runtime_error("invalid query row");

        faiss::HNSWOnnxModel reference_model(argv[4], "cpu");
        faiss::HNSWOnnxModel model(argv[4], provider, 0, depth);
        const int k = model.topk();
        std::vector<float> distances(k);
        std::vector<faiss::idx_t> labels(k);
        faiss::HNSWFeatureCollector collector;
        faiss::SearchParametersHNSW params;
        params.efSearch = 200;
        params.onnx_model = &reference_model;
        params.onnx_threshold = 0.5f;
        params.feature_collector = &collector;

        // Untimed warm-up: graph/feature/ONNX only, no payload reads.
        index->search(1, query.data(), k, distances.data(), labels.data(), &params);
        const auto reference = collector.queries[0];
        const auto reference_labels = labels;
        // Warm up the actual provider and reusable inference worker too.
        // Each subsequent query starts with a new zeroed recurrent state.
        params.onnx_model = &model;
        params.onnx_async_inference = true;
        index->search(1, query.data(), k, distances.data(), labels.data(), &params);
        faiss::HNSWRawDataPrefetcher raw(argv[5], 4096, 4, 4096, true);
        faiss::HNSWSearchProfile profile;
        params.raw_data_prefetcher = &raw;
        params.prefetch_enabled = true;
        params.onnx_async_inference = true;
        params.search_profile = &profile;
        index->search(1, query.data(), k, distances.data(), labels.data(), &params);
        const auto drain_started = std::chrono::steady_clock::now();
        raw.drain(); // FP requests can outlive final top-k readiness.
        const double drain_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - drain_started).count();
        const auto summary = raw.summary();
        const auto& trace = collector.queries[0];
        if (labels != reference_labels || trace.timesteps.size() != reference.timesteps.size()) {
            throw std::runtime_error("profile changed search results");
        }
        float max_error = 0;
        for (size_t t = 0; t < trace.timesteps.size(); ++t) {
            if (trace.timesteps[t].prefetch_decisions != reference.timesteps[t].prefetch_decisions) {
                throw std::runtime_error("profile changed prefetch decisions");
            }
            for (int r = 0; r < k; ++r) {
                max_error = std::max(max_error, std::abs(
                        trace.timesteps[t].prediction_probabilities[r] -
                        reference.timesteps[t].prediction_probabilities[r]));
            }
        }
        if (profile.inference_events.size() != trace.timesteps.size() || max_error > 1e-4f) {
            throw std::runtime_error("profile inference validation failed");
        }
        if (std::abs(profile.data_ready_ms - summary.end_to_end_latency.mean_us / 1000) > 1e-6) {
            throw std::runtime_error("profile and raw-data timing boundaries differ");
        }

        std::ofstream out(argv[6]);
        if (!out) throw std::runtime_error("cannot open profile JSON");
        out << std::fixed << std::setprecision(9)
            << "{\n  \"query_file\": " << std::quoted(argv[2])
            << ",\n  \"query_row\": " << row
            << ",\n  \"model\": " << std::quoted(argv[4])
            << ",\n  \"k\": " << k
            << ",\n  \"ef_search\": 200,\n  \"threshold\": 0.5,\n"
            << "  \"provider\": " << std::quoted(model.provider())
            << ",\n  \"gpu_inflight_limit\": " << depth
            << ",\n  \"gpu_peak_inflight\": " << trace.gpu_pipeline_peak_inflight
            << ",\n  \"io_workers\": 4,\n"
            << "  \"payload_bytes\": 4096,\n  \"evict_file_cache_advised\": true,\n"
            << "  \"worker_startup_excluded\": true,\n  \"untimed_provider_warmup\": true,\n"
            << "  \"profiling_enabled\": true,\n  \"timesteps\": " << trace.timesteps.size()
            << ",\n  \"end_to_end_ms\": " << profile.data_ready_ms
            << ",\n  \"phases_ms\": {\n"
            << "    \"upper_levels\": " << profile.upper_end_ms
            << ",\n    \"query_setup\": " << profile.setup_end_ms - profile.upper_end_ms
            << ",\n    \"level0\": " << profile.graph_end_ms - profile.setup_end_ms
            << ",\n    \"inference_drain_and_writeback\": " << profile.inference_end_ms - profile.graph_end_ms
            << ",\n    \"final_result_extraction\": " << profile.search_end_ms - profile.inference_end_ms
            << ",\n    \"final_request_lookup_and_submit\": " << profile.requests_submitted_ms - profile.search_end_ms
            << ",\n    \"final_payload_wait\": " << profile.data_ready_ms - profile.requests_submitted_ms
            << "\n  },\n  \"nested_work_ms\": {\n"
            << "    \"feature\": " << profile.feature_ms
            << ",\n    \"snapshot_enqueue\": " << profile.enqueue_ms
            << ",\n    \"predict_total\": " << profile.predict_ms
            << ",\n    \"predict_mean\": " << profile.predict_ms / trace.timesteps.size()
            << ",\n    \"threshold_and_io_submit\": " << profile.io_submit_ms
            << "\n  },\n  \"boundaries_ms\": {\n"
            << "    \"upper_end\": " << profile.upper_end_ms
            << ",\n    \"setup_end\": " << profile.setup_end_ms
            << ",\n    \"graph_end\": " << profile.graph_end_ms
            << ",\n    \"inference_end\": " << profile.inference_end_ms
            << ",\n    \"search_end\": " << profile.search_end_ms
            << ",\n    \"requests_submitted\": " << profile.requests_submitted_ms
            << ",\n    \"data_ready\": " << profile.data_ready_ms
            << "\n  },\n  \"peak_snapshot_queue\": " << profile.peak_snapshot_queue
            << ",\n  \"tp\": " << summary.prefetch_tp
            << ",\n  \"fp\": " << summary.prefetch_fp
            << ",\n  \"fn\": " << summary.prefetch_fn
            << ",\n  \"read_count\": " << summary.async_read_count
            << ",\n  \"read_mean_ms\": " << summary.mmap_read_latency.mean_us / 1000
            << ",\n  \"io_queue_wait_mean_ms\": " << summary.queue_wait_latency.mean_us / 1000
            << ",\n  \"post_search_drain_ms\": " << drain_ms
            << ",\n  \"max_probability_error_vs_sync\": " << max_error
            << ",\n  \"inference_events\": [\n";
        for (size_t i = 0; i < profile.inference_events.size(); ++i) {
            const auto& event = profile.inference_events[i];
            if (i) out << ",\n";
            out << "    {\"timestep\":" << event.timestep
                << ",\"enqueued_ms\":" << event.enqueued_ms
                << ",\"started_ms\":" << event.started_ms
                << ",\"gpu_enqueued_ms\":" << event.gpu_enqueued_ms
                << ",\"collect_started_ms\":" << event.collect_started_ms
                << ",\"predicted_ms\":" << event.predicted_ms
                << ",\"submitted_ms\":" << event.submitted_ms
                << ",\"input_tensor_ms\":" << event.predict.input_tensor_ms
                << ",\"session_run_ms\":" << event.predict.session_run_ms
                << ",\"output_metadata_ms\":" << event.predict.output_metadata_ms
                << ",\"probability_copy_ms\":" << event.predict.probability_copy_ms
                << ",\"state_update_ms\":" << event.predict.state_update_ms
                << ",\"transfer_enqueue_ms\":" << event.predict.transfer_enqueue_ms
                << ",\"collect_wait_ms\":" << event.predict.collect_wait_ms << "}";
        }
        out << "\n  ]\n}\n";
        if (!out) throw std::runtime_error("failed writing profile JSON");
        std::cout << "profile=" << argv[6] << "\nend_to_end_ms=" << profile.data_ready_ms
                  << "\ntimesteps=" << trace.timesteps.size()
                  << "\nvalidation_passed=true\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
