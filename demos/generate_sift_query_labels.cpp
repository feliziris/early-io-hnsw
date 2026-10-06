/*
 * Generate complete per-timestep traces for all SIFT queries searched through
 * an IndexHNSW. The output includes node IDs, query distances, the six runtime
 * features and final-top-k membership labels.
 */

#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSWOnnxInference.h>
#include <faiss/impl/HNSWRawDataPrefetch.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kFormatVersion = 1;
constexpr char kMagic[8] = {'H', 'N', 'S', 'W', 'T', 'R', 'C', '1'};

struct FvecsData {
    std::int32_t dimension = 0;
    std::uint32_t count = 0;
    std::vector<float> vectors;
};

struct LabelStats {
    std::uint64_t timesteps = 0;
    std::uint64_t labels = 0;
    std::uint64_t positives = 0;
};

template <typename T>
void write_value(std::ostream& output, const T& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!output) {
        throw std::runtime_error("failed writing label output");
    }
}

void write_header(
        std::ostream& output,
        std::uint32_t nq,
        std::uint32_t k,
        std::uint32_t ef_search,
        const LabelStats& stats) {
    output.write(kMagic, sizeof(kMagic));
    write_value(output, kFormatVersion);
    write_value(output, nq);
    write_value(output, k);
    write_value(output, ef_search);
    const std::uint32_t feature_count = faiss::HNSW_FEATURE_COUNT;
    const std::uint32_t reserved = 0;
    write_value(output, feature_count);
    write_value(output, reserved);
    write_value(output, stats.timesteps);
    write_value(output, stats.labels);
    write_value(output, stats.positives);
}

long parse_positive(const char* value, const char* name) {
    std::size_t parsed = 0;
    const long result = std::stol(value, &parsed);
    if (value[parsed] != '\0' || result <= 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
    return result;
}

float parse_probability(const char* value, const char* name) {
    std::size_t parsed = 0;
    const float result = std::stof(value, &parsed);
    if (value[parsed] != '\0' || result < 0.0f || result > 1.0f) {
        throw std::runtime_error(
                std::string(name) + " must be between 0 and 1");
    }
    return result;
}

void print_latency(
        const char* name,
        const faiss::HNSWLatencyStatistics& latency) {
    std::cout << name << "_count=" << latency.count << '\n'
              << name << "_mean_us=" << latency.mean_us << '\n'
              << name << "_p50_us=" << latency.p50_us << '\n'
              << name << "_p95_us=" << latency.p95_us << '\n'
              << name << "_p99_us=" << latency.p99_us << '\n'
              << name << "_max_us=" << latency.max_us << '\n';
}

FvecsData read_all_fvecs(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open query file: " + path);
    }

    const std::streamoff file_size = input.tellg();
    input.seekg(0);

    FvecsData data;
    input.read(
            reinterpret_cast<char*>(&data.dimension), sizeof(data.dimension));
    if (!input || data.dimension <= 0) {
        throw std::runtime_error("invalid fvecs dimension header");
    }

    const std::uint64_t record_size = sizeof(std::int32_t) +
            sizeof(float) * static_cast<std::uint64_t>(data.dimension);
    if (file_size <= 0 ||
        static_cast<std::uint64_t>(file_size) % record_size != 0) {
        throw std::runtime_error("query file size is not valid fvecs data");
    }
    const std::uint64_t count =
            static_cast<std::uint64_t>(file_size) / record_size;
    if (count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("too many query vectors");
    }
    data.count = static_cast<std::uint32_t>(count);
    data.vectors.resize(
            count * static_cast<std::uint64_t>(data.dimension));

    input.seekg(0);
    for (std::uint32_t query = 0; query < data.count; ++query) {
        std::int32_t dimension = 0;
        input.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
        if (!input || dimension != data.dimension) {
            throw std::runtime_error(
                    "inconsistent fvecs dimension at query " +
                    std::to_string(query));
        }
        input.read(
                reinterpret_cast<char*>(
                        data.vectors.data() +
                        static_cast<std::uint64_t>(query) * data.dimension),
                sizeof(float) * data.dimension);
        if (!input) {
            throw std::runtime_error("truncated query vector data");
        }
    }
    return data;
}

bool is_final_result(
        const std::vector<std::int32_t>& final_ids,
        std::int32_t node_id) {
    return std::find(final_ids.begin(), final_ids.end(), node_id) !=
            final_ids.end();
}

void write_query_record(
        std::ostream& output,
        std::ostream& index_output,
        std::uint32_t query_id,
        const faiss::HNSWQueryTrace& trace,
        std::uint32_t k,
        LabelStats& total_stats) {
    if (trace.final_top_k_ids.size() != k ||
        trace.final_top_k_distances.size() != k) {
        throw std::runtime_error(
                "incomplete final top-k for query " +
                std::to_string(query_id));
    }
    if (trace.timesteps.size() >
        std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("too many timesteps in one query");
    }

    const std::uint64_t record_offset =
            static_cast<std::uint64_t>(output.tellp());
    const std::uint32_t timestep_count =
            static_cast<std::uint32_t>(trace.timesteps.size());
    write_value(output, query_id);
    write_value(output, timestep_count);
    write_value(output, trace.entry_distance);

    for (std::uint32_t rank = 0; rank < k; ++rank) {
        write_value(output, trace.final_top_k_ids[rank]);
    }
    for (std::uint32_t rank = 0; rank < k; ++rank) {
        write_value(output, trace.final_top_k_distances[rank]);
    }

    std::uint64_t query_positives = 0;
    for (const faiss::HNSWFeatureTimestep& timestep : trace.timesteps) {
        if (timestep.node_ids.size() != k ||
            timestep.query_distances.size() != k ||
            timestep.features.size() !=
                    static_cast<std::size_t>(k) *
                            faiss::HNSW_FEATURE_COUNT) {
            throw std::runtime_error(
                    "incomplete timestep trace for query " +
                    std::to_string(query_id));
        }
        const std::uint64_t pop_count = timestep.pop_count;
        write_value(output, pop_count);
        write_value(output, timestep.expanded_node_id);
        write_value(output, timestep.first_distance);
        write_value(output, timestep.search_progress);

        for (std::uint32_t rank = 0; rank < k; ++rank) {
            write_value(output, timestep.node_ids[rank]);
        }
        for (std::uint32_t rank = 0; rank < k; ++rank) {
            write_value(output, timestep.query_distances[rank]);
        }
        for (float feature : timestep.features) {
            write_value(output, feature);
        }
        for (std::uint32_t rank = 0; rank < k; ++rank) {
            const std::int32_t node_id = timestep.node_ids[rank];
            const std::uint8_t label = static_cast<std::uint8_t>(
                    is_final_result(trace.final_top_k_ids, node_id));
            write_value(output, label);
            query_positives += label;
        }
    }

    const std::uint64_t query_labels =
            static_cast<std::uint64_t>(timestep_count) * k;
    index_output << query_id << '\t' << record_offset << '\t'
                 << timestep_count << '\t' << query_labels << '\t'
                 << query_positives << '\t'
                 << (query_labels - query_positives) << '\n';
    if (!index_output) {
        throw std::runtime_error("failed writing query label index");
    }

    total_stats.timesteps += timestep_count;
    total_stats.labels += query_labels;
    total_stats.positives += query_positives;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr
                << "usage: " << argv[0]
                << " <index> <query.fvecs> <trace.bin> <trace-index.tsv> "
                   "<k> <efSearch> <batch-size> "
                   "[--onnx-model MODEL_ONNX] [--threshold VALUE] "
                   "[--prefetch on|off --raw-data FILE "
                   "--payload-size BYTES --io-workers N "
                   "--io-queue-size N [--evict-file-cache] "
                   "[--oracle-prefetch]]\n";
        return 2;
    }

    try {
        const long k_arg = parse_positive(argv[5], "k");
        const long ef_search_arg = parse_positive(argv[6], "efSearch");
        const long batch_size_arg = parse_positive(argv[7], "batch-size");
        if (k_arg > std::numeric_limits<std::uint32_t>::max() ||
            ef_search_arg > std::numeric_limits<std::uint32_t>::max() ||
            batch_size_arg > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("numeric argument is too large");
        }
        const std::uint32_t k = static_cast<std::uint32_t>(k_arg);
        const std::uint32_t ef_search =
                static_cast<std::uint32_t>(ef_search_arg);
        const std::uint32_t batch_size =
                static_cast<std::uint32_t>(batch_size_arg);
        std::string onnx_model_path;
        float onnx_threshold = 0.5f;
        bool threshold_was_set = false;
        std::string prefetch_mode;
        std::string raw_data_path;
        long payload_size = 512 * 1024;
        long io_workers = 4;
        long io_queue_size = 4096;
        bool evict_file_cache = false;
        bool oracle_prefetch = false;
        for (int argument = 8; argument < argc;) {
            const std::string option = argv[argument++];
            if (option == "--evict-file-cache") {
                evict_file_cache = true;
                continue;
            }
            if (option == "--oracle-prefetch") {
                // ORACLE EVALUATION ONLY: the measured search still runs the
                // real model, but I/O submissions known to be false positives
                // are filtered using a separately precomputed final top-k.
                oracle_prefetch = true;
                continue;
            }
            if (argument >= argc) {
                throw std::runtime_error(option + " requires a value");
            }
            if (option == "--onnx-model") {
                onnx_model_path = argv[argument++];
            } else if (option == "--threshold") {
                onnx_threshold =
                        parse_probability(argv[argument++], "threshold");
                threshold_was_set = true;
            } else if (option == "--prefetch") {
                prefetch_mode = argv[argument++];
                if (prefetch_mode != "on" && prefetch_mode != "off") {
                    throw std::runtime_error(
                            "--prefetch must be either on or off");
                }
            } else if (option == "--raw-data") {
                raw_data_path = argv[argument++];
            } else if (option == "--payload-size") {
                payload_size = parse_positive(
                        argv[argument++], "payload-size");
            } else if (option == "--io-workers") {
                io_workers = parse_positive(argv[argument++], "io-workers");
            } else if (option == "--io-queue-size") {
                io_queue_size = parse_positive(
                        argv[argument++], "io-queue-size");
            } else {
                throw std::runtime_error("unknown option: " + option);
            }
        }
        const bool onnx_enabled = !onnx_model_path.empty();
        if (threshold_was_set && !onnx_enabled) {
            throw std::runtime_error(
                    "--threshold requires --onnx-model");
        }
        const bool raw_io_enabled = !raw_data_path.empty();
        if (raw_io_enabled != !prefetch_mode.empty()) {
            throw std::runtime_error(
                    "--raw-data and --prefetch must be specified together");
        }
        if (prefetch_mode == "on" && !onnx_enabled) {
            throw std::runtime_error(
                    "--prefetch on requires --onnx-model");
        }
        if (prefetch_mode == "off" && onnx_enabled) {
            throw std::runtime_error(
                    "baseline --prefetch off must not use --onnx-model");
        }
        if (oracle_prefetch && prefetch_mode != "on") {
            throw std::runtime_error(
                    "--oracle-prefetch requires --prefetch on");
        }

        std::unique_ptr<faiss::Index> index = faiss::read_index_up(argv[1]);
        auto* hnsw_index = dynamic_cast<faiss::IndexHNSW*>(index.get());
        if (!hnsw_index) {
            throw std::runtime_error("loaded index is not an IndexHNSW");
        }

        FvecsData queries = read_all_fvecs(argv[2]);
        if (queries.dimension != index->d) {
            throw std::runtime_error("query dimension does not match index");
        }
        if (k > static_cast<std::uint64_t>(index->ntotal)) {
            throw std::runtime_error("k exceeds index ntotal");
        }

        const auto feature_started = std::chrono::steady_clock::now();
        hnsw_index->compute_level0_avg_neighbor_distance();
        const double feature_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - feature_started)
                                               .count();

        std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
        std::ofstream index_output(argv[4], std::ios::trunc);
        if (!output) {
            throw std::runtime_error("cannot open label output");
        }
        if (!index_output) {
            throw std::runtime_error("cannot open label index output");
        }

        LabelStats total_stats;
        write_header(output, queries.count, k, ef_search, total_stats);
        index_output
                << "query_id\trecord_offset\ttimestep_count\tlabel_count\t"
                   "positive_count\tnegative_count\n";

        faiss::SearchParametersHNSW params;
        params.efSearch = static_cast<int>(ef_search);
        params.check_relative_distance = true;
        params.bounded_queue = true;
        std::unique_ptr<faiss::HNSWOnnxModel> onnx_model;
        if (onnx_enabled) {
            onnx_model = std::make_unique<faiss::HNSWOnnxModel>(
                    onnx_model_path);
            params.onnx_model = onnx_model.get();
            params.onnx_threshold = onnx_threshold;
        }

        std::vector<faiss::idx_t> oracle_final_ids;
        double oracle_precompute_seconds = 0.0;
        if (oracle_prefetch) {
            // ORACLE EVALUATION ONLY: run the same deterministic HNSW search
            // once without ONNX or raw I/O to obtain ground-truth final top-k
            // IDs. This precomputation is outside all reported query latency.
            oracle_final_ids.resize(
                    static_cast<std::uint64_t>(queries.count) * k);
            std::vector<float> oracle_distances(
                    static_cast<std::uint64_t>(queries.count) * k);
            faiss::SearchParametersHNSW oracle_params;
            oracle_params.efSearch = static_cast<int>(ef_search);
            oracle_params.check_relative_distance = true;
            oracle_params.bounded_queue = true;
            const auto oracle_started = std::chrono::steady_clock::now();
            index->search(
                    queries.count,
                    queries.vectors.data(),
                    k,
                    oracle_distances.data(),
                    oracle_final_ids.data(),
                    &oracle_params);
            oracle_precompute_seconds = std::chrono::duration<double>(
                                                std::chrono::steady_clock::now() -
                                                oracle_started)
                                                .count();
            params.oracle_prefetch = true;
            params.oracle_k = k;
        }
        std::unique_ptr<faiss::HNSWRawDataPrefetcher> raw_data_prefetcher;
        if (raw_io_enabled) {
            raw_data_prefetcher =
                    std::make_unique<faiss::HNSWRawDataPrefetcher>(
                            raw_data_path,
                            static_cast<std::size_t>(payload_size),
                            static_cast<std::size_t>(io_workers),
                            static_cast<std::size_t>(io_queue_size),
                            evict_file_cache);
            params.raw_data_prefetcher = raw_data_prefetcher.get();
            params.prefetch_enabled = prefetch_mode == "on";
        }
        faiss::HNSWFeatureCollector collector;
        params.feature_collector = &collector;

        const auto search_started = std::chrono::steady_clock::now();
        std::uint64_t prediction_timesteps = 0;
        std::uint64_t prefetch_decisions = 0;
        for (std::uint32_t first = 0; first < queries.count;
             first += batch_size) {
            const std::uint32_t count =
                    std::min(batch_size, queries.count - first);
            std::vector<float> distances(
                    static_cast<std::uint64_t>(count) * k);
            std::vector<faiss::idx_t> labels(
                    static_cast<std::uint64_t>(count) * k);

            if (oracle_prefetch) {
                params.oracle_final_ids = oracle_final_ids.data() +
                        static_cast<std::uint64_t>(first) * k;
                params.oracle_query_count = count;
            }

            index->search(
                    count,
                    queries.vectors.data() +
                            static_cast<std::uint64_t>(first) *
                                    queries.dimension,
                    k,
                    distances.data(),
                    labels.data(),
                    &params);
            if (collector.queries.size() != count) {
                throw std::runtime_error("collector batch size mismatch");
            }

            for (std::uint32_t local = 0; local < count; ++local) {
                const std::uint32_t query_id = first + local;
                const faiss::HNSWQueryTrace& trace = collector.queries[local];
                if (onnx_enabled) {
                    for (const auto& timestep : trace.timesteps) {
                        if (timestep.prediction_probabilities.size() != k ||
                            timestep.prefetch_decisions.size() != k) {
                            throw std::runtime_error(
                                    "incomplete ONNX prediction for query " +
                                    std::to_string(query_id));
                        }
                        ++prediction_timesteps;
                        prefetch_decisions += std::count(
                                timestep.prefetch_decisions.begin(),
                                timestep.prefetch_decisions.end(),
                                static_cast<std::uint8_t>(1));
                    }
                }
                for (std::uint32_t rank = 0; rank < k; ++rank) {
                    const std::uint64_t result_offset =
                            static_cast<std::uint64_t>(local) * k + rank;
                    if (trace.final_top_k_ids[rank] !=
                                labels[result_offset] ||
                        trace.final_top_k_distances[rank] !=
                                distances[result_offset]) {
                        throw std::runtime_error(
                                "trace final top-k mismatch for query " +
                                std::to_string(query_id));
                    }
                    if (oracle_prefetch &&
                        labels[result_offset] !=
                                oracle_final_ids
                                        [static_cast<std::uint64_t>(query_id) *
                                                 k +
                                         rank]) {
                        throw std::runtime_error(
                                "oracle precompute mismatch for query " +
                                std::to_string(query_id));
                    }
                }
                write_query_record(
                        output,
                        index_output,
                        query_id,
                        trace,
                        k,
                        total_stats);
            }

            const std::uint32_t completed = first + count;
            if (completed == queries.count || completed % 1000 == 0) {
                std::cout << "completed_queries=" << completed << '/'
                          << queries.count << '\n';
            }
        }

        output.seekp(0);
        write_header(output, queries.count, k, ef_search, total_stats);
        output.flush();
        index_output.flush();
        if (!output || !index_output) {
            throw std::runtime_error("failed finalizing label outputs");
        }
        if (raw_data_prefetcher) {
            raw_data_prefetcher->drain();
        }

        const double search_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - search_started)
                                              .count();
        std::cout << std::fixed << std::setprecision(6)
                  << "queries=" << queries.count << " dimension="
                  << queries.dimension << " k=" << k
                  << " efSearch=" << ef_search << '\n'
                  << "avg_neighbor_distance_seconds=" << feature_seconds
                  << '\n'
                  << "search_and_write_seconds=" << search_seconds << '\n'
                  << "timesteps=" << total_stats.timesteps << '\n'
                  << "labels=" << total_stats.labels << '\n'
                  << "positive_labels=" << total_stats.positives << '\n'
                  << "negative_labels="
                  << (total_stats.labels - total_stats.positives) << '\n'
                  << "onnx_enabled=" << onnx_enabled << '\n'
                  << "onnx_threshold=" << onnx_threshold << '\n'
                  << "oracle_prefetch=" << oracle_prefetch << '\n'
                  << "oracle_precompute_seconds="
                  << oracle_precompute_seconds << '\n'
                  << "prediction_timesteps=" << prediction_timesteps << '\n'
                  << "prefetch_decisions=" << prefetch_decisions << '\n'
                  << "trace_path=" << argv[3] << '\n'
                  << "index_path=" << argv[4] << '\n';
        if (raw_data_prefetcher) {
            const faiss::HNSWRawDataSummary raw =
                    raw_data_prefetcher->summary();
            const uint64_t useful_payload_count =
                    raw.prefetch_tp + raw.prefetch_fn;
            const double wasted_io_ratio = useful_payload_count == 0
                    ? 0.0
                    : static_cast<double>(raw.prefetch_fp) /
                            useful_payload_count;
            std::cout << "raw_prefetch_enabled=" << raw.prefetch_enabled
                      << '\n'
                      << "raw_cache_eviction_advised="
                      << raw.cache_eviction_advised << '\n'
                      << "raw_payload_size=" << raw.payload_size << '\n'
                      << "raw_payload_count=" << raw.payload_count << '\n'
                      << "raw_query_count=" << raw.query_count << '\n'
                      << "raw_async_read_count=" << raw.async_read_count
                      << '\n'
                      << "raw_demand_read_count=" << raw.demand_read_count
                      << '\n'
                      << "raw_prefetch_tp=" << raw.prefetch_tp << '\n'
                      << "raw_prefetch_fp=" << raw.prefetch_fp << '\n'
                      << "raw_prefetch_fn=" << raw.prefetch_fn << '\n'
                      << "raw_wasted_io_ratio=" << wasted_io_ratio << '\n'
                      << "raw_total_bytes_read=" << raw.total_bytes_read
                      << '\n';
            print_latency("raw_search", raw.search_latency);
            print_latency(
                    "raw_data_ready_wait", raw.data_ready_wait_latency);
            print_latency("raw_end_to_end", raw.end_to_end_latency);
            print_latency("raw_queue_wait", raw.queue_wait_latency);
            print_latency("raw_mmap_read", raw.mmap_read_latency);
            print_latency(
                    "raw_decision_to_completion",
                    raw.decision_to_completion_latency);
            print_latency(
                    "raw_late_demand_read", raw.late_demand_read_latency);
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
