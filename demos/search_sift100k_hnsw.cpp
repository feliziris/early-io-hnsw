/*
 * Load the sampled SIFT100K HNSW index and search SIFT query vectors.
 */

#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> read_fvecs(
        const std::string& path,
        std::size_t count,
        int expected_dimension) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open query file: " + path);
    }

    std::vector<float> vectors(count * expected_dimension);
    for (std::size_t i = 0; i < count; ++i) {
        std::int32_t dimension = 0;
        input.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
        if (!input) {
            throw std::runtime_error(
                    "query file contains fewer vectors than requested");
        }
        if (dimension != expected_dimension) {
            throw std::runtime_error(
                    "unexpected query dimension: " +
                    std::to_string(dimension));
        }
        input.read(
                reinterpret_cast<char*>(vectors.data() + i * dimension),
                sizeof(float) * dimension);
        if (!input) {
            throw std::runtime_error("truncated query vector");
        }
    }
    return vectors;
}

std::vector<faiss::idx_t> read_mapping(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open mapping file: " + path);
    }

    std::string internal_header;
    std::string original_header;
    input >> internal_header >> original_header;
    if (internal_header != "internal_id" || original_header != "original_id") {
        throw std::runtime_error("invalid mapping header");
    }

    std::vector<faiss::idx_t> original_ids;
    faiss::idx_t internal_id = -1;
    faiss::idx_t original_id = -1;
    while (input >> internal_id >> original_id) {
        if (internal_id != static_cast<faiss::idx_t>(original_ids.size())) {
            throw std::runtime_error("mapping internal IDs are not sequential");
        }
        original_ids.push_back(original_id);
    }
    if (!input.eof()) {
        throw std::runtime_error("failed parsing mapping file");
    }
    return original_ids;
}

long parse_positive(const char* value, const char* name) {
    std::size_t parsed = 0;
    const long result = std::stol(value, &parsed);
    if (value[parsed] != '\0' || result <= 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
    return result;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: " << argv[0]
                  << " <index> <query.fvecs> <mapping.tsv> <nq> <k>\n";
        return 2;
    }

    try {
        const long nq = parse_positive(argv[4], "nq");
        const long k = parse_positive(argv[5], "k");

        std::unique_ptr<faiss::Index> index = faiss::read_index_up(argv[1]);
        auto* hnsw_index = dynamic_cast<faiss::IndexHNSW*>(index.get());
        if (!hnsw_index) {
            throw std::runtime_error("loaded index is not an IndexHNSW");
        }
        hnsw_index->compute_level0_avg_neighbor_distance();

        std::vector<faiss::idx_t> original_ids = read_mapping(argv[3]);
        if (original_ids.size() != static_cast<std::size_t>(index->ntotal)) {
            throw std::runtime_error("mapping size does not match index ntotal");
        }
        std::vector<float> queries = read_fvecs(argv[2], nq, index->d);
        std::vector<float> distances(static_cast<std::size_t>(nq * k));
        std::vector<faiss::idx_t> labels(static_cast<std::size_t>(nq * k));

        faiss::SearchParametersHNSW params;
        params.efSearch = 200;
        params.check_relative_distance = true;
        params.bounded_queue = true;
        faiss::HNSWFeatureCollector feature_collector;
        params.feature_collector = &feature_collector;

        faiss::SearchParametersHNSW baseline_params = params;
        baseline_params.feature_collector = nullptr;
        std::vector<float> baseline_distances(
                static_cast<std::size_t>(nq * k));
        std::vector<faiss::idx_t> baseline_labels(
                static_cast<std::size_t>(nq * k));
        index->search(
                nq,
                queries.data(),
                k,
                baseline_distances.data(),
                baseline_labels.data(),
                &baseline_params);

        std::cout << "index_type=IndexHNSW ntotal=" << index->ntotal
                  << " dimension=" << index->d
                  << " max_level=" << hnsw_index->hnsw.max_level << '\n';
        std::cout << "efConstruction=" << hnsw_index->hnsw.efConstruction
                  << " stored_efSearch=" << hnsw_index->hnsw.efSearch
                  << " query_efSearch=" << params.efSearch
                  << " nq=" << nq << " k=" << k << '\n';

        faiss::hnsw_stats.reset();
        const auto started = std::chrono::steady_clock::now();
        index->search(
                nq,
                queries.data(),
                k,
                distances.data(),
                labels.data(),
                &params);
        const double seconds = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - started)
                                       .count();

        std::cout << std::fixed << std::setprecision(6)
                  << "search_seconds=" << seconds
                  << " qps=" << nq / seconds << '\n';
        std::cout << "hnsw_stats: n1=" << faiss::hnsw_stats.n1
                  << " n2=" << faiss::hnsw_stats.n2
                  << " ndis=" << faiss::hnsw_stats.ndis
                  << " nhops=" << faiss::hnsw_stats.nhops << '\n';

        if (labels != baseline_labels || distances != baseline_distances) {
            throw std::runtime_error(
                    "feature collection changed the HNSW search result");
        }
        std::cout << "feature_collection_preserves_results=true\n";

        if (feature_collector.queries.size() !=
            static_cast<std::size_t>(nq)) {
            throw std::runtime_error("feature collector query count mismatch");
        }
        for (long query_id = 0; query_id < nq; ++query_id) {
            const auto& trace = feature_collector.queries[query_id];
            if (trace.final_top_k_ids.size() !=
                static_cast<std::size_t>(k)) {
                throw std::runtime_error("feature trace final top-k is incomplete");
            }
            for (long rank = 0; rank < k; ++rank) {
                const std::size_t offset = query_id * k + rank;
                if (trace.final_top_k_ids[rank] != labels[offset] ||
                    trace.final_top_k_distances[rank] != distances[offset]) {
                    throw std::runtime_error(
                            "feature trace final top-k does not match search result");
                }
            }
        }

        const auto& first_trace = feature_collector.queries.front();
        std::cout << "feature_trace_query=0 timesteps="
                  << first_trace.timesteps.size()
                  << " entry_distance=" << first_trace.entry_distance << '\n';
        if (!first_trace.timesteps.empty()) {
            const auto& step = first_trace.timesteps.front();
            std::cout << "first_timestep pop_count=" << step.pop_count
                      << " expanded_node_id=" << step.expanded_node_id
                      << " search_progress=" << step.search_progress << '\n';
            for (long rank = 0; rank < k; ++rank) {
                const float* features = step.features.data() +
                        rank * faiss::HNSW_FEATURE_COUNT;
                std::cout << "  feature_rank=" << rank
                          << " node_id=" << step.node_ids[rank]
                          << " ratio_to_entry="
                          << features[faiss::HNSW_FEATURE_RATIO_TO_ENTRY]
                          << " ratio_to_first="
                          << features[faiss::HNSW_FEATURE_RATIO_TO_FIRST]
                          << " progress="
                          << features[faiss::HNSW_FEATURE_SEARCH_PROGRESS]
                          << " rank_norm="
                          << features[faiss::HNSW_FEATURE_RANK_NORM]
                          << " persistence="
                          << features[faiss::HNSW_FEATURE_PERSISTENCE_RATE]
                          << " avg_neigh_dist="
                          << features[faiss::HNSW_FEATURE_AVG_NEIGH_DIST]
                          << '\n';
            }
        }

        const long rows_to_print = std::min<long>(nq, 3);
        for (long query_id = 0; query_id < rows_to_print; ++query_id) {
            std::cout << "query=" << query_id << '\n';
            for (long rank = 0; rank < k; ++rank) {
                const std::size_t offset = query_id * k + rank;
                const faiss::idx_t internal_id = labels[offset];
                if (internal_id < 0 ||
                    internal_id >= static_cast<faiss::idx_t>(original_ids.size())) {
                    throw std::runtime_error("search returned an invalid label");
                }
                std::cout << "  rank=" << rank
                          << " internal_id=" << internal_id
                          << " original_id=" << original_ids[internal_id]
                          << " distance=" << distances[offset] << '\n';
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
