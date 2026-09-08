/*
 * Build a deterministic HNSW index from a fixed random sample of SIFT1M.
 */

#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kSampleCount = 100000;
constexpr std::uint32_t kSamplingSeed = 42;
constexpr int kExpectedDimension = 128;
constexpr int kM = 32;
constexpr int kEfConstruction = 200;
constexpr int kEfSearch = 200;

struct FvecsMetadata {
    std::int32_t dimension;
    std::size_t count;
    std::size_t record_bytes;
};

FvecsMetadata inspect_fvecs(std::ifstream& input) {
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    if (end <= 0) {
        throw std::runtime_error("empty or unreadable fvecs file");
    }

    input.seekg(0, std::ios::beg);
    std::int32_t dimension = 0;
    input.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
    if (!input || dimension <= 0) {
        throw std::runtime_error("invalid fvecs dimension header");
    }

    const std::size_t record_bytes =
            sizeof(std::int32_t) + sizeof(float) * dimension;
    const auto file_bytes = static_cast<std::size_t>(end);
    if (file_bytes % record_bytes != 0) {
        throw std::runtime_error("fvecs file size is not record-aligned");
    }

    input.clear();
    return {dimension, file_bytes / record_bytes, record_bytes};
}

std::vector<faiss::idx_t> sample_ids(std::size_t count) {
    if (count < kSampleCount) {
        throw std::runtime_error("fvecs file contains fewer than 100000 vectors");
    }

    std::vector<faiss::idx_t> ids(count);
    std::iota(ids.begin(), ids.end(), faiss::idx_t{0});
    std::mt19937 rng(kSamplingSeed);
    std::shuffle(ids.begin(), ids.end(), rng);
    ids.resize(kSampleCount);
    return ids;
}

std::vector<float> read_sample(
        std::ifstream& input,
        const FvecsMetadata& metadata,
        const std::vector<faiss::idx_t>& original_ids) {
    // Read in source-file order for mostly sequential I/O, but place each
    // vector in its shuffled sample slot. The HNSW insertion order therefore
    // remains the deterministic shuffled order represented by original_ids.
    std::vector<std::pair<faiss::idx_t, std::size_t>> requests;
    requests.reserve(original_ids.size());
    for (std::size_t slot = 0; slot < original_ids.size(); ++slot) {
        requests.emplace_back(original_ids[slot], slot);
    }
    std::sort(requests.begin(), requests.end());

    std::vector<float> vectors(
            original_ids.size() * static_cast<std::size_t>(metadata.dimension));
    for (const auto& [original_id, slot] : requests) {
        const auto offset = static_cast<std::streamoff>(original_id) *
                static_cast<std::streamoff>(metadata.record_bytes);
        input.seekg(offset, std::ios::beg);

        std::int32_t dimension = 0;
        input.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
        if (!input || dimension != metadata.dimension) {
            throw std::runtime_error(
                    "inconsistent fvecs dimension at original ID " +
                    std::to_string(original_id));
        }

        float* destination = vectors.data() + slot * metadata.dimension;
        input.read(
                reinterpret_cast<char*>(destination),
                sizeof(float) * metadata.dimension);
        if (!input) {
            throw std::runtime_error(
                    "failed reading vector at original ID " +
                    std::to_string(original_id));
        }
    }
    return vectors;
}

void write_mapping(
        const std::string& path,
        const std::vector<faiss::idx_t>& original_ids) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot open mapping output: " + path);
    }
    output << "internal_id\toriginal_id\n";
    for (std::size_t internal_id = 0; internal_id < original_ids.size();
         ++internal_id) {
        output << internal_id << '\t' << original_ids[internal_id] << '\n';
    }
    if (!output) {
        throw std::runtime_error("failed writing mapping output: " + path);
    }
}

double elapsed_seconds(const std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - start)
            .count();
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: " << argv[0]
                  << " <sift_base.fvecs> <output.index> <mapping.tsv>\n";
        return 2;
    }

    const std::string input_path = argv[1];
    const std::string index_path = argv[2];
    const std::string mapping_path = argv[3];

    try {
        std::ifstream input(input_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("cannot open input: " + input_path);
        }

        const FvecsMetadata metadata = inspect_fvecs(input);
        if (metadata.dimension != kExpectedDimension) {
            throw std::runtime_error(
                    "expected 128 dimensions, found " +
                    std::to_string(metadata.dimension));
        }

        std::cout << "input_vectors=" << metadata.count
                  << " dimension=" << metadata.dimension << '\n';
        std::cout << "sample_count=" << kSampleCount
                  << " sampling_seed=" << kSamplingSeed << '\n';

        auto started = std::chrono::steady_clock::now();
        std::vector<faiss::idx_t> original_ids = sample_ids(metadata.count);
        std::vector<float> sampled_vectors =
                read_sample(input, metadata, original_ids);
        std::cout << std::fixed << std::setprecision(3)
                  << "sampling_seconds=" << elapsed_seconds(started) << '\n';

        faiss::hnsw_deterministic_build = true;
        faiss::IndexHNSWFlat index(
                metadata.dimension, kM, faiss::METRIC_L2);
        index.hnsw.efConstruction = kEfConstruction;
        index.hnsw.efSearch = kEfSearch;

        std::cout << "M=" << kM
                  << " efConstruction=" << kEfConstruction
                  << " efSearch=" << kEfSearch
                  << " deterministic=true\n";

        started = std::chrono::steady_clock::now();
        index.add(kSampleCount, sampled_vectors.data());
        const double build_seconds = elapsed_seconds(started);
        std::cout << "build_seconds=" << build_seconds << '\n';
        std::cout << "ntotal=" << index.ntotal
                  << " max_level=" << index.hnsw.max_level
                  << " entry_point=" << index.hnsw.entry_point << '\n';

        started = std::chrono::steady_clock::now();
        faiss::write_index(&index, index_path.c_str());
        write_mapping(mapping_path, original_ids);
        std::cout << "write_seconds=" << elapsed_seconds(started) << '\n';
        std::cout << "index_path=" << index_path << '\n';
        std::cout << "mapping_path=" << mapping_path << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
