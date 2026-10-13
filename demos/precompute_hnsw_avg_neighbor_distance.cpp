/*
 * Compute the query-independent level-0 average-neighbor-distance feature
 * once and persist it as a validated sidecar for subsequent HNSW searches.
 */

#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0]
                  << " <index.faiss> <avg-neighbor-distance.bin>\n";
        return 2;
    }

    try {
        const auto started = std::chrono::steady_clock::now();
        std::unique_ptr<faiss::Index> index = faiss::read_index_up(argv[1]);
        auto* hnsw_index = dynamic_cast<faiss::IndexHNSW*>(index.get());
        if (!hnsw_index) {
            throw std::runtime_error("loaded index is not an IndexHNSW");
        }
        const auto loaded = std::chrono::steady_clock::now();

        hnsw_index->compute_level0_avg_neighbor_distance();
        const auto computed = std::chrono::steady_clock::now();
        hnsw_index->write_level0_avg_neighbor_distance(argv[2]);
        const auto written = std::chrono::steady_clock::now();

        std::cout << std::fixed << std::setprecision(6)
                  << "index_path=" << argv[1] << '\n'
                  << "sidecar_path=" << argv[2] << '\n'
                  << "ntotal=" << hnsw_index->ntotal << '\n'
                  << "dimension=" << hnsw_index->d << '\n'
                  << "index_load_seconds="
                  << std::chrono::duration<double>(loaded - started).count()
                  << '\n'
                  << "compute_seconds="
                  << std::chrono::duration<double>(computed - loaded).count()
                  << '\n'
                  << "write_seconds="
                  << std::chrono::duration<double>(written - computed).count()
                  << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
