/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace faiss {

struct HNSWLatencyStatistics {
    uint64_t count = 0;
    double mean_us = 0.0;
    double min_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
    double max_us = 0.0;
};

struct HNSWRawDataSummary {
    bool prefetch_enabled = false;
    bool cache_eviction_advised = false;
    size_t payload_size = 0;
    size_t payload_count = 0;
    uint64_t query_count = 0;
    uint64_t async_read_count = 0;
    uint64_t demand_read_count = 0;
    uint64_t prefetch_tp = 0;
    uint64_t prefetch_fp = 0;
    uint64_t prefetch_fn = 0;
    uint64_t total_bytes_read = 0;
    HNSWLatencyStatistics search_latency;
    HNSWLatencyStatistics data_ready_wait_latency;
    HNSWLatencyStatistics end_to_end_latency;
    HNSWLatencyStatistics queue_wait_latency;
    HNSWLatencyStatistics mmap_read_latency;
    HNSWLatencyStatistics decision_to_completion_latency;
    HNSWLatencyStatistics late_demand_read_latency;
};

class HNSWRawDataQuery;

/** Shared mmap and bounded worker queue for HNSW payload reads. */
class HNSWRawDataPrefetcher {
   public:
    HNSWRawDataPrefetcher(
            const std::string& path,
            size_t payload_size,
            size_t worker_count,
            size_t queue_capacity,
            bool evict_file_cache);
    ~HNSWRawDataPrefetcher();

    HNSWRawDataPrefetcher(const HNSWRawDataPrefetcher&) = delete;
    HNSWRawDataPrefetcher& operator=(const HNSWRawDataPrefetcher&) = delete;

    std::unique_ptr<HNSWRawDataQuery> create_query(bool prefetch_enabled);
    void drain();
    HNSWRawDataSummary summary() const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class HNSWRawDataQuery;
};

/** Per-query deduplication, request handles, and data-ready boundary. */
class HNSWRawDataQuery {
   public:
    ~HNSWRawDataQuery();
    HNSWRawDataQuery(HNSWRawDataQuery&&) noexcept;
    HNSWRawDataQuery& operator=(HNSWRawDataQuery&&) noexcept;

    HNSWRawDataQuery(const HNSWRawDataQuery&) = delete;
    HNSWRawDataQuery& operator=(const HNSWRawDataQuery&) = delete;

    void submit(int32_t node_id);
    void finish(const std::vector<int32_t>& final_ids);

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    explicit HNSWRawDataQuery(std::unique_ptr<Impl> impl);
    friend class HNSWRawDataPrefetcher;
};

} // namespace faiss
