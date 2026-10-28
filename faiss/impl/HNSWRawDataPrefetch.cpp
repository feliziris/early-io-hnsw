/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/HNSWRawDataPrefetch.h>
#include <faiss/impl/HNSWSearchProfile.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace faiss {

namespace {

using Clock = std::chrono::steady_clock;

uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
    return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                    .count());
}

HNSWLatencyStatistics summarize(const std::vector<uint64_t>& values_ns) {
    HNSWLatencyStatistics result;
    result.count = values_ns.size();
    if (values_ns.empty()) {
        return result;
    }
    std::vector<uint64_t> sorted = values_ns;
    std::sort(sorted.begin(), sorted.end());
    long double total = 0.0;
    for (uint64_t value : values_ns) {
        total += value;
    }
    auto percentile = [&](double fraction) {
        const double position = fraction * (sorted.size() - 1);
        const size_t lower = static_cast<size_t>(position);
        const size_t upper = std::min(lower + 1, sorted.size() - 1);
        const double weight = position - lower;
        return (sorted[lower] * (1.0 - weight) + sorted[upper] * weight) /
                1000.0;
    };
    result.mean_us = static_cast<double>(total / values_ns.size() / 1000.0L);
    result.min_us = sorted.front() / 1000.0;
    result.p50_us = percentile(0.50);
    result.p95_us = percentile(0.95);
    result.p99_us = percentile(0.99);
    result.max_us = sorted.back() / 1000.0;
    return result;
}

struct ReadRequest {
    int32_t node_id = -1;
    bool late_demand = false;
    Clock::time_point decision_time;
    std::mutex mutex;
    std::condition_variable completed;
    bool done = false;
    std::exception_ptr error;

    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        completed.wait(lock, [&] { return done; });
        if (error) {
            std::rethrow_exception(error);
        }
    }
};

} // namespace

struct HNSWRawDataPrefetcher::Impl {
    std::string path;
    size_t payload_size;
    size_t payload_count;
    size_t queue_capacity;
    bool cache_eviction_advised;
    int file_descriptor = -1;
    size_t file_size = 0;
    const uint8_t* mapping = nullptr;

    mutable std::mutex queue_mutex;
    std::condition_variable queue_not_empty;
    std::condition_variable queue_not_full;
    std::condition_variable queue_drained;
    std::deque<std::shared_ptr<ReadRequest>> queue;
    size_t active_workers = 0;
    bool stopping = false;
    std::vector<std::thread> workers;

    mutable std::mutex stats_mutex;
    std::vector<uint64_t> search_ns;
    std::vector<uint64_t> data_wait_ns;
    std::vector<uint64_t> end_to_end_ns;
    std::vector<uint64_t> queue_wait_ns;
    std::vector<uint64_t> async_read_ns;
    std::vector<uint64_t> completion_ns;
    std::vector<uint64_t> demand_read_ns;
    uint64_t prefetch_tp = 0;
    uint64_t prefetch_fp = 0;
    uint64_t prefetch_fn = 0;
    bool prefetch_enabled = false;
    bool prefetch_mode_initialized = false;
    std::vector<std::exception_ptr> errors;
    std::atomic<uint64_t> read_checksum{0};

    Impl(
            const std::string& input_path,
            size_t input_payload_size,
            size_t worker_count,
            size_t input_queue_capacity,
            bool evict_file_cache)
            : path(input_path),
              payload_size(input_payload_size),
              payload_count(0),
              queue_capacity(input_queue_capacity),
              cache_eviction_advised(evict_file_cache) {
        if (payload_size == 0 || worker_count == 0 || queue_capacity == 0) {
            throw std::invalid_argument(
                    "payload size, worker count, and queue capacity must be positive");
        }
        file_descriptor = open(path.c_str(), O_RDONLY);
        if (file_descriptor < 0) {
            throw std::runtime_error("cannot open raw data file: " + path);
        }
        struct stat status {};
        if (fstat(file_descriptor, &status) != 0 || status.st_size <= 0) {
            close(file_descriptor);
            throw std::runtime_error("cannot stat raw data file: " + path);
        }
        file_size = static_cast<size_t>(status.st_size);
        if (file_size % payload_size != 0) {
            close(file_descriptor);
            throw std::runtime_error(
                    "raw data size is not divisible by payload size");
        }
        payload_count = file_size / payload_size;
        if (evict_file_cache &&
            posix_fadvise(file_descriptor, 0, 0, POSIX_FADV_DONTNEED) != 0) {
            close(file_descriptor);
            throw std::runtime_error("POSIX_FADV_DONTNEED failed");
        }
        void* address = mmap(
                nullptr, file_size, PROT_READ, MAP_SHARED, file_descriptor, 0);
        if (address == MAP_FAILED) {
            close(file_descriptor);
            throw std::runtime_error("mmap failed for raw data file");
        }
        mapping = static_cast<const uint8_t*>(address);
        madvise(const_cast<uint8_t*>(mapping), file_size, MADV_RANDOM);
        workers.reserve(worker_count);
        for (size_t worker = 0; worker < worker_count; ++worker) {
            workers.emplace_back([this] { worker_loop(); });
        }
    }

    ~Impl() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_drained.wait(
                    lock, [&] { return queue.empty() && active_workers == 0; });
            stopping = true;
        }
        queue_not_empty.notify_all();
        for (std::thread& worker : workers) {
            worker.join();
        }
        munmap(const_cast<uint8_t*>(mapping), file_size);
        close(file_descriptor);
    }

    void validate_node(int32_t node_id) const {
        if (node_id < 0 || static_cast<size_t>(node_id) >= payload_count) {
            throw std::out_of_range("node ID is outside raw payload range");
        }
    }

    void read_payload(int32_t node_id, std::vector<uint8_t>& buffer) {
        validate_node(node_id);
        buffer.resize(payload_size);
        const size_t offset = static_cast<size_t>(node_id) * payload_size;
        std::memcpy(buffer.data(), mapping + offset, payload_size);
        uint64_t checksum = 0;
        constexpr size_t page_size = 4096;
        for (size_t position = 0; position < payload_size;
             position += page_size) {
            checksum += buffer[position];
        }
        checksum += buffer.back();
        read_checksum.fetch_add(checksum, std::memory_order_relaxed);
    }

    std::shared_ptr<ReadRequest> submit(
            int32_t node_id,
            bool late_demand) {
        validate_node(node_id);
        auto request = std::make_shared<ReadRequest>();
        request->node_id = node_id;
        request->late_demand = late_demand;
        request->decision_time = Clock::now();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_not_full.wait(lock, [&] {
                return stopping || queue.size() < queue_capacity;
            });
            if (stopping) {
                throw std::runtime_error("raw prefetcher is stopping");
            }
            queue.push_back(request);
        }
        queue_not_empty.notify_one();
        return request;
    }

    void worker_loop() {
        std::vector<uint8_t> buffer;
        while (true) {
            std::shared_ptr<ReadRequest> request;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_not_empty.wait(lock, [&] {
                    return stopping || !queue.empty();
                });
                if (stopping && queue.empty()) {
                    return;
                }
                request = std::move(queue.front());
                queue.pop_front();
                ++active_workers;
            }
            queue_not_full.notify_one();

            const Clock::time_point started = Clock::now();
            try {
                read_payload(request->node_id, buffer);
                const Clock::time_point finished = Clock::now();
                std::lock_guard<std::mutex> lock(stats_mutex);
                queue_wait_ns.push_back(
                        elapsed_ns(request->decision_time, started));
                async_read_ns.push_back(elapsed_ns(started, finished));
                if (request->late_demand) {
                    demand_read_ns.push_back(elapsed_ns(started, finished));
                }
                completion_ns.push_back(
                        elapsed_ns(request->decision_time, finished));
            } catch (...) {
                request->error = std::current_exception();
                std::lock_guard<std::mutex> lock(stats_mutex);
                errors.push_back(request->error);
            }
            {
                std::lock_guard<std::mutex> lock(request->mutex);
                request->done = true;
            }
            request->completed.notify_all();
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                --active_workers;
                if (queue.empty() && active_workers == 0) {
                    queue_drained.notify_all();
                }
            }
        }
    }

    void drain(bool rethrow_errors) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_drained.wait(
                lock, [&] { return queue.empty() && active_workers == 0; });
        lock.unlock();
        if (rethrow_errors) {
            std::lock_guard<std::mutex> stats_lock(stats_mutex);
            if (!errors.empty()) {
                std::rethrow_exception(errors.front());
            }
        }
    }
};

struct HNSWRawDataQuery::Impl {
    HNSWRawDataPrefetcher::Impl& owner;
    bool prefetch_enabled;
    Clock::time_point search_started = Clock::now();
    std::unordered_map<int32_t, std::shared_ptr<ReadRequest>> requests;
    bool finished = false;
    std::mutex requests_mutex;
    HNSWSearchProfile* profile = nullptr;

    Impl(HNSWRawDataPrefetcher::Impl& input_owner, bool enabled)
            : owner(input_owner), prefetch_enabled(enabled) {}
};

HNSWRawDataPrefetcher::HNSWRawDataPrefetcher(
        const std::string& path,
        size_t payload_size,
        size_t worker_count,
        size_t queue_capacity,
        bool evict_file_cache)
        : impl_(std::make_unique<Impl>(
                  path,
                  payload_size,
                  worker_count,
                  queue_capacity,
                  evict_file_cache)) {}

HNSWRawDataPrefetcher::~HNSWRawDataPrefetcher() = default;

std::unique_ptr<HNSWRawDataQuery> HNSWRawDataPrefetcher::create_query(
        bool prefetch_enabled, HNSWSearchProfile* profile) {
    {
        std::lock_guard<std::mutex> lock(impl_->stats_mutex);
        if (impl_->prefetch_mode_initialized &&
            impl_->prefetch_enabled != prefetch_enabled) {
            throw std::runtime_error(
                    "cannot mix prefetch on/off queries in one prefetcher");
        }
        impl_->prefetch_enabled = prefetch_enabled;
        impl_->prefetch_mode_initialized = true;
    }
    auto state = std::make_unique<HNSWRawDataQuery::Impl>(
            *impl_, prefetch_enabled);
    state->profile = profile;
    if (profile) {
        profile->start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    state->search_started.time_since_epoch())
                                    .count();
    }
    return std::unique_ptr<HNSWRawDataQuery>(
            new HNSWRawDataQuery(std::move(state)));
}

void HNSWRawDataPrefetcher::drain() {
    impl_->drain(true);
}

HNSWRawDataSummary HNSWRawDataPrefetcher::summary() const {
    std::lock_guard<std::mutex> lock(impl_->stats_mutex);
    HNSWRawDataSummary result;
    result.prefetch_enabled = impl_->prefetch_enabled;
    result.cache_eviction_advised = impl_->cache_eviction_advised;
    result.payload_size = impl_->payload_size;
    result.payload_count = impl_->payload_count;
    result.query_count = impl_->search_ns.size();
    result.async_read_count = impl_->async_read_ns.size();
    result.demand_read_count = impl_->demand_read_ns.size();
    result.prefetch_tp = impl_->prefetch_tp;
    result.prefetch_fp = impl_->prefetch_fp;
    result.prefetch_fn = impl_->prefetch_fn;
    result.total_bytes_read = result.async_read_count * result.payload_size;
    result.search_latency = summarize(impl_->search_ns);
    result.data_ready_wait_latency = summarize(impl_->data_wait_ns);
    result.end_to_end_latency = summarize(impl_->end_to_end_ns);
    result.queue_wait_latency = summarize(impl_->queue_wait_ns);
    result.mmap_read_latency = summarize(impl_->async_read_ns);
    result.decision_to_completion_latency = summarize(impl_->completion_ns);
    result.late_demand_read_latency = summarize(impl_->demand_read_ns);
    return result;
}

HNSWRawDataQuery::HNSWRawDataQuery(std::unique_ptr<Impl> impl)
        : impl_(std::move(impl)) {}
HNSWRawDataQuery::~HNSWRawDataQuery() = default;
HNSWRawDataQuery::HNSWRawDataQuery(HNSWRawDataQuery&&) noexcept = default;
HNSWRawDataQuery& HNSWRawDataQuery::operator=(
        HNSWRawDataQuery&&) noexcept = default;

void HNSWRawDataQuery::submit(int32_t node_id) {
    std::lock_guard<std::mutex> lock(impl_->requests_mutex);
    if (!impl_->prefetch_enabled || impl_->finished) {
        return;
    }
    if (impl_->requests.find(node_id) == impl_->requests.end()) {
        impl_->requests.emplace(
                node_id, impl_->owner.submit(node_id, false));
    }
}

void HNSWRawDataQuery::finish(const std::vector<int32_t>& final_ids) {
    std::unique_lock<std::mutex> lock(impl_->requests_mutex);
    if (impl_->finished) {
        throw std::runtime_error("raw data query was already finished");
    }
    impl_->finished = true;
    // No further speculative submissions can mutate requests after this point.
    // Release before queue submissions/waits so the inference worker can exit.
    lock.unlock();
    const Clock::time_point search_ended = Clock::now();
    if (impl_->profile) {
        impl_->profile->search_end_ms =
                elapsed_ns(impl_->search_started, search_ended) / 1e6;
    }
    const Clock::time_point data_wait_started = search_ended;
    uint64_t tp = 0;
    uint64_t fn = 0;
    std::vector<std::shared_ptr<ReadRequest>> final_requests;
    final_requests.reserve(final_ids.size());
    for (int32_t node_id : final_ids) {
        auto request = impl_->requests.find(node_id);
        if (impl_->prefetch_enabled && request != impl_->requests.end()) {
            final_requests.push_back(request->second);
            ++tp;
        } else {
            final_requests.push_back(impl_->owner.submit(node_id, true));
            ++fn;
        }
    }
    if (impl_->profile) {
        impl_->profile->requests_submitted_ms =
                elapsed_ns(impl_->search_started, Clock::now()) / 1e6;
    }
    for (const auto& request : final_requests) {
        request->wait();
    }
    const Clock::time_point data_ready = Clock::now();
    if (impl_->profile) {
        impl_->profile->data_ready_ms =
                elapsed_ns(impl_->search_started, data_ready) / 1e6;
    }
    const uint64_t fp = impl_->prefetch_enabled
            ? impl_->requests.size() - tp
            : 0;
    {
        std::lock_guard<std::mutex> lock(impl_->owner.stats_mutex);
        impl_->owner.search_ns.push_back(
                elapsed_ns(impl_->search_started, search_ended));
        impl_->owner.data_wait_ns.push_back(
                elapsed_ns(data_wait_started, data_ready));
        impl_->owner.end_to_end_ns.push_back(
                elapsed_ns(impl_->search_started, data_ready));
        impl_->owner.prefetch_tp += tp;
        impl_->owner.prefetch_fp += fp;
        impl_->owner.prefetch_fn += fn;
    }
}

} // namespace faiss
