#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace faiss {

/** Diagnostic timestamps relative to start_ns, all durations in milliseconds.
 * One profile belongs to one query; inspect it only after search() returns.
 * Inference work overlaps the search phases and must not be added to them.
 */
struct HNSWOnnxPredictProfile {
    double input_tensor_ms = 0;
    double session_run_ms = 0;
    double output_metadata_ms = 0;
    double probability_copy_ms = 0;
    double state_update_ms = 0;
    // CUDA pipeline CPU-side timings, not GPU kernel durations.
    double transfer_enqueue_ms = 0;
    double collect_wait_ms = 0;
};

struct HNSWInferenceProfileEvent {
    size_t timestep = 0;
    double enqueued_ms = 0;
    double started_ms = 0;
    double gpu_enqueued_ms = 0;
    double collect_started_ms = 0;
    double predicted_ms = 0;
    double submitted_ms = 0;
    HNSWOnnxPredictProfile predict;
};

struct HNSWSearchProfile {
    int64_t start_ns = 0;
    double upper_end_ms = 0;
    double setup_end_ms = 0;
    double graph_end_ms = 0;
    // May follow data_ready_ms when cancelled inference needs final cleanup.
    double inference_end_ms = 0;
    double search_end_ms = 0;
    double requests_submitted_ms = 0;
    double data_ready_ms = 0;
    double feature_ms = 0;
    double enqueue_ms = 0;
    double predict_ms = 0;
    double io_submit_ms = 0;
    size_t peak_snapshot_queue = 0;
    std::vector<HNSWInferenceProfileEvent> inference_events;
};

} // namespace faiss
