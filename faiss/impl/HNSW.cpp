/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/HNSW.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <deque>
#include <exception>
#include <future>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_set>

#include <faiss/IndexHNSW.h>

#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/HNSWOnnxInference.h>
#include <faiss/impl/HNSWRawDataPrefetch.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/impl/hnsw/LockVector.h>
#include <faiss/impl/hnsw/MinimaxHeap.h>

namespace faiss {

/**************************************************************
 * HNSW structure implementation
 **************************************************************/

int HNSW::nb_neighbors(int layer_no) const {
    FAISS_THROW_IF_NOT(
            static_cast<size_t>(layer_no + 1) < cum_nneighbor_per_level.size());
    return cum_nneighbor_per_level[layer_no + 1] -
            cum_nneighbor_per_level[layer_no];
}

void HNSW::set_nb_neighbors(int level_no, int n) {
    FAISS_THROW_IF_NOT(levels.size() == 0);
    int cur_n = nb_neighbors(level_no);
    for (size_t i = level_no + 1; i < cum_nneighbor_per_level.size(); i++) {
        cum_nneighbor_per_level[i] += n - cur_n;
    }
}

int HNSW::cum_nb_neighbors(int layer_no) const {
    FAISS_CHECK_RANGE_DEBUG(layer_no, 0, (int)cum_nneighbor_per_level.size());
    return cum_nneighbor_per_level[layer_no];
}

void HNSW::neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end)
        const {
    FAISS_CHECK_RANGE_DEBUG(no, 0, (idx_t)offsets.size());
    FAISS_CHECK_RANGE_DEBUG(
            layer_no, 0, (int)cum_nneighbor_per_level.size() - 1);
    size_t o = offsets[no];
    *begin = o + cum_nb_neighbors(layer_no);
    *end = o + cum_nb_neighbors(layer_no + 1);
}

HNSW::HNSW(int M) : rng(12345) {
    set_default_probas(M, 1.0 / log(M));
    offsets.push_back(0);
}

int HNSW::random_level() {
    double f = rng.rand_float();
    // could be a bit faster with bisection
    for (size_t level = 0; level < assign_probas.size(); level++) {
        if (f < assign_probas[level]) {
            return level;
        }
        f -= assign_probas[level];
    }
    // happens with exponentially low probability
    return assign_probas.size() - 1;
}

void HNSW::set_default_probas(int M, float levelMult) {
    int nn = 0;
    cum_nneighbor_per_level.push_back(0);
    for (int level = 0;; level++) {
        float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
        if (proba < 1e-9) {
            break;
        }
        assign_probas.push_back(proba);
        nn += level == 0 ? M * 2 : M;
        cum_nneighbor_per_level.push_back(nn);
    }
}

void HNSW::clear_neighbor_tables(int level) {
    for (size_t i = 0; i < levels.size(); i++) {
        size_t begin, end;
        neighbor_range(i, level, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            neighbors[j] = -1;
        }
    }
}

void HNSW::reset() {
    max_level = -1;
    entry_point = -1;
    offsets.clear();
    offsets.push_back(0);
    levels.clear();
    neighbors.clear();
}

void HNSW::print_neighbor_stats(int level) const {
    FAISS_THROW_IF_NOT(
            static_cast<size_t>(level) < cum_nneighbor_per_level.size());
    printf("stats on level %d, max %d neighbors per vertex:\n",
           level,
           nb_neighbors(level));
    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
#pragma omp parallel for reduction(+ : tot_neigh) reduction(+ : tot_common) \
        reduction(+ : tot_reciprocal) reduction(+ : n_node)
    for (idx_t i = 0; i < static_cast<idx_t>(levels.size()); i++) {
        if (levels[i] > level) {
            n_node++;
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            std::unordered_set<int> neighset;
            for (size_t j = begin; j < end; j++) {
                if (neighbors[j] < 0) {
                    break;
                }
                neighset.insert(neighbors[j]);
            }
            size_t n_neigh = neighset.size();
            int n_common = 0;
            int n_reciprocal = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = neighbors[j];
                if (i2 < 0) {
                    break;
                }
                FAISS_ASSERT(i2 != i);
                size_t begin2, end2;
                neighbor_range(i2, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2++) {
                    storage_idx_t i3 = neighbors[j2];
                    if (i3 < 0) {
                        break;
                    }
                    if (i3 == i) {
                        n_reciprocal++;
                        continue;
                    }
                    if (neighset.count(i3)) {
                        neighset.erase(i3);
                        n_common++;
                    }
                }
            }
            tot_neigh += n_neigh;
            tot_common += n_common;
            tot_reciprocal += n_reciprocal;
        }
    }
    float normalizer = n_node;
    printf("   nb of nodes at that level %zd\n", n_node);
    printf("   neighbors per node: %.2f (%zd)\n",
           tot_neigh / normalizer,
           tot_neigh);
    printf("   nb of reciprocal neighbors: %.2f\n",
           tot_reciprocal / normalizer);
    printf("   nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer,
           tot_common);
}

void HNSW::fill_with_random_links(size_t n) {
    if (n == 0) {
        return;
    }
    max_level = prepare_level_tab(n);
    entry_point = 0;

    RandomGenerator rng2(456);

    for (int level = max_level - 1; level >= 0; --level) {
        std::vector<int> elts;
        for (size_t i = 0; i < n; i++) {
            if (levels[i] > level) {
                elts.push_back(i);
            }
        }

        if (elts.size() == 1) {
            continue;
        }

        for (size_t ii = 0; ii < elts.size(); ii++) {
            int i = elts[ii];
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            for (size_t j = begin; j < end; j++) {
                int other = 0;
                do {
                    other = elts[rng2.rand_int(elts.size())];
                } while (other == i);

                neighbors[j] = other;
            }
        }
    }
}

int HNSW::prepare_level_tab(size_t n, bool preset_levels) {
    size_t n0 = offsets.size() - 1;

    if (preset_levels) {
        FAISS_ASSERT(n0 + n == levels.size());
    } else {
        FAISS_ASSERT(n0 == levels.size());
        for (size_t i = 0; i < n; i++) {
            int pt_level = random_level();
            levels.push_back(pt_level + 1);
        }
    }

    int local_max_level = 0;
    for (size_t i = 0; i < n; i++) {
        int pt_level = levels[i + n0] - 1;
        if (pt_level > local_max_level) {
            local_max_level = pt_level;
        }
        offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
    }
    neighbors.resize(offsets.back(), -1);

    return local_max_level;
}

/** Enumerate vertices from nearest to farthest from query, keep a
 * neighbor only if there is no previous neighbor that is closer to
 * that vertex than the query.
 */
template <class Comp>
void HNSW::shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistFartherT<Comp>>& input,
        std::vector<NodeDistFartherT<Comp>>& output,
        size_t max_size,
        bool keep_max_size_level0) {
    // This prevents number of neighbors at
    // level 0 from being shrunk to less than 2 * M.
    // This is essential in making sure
    // `faiss::gpu::GpuIndexCagra::copyFrom(IndexHNSWCagra*)` is functional
    std::vector<NodeDistFartherT<Comp>> outsiders;

    while (input.size() > 0) {
        NodeDistFartherT<Comp> v1 = input.top();
        input.pop();
        float dist_v1_q = v1.d;

        bool good = true;
        for (NodeDistFartherT<Comp> v2 : output) {
            float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);

            // "v1 is bad" if some previously-kept neighbor v2 is closer
            // (more similar, under CMin) to v1 than the query is. Encoded
            // generically as: v1v2 is "better than" v1q under Comp.
            if (Comp::cmp(dist_v1_q, dist_v1_v2)) {
                good = false;
                break;
            }
        }

        if (good) {
            output.push_back(v1);
            if (output.size() >= static_cast<size_t>(max_size)) {
                return;
            }
        } else if (keep_max_size_level0) {
            outsiders.push_back(v1);
        }
    }
    size_t idx = 0;
    while (keep_max_size_level0 &&
           (output.size() < static_cast<size_t>(max_size)) &&
           (idx < outsiders.size())) {
        output.push_back(outsiders[idx++]);
    }
}

// Explicit instantiations for the two supported comparators.
template void HNSW::shrink_neighbor_list<HNSW::C_distance>(
        DistanceComputer&,
        std::priority_queue<HNSW::NodeDistFartherT<HNSW::C_distance>>&,
        std::vector<HNSW::NodeDistFartherT<HNSW::C_distance>>&,
        size_t,
        bool);
template void HNSW::shrink_neighbor_list<HNSW::C_similarity>(
        DistanceComputer&,
        std::priority_queue<HNSW::NodeDistFartherT<HNSW::C_similarity>>&,
        std::vector<HNSW::NodeDistFartherT<HNSW::C_similarity>>&,
        size_t,
        bool);

namespace {

using storage_idx_t = HNSW::storage_idx_t;

inline size_t pruned_neighbor_size(size_t max_size, float prune_headroom) {
    return static_cast<size_t>(
            max_size - max_size * std::clamp(prune_headroom, 0.0f, 0.5f));
}

// Map a (high-level) HNSW comparator C — which uses int64_t IDs — to the
// (low-level) MinimaxHeap comparator HC, which uses int32_t IDs.
template <class C>
using HC_for = std::
        conditional_t<C::is_max, CMax<float, int32_t>, CMin<float, int32_t>>;

// Priority queue types used by the unbounded search variant. For CMax
// (distance) "top_candidates" is a max-heap of the kept-so-far results
// (top is the farthest) and "candidates" is a min-heap of the next nodes
// to explore (top is the closest). For CMin (similarity) the orderings are
// swapped: top_candidates is a min-heap (top is the least similar) and
// candidates is a max-heap (top is the most similar).
template <class C>
using TopCandidatesQueue = std::conditional_t<
        C::is_max,
        std::priority_queue<HNSW::Node>,
        std::priority_queue<
                HNSW::Node,
                std::vector<HNSW::Node>,
                std::greater<HNSW::Node>>>;

template <class C>
using CandidatesQueue = std::conditional_t<
        C::is_max,
        std::priority_queue<
                HNSW::Node,
                std::vector<HNSW::Node>,
                std::greater<HNSW::Node>>,
        std::priority_queue<HNSW::Node>>;

/**************************************************************
 * Addition subroutines
 **************************************************************/

/// remove neighbors from the list to make it smaller than max_size
template <class C>
void shrink_neighbor_list_inner(
        DistanceComputer& qdis,
        std::priority_queue<HNSW::NodeDistCloserT<C>>& resultSet1,
        size_t max_size,
        bool keep_max_size_level0 = false) {
    if (resultSet1.size() < static_cast<size_t>(max_size)) {
        return;
    }
    std::priority_queue<HNSW::NodeDistFartherT<C>> resultSet;
    std::vector<HNSW::NodeDistFartherT<C>> returnlist;

    while (resultSet1.size() > 0) {
        resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
        resultSet1.pop();
    }

    HNSW::shrink_neighbor_list<C>(
            qdis, resultSet, returnlist, max_size, keep_max_size_level0);

    for (HNSW::NodeDistFartherT<C> curen2 : returnlist) {
        resultSet1.emplace(curen2.d, curen2.id);
    }
}

/// add a link between two elements, possibly shrinking the list
/// of links to make room for it.
template <class C>
void add_link_tpl(
        HNSW& hnsw,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level,
        bool keep_max_size_level0 = false) {
    size_t begin, end;
    hnsw.neighbor_range(src, level, &begin, &end);
    if (hnsw.neighbors[end - 1] == -1) {
        // there is enough room, find a slot to add it
        size_t i = end;
        while (i > begin) {
            if (hnsw.neighbors[i - 1] != -1) {
                break;
            }
            i--;
        }
        hnsw.neighbors[i] = dest;
        return;
    }

    // otherwise we let them fight out which to keep

    // copy to resultSet...
    std::priority_queue<HNSW::NodeDistCloserT<C>> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < end; i++) {
        storage_idx_t neigh = hnsw.neighbors[i];
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }

    size_t max_size = end - begin;
    max_size -= max_size * std::clamp(hnsw.prune_headroom, 0.0f, 0.5f);
    shrink_neighbor_list_inner<C>(
            qdis, resultSet, max_size, keep_max_size_level0);

    // ...and back
    size_t i = begin;
    while (resultSet.size()) {
        hnsw.neighbors[i++] = resultSet.top().id;
        resultSet.pop();
    }
    // they may have shrunk more than just by 1 element
    while (i < end) {
        hnsw.neighbors[i++] = -1;
    }
}

/** Templated body of `search_neighbors_to_add` — instantiated once per final
 * VisitedTable subclass × comparator so that `vt.set/advance` are inlined
 * and the cost of virtual dispatch is paid only once at the top of the call.
 */
template <typename VTType, class C>
static void search_neighbors_to_add_fixVT(
        HNSW& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<HNSW::NodeDistCloserT<C>>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VTType& vt,
        bool reference_version) {
    // top is nearest candidate
    std::priority_queue<HNSW::NodeDistFartherT<C>> candidates;

    HNSW::NodeDistFartherT<C> ev(d_entry_point, entry_point);
    candidates.push(ev);
    results.emplace(d_entry_point, entry_point);
    vt.set(entry_point);

    while (!candidates.empty()) {
        // get nearest
        const HNSW::NodeDistFartherT<C>& currEv = candidates.top();

        if (C::cmp(currEv.d, results.top().d)) {
            break;
        }
        int currNode = currEv.id;
        candidates.pop();

        // loop over neighbors
        size_t begin, end;
        hnsw.neighbor_range(currNode, level, &begin, &end);

        // The reference version is not used, but kept here because:
        // 1. It is easier to switch back if the optimized version has a problem
        // 2. It serves as a starting point for new optimizations
        // 3. It helps understand the code
        // 4. It ensures the reference version is still compilable if the
        // optimized version changes
        // The reference and the optimized versions' results are compared in
        // test_hnsw.cpp
        if (reference_version) {
            // a reference version
            for (size_t i = begin; i < end; i++) {
                HNSW::storage_idx_t nodeId = hnsw.neighbors[i];
                if (nodeId < 0) {
                    break;
                }
                if (!vt.set(nodeId)) {
                    continue;
                }

                float dis = qdis(nodeId);
                HNSW::NodeDistFartherT<C> evE1(dis, nodeId);

                if (results.size() < static_cast<size_t>(hnsw.efConstruction) ||
                    C::cmp(results.top().d, dis)) {
                    results.emplace(dis, nodeId);
                    candidates.emplace(dis, nodeId);
                    if (results.size() >
                        static_cast<size_t>(hnsw.efConstruction)) {
                        results.pop();
                    }
                }
            }
        } else {
            // a faster version

            // the following version processes 4 neighbors at a time
            auto update_with_candidate = [&](const HNSW::storage_idx_t idx,
                                             const float dis) {
                if (results.size() < static_cast<size_t>(hnsw.efConstruction) ||
                    C::cmp(results.top().d, dis)) {
                    results.emplace(dis, idx);
                    candidates.emplace(dis, idx);
                    if (results.size() >
                        static_cast<size_t>(hnsw.efConstruction)) {
                        results.pop();
                    }
                }
            };

            int n_buffered = 0;
            HNSW::storage_idx_t buffered_ids[4];

            for (size_t j = begin; j < end; j++) {
                HNSW::storage_idx_t nodeId = hnsw.neighbors[j];
                if (nodeId < 0) {
                    break;
                }
                if (!vt.set(nodeId)) {
                    continue;
                }

                buffered_ids[n_buffered] = nodeId;
                n_buffered += 1;

                if (n_buffered == 4) {
                    float dis[4];
                    qdis.distances_batch_4(
                            buffered_ids[0],
                            buffered_ids[1],
                            buffered_ids[2],
                            buffered_ids[3],
                            dis[0],
                            dis[1],
                            dis[2],
                            dis[3]);

                    for (size_t id4 = 0; id4 < 4; id4++) {
                        update_with_candidate(buffered_ids[id4], dis[id4]);
                    }

                    n_buffered = 0;
                }
            }

            // process leftovers
            for (int icnt = 0; icnt < n_buffered; icnt++) {
                float dis = qdis(buffered_ids[icnt]);
                update_with_candidate(buffered_ids[icnt], dis);
            }
        }
    }

    vt.advance();
}

/// Dispatches the VisitedTable concrete type for a given C, then calls
/// the templated `search_neighbors_to_add_fixVT<VTType, C>`.
template <class C>
void search_neighbors_to_add_dispatch(
        HNSW& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<HNSW::NodeDistCloserT<C>>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt,
        bool reference_version) {
    auto call = [&]<typename VTType>(VTType& vt_concrete) {
        search_neighbors_to_add_fixVT<VTType, C>(
                hnsw,
                qdis,
                results,
                entry_point,
                d_entry_point,
                level,
                vt_concrete,
                reference_version);
    };
    if (VisitedTableVector* vtv = dynamic_cast<VisitedTableVector*>(&vt)) {
        call(*vtv);
        return;
    }
    VisitedTableSet& vts = dynamic_cast<VisitedTableSet&>(vt);
    call(vts);
}

/// Templated implementation of `HNSW::add_links_starting_from`.
template <class C>
void add_links_starting_from_impl(
        HNSW& hnsw,
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        LockVector& locks,
        VisitedTable& vt,
        bool keep_max_size_level0) {
    std::priority_queue<HNSW::NodeDistCloserT<C>> link_targets;

    search_neighbors_to_add_dispatch<C>(
            hnsw, ptdis, link_targets, nearest, d_nearest, level, vt, false);

    // but we can afford only this many neighbors
    int M = hnsw.nb_neighbors(level);

    shrink_neighbor_list_inner<C>(ptdis, link_targets, M, keep_max_size_level0);

    std::vector<storage_idx_t> neighbors_to_add;
    neighbors_to_add.reserve(link_targets.size());
    while (!link_targets.empty()) {
        storage_idx_t other_id = link_targets.top().id;
        add_link_tpl<C>(
                hnsw, ptdis, pt_id, other_id, level, keep_max_size_level0);
        neighbors_to_add.push_back(other_id);
        link_targets.pop();
    }

    locks.unlock(pt_id);
    for (storage_idx_t other_id : neighbors_to_add) {
        locks.lock(other_id);
        add_link_tpl<C>(
                hnsw, ptdis, other_id, pt_id, level, keep_max_size_level0);
        locks.unlock(other_id);
    }
    locks.lock(pt_id);
}

} // namespace

/// Finds neighbors and builds links with them, starting from an entry
/// point. The own neighbor list is assumed to be locked.
void HNSW::add_links_starting_from(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        LockVector& locks,
        VisitedTable& vt,
        bool keep_max_size_level0) {
    if (is_similarity) {
        add_links_starting_from_impl<C_similarity>(
                *this,
                ptdis,
                pt_id,
                nearest,
                d_nearest,
                level,
                locks,
                vt,
                keep_max_size_level0);
    } else {
        add_links_starting_from_impl<C_distance>(
                *this,
                ptdis,
                pt_id,
                nearest,
                d_nearest,
                level,
                locks,
                vt,
                keep_max_size_level0);
    }
}

/// search neighbors on a single level, starting from an entry point.
/// Public dispatcher: always operates in distance (CMax) mode because its
/// `priority_queue<HNSW::NodeDistCloser>` signature is the back-compat
/// distance flavor. Internal callers that need similarity mode reach the
/// templated implementation directly via `search_neighbors_to_add_dispatch`.
void hnsw_detail::search_neighbors_to_add(
        HNSW& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<HNSW::NodeDistCloser>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt,
        bool reference_version) {
    search_neighbors_to_add_dispatch<HNSW::C_distance>(
            hnsw,
            qdis,
            results,
            entry_point,
            d_entry_point,
            level,
            vt,
            reference_version);
}

/**************************************************************
 * Building, parallel
 **************************************************************/

namespace {

/// Greedy update of the nearest entry point at a given level.
template <class C>
HNSWStats greedy_update_nearest_impl(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {
    HNSWStats stats;

    for (;;) {
        storage_idx_t prev_nearest = nearest;

        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);

        size_t ndis = 0;

        // a faster version: reference version in unit test test_hnsw.cpp
        // the following version processes 4 neighbors at a time
        auto update_with_candidate = [&](const storage_idx_t idx,
                                         const float dis) {
            if (C::cmp(d_nearest, dis)) {
                nearest = idx;
                d_nearest = dis;
            }
        };

        int n_buffered = 0;
        storage_idx_t buffered_ids[4];

        for (size_t j = begin; j < end; j++) {
            storage_idx_t v = hnsw.neighbors[j];
            if (v < 0) {
                break;
            }
            ndis += 1;

            buffered_ids[n_buffered] = v;
            n_buffered += 1;

            if (n_buffered == 4) {
                float dis[4];
                qdis.distances_batch_4(
                        buffered_ids[0],
                        buffered_ids[1],
                        buffered_ids[2],
                        buffered_ids[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    update_with_candidate(buffered_ids[id4], dis[id4]);
                }

                n_buffered = 0;
            }
        }

        // process leftovers
        for (int icnt = 0; icnt < n_buffered; icnt++) {
            float dis = qdis(buffered_ids[icnt]);
            update_with_candidate(buffered_ids[icnt], dis);
        }

        // update stats
        stats.ndis += ndis;
        stats.nhops += 1;

        if (nearest == prev_nearest) {
            return stats;
        }
    }
}

} // namespace

/// greedily update a nearest vector at a given level
HNSWStats hnsw_detail::greedy_update_nearest(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {
    if (hnsw.is_similarity) {
        return greedy_update_nearest_impl<HNSW::C_similarity>(
                hnsw, qdis, level, nearest, d_nearest);
    }
    return greedy_update_nearest_impl<HNSW::C_distance>(
            hnsw, qdis, level, nearest, d_nearest);
}

/**************************************************************
 * Deterministic addition subroutines
 **************************************************************/

namespace {

template <class C>
void compute_forward_links_impl(
        HNSW& hnsw,
        DistanceComputer& ptdis,
        int pt_level,
        storage_idx_t pt_id,
        VisitedTable& vt,
        std::vector<std::pair<storage_idx_t, int>>& pt_reverse_edges,
        bool keep_max_size_level0) {
    storage_idx_t nearest = hnsw.entry_point;
    FAISS_ASSERT(nearest >= 0);
    float d_nearest = ptdis(nearest);

    int level = hnsw.max_level;
    // greedy descent on the upper levels the point does not live on
    for (; level > pt_level; level--) {
        greedy_update_nearest_impl<C>(hnsw, ptdis, level, nearest, d_nearest);
    }

    // levels the point lives on: search from the same entry point
    for (; level >= 0; level--) {
        std::priority_queue<HNSW::NodeDistCloserT<C>> link_targets;
        search_neighbors_to_add_dispatch<C>(
                hnsw,
                ptdis,
                link_targets,
                nearest,
                d_nearest,
                level,
                vt,
                false);

        int M = hnsw.nb_neighbors(level);
        shrink_neighbor_list_inner<C>(
                ptdis, link_targets, M, keep_max_size_level0 && (level == 0));

        // pt_id is not reachable yet, so the snapshot stays immutable for the
        // other points in this batch
        size_t begin, end;
        hnsw.neighbor_range(pt_id, level, &begin, &end);
        size_t i = begin;
        while (!link_targets.empty()) {
            storage_idx_t other_id = link_targets.top().id;
            link_targets.pop();
            // pt_id is in its own candidate list when it is the entry point
            if (other_id == pt_id) {
                continue;
            }
            FAISS_ASSERT(i < end);
            hnsw.neighbors[i++] = other_id;
            pt_reverse_edges.emplace_back(other_id, level);
        }
        while (i < end) {
            hnsw.neighbors[i++] = -1;
        }
    }
}

template <class C>
void merge_reverse_links_impl(
        HNSW& hnsw,
        DistanceComputer& dis,
        storage_idx_t node,
        int level,
        std::vector<storage_idx_t>& incoming,
        bool keep_max_size_level0) {
    size_t begin, end;
    hnsw.neighbor_range(node, level, &begin, &end);
    size_t max_size = end - begin;

    std::vector<storage_idx_t> cands;
    cands.reserve(max_size + incoming.size());
    for (size_t i = begin; i < end; i++) {
        if (hnsw.neighbors[i] < 0) {
            break;
        }
        cands.push_back(hnsw.neighbors[i]);
    }
    size_t n_existing = cands.size();

    std::sort(incoming.begin(), incoming.end());
    incoming.erase(
            std::unique(incoming.begin(), incoming.end()), incoming.end());
    for (storage_idx_t v : incoming) {
        if (v == node) {
            continue;
        }
        bool present = false;
        for (size_t i = 0; i < n_existing; i++) {
            if (cands[i] == v) {
                present = true;
                break;
            }
        }
        if (!present) {
            cands.push_back(v);
        }
    }

    if (cands.size() <= max_size) {
        size_t i = begin;
        for (storage_idx_t v : cands) {
            hnsw.neighbors[i++] = v;
        }
        while (i < end) {
            hnsw.neighbors[i++] = -1;
        }
        return;
    }

    // Ties broken by id, so the result is independent of collection order.
    std::vector<std::pair<float, storage_idx_t>> arr;
    arr.reserve(cands.size());
    for (storage_idx_t v : cands) {
        arr.emplace_back(dis.symmetric_dis(node, v), v);
    }
    std::sort(
            arr.begin(),
            arr.end(),
            [](const std::pair<float, storage_idx_t>& a,
               const std::pair<float, storage_idx_t>& b) {
                if (a.first != b.first) {
                    // C::cmp(x, y) is true when y is "better" than x, so
                    // a before b iff a is better than b.
                    return C::cmp(b.first, a.first);
                }
                return a.second < b.second;
            });

    // Same headroom as add_link's reciprocal pruning, so degrees stay
    // comparable between the two builds.
    size_t pruned_size = pruned_neighbor_size(max_size, hnsw.prune_headroom);
    if (pruned_size < 1) {
        pruned_size = 1;
    }

    std::vector<std::pair<float, storage_idx_t>> kept;
    std::vector<std::pair<float, storage_idx_t>> outsiders;
    for (const auto& cand : arr) {
        float dist_v1_q = cand.first;
        bool good = true;
        for (const auto& k : kept) {
            float dist_v1_v2 = dis.symmetric_dis(k.second, cand.second);
            if (C::cmp(dist_v1_q, dist_v1_v2)) {
                good = false;
                break;
            }
        }
        if (good) {
            kept.push_back(cand);
            if (kept.size() >= pruned_size) {
                break;
            }
        } else if (keep_max_size_level0) {
            outsiders.push_back(cand);
        }
    }
    for (size_t idx = 0; keep_max_size_level0 && kept.size() < max_size &&
         idx < outsiders.size();
         idx++) {
        kept.push_back(outsiders[idx]);
    }

    size_t i = begin;
    for (const auto& k : kept) {
        hnsw.neighbors[i++] = k.second;
    }
    while (i < end) {
        hnsw.neighbors[i++] = -1;
    }
}

template <class C>
void add_with_locks_impl(
        HNSW& hnsw,
        DistanceComputer& ptdis,
        int pt_level,
        int pt_id,
        LockVector& locks,
        VisitedTable& vt,
        bool keep_max_size_level0) {
    storage_idx_t nearest = hnsw.entry_point;
    if (nearest == -1) { // avoid locking after the first point.
#pragma omp critical
        if (hnsw.entry_point == -1) { // double-check under lock.
            hnsw.max_level = pt_level;
            hnsw.entry_point = pt_id;
            // leave nearest = -1 to trigger early exit after critical block.
        } else {
            // else: Another thread set the entry point.
            nearest = hnsw.entry_point;
        }
    }

    if (nearest < 0) {
        return;
    }

    locks.lock(pt_id);

    int level = hnsw.max_level; // level at which we start adding neighbors
    float d_nearest = ptdis(nearest);

    //  greedy search on upper levels
    for (; level > pt_level; level--) {
        greedy_update_nearest_impl<C>(hnsw, ptdis, level, nearest, d_nearest);
    }

    for (; level >= 0; level--) {
        add_links_starting_from_impl<C>(
                hnsw,
                ptdis,
                pt_id,
                nearest,
                d_nearest,
                level,
                locks,
                vt,
                keep_max_size_level0);
    }

    locks.unlock(pt_id);

#pragma omp critical
    {
        if (pt_level > hnsw.max_level) {
            hnsw.max_level = pt_level;
            hnsw.entry_point = pt_id;
        }
    }
}

} // namespace

void HNSW::add_with_locks(
        DistanceComputer& ptdis,
        int pt_level,
        int pt_id,
        LockVector& locks,
        VisitedTable& vt,
        bool keep_max_size_level0) {
    if (is_similarity) {
        add_with_locks_impl<C_similarity>(
                *this, ptdis, pt_level, pt_id, locks, vt, keep_max_size_level0);
    } else {
        add_with_locks_impl<C_distance>(
                *this, ptdis, pt_level, pt_id, locks, vt, keep_max_size_level0);
    }
}

void HNSW::compute_forward_links_deterministic(
        DistanceComputer& ptdis,
        int pt_level,
        storage_idx_t pt_id,
        VisitedTable& vt,
        std::vector<std::pair<storage_idx_t, int>>& pt_reverse_edges,
        bool keep_max_size_level0) {
    if (is_similarity) {
        compute_forward_links_impl<C_similarity>(
                *this,
                ptdis,
                pt_level,
                pt_id,
                vt,
                pt_reverse_edges,
                keep_max_size_level0);
    } else {
        compute_forward_links_impl<C_distance>(
                *this,
                ptdis,
                pt_level,
                pt_id,
                vt,
                pt_reverse_edges,
                keep_max_size_level0);
    }
}

void HNSW::merge_reverse_links_deterministic(
        DistanceComputer& dis,
        storage_idx_t node,
        int level,
        std::vector<storage_idx_t>& incoming,
        bool keep_max_size_level0) {
    if (is_similarity) {
        merge_reverse_links_impl<C_similarity>(
                *this, dis, node, level, incoming, keep_max_size_level0);
    } else {
        merge_reverse_links_impl<C_distance>(
                *this, dis, node, level, incoming, keep_max_size_level0);
    }
}

/**************************************************************
 * Searching
 **************************************************************/

namespace {

/** Helper to extract search parameters from HNSW and SearchParameters */
inline void extract_search_params(
        const HNSW& hnsw,
        const SearchParameters* params,
        bool& do_dis_check,
        int& efSearch,
        const IDSelector*& sel) {
    // can be overridden by search params
    do_dis_check = hnsw.check_relative_distance;
    efSearch = hnsw.efSearch;
    sel = nullptr;
    if (params) {
        if (const SearchParametersHNSW* hnsw_params =
                    dynamic_cast<const SearchParametersHNSW*>(params)) {
            do_dis_check = hnsw_params->check_relative_distance;
            efSearch = hnsw_params->efSearch;
        }
        sel = params->sel;
    }
}

template <class C>
std::vector<size_t> sorted_result_slots(ResultHandler& res, int k) {
    auto* heap = dynamic_cast<HeapResultHandler<C>*>(&res);
    FAISS_THROW_IF_NOT_MSG(
            heap,
            "HNSW feature collection requires a heap top-k result handler");

    std::vector<size_t> order;
    order.reserve(k);
    for (int slot = 0; slot < k; ++slot) {
        if (heap->heap_ids[slot] >= 0) {
            order.push_back(slot);
        }
    }

    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const float da = heap->heap_dis[a];
        const float db = heap->heap_dis[b];
        if (da == db) {
            return heap->heap_ids[a] < heap->heap_ids[b];
        }
        // C::cmp(x, y) means y is better than x. Therefore a is better
        // than b iff C::cmp(db, da).
        return C::cmp(db, da);
    });
    return order;
}

inline float stable_ratio(float numerator, float denominator) {
    constexpr float epsilon = 1e-12f;
    return (numerator + epsilon) / (denominator + epsilon);
}

template <class C>
void collect_feature_timestep(
        ResultHandler& res,
        const IndexHNSW& index,
        HNSWQueryTrace& trace,
        int32_t expanded_node_id,
        size_t pop_count,
        std::vector<int32_t>& previous_rank_ids,
        std::vector<size_t>& rank_persistence_counts) {
    std::vector<size_t> order = sorted_result_slots<C>(res, trace.k);
    if (order.size() < static_cast<size_t>(trace.k)) {
        return;
    }

    auto* heap = dynamic_cast<HeapResultHandler<C>*>(&res);
    FAISS_ASSERT(heap);

    HNSWFeatureTimestep timestep;
    timestep.expanded_node_id = expanded_node_id;
    timestep.pop_count = pop_count;
    timestep.first_distance = heap->heap_dis[order[0]];
    timestep.search_progress = static_cast<float>(pop_count) /
            static_cast<float>(trace.ef_search);
    timestep.node_ids.resize(trace.k);
    timestep.query_distances.resize(trace.k);
    timestep.features.resize(trace.k * HNSW_FEATURE_COUNT);

    const float timestep_count =
            static_cast<float>(trace.timesteps.size() + 1);
    for (int rank = 0; rank < trace.k; ++rank) {
        const size_t slot = order[rank];
        const int32_t node_id = static_cast<int32_t>(heap->heap_ids[slot]);
        const float node_distance = heap->heap_dis[slot];

        if (previous_rank_ids[rank] == node_id) {
            ++rank_persistence_counts[rank];
        } else {
            previous_rank_ids[rank] = node_id;
            rank_persistence_counts[rank] = 1;
        }

        timestep.node_ids[rank] = node_id;
        timestep.query_distances[rank] = node_distance;

        float* features = timestep.features.data() +
                rank * HNSW_FEATURE_COUNT;
        features[HNSW_FEATURE_RATIO_TO_ENTRY] =
                stable_ratio(node_distance, trace.entry_distance);
        features[HNSW_FEATURE_RATIO_TO_FIRST] = stable_ratio(
                timestep.first_distance, node_distance);
        features[HNSW_FEATURE_SEARCH_PROGRESS] =
                timestep.search_progress;
        features[HNSW_FEATURE_RANK_NORM] = trace.k == 1
                ? 0.0f
                : static_cast<float>(rank) /
                        static_cast<float>(trace.k - 1);
        features[HNSW_FEATURE_PERSISTENCE_RATE] =
                static_cast<float>(rank_persistence_counts[rank]) /
                timestep_count;
        features[HNSW_FEATURE_AVG_NEIGH_DIST] =
                index.level0_avg_neighbor_distance[node_id];
    }

    trace.timesteps.push_back(std::move(timestep));
}

template <class C>
void collect_final_top_k(ResultHandler& res, HNSWQueryTrace& trace) {
    std::vector<size_t> order = sorted_result_slots<C>(res, trace.k);
    auto* heap = dynamic_cast<HeapResultHandler<C>*>(&res);
    FAISS_ASSERT(heap);

    trace.final_top_k_ids.clear();
    trace.final_top_k_distances.clear();
    trace.final_top_k_ids.reserve(order.size());
    trace.final_top_k_distances.reserve(order.size());
    for (size_t slot : order) {
        trace.final_top_k_ids.push_back(
                static_cast<int32_t>(heap->heap_ids[slot]));
        trace.final_top_k_distances.push_back(heap->heap_dis[slot]);
    }
}

/** Persistent executor owned by the calling search thread. Queries on the
 * same search thread reuse its inference worker, including across search()
 * calls. Concurrent OpenMP search threads each have a separate executor.
 * Jobs must finish before their query-owned data is destroyed.
 */
int64_t profile_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
}

double profile_elapsed_ms(const HNSWSearchProfile* profile) {
    return (profile_now_ns() - profile->start_ns) / 1e6;
}

class HNSWAsyncOnnxWorker {
   public:
    HNSWAsyncOnnxWorker() : worker_([this] { run(); }) {
        // Do not start query latency accounting until the OS thread has
        // actually entered run(), including on the first async query.
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return started_; });
    }

    ~HNSWAsyncOnnxWorker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_one();
        worker_.join();
    }

    std::future<void> submit(std::packaged_task<void()> job) {
        auto completion = job.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            jobs_.push_back(std::move(job));
        }
        ready_.notify_one();
        return completion;
    }

   private:
    void run() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            started_ = true;
        }
        ready_.notify_one();
        while (true) {
            std::packaged_task<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [&] { return stopping_ || !jobs_.empty(); });
                if (jobs_.empty() && stopping_) {
                    return;
                }
                job = std::move(jobs_.front());
                jobs_.pop_front();
            }
            job();
        }
    }

    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::packaged_task<void()>> jobs_;
    bool stopping_ = false;
    bool started_ = false;
    // Start only after all state used by run() has been constructed.
    std::thread worker_;
};

HNSWAsyncOnnxWorker& async_onnx_worker() {
    thread_local HNSWAsyncOnnxWorker worker;
    return worker;
}

/** Asynchronous inference pipeline for one query.
 *
 * The HNSW thread deep-copies each observable result-queue snapshot into a
 * FIFO. A reusable worker processes the recurrent HNSWOnnxQuery, preserving
 * LSTM state order, and submits predicted raw-data reads as soon as each
 * prediction completes. At traversal completion, cancel_pending() discards
 * queued snapshots. Final payload I/O starts before the active prediction is
 * joined; the raw-data query synchronizes request-map access.
 */
class HNSWAsyncOnnxQuery {
   public:
    HNSWAsyncOnnxQuery(
            std::unique_ptr<HNSWOnnxQuery> query,
            float threshold,
            HNSWRawDataQuery* raw_data_query,
            HNSWSearchProfile* profile)
            : query_(std::move(query)),
              threshold_(threshold),
              raw_data_query_(raw_data_query),
              profile_(profile) {
        completion_ = async_onnx_worker().submit(
                std::packaged_task<void()>([this] { run(); }));
    }

    ~HNSWAsyncOnnxQuery() {
        close_and_wait();
    }

    void enqueue(size_t timestep_index, const HNSWFeatureTimestep& timestep) {
        Task task;
        task.timestep_index = timestep_index;
        task.node_ids = timestep.node_ids;
        task.features = timestep.features;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            FAISS_THROW_IF_NOT_MSG(!closed_, "async ONNX queue is closed");
            if (profile_) {
                task.enqueued_ms = profile_elapsed_ms(profile_);
                profile_->peak_snapshot_queue = std::max(
                        profile_->peak_snapshot_queue, tasks_.size() + 1);
            }
            tasks_.push_back(std::move(task));
        }
        ready_.notify_one();
    }

    void cancel_pending() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cancelled_.store(true, std::memory_order_relaxed);
            closed_ = true;
            tasks_.clear();
        }
        ready_.notify_one();
    }

    void finish(HNSWQueryTrace& trace) {
        close_and_wait();
        if (error_) {
            std::rethrow_exception(error_);
        }
        FAISS_THROW_IF_NOT_FMT(
                cancelled_.load(std::memory_order_relaxed) ||
                        completed_.size() == trace.timesteps.size(),
                "async ONNX completed %zu of %zu timesteps",
                completed_.size(),
                trace.timesteps.size());
        for (Result& result : completed_) {
            FAISS_THROW_IF_NOT_FMT(
                    result.timestep_index < trace.timesteps.size(),
                    "async ONNX result timestep %zu is out of range",
                    result.timestep_index);
            HNSWFeatureTimestep& timestep =
                    trace.timesteps[result.timestep_index];
            timestep.prediction_probabilities =
                    std::move(result.probabilities);
            timestep.prefetch_decisions = std::move(result.decisions);
        }
        trace.gpu_pipeline_submitted = gpu_submitted_;
        trace.gpu_pipeline_peak_inflight = gpu_peak_inflight_;
    }

   private:
    struct Task {
        size_t timestep_index = 0;
        double enqueued_ms = 0;
        std::vector<int32_t> node_ids;
        std::vector<float> features;
    };

    struct Result {
        size_t timestep_index = 0;
        std::vector<float> probabilities;
        std::vector<uint8_t> decisions;
    };

    void close_and_wait() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        ready_.notify_one();
        if (completion_.valid()) {
            completion_.wait();
        }
    }

    void run() noexcept {
        try {
            if (query_->gpu_pipeline_depth()) {
                run_gpu_pipeline();
                return;
            }
            while (true) {
                Task task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    ready_.wait(lock, [&] { return closed_ || !tasks_.empty(); });
                    if (tasks_.empty()) {
                        if (closed_) {
                            break;
                        }
                        continue;
                    }
                    task = std::move(tasks_.front());
                    tasks_.pop_front();
                }

                Result result;
                result.timestep_index = task.timestep_index;
                HNSWInferenceProfileEvent event;
                if (profile_) {
                    event.timestep = task.timestep_index;
                    event.enqueued_ms = task.enqueued_ms;
                    event.started_ms = profile_elapsed_ms(profile_);
                }
                result.probabilities = query_->predict(
                        task.features.data(), task.features.size(),
                        profile_ ? &event.predict : nullptr);
                if (profile_) {
                    event.predicted_ms = profile_elapsed_ms(profile_);
                }
                result.decisions.resize(result.probabilities.size());
                for (size_t rank = 0; rank < result.probabilities.size();
                     ++rank) {
                    result.decisions[rank] = static_cast<uint8_t>(
                            result.probabilities[rank] >= threshold_);
                    if (!cancelled_.load(std::memory_order_relaxed) &&
                        raw_data_query_ && result.decisions[rank] &&
                        task.node_ids[rank] >= 0) {
                        raw_data_query_->submit(task.node_ids[rank]);
                    }
                }
                if (profile_) {
                    event.submitted_ms = profile_elapsed_ms(profile_);
                    profile_->predict_ms += event.predicted_ms - event.started_ms;
                    profile_->io_submit_ms += event.submitted_ms - event.predicted_ms;
                    profile_->inference_events.push_back(event);
                }
                completed_.push_back(std::move(result));
            }
        } catch (...) {
            error_ = std::current_exception();
        }
    }

    void run_gpu_pipeline() {
        struct Pending {
            Task snapshot;
            HNSWInferenceProfileEvent event;
        };
        std::deque<Pending> pending;
        auto complete = [&](std::vector<float> probabilities) {
            Task snapshot = std::move(pending.front().snapshot);
            HNSWInferenceProfileEvent event = pending.front().event;
            pending.pop_front();
            Result result;
            result.timestep_index = snapshot.timestep_index;
            result.probabilities = std::move(probabilities);
            result.decisions.resize(result.probabilities.size());
            for (size_t rank = 0; rank < result.probabilities.size(); ++rank) {
                result.decisions[rank] = static_cast<uint8_t>(
                        result.probabilities[rank] >= threshold_);
                if (!cancelled_.load(std::memory_order_relaxed) &&
                    raw_data_query_ && result.decisions[rank] &&
                    snapshot.node_ids[rank] >= 0) {
                    raw_data_query_->submit(snapshot.node_ids[rank]);
                }
            }
            if (profile_) {
                event.submitted_ms = profile_elapsed_ms(profile_);
                profile_->predict_ms += event.gpu_enqueued_ms - event.started_ms +
                        event.predicted_ms - event.collect_started_ms;
                profile_->io_submit_ms += event.submitted_ms - event.predicted_ms;
                profile_->inference_events.push_back(event);
            }
            completed_.push_back(std::move(result));
        };
        auto collect = [&](std::vector<float>& probabilities, bool wait) {
            auto& event = pending.front().event;
            const double started = profile_ ? profile_elapsed_ms(profile_) : 0;
            const bool ready = query_->collect_gpu(
                    probabilities, wait, profile_ ? &event.predict : nullptr);
            if (profile_ && ready) {
                event.collect_started_ms = started;
                event.predicted_ms = profile_elapsed_ms(profile_);
            }
            return ready;
        };
        while (true) {
            // Reap a ready result before submitting more work so prefetch
            // decisions are not held until every GPU request has completed.
            std::vector<float> probabilities;
            if (!pending.empty() && collect(probabilities, false)) {
                complete(std::move(probabilities));
                continue;
            }
            Task task;
            bool have_task = false;
            bool closed_and_empty = false;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                if (pending.empty()) {
                    ready_.wait(lock, [&] { return closed_ || !tasks_.empty(); });
                }
                if (!tasks_.empty() &&
                    pending.size() < query_->gpu_pipeline_depth()) {
                    task = std::move(tasks_.front());
                    tasks_.pop_front();
                    have_task = true;
                }
                closed_and_empty = closed_ && tasks_.empty();
            }
            if (have_task) {
                HNSWInferenceProfileEvent event;
                if (profile_) {
                    event.timestep = task.timestep_index;
                    event.enqueued_ms = task.enqueued_ms;
                    event.started_ms = profile_elapsed_ms(profile_);
                }
                query_->submit_gpu(task.features.data(), task.features.size(),
                                   profile_ ? &event.predict : nullptr);
                if (profile_) event.gpu_enqueued_ms = profile_elapsed_ms(profile_);
                pending.push_back({std::move(task), event});
                ++gpu_submitted_;
                gpu_peak_inflight_ = std::max(gpu_peak_inflight_, pending.size());
            } else if (!pending.empty() &&
                       (pending.size() == query_->gpu_pipeline_depth() ||
                        closed_and_empty)) {
                // Wait only for bounded backpressure or final drain. While
                // the producer is active and slots are available, a new
                // snapshot must not be blocked behind the oldest GPU result.
                FAISS_THROW_IF_NOT(collect(probabilities, true));
                complete(std::move(probabilities));
            } else if (!pending.empty()) {
                // CUDA completion events cannot notify this CPU condition
                // variable. Poll them with a short timed wait, but wake
                // immediately when search enqueues another snapshot.
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait_for(lock, std::chrono::microseconds(50), [&] {
                    return closed_ || !tasks_.empty();
                });
            } else if (closed_and_empty) {
                break;
            }
        }
    }

    std::unique_ptr<HNSWOnnxQuery> query_;
    float threshold_;
    HNSWRawDataQuery* raw_data_query_;
    HNSWSearchProfile* profile_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<Task> tasks_;
    bool closed_ = false;
    std::atomic<bool> cancelled_{false};
    std::vector<Result> completed_;
    std::exception_ptr error_;
    std::future<void> completion_;
    size_t gpu_submitted_ = 0;
    size_t gpu_peak_inflight_ = 0;
};

/** Templated body of `search_from_candidates` — instantiated once per
 * VisitedTable subclass × comparator.
 */
template <typename VTType, class C>
int search_from_candidates_fixVT(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        ResultHandler& res,
        MinimaxHeapT<HC_for<C>>& candidates,
        VTType& vt,
        HNSWStats& stats,
        int level,
        int nres_in,
        const SearchParameters* params,
        const IndexHNSW* index = nullptr,
        HNSWQueryTrace* feature_trace = nullptr,
        HNSWOnnxQuery* onnx_query = nullptr,
        HNSWAsyncOnnxQuery* async_onnx_query = nullptr,
        float onnx_threshold = 0.5f,
        HNSWRawDataQuery* raw_data_query = nullptr,
        const std::unordered_set<int32_t>* oracle_final_ids = nullptr,
        const HNSWPrefetchReplayQuery* prefetch_replay = nullptr) {
    int nres = nres_in;
    int ndis = 0;

    bool do_dis_check;
    int efSearch;
    const IDSelector* sel;
    extract_search_params(hnsw, params, do_dis_check, efSearch, sel);

    const auto* profile_params = dynamic_cast<const SearchParametersHNSW*>(params);
    HNSWSearchProfile* profile =
            profile_params ? profile_params->search_profile : nullptr;

    vt.reserve(efSearch);

    typename C::T threshold = res.threshold;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (C::cmp(threshold, d)) {
                if (res.add_result(d, v1)) {
                    threshold = res.threshold;
                }
            }
        }
        vt.set(v1);
    }

    int nstep = 0;
    std::vector<int32_t> previous_rank_ids;
    std::vector<size_t> rank_persistence_counts;
    if (feature_trace) {
        FAISS_THROW_IF_NOT_MSG(
                level == 0 && index,
                "HNSW feature collection is only available for IndexHNSW "
                "level-0 search");
        previous_rank_ids.assign(feature_trace->k, -1);
        rank_persistence_counts.assign(feature_trace->k, 0);
    }

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0

            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;
            }
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        // a faster version: reference version in unit test test_hnsw.cpp
        // the following version processes 4 neighbors at a time
        size_t jmax = begin;
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }

            vt.prefetch(v1);
            jmax += 1;
        }

        int counter = 0;
        size_t saved_j[4];

        threshold = res.threshold;

        auto add_to_heap = [&](const size_t idx, const float dis) {
            if (!sel || sel->is_member(idx)) {
                if (C::cmp(threshold, dis)) {
                    if (res.add_result(dis, idx)) {
                        threshold = res.threshold;
                        nres += 1;
                    }
                }
            }
            candidates.push(idx, dis);
        };

        for (size_t j = begin; j < jmax; j++) {
            int v1 = hnsw.neighbors[j];

            saved_j[counter] = v1;
            counter += vt.set(v1) ? 1 : 0;

            if (counter == 4) {
                float dis[4];
                qdis.distances_batch_4(
                        saved_j[0],
                        saved_j[1],
                        saved_j[2],
                        saved_j[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    add_to_heap(saved_j[id4], dis[id4]);
                }

                ndis += 4;

                counter = 0;
            }
        }

        for (int icnt = 0; icnt < counter; icnt++) {
            float dis = qdis(saved_j[icnt]);
            add_to_heap(saved_j[icnt], dis);

            ndis += 1;
        }

        nstep++;
        if (feature_trace) {
            const int64_t feature_started = profile ? profile_now_ns() : 0;
            collect_feature_timestep<C>(
                    res,
                    *index,
                    *feature_trace,
                    v0,
                    nstep,
                    previous_rank_ids,
                    rank_persistence_counts);
            if (profile) {
                profile->feature_ms += (profile_now_ns() - feature_started) / 1e6;
            }
            if (prefetch_replay && !feature_trace->timesteps.empty()) {
                const size_t timestep_index = static_cast<size_t>(nstep - 1);
                // A replay run may execute trailing efSearch-boundary steps
                // that were absent while recording. There is no recorded
                // model decision for those steps, so they intentionally submit
                // nothing; all recorded decisions retain their exact timing.
                if (timestep_index <
                    prefetch_replay->timestep_node_ids.size()) {
                    for (int32_t node_id :
                         prefetch_replay->timestep_node_ids[timestep_index]) {
                        if (raw_data_query && node_id >= 0) {
                            raw_data_query->submit(node_id);
                        }
                    }
                }
            } else if (
                    oracle_final_ids && !feature_trace->timesteps.empty()) {
                // IDEAL ORACLE EVALUATION ONLY: the model has zero execution
                // cost and every final top-k node is submitted as soon as it
                // appears in the observable result queue. submit() deduplicates
                // nodes that remain in the queue across later timesteps.
                const HNSWFeatureTimestep& timestep =
                        feature_trace->timesteps.back();
                for (int32_t node_id : timestep.node_ids) {
                    if (raw_data_query && node_id >= 0 &&
                        oracle_final_ids->count(node_id) != 0) {
                        raw_data_query->submit(node_id);
                    }
                }
            } else if (
                    async_onnx_query && !feature_trace->timesteps.empty()) {
                const int64_t enqueue_started = profile ? profile_now_ns() : 0;
                async_onnx_query->enqueue(
                        feature_trace->timesteps.size() - 1,
                        feature_trace->timesteps.back());
                if (profile) {
                    profile->enqueue_ms += (profile_now_ns() - enqueue_started) / 1e6;
                }
            } else if (onnx_query && !feature_trace->timesteps.empty()) {
                HNSWFeatureTimestep& timestep =
                        feature_trace->timesteps.back();
                timestep.prediction_probabilities = onnx_query->predict(
                        timestep.features.data(), timestep.features.size());
                timestep.prefetch_decisions.resize(
                        timestep.prediction_probabilities.size());
                for (size_t rank = 0;
                     rank < timestep.prediction_probabilities.size();
                     ++rank) {
                    timestep.prefetch_decisions[rank] =
                            static_cast<uint8_t>(
                                    timestep.prediction_probabilities[rank] >=
                                    onnx_threshold);
                    if (raw_data_query &&
                        timestep.prefetch_decisions[rank] &&
                        timestep.node_ids[rank] >= 0) {
                        raw_data_query->submit(timestep.node_ids[rank]);
                    }
                }
            }
        }
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    FAISS_THROW_IF_NOT_FMT(
            !prefetch_replay ||
                    static_cast<size_t>(nstep) >=
                            prefetch_replay->timestep_node_ids.size(),
            "prefetch replay timestep count %zu does not match search %d",
            prefetch_replay ? prefetch_replay->timestep_node_ids.size() : 0,
            nstep);

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.ndis += ndis;
        stats.nhops += nstep;
    }

    return nres;
}

/// Dispatches the VisitedTable concrete type for a given C, then calls
/// the templated `search_from_candidates_fixVT<VTType, C>`.
template <class C>
int search_from_candidates_dispatch(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        ResultHandler& res,
        MinimaxHeapT<HC_for<C>>& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in,
        const SearchParameters* params,
        const IndexHNSW* index = nullptr,
        HNSWQueryTrace* feature_trace = nullptr,
        HNSWOnnxQuery* onnx_query = nullptr,
        HNSWAsyncOnnxQuery* async_onnx_query = nullptr,
        float onnx_threshold = 0.5f,
        HNSWRawDataQuery* raw_data_query = nullptr,
        const std::unordered_set<int32_t>* oracle_final_ids = nullptr,
        const HNSWPrefetchReplayQuery* prefetch_replay = nullptr) {
    auto call = [&]<typename VTType>(VTType& vt_concrete) -> int {
        return search_from_candidates_fixVT<VTType, C>(
                hnsw,
                qdis,
                res,
                candidates,
                vt_concrete,
                stats,
                level,
                nres_in,
                params,
                index,
                feature_trace,
                onnx_query,
                async_onnx_query,
                onnx_threshold,
                raw_data_query,
                oracle_final_ids,
                prefetch_replay);
    };
    if (VisitedTableVector* vtv = dynamic_cast<VisitedTableVector*>(&vt)) {
        return call(*vtv);
    }
    VisitedTableSet& vts = dynamic_cast<VisitedTableSet&>(vt);
    return call(vts);
}

} // namespace

/** Do a BFS on the candidates list. Public dispatcher: only handles the
 *  distance (CMax) flavor because its `MinimaxHeap` parameter is the
 *  CMax instantiation. */
int hnsw_detail::search_from_candidates(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        ResultHandler& res,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in,
        const SearchParameters* params) {
    return search_from_candidates_dispatch<HNSW::C_distance>(
            hnsw, qdis, res, candidates, vt, stats, level, nres_in, params);
}

int hnsw_detail::search_from_candidates_panorama(
        const HNSW& hnsw,
        const IndexHNSW* index,
        DistanceComputer& qdis,
        ResultHandler& res,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in,
        const SearchParameters* params) {
    // Panorama's progressive-bound math is L2-specific: refuse to run in
    // similarity mode.
    FAISS_THROW_IF_MSG(
            hnsw.is_similarity,
            "search_from_candidates_panorama does not support is_similarity=true");

    using C = HNSW::C_distance;

    int nres = nres_in;
    int ndis = 0;

    bool do_dis_check;
    int efSearch;
    const IDSelector* sel;
    extract_search_params(hnsw, params, do_dis_check, efSearch, sel);

    C::T threshold = res.threshold;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (C::cmp(threshold, d)) {
                if (res.add_result(d, v1)) {
                    threshold = res.threshold;
                }
            }
        }
        vt.set(v1);
    }

    // Validate the index type so we can access cumulative sums, n_levels, and
    // get the ability to compute partial dot products.
    const auto* panorama_index =
            dynamic_cast<const IndexHNSWFlatPanorama*>(index);
    FAISS_THROW_IF_NOT_MSG(
            panorama_index, "Index must be a IndexHNSWFlatPanorama");
    auto* flat_codes_qdis = dynamic_cast<FlatCodesDistanceComputer*>(&qdis);
    FAISS_THROW_IF_NOT_MSG(
            flat_codes_qdis,
            "DistanceComputer must be a FlatCodesDistanceComputer");

    const auto& pano = panorama_index->pano;
    const size_t nb_per_parent = static_cast<size_t>(hnsw.nb_neighbors(level));
    const size_t num_panorama_levels = pano.n_levels;
    const size_t level_width_floats = pano.level_width_floats;

    constexpr size_t kTargetBatch = 64;
    const size_t buf_cap = kTargetBatch + nb_per_parent;
    std::vector<uint32_t> index_array(buf_cap);
    std::vector<float> exact_distances(buf_cap);
    std::vector<float> dot_buffer(buf_cap);
    std::vector<float> query_cum_sums_buf(num_panorama_levels + 1);

    const float* query = flat_codes_qdis->q;
    const size_t d = static_cast<size_t>(panorama_index->d);

    const float* cum_base = panorama_index->get_cum_sum(0);
    const size_t cum_stride = num_panorama_levels + 1;
    const auto* flat_storage =
            static_cast<const IndexFlat*>(panorama_index->storage);
    const float* xb_base =
            reinterpret_cast<const float*>(flat_storage->codes.data());
    const size_t feat_stride = d;

    pano.compute_query_cum_sums(query, query_cum_sums_buf.data());
    const float* query_cum_sums = query_cum_sums_buf.data();
    const float query_norm_sq = query_cum_sums[0] * query_cum_sums[0];

    int nstep = 0;

    PanoramaStats local_pano_stats;
    local_pano_stats.reset();

    bool stop_flag = false;
    while (candidates.size() > 0 && !stop_flag) {
        size_t initial_size = 0;
        size_t k_popped = 0;
        while (initial_size < kTargetBatch && candidates.size() > 0) {
            float d0 = 0;
            int v0 = candidates.pop_min(&d0);
            if (do_dis_check) {
                int n_dis_below = candidates.count_below(d0);
                if (n_dis_below >= efSearch) {
                    if (k_popped == 0) {
                        // Standard early-stop: nothing queued this
                        // iteration, terminate the outer loop entirely
                        // (matches the single-pop path's behavior).
                        stop_flag = true;
                    } else {
                        // We already have parents queued; un-pop this
                        // one so the next outer iteration sees it and
                        // re-applies the stop check from a clean state.
                        candidates.push(v0, d0);
                    }
                    break;
                }
            }
            k_popped++;

            size_t begin, end;
            hnsw.neighbor_range(v0, level, &begin, &end);
            for (size_t j = begin; j < end; j++) {
                int v1 = hnsw.neighbors[j];
                if (v1 < 0) {
                    break;
                }

                bool is_new = vt.set(v1);
                bool is_selected = !sel || sel->is_member(v1);
                if (is_new && is_selected) {
                    const float vsum =
                            cum_base[static_cast<size_t>(v1) * cum_stride];
                    index_array[initial_size] = v1;
                    exact_distances[initial_size] = query_norm_sq + vsum * vsum;
                    initial_size++;
                }
            }
        }

        local_pano_stats.total_dims += initial_size * d;

        size_t batch_size = initial_size;
        size_t curr_panorama_level = 0;
        while (curr_panorama_level < num_panorama_levels && batch_size > 0) {
            const size_t cs_level_idx = curr_panorama_level + 1;
            const float query_cum_norm = query_cum_sums[cs_level_idx];
            const float two_qc = 2.0f * query_cum_norm;

            const size_t start_dim = curr_panorama_level * level_width_floats;
            size_t end_dim = (curr_panorama_level + 1) * level_width_floats;
            end_dim = std::min(end_dim, d);
            const size_t dim_span = end_dim - start_dim;

            const float* level_base = xb_base + start_dim;
            with_level_width(dim_span, [&]<size_t W>() {
                compute_level_dot_kernel<false, W>(
                        query + start_dim,
                        level_base,
                        index_array.data(),
                        batch_size,
                        dim_span,
                        dot_buffer.data(),
                        feat_stride);
            });
            ndis += batch_size;

            size_t next_batch_size = 0;

            for (size_t i = 0; i < batch_size; i++) {
                float ne = exact_distances[i] - 2.0f * dot_buffer[i];
                float cum = cum_base
                        [static_cast<size_t>(index_array[i]) * cum_stride +
                         cs_level_idx];
                float lb = ne - two_qc * cum;
                if (lb <= threshold) {
                    exact_distances[next_batch_size] = ne;
                    index_array[next_batch_size] = index_array[i];
                    next_batch_size++;
                } else {
                    candidates.push(index_array[i], ne);
                }
            }

            local_pano_stats.total_dims_scanned += batch_size * dim_span;
            batch_size = next_batch_size;
            curr_panorama_level++;
        }

        // Add surviving candidates to the result handler.
        for (size_t i = 0; i < batch_size; i++) {
            idx_t idx = index_array[i];
            if (res.add_result(exact_distances[i], idx)) {
                threshold = res.threshold;
                nres += 1;
            }
            candidates.push(idx, exact_distances[i]);
        }

        nstep += static_cast<int>(k_popped);
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.ndis += ndis;
        stats.nhops += nstep;
    }

    indexPanorama_stats.add(local_pano_stats);
    return nres;
}

namespace {

template <typename T, typename Container, typename Compare>
void reservePriorityQueue(
        std::priority_queue<T, Container, Compare>& q,
        std::size_t size) {
    struct Access : std::priority_queue<T, Container, Compare> {
        using std::priority_queue<T, Container, Compare>::c;
    };
    Access access{std::move(q)};
    access.c.reserve(size);
    q = std::move(access);
}

/// Templated body of `search_from_candidate_unbounded`. The choice of
/// max-heap vs min-heap for both `top_candidates` and `candidates` is
/// derived from C via `TopCandidatesQueue` / `CandidatesQueue`.
template <typename VTType, class C>
TopCandidatesQueue<C> search_from_candidate_unbounded_fixVT(
        const HNSW& hnsw,
        const HNSW::Node& node,
        DistanceComputer& qdis,
        int ef,
        VTType& vt,
        HNSWStats& stats) {
    int ndis = 0;
    TopCandidatesQueue<C> top_candidates;
    reservePriorityQueue(top_candidates, ef);

    CandidatesQueue<C> candidates;
    reservePriorityQueue(candidates, ef);

    top_candidates.push(node);
    candidates.push(node);

    vt.set(node.second);

    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (C::cmp(d0, top_candidates.top().first)) {
            break;
        }

        candidates.pop();

        size_t begin, end;
        hnsw.neighbor_range(v0, 0, &begin, &end);

        // a faster version: reference version in unit test test_hnsw.cpp
        // the following version processes 4 neighbors at a time
        size_t jmax = begin;
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }

            vt.prefetch(v1);
            jmax += 1;
        }

        int counter = 0;
        size_t saved_j[4];

        auto add_to_heap = [&](const size_t idx, const float dis) {
            if (C::cmp(top_candidates.top().first, dis) ||
                top_candidates.size() < static_cast<size_t>(ef)) {
                candidates.emplace(dis, idx);
                top_candidates.emplace(dis, idx);

                if (top_candidates.size() > static_cast<size_t>(ef)) {
                    top_candidates.pop();
                }
            }
        };

        for (size_t j = begin; j < jmax; j++) {
            int v1 = hnsw.neighbors[j];

            saved_j[counter] = v1;
            counter += vt.set(v1) ? 1 : 0;

            if (counter == 4) {
                float dis[4];
                qdis.distances_batch_4(
                        saved_j[0],
                        saved_j[1],
                        saved_j[2],
                        saved_j[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    add_to_heap(saved_j[id4], dis[id4]);
                }

                ndis += 4;

                counter = 0;
            }
        }

        for (int icnt = 0; icnt < counter; icnt++) {
            float dis = qdis(saved_j[icnt]);
            add_to_heap(saved_j[icnt], dis);

            ndis += 1;
        }

        stats.nhops += 1;
    }

    ++stats.n1;
    if (candidates.size() == 0) {
        ++stats.n2;
    }
    stats.ndis += ndis;

    return top_candidates;
}

} // namespace

/// Public dispatcher: only the distance (CMax) flavor is exposed because
/// its return type — `std::priority_queue<HNSW::Node>` — is the CMax
/// max-heap. Internal callers that need similarity mode use the same
/// dispatch pattern inline.
std::priority_queue<HNSW::Node> hnsw_detail::search_from_candidate_unbounded(
        const HNSW& hnsw,
        const HNSW::Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        HNSWStats& stats) {
    using C = HNSW::C_distance;
    auto call = [&]<typename VTType>(VTType& vt_concrete) {
        return search_from_candidate_unbounded_fixVT<VTType, C>(
                hnsw, node, qdis, ef, vt_concrete, stats);
    };
    if (VisitedTableVector* vtv = dynamic_cast<VisitedTableVector*>(vt)) {
        return call(*vtv);
    }
    VisitedTableSet& vts = dynamic_cast<VisitedTableSet&>(*vt);
    return call(vts);
}

namespace {

// just used as a lower bound for the minmaxheap, but it is set for heap search
template <class C>
int extract_k_from_ResultHandler(ResultHandler& res) {
    using RH = HeapBlockResultHandler<C>;
    if (auto hres = dynamic_cast<typename RH::SingleResultHandler*>(&res)) {
        return hres->k;
    }
    return 1;
}

template <class C>
HNSWStats search_impl(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        const IndexHNSW* index,
        ResultHandler& res,
        VisitedTable& vt,
        const SearchParameters* params,
        HNSWQueryTrace* feature_trace) {
    HNSWStats stats;
    if (hnsw.entry_point == -1) {
        return stats;
    }
    int k = extract_k_from_ResultHandler<C>(res);

    bool bounded_queue = hnsw.search_bounded_queue;
    int cur_efSearch = hnsw.efSearch;
    HNSWOnnxModel* onnx_model = nullptr;
    float onnx_threshold = 0.5f;
    bool onnx_async_inference = false;
    HNSWSearchProfile* profile = nullptr;
    HNSWRawDataPrefetcher* raw_data_prefetcher = nullptr;
    bool prefetch_enabled = false;
    bool oracle_prefetch = false;
    const idx_t* oracle_final_ids = nullptr;
    size_t oracle_k = 0;
    const HNSWPrefetchReplayQuery* prefetch_replay = nullptr;
    if (params) {
        if (const SearchParametersHNSW* hnsw_params =
                    dynamic_cast<const SearchParametersHNSW*>(params)) {
            bounded_queue = hnsw_params->bounded_queue;
            cur_efSearch = hnsw_params->efSearch;
            onnx_model = hnsw_params->onnx_model;
            onnx_threshold = hnsw_params->onnx_threshold;
            onnx_async_inference = hnsw_params->onnx_async_inference;
            profile = hnsw_params->search_profile;
            raw_data_prefetcher = hnsw_params->raw_data_prefetcher;
            prefetch_enabled = hnsw_params->prefetch_enabled;
            oracle_prefetch = hnsw_params->oracle_prefetch;
            oracle_final_ids = hnsw_params->oracle_final_ids;
            oracle_k = hnsw_params->oracle_k;
            prefetch_replay = hnsw_params->prefetch_replay_queries;
        }
    }

    FAISS_THROW_IF_NOT_MSG(
            !prefetch_enabled || raw_data_prefetcher,
            "prefetch_enabled=true requires a raw data prefetcher");
    FAISS_THROW_IF_NOT_MSG(
            !prefetch_enabled || onnx_model || oracle_prefetch ||
                    prefetch_replay,
            "prefetch_enabled=true requires ONNX, oracle, or replay mode");
    FAISS_THROW_IF_NOT_MSG(
            !(onnx_model && prefetch_replay),
            "prefetch replay must not execute an ONNX model");
    FAISS_THROW_IF_NOT_MSG(
            !onnx_model || !onnx_model->gpu_pipeline_depth() || onnx_async_inference,
            "GPU pipeline requires onnx_async_inference=true");
    FAISS_THROW_IF_NOT_MSG(
            !(oracle_prefetch && prefetch_replay),
            "prefetch replay and oracle mode are mutually exclusive");
    FAISS_THROW_IF_NOT_MSG(
            !oracle_prefetch || prefetch_enabled,
            "oracle prefetch requires prefetch_enabled=true");
    FAISS_THROW_IF_NOT_MSG(
            !oracle_prefetch || oracle_final_ids,
            "oracle prefetch requires final top-k IDs");
    FAISS_THROW_IF_NOT_FMT(
            !oracle_prefetch || oracle_k == static_cast<size_t>(k),
            "oracle top-k %zu does not match search k %d",
            oracle_k,
            k);

    std::unordered_set<int32_t> oracle_final_set;
    if (oracle_prefetch) {
        // ORACLE EVALUATION ONLY: materialize the known final result as an
        // O(1)-membership filter for I/O submission. This set must never be
        // used to alter HNSW traversal, result insertion, or model inference.
        oracle_final_set.reserve(oracle_k);
        for (size_t rank = 0; rank < oracle_k; ++rank) {
            FAISS_THROW_IF_NOT_MSG(
                    oracle_final_ids[rank] >= 0,
                    "oracle final top-k contains an invalid node ID");
            oracle_final_set.insert(
                    static_cast<int32_t>(oracle_final_ids[rank]));
        }
    }
    if (onnx_model && onnx_async_inference) {
        // Create and start the persistent inference thread before
        // create_query() records search_started. Later queries reuse it.
        async_onnx_worker();
    }
    std::unique_ptr<HNSWRawDataQuery> raw_data_query;
    if (profile) {
        *profile = HNSWSearchProfile{};
        profile->start_ns = profile_now_ns();
    }
    if (raw_data_prefetcher) {
        raw_data_query =
                raw_data_prefetcher->create_query(prefetch_enabled, profile);
    }

    //  greedy search on upper levels
    storage_idx_t nearest = hnsw.entry_point;
    float d_nearest = qdis(nearest);

    for (int level = hnsw.max_level; level >= 1; level--) {
        HNSWStats local_stats = greedy_update_nearest_impl<C>(
                hnsw, qdis, level, nearest, d_nearest);
        stats.combine(local_stats);
    }

    if (profile) {
        profile->upper_end_ms = profile_elapsed_ms(profile);
    }
    HNSWQueryTrace onnx_only_trace;
    std::unique_ptr<HNSWOnnxQuery> onnx_query;
    std::unique_ptr<HNSWAsyncOnnxQuery> async_onnx_query;
    if (onnx_model) {
        FAISS_THROW_IF_NOT_MSG(
                index,
                "HNSW ONNX inference requires an IndexHNSW instance");
        FAISS_THROW_IF_NOT_MSG(
                index->metric_type == METRIC_L2,
                "HNSW ONNX inference currently supports METRIC_L2 only");
        FAISS_THROW_IF_NOT_MSG(
                bounded_queue,
                "HNSW ONNX inference requires bounded_queue=true");
        FAISS_THROW_IF_NOT_MSG(
                !hnsw.is_panorama,
                "HNSW ONNX inference does not support Panorama search");
        FAISS_THROW_IF_NOT_FMT(
                onnx_model->topk() == k,
                "ONNX model top-k %d does not match search k %d",
                onnx_model->topk(),
                k);
        FAISS_THROW_IF_NOT_MSG(
                onnx_threshold >= 0.0f && onnx_threshold <= 1.0f,
                "HNSW ONNX threshold must be between 0 and 1");
        FAISS_THROW_IF_NOT_MSG(
                index->level0_avg_neighbor_distance.size() ==
                        static_cast<size_t>(index->ntotal),
                "level-0 average neighbor distances are unavailable; call "
                "compute_level0_avg_neighbor_distance() before ONNX search");
        if (!feature_trace) {
            feature_trace = &onnx_only_trace;
        }
        onnx_query = onnx_model->create_query();
        if (onnx_async_inference) {
            async_onnx_query = std::make_unique<HNSWAsyncOnnxQuery>(
                    std::move(onnx_query),
                    onnx_threshold,
                    raw_data_query.get(),
                    profile);
        }
    }

    if (feature_trace) {
        FAISS_THROW_IF_NOT_MSG(
                index,
                "HNSW feature collection requires an IndexHNSW instance");
        FAISS_THROW_IF_NOT_MSG(
                bounded_queue,
                "HNSW feature collection requires bounded_queue=true");
        FAISS_THROW_IF_NOT_MSG(
                !hnsw.is_panorama,
                "HNSW feature collection does not support Panorama search");
        feature_trace->k = k;
        feature_trace->ef_search = cur_efSearch;
        feature_trace->entry_distance = d_nearest;
        feature_trace->timesteps.clear();
        feature_trace->final_top_k_ids.clear();
        feature_trace->final_top_k_distances.clear();
    }

    if (profile) {
        profile->setup_end_ms = profile_elapsed_ms(profile);
    }
    int ef = std::max(cur_efSearch, k);
    if (bounded_queue) { // this is the most common branch, for now we only
                         // support Panorama search in this branch
        MinimaxHeapT<HC_for<C>> candidates(ef);

        candidates.push(nearest, d_nearest);

        if (!hnsw.is_panorama) {
            search_from_candidates_dispatch<C>(
                    hnsw,
                    qdis,
                    res,
                    candidates,
                    vt,
                    stats,
                    0,
                    0,
                    params,
                    index,
                    feature_trace,
                    onnx_query.get(),
                    async_onnx_query.get(),
                    onnx_threshold,
                    raw_data_query.get(),
                    oracle_prefetch ? &oracle_final_set : nullptr,
                    prefetch_replay);
        } else {
            // Panorama is L2-specific and is only valid for C_distance.
            // The public dispatch ensures we never reach this code path
            // with C != C_distance, but assert in debug builds.
            if constexpr (std::is_same_v<C, HNSW::C_distance>) {
                hnsw_detail::search_from_candidates_panorama(
                        hnsw,
                        index,
                        qdis,
                        res,
                        candidates,
                        vt,
                        stats,
                        0,
                        0,
                        params);
            } else {
                FAISS_THROW_MSG(
                        "Panorama search is not supported with is_similarity=true");
            }
        }
    } else {
        auto call = [&]<typename VTType>(VTType& vt_concrete) {
            return search_from_candidate_unbounded_fixVT<VTType, C>(
                    hnsw,
                    HNSW::Node(d_nearest, nearest),
                    qdis,
                    ef,
                    vt_concrete,
                    stats);
        };
        TopCandidatesQueue<C> top_candidates;
        if (VisitedTableVector* vtv = dynamic_cast<VisitedTableVector*>(&vt)) {
            top_candidates = call(*vtv);
        } else {
            VisitedTableSet& vts = dynamic_cast<VisitedTableSet&>(vt);
            top_candidates = call(vts);
        }

        while (top_candidates.size() > static_cast<size_t>(k)) {
            top_candidates.pop();
        }

        while (!top_candidates.empty()) {
            float d;
            storage_idx_t label;
            std::tie(d, label) = top_candidates.top();
            res.add_result(d, label);
            top_candidates.pop();
        }
    }

    if (profile) {
        profile->graph_end_ms = profile_elapsed_ms(profile);
    }
    // With payload I/O, stop speculative work as soon as traversal is done.
    // Do not delay demand reads behind the snapshot backlog. Without payload
    // I/O, retain complete predictions for offline validation/trace generation.
    const bool stop_predictions = async_onnx_query && raw_data_query;
    if (stop_predictions) {
        async_onnx_query->cancel_pending();
    } else if (async_onnx_query) {
        async_onnx_query->finish(*feature_trace);
    }
    if (profile && !stop_predictions) {
        profile->inference_end_ms = profile_elapsed_ms(profile);
    }

    if (feature_trace) {
        collect_final_top_k<C>(res, *feature_trace);
    }

    if (raw_data_query) {
        if (feature_trace) {
            raw_data_query->finish(feature_trace->final_top_k_ids);
        } else {
            HNSWQueryTrace final_result;
            final_result.k = k;
            collect_final_top_k<C>(res, final_result);
            raw_data_query->finish(final_result.final_top_k_ids);
        }
    }

    if (stop_predictions) {
        // Only an already-running CPU prediction / submitted CUDA slots remain.
        // Their cleanup overlaps final I/O; joining preserves caller-owned
        // model, trace, profile and raw-query lifetimes before search returns.
        async_onnx_query->finish(*feature_trace);
        if (profile) {
            profile->inference_end_ms = profile_elapsed_ms(profile);
        }
    }

    vt.advance();

    return stats;
}

template <class C>
void search_level_0_impl(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        ResultHandler& res,
        idx_t nprobe,
        const storage_idx_t* nearest_i,
        const float* nearest_d,
        int search_type,
        HNSWStats& search_stats,
        VisitedTable& vt,
        const SearchParameters* params) {
    auto cur_efSearch = hnsw.efSearch;
    if (params) {
        if (const SearchParametersHNSW* hnsw_params =
                    dynamic_cast<const SearchParametersHNSW*>(params)) {
            cur_efSearch = hnsw_params->efSearch;
        }
    }

    int k = extract_k_from_ResultHandler<C>(res);

    if (search_type == 1) {
        int nres = 0;

        for (idx_t j = 0; j < nprobe; j++) {
            storage_idx_t cj = nearest_i[j];

            if (cj < 0) {
                break;
            }

            if (vt.get(cj)) {
                continue;
            }

            int candidates_size = std::max(cur_efSearch, k);
            MinimaxHeapT<HC_for<C>> candidates(candidates_size);

            candidates.push(cj, nearest_d[j]);

            nres = search_from_candidates_dispatch<C>(
                    hnsw,
                    qdis,
                    res,
                    candidates,
                    vt,
                    search_stats,
                    0,
                    nres,
                    params);
            nres = std::min(nres, candidates_size);
        }
    } else if (search_type == 2) {
        int candidates_size = std::max(cur_efSearch, int(k));
        candidates_size = std::max(candidates_size, int(nprobe));

        MinimaxHeapT<HC_for<C>> candidates(candidates_size);
        for (idx_t j = 0; j < nprobe; j++) {
            storage_idx_t cj = nearest_i[j];

            if (cj < 0) {
                break;
            }
            candidates.push(cj, nearest_d[j]);
        }

        search_from_candidates_dispatch<C>(
                hnsw, qdis, res, candidates, vt, search_stats, 0, 0, params);
    }
}

} // namespace

HNSWStats HNSW::search(
        DistanceComputer& qdis,
        const IndexHNSW* index,
        ResultHandler& res,
        VisitedTable& vt,
        const SearchParameters* params) const {
    return search(qdis, index, res, vt, params, nullptr);
}

HNSWStats HNSW::search(
        DistanceComputer& qdis,
        const IndexHNSW* index,
        ResultHandler& res,
        VisitedTable& vt,
        const SearchParameters* params,
        HNSWQueryTrace* feature_trace) const {
    if (is_similarity) {
        return search_impl<C_similarity>(
                *this, qdis, index, res, vt, params, feature_trace);
    }
    return search_impl<C_distance>(
            *this, qdis, index, res, vt, params, feature_trace);
}

void HNSW::search_level_0(
        DistanceComputer& qdis,
        ResultHandler& res,
        idx_t nprobe,
        const storage_idx_t* nearest_i,
        const float* nearest_d,
        int search_type,
        HNSWStats& search_stats,
        VisitedTable& vt,
        const SearchParameters* params) const {
    if (is_similarity) {
        search_level_0_impl<C_similarity>(
                *this,
                qdis,
                res,
                nprobe,
                nearest_i,
                nearest_d,
                search_type,
                search_stats,
                vt,
                params);
    } else {
        search_level_0_impl<C_distance>(
                *this,
                qdis,
                res,
                nprobe,
                nearest_i,
                nearest_d,
                search_type,
                search_stats,
                vt,
                params);
    }
}

void HNSW::permute_entries(const idx_t* map) {
    // remap levels
    storage_idx_t ntotal = levels.size();
    std::vector<storage_idx_t> imap(ntotal); // inverse mapping
    // map: new index -> old index
    // imap: old index -> new index
    for (int i = 0; i < ntotal; i++) {
        assert(map[i] >= 0 && map[i] < ntotal);
        imap[map[i]] = i;
    }
    if (entry_point != -1) {
        entry_point = imap[entry_point];
    }
    std::vector<int> new_levels(ntotal);
    std::vector<size_t> new_offsets(ntotal + 1);
    std::vector<storage_idx_t> new_neighbors(neighbors.size());
    size_t no = 0;
    for (int i = 0; i < ntotal; i++) {
        storage_idx_t o = map[i]; // corresponding "old" index
        new_levels[i] = levels[o];
        for (size_t j = offsets[o]; j < offsets[o + 1]; j++) {
            storage_idx_t neigh = neighbors[j];
            new_neighbors[no++] = neigh >= 0 ? imap[neigh] : neigh;
        }
        new_offsets[i + 1] = no;
    }
    assert(new_offsets[ntotal] == offsets[ntotal]);
    // swap everyone
    std::swap(levels, new_levels);
    std::swap(offsets, new_offsets);
    neighbors = std::move(new_neighbors);
}

} // namespace faiss
