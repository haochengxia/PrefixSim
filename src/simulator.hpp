// The three-phase prefix-cache simulator.
//
// See doc/design.md for the full rationale. Each request is served as:
//   1. match     -- read-only probe of every block; produces all statistics
//   2. allocate  -- evict until there is room for the *missing* blocks, without
//                   counting victims that this same request is about to reuse
//   3. access    -- replay the request through the eviction algorithm, deepest
//                   block first
//
// Phase 2 is the piece libCacheSim does not provide on its own: an LLM request
// needs all of its blocks resident simultaneously, and freeing space naively
// evicts blocks the arriving request was about to reuse.

#pragma once

#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

#include "cost_model.hpp"
#include "flat_id_set.hpp"
#include "trace.hpp"

extern "C" {
#include "libCacheSim/cache.h"
#include "libCacheSim/prefetchAlgo.h"
}

namespace prefixsim {

struct Stats {
  // ---- request level ----
  int64_t n_requests = 0;
  /// Requests with more distinct blocks than the entire cache. They can never
  /// be fully resident, so they are skipped and excluded from every ratio.
  int64_t n_requests_skipped = 0;

  // ---- block level: the headline numbers ----
  int64_t n_blocks = 0;  ///< Block accesses across all served requests.
  int64_t n_block_hits =
      0;  ///< Blocks already resident when the request arrived.
  double total_cost = 0.0;  ///< Sum of block_cost() over all blocks.
  double saved_cost = 0.0;  ///< Sum of block_cost() over hit blocks.

  // ---- diagnostics ----
  int64_t n_evictions = 0;  ///< Objects evicted during phase 2.
  int64_t n_self_evictions =
      0;  ///< ...that the arriving request itself needed.
  /// Evictions triggered from inside phase 3. Must stay 0: phase 2 reserves
  /// exactly enough room. Non-zero means the allocation arithmetic and the
  /// eviction algorithm disagree, and the run is not trustworthy.
  int64_t n_unexpected_evictions = 0;
  /// Allocation rounds where the algorithm evicted without reporting a victim
  /// (it evicts inside sub-caches whose prefetcher is not ours), so the deficit
  /// was recomputed by re-probing instead. Diagnostic only: n_self_evictions is
  /// still exact, because the same re-probe detects Alpha's blocks going away.
  int64_t n_unobserved_eviction_rounds = 0;

  double block_hit_ratio() const {
    return n_blocks ? static_cast<double>(n_block_hits) /
                          static_cast<double>(n_blocks)
                    : 0.0;
  }
  double compute_saving_ratio() const {
    return total_cost > 0.0 ? saved_cost / total_cost : 0.0;
  }

  /// What the run did between `earlier`, a snapshot of the same run, and now.
  Stats since(const Stats &earlier) const {
    Stats d;
    d.n_requests = n_requests - earlier.n_requests;
    d.n_requests_skipped = n_requests_skipped - earlier.n_requests_skipped;
    d.n_blocks = n_blocks - earlier.n_blocks;
    d.n_block_hits = n_block_hits - earlier.n_block_hits;
    d.total_cost = total_cost - earlier.total_cost;
    d.saved_cost = saved_cost - earlier.saved_cost;
    d.n_evictions = n_evictions - earlier.n_evictions;
    d.n_self_evictions = n_self_evictions - earlier.n_self_evictions;
    d.n_unexpected_evictions =
        n_unexpected_evictions - earlier.n_unexpected_evictions;
    d.n_unobserved_eviction_rounds =
        n_unobserved_eviction_rounds - earlier.n_unobserved_eviction_rounds;
    return d;
  }
};

/// One interval report; see SimulatorConfig::report_interval_sec.
struct IntervalReport {
  /// Trace time of the request that closed the interval, in seconds since the
  /// trace's first request.
  double time_sec = 0.0;
  Stats total;     ///< The run so far.
  Stats interval;  ///< Since the previous report.
};

struct SimulatorConfig {
  int64_t cache_size_blocks = 0;
  CostModel cost_model;  ///< uniform unless set
  /// After phase 3, re-probe every block of the request and fail if any is
  /// missing. Catches an eviction algorithm whose find() does not report what
  /// it holds, which would silently invalidate the phase-1 match.
  bool verify = true;
  /// Make the arriving request's own blocks unevictable while phase 2 frees
  /// space for it, the way a real engine ref-counts the blocks of a running
  /// request. Without it the algorithm may evict a block this request already
  /// holds; the block still counts as a hit (phase 1 matched it before any
  /// eviction), but it re-enters through insert() and so loses its place in
  /// the policy's structure -- an S3FIFO main-queue block, for instance, comes
  /// back in the small queue because a main-queue eviction writes no ghost
  /// entry. Off reproduces the numbers from before pinning existed.
  ///
  /// Only RandomCompute and algorithms whose victim selection is built on LRU,
  /// FIFO, Clock, Sieve or SLRU honour the pins; the others still evict such
  /// blocks, and n_self_evictions counts them.
  bool pin_request_blocks = true;
  /// Phase 3 replay order. True (the default) touches the deepest block
  /// first, so recency-ordered algorithms leave the prefix root at the MRU end
  /// and eviction eats the prefix from the deep end inward (see
  /// doc/design.md "Replay order"). False replays in prefix order, which
  /// leaves the root nearest the LRU end, so eviction starts from the
  /// beginning.
  bool replay_deepest_first = true;
  /// When set, append one line per served request describing that request's
  /// holes -- the contiguous runs of blocks it had to recompute. Empty
  /// disables it. See Simulator::dump_holes.
  std::string dump_holes_path;
  /// Trace time, in seconds, between interval reports to `on_report`; 0
  /// disables them. The rule is cachesim's --report-interval: time is a
  /// request's `timestamp` less the trace's first, and a served request that
  /// lands at least this long after the previous report closes an interval
  /// that includes it. What follows the last such request is reported when the
  /// trace ends, which cachesim does not do. Skipped requests close nothing.
  int64_t report_interval_sec = 0;
  std::function<void(const IntervalReport &)> on_report;
};

/// Whether the phase-2 pin predicate (SimulatorConfig::pin_request_blocks) is
/// implemented for this algorithm. False for an unknown name, and for the
/// algorithms that assume an eviction removes the exact object they last looked
/// at. True does not mean the algorithm honours it; see pin_request_blocks.
bool algorithm_supports_pinning(const std::string &algorithm);

/// Runs one eviction algorithm over a trace.
class Simulator {
 public:
  Simulator(cache_t *cache, const SimulatorConfig &config);
  ~Simulator();

  Simulator(const Simulator &) = delete;
  Simulator &operator=(const Simulator &) = delete;

  /// Serve every request in order. Returns false and fills `error` if an
  /// invariant broke.
  bool run(const std::vector<Request> &requests, std::string &error);

  /// Serve a trace as it streams, one request in memory at a time. The reader
  /// must be annotated(): next access cannot be known in a forward pass. Its
  /// requests must also all fit in the cache, since the stored next-access
  /// times count every request; load_trace() handles a trace where some do
  /// not.
  bool run(TraceReader &reader, std::string &error);

  const Stats &stats() const { return stats_; }

 private:
  bool serve(const Request &request, std::string &error);

  /// Phase 1: is this block resident right now? Asks the algorithm's own
  /// find() with update_cache false, which must leave the cache unchanged.
  bool probe(obj_id_t id) const;

  /// Phase 2: evict until `needed` blocks that this request does *not* want
  /// have been freed.
  bool allocate(const Request &request, int64_t needed, std::string &error);

  /// Phase 3: replay the request, deepest block first.
  void access(const Request &request);

  /// Point req_buf_ at block `i` of `request`.
  void fill_request(const Request &request, size_t i);

  /// Write this request's holes, one line per request (see dump_holes()).
  void dump_holes(const Request &request);

  /// Close an interval at trace time `time_sec` (see report_interval_sec).
  void report_interval(double time_sec);

  /// Report what the last interval left unreported, at the end of a run.
  void report_rest();

  cache_t *cache_ = nullptr;
  SimulatorConfig config_;
  Stats stats_;

  // Interval reports, in seconds since the first request's timestamp.
  bool clock_started_ = false;
  double first_timestamp_ = 0.0;
  double last_served_time_ = 0.0;
  double last_report_time_ = 0.0;
  Stats last_report_stats_;

  request_t *req_buf_ = nullptr;
  prefetcher_t *recorder_ = nullptr;
  FILE *holes_file_ = nullptr;

  // Scratch reused across requests so the hot loop does not allocate.
  std::vector<char> resident_;
  FlatIdSet alpha_;
  FlatIdSet missing_;
  std::unordered_set<obj_id_t> alpha_resident_;
  std::vector<obj_id_t> victims_;
};

/// Create a cache by algorithm name (case-insensitive). Returns nullptr for
/// unknown names. `cache_size_blocks` is in blocks: every object has size 1.
cache_t *create_cache_by_name(const std::string &algorithm,
                              int64_t cache_size_blocks,
                              const char *cache_specific_params = nullptr,
                              int64_t n_total_req = 0);

/// The algorithms prefixsim can drive: each completes a whole trace with the
/// post-request residency check on.
const std::vector<std::string> &supported_algorithms();

}  // namespace prefixsim
