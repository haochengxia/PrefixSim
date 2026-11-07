// A set of block ids for the Simulator's per-request scratch sets. Open
// addressing, cleared in O(1) by bumping an epoch, so one instance serves every
// request without allocating. count() is a multiply and an array read: the pin
// predicate calls it for every pinned object cache_skip_pinned walks past (~9
// per block accessed, LRU at 64k blocks), which made std::unordered_set's modulo
// and node chase the hot spot. Iterates in insertion order.

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace prefixsim {

class FlatIdSet {
 public:
  void clear() {
    ids_.clear();
    if (++epoch_ == 0) {  // wrapped: old stamps would alias the new epoch
      std::fill(stamps_.begin(), stamps_.end(), 0u);
      epoch_ = 1;
    }
  }
  bool insert(uint64_t id) {
    if (2 * (ids_.size() + 1) > stamps_.size()) grow();
    size_t h = slot(id);
    for (; stamps_[h] == epoch_; h = (h + 1) & mask_) {
      if (keys_[h] == id) return false;
    }
    stamps_[h] = epoch_;
    keys_[h] = id;
    ids_.push_back(id);
    return true;
  }
  size_t count(uint64_t id) const {
    if (stamps_.empty()) return 0;
    for (size_t h = slot(id); stamps_[h] == epoch_; h = (h + 1) & mask_) {
      if (keys_[h] == id) return 1;
    }
    return 0;
  }
  size_t size() const { return ids_.size(); }
  std::vector<uint64_t>::const_iterator begin() const { return ids_.begin(); }
  std::vector<uint64_t>::const_iterator end() const { return ids_.end(); }

 private:
  size_t slot(uint64_t id) const { return (id * 0x9E3779B97F4A7C15ull >> 32) & mask_; }
  void grow() {  // to twice the size, keeping the current ids
    const size_t cap = std::max<size_t>(64, 2 * stamps_.size());
    keys_.assign(cap, 0);
    stamps_.assign(cap, 0);
    mask_ = cap - 1;
    epoch_ = 1;
    std::vector<uint64_t> ids;
    ids.swap(ids_);
    for (const uint64_t id : ids) insert(id);
  }

  std::vector<uint64_t> keys_;
  std::vector<uint32_t> stamps_;
  std::vector<uint64_t> ids_;
  uint32_t epoch_ = 1;
  size_t mask_ = 0;
};

}  // namespace prefixsim
