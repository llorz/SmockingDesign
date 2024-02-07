#pragma once
#include <vector>

namespace Optiz {
template <typename K, typename V>
class VectorMap {
 public:
  VectorMap(int reserved_size = 10) { values.reserve(reserved_size); }
  VectorMap(const VectorMap&) = default;
  VectorMap(VectorMap&&) noexcept = default;
  VectorMap& operator=(const VectorMap&) = default;
  VectorMap& operator=(VectorMap&&) = default;

  V& operator[](const K& key) {
    for (auto& val : values) {
      if (val.first == key) {
        return val.second;
      }
    }
    values.push_back(std::make_pair(key, V()));
    return values.back().second;
  }

  V operator[](const K& key) const {
    for (auto& val : values) {
      if (val.first == key) {
        return val.second;
      }
    }
    return V();
  }

  class iterator {
   public:
    iterator(typename std::vector<std::pair<K, V>>::iterator it) : it(it) {}

    std::pair<K, V>& operator*() { return *it; }
    iterator& operator++() {
      ++it;
      return *this;
    }
    bool operator!=(const iterator& other) { return it != other.it; }

   private:
    typename std::vector<std::pair<K, V>>::iterator it;
  };

  class const_iterator {
   public:
    const_iterator(typename std::vector<std::pair<K, V>>::const_iterator it)
        : it(it) {}

    const std::pair<K, V>& operator*() const { return *it; }
    const_iterator& operator++() {
      ++it;
      return *this;
    }
    bool operator!=(const const_iterator& other) { return it != other.it; }

   private:
    typename std::vector<std::pair<K, V>>::const_iterator it;
  };

  iterator begin() { return iterator(values.begin()); }
  iterator end() { return iterator(values.end()); }
  const_iterator begin() const { return const_iterator(values.begin()); }
  const_iterator end() const { return const_iterator(values.end()); }

  size_t size() const { return values.size(); }

  std::pair<iterator, bool> try_emplace(const K& key, const V& val) {
    for (auto it = values.begin(); it != values.end(); ++it) {
      if (it->first == key) {
        return std::make_pair(iterator(it), false);
      }
    }
    values.push_back({key, val});
    return std::make_pair(iterator(values.end() - 1), true);
  }

 private:
  std::vector<std::pair<K, V>> values;
};
}  // namespace Optiz
