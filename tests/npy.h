// Minimal NPY reader for the reference files written by
// reference/dump_reference.cxx (little-endian <f8 / <i4, C order).

#ifndef OPG_TESTS_NPY_H
#define OPG_TESTS_NPY_H

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef OPG_TEST_DATA_DIR
#define OPG_TEST_DATA_DIR "tests/data"
#endif

namespace npy {

  template <class T> struct Array {
      std::vector<size_t> shape;
      std::vector<T>      data;

      size_t size() const { return data.size(); }
      const T& operator[](size_t i) const { return data[i]; }
  };

  inline std::string path(const std::string& name)
  {
    return std::string(OPG_TEST_DATA_DIR) + "/" + name;
  }

  inline bool exists(const std::string& name)
  {
    std::ifstream f(path(name), std::ios::binary);
    return bool(f);
  }

  template <class T> Array<T> load(const std::string& name)
  {
    std::ifstream f(path(name), std::ios::binary);
    if (!f)
      throw std::runtime_error("npy: cannot open " + path(name) +
                               " (run scripts/fetch_test_data.sh, or "
                               "reference/make_reference.sh to regenerate)");
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
      throw std::runtime_error("npy: bad magic in " + name);
    uint16_t hlen;
    f.read(reinterpret_cast<char*>(&hlen), 2);
    std::string header(hlen, ' ');
    f.read(&header[0], hlen);

    const char* want = sizeof(T) == 8 ? "<f8" : "<i4";
    if (header.find(want) == std::string::npos)
      throw std::runtime_error("npy: unexpected dtype in " + name);

    Array<T> a;
    size_t   p0 = header.find('(') + 1, p1 = header.find(')');
    std::string s = header.substr(p0, p1 - p0);
    size_t      n = 1;
    size_t      pos = 0;
    while (pos < s.size()) {
      size_t comma = s.find(',', pos);
      std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos
                                                                 : comma - pos);
      if (tok.find_first_of("0123456789") != std::string::npos) {
        size_t d = std::stoul(tok);
        a.shape.push_back(d);
        n *= d;
      }
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
    a.data.resize(n);
    f.read(reinterpret_cast<char*>(a.data.data()), n * sizeof(T));
    if (!f) throw std::runtime_error("npy: short read in " + name);
    return a;
  }

} // namespace npy

#endif
