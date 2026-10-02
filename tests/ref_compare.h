// Helpers to compare a Propagator against the OscProb reference files.

#ifndef OPG_TESTS_REF_COMPARE_H
#define OPG_TESTS_REF_COMPARE_H

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "npy.h"
#include "opg/propagator.h"

namespace refcmp {

  struct Result {
      double max_abs = 0;   ///< max |P - P_ref|
      size_t n       = 0;   ///< number of values compared
      size_t n_exact = 0;   ///< number of bit-identical values
  };

  inline void accumulate(Result& r, double a, double b)
  {
    r.max_abs = std::max(r.max_abs, std::fabs(a - b));
    r.n++;
    r.n_exact += (a == b);
  }

  inline const std::vector<opg::Segment<double>>& test_path()
  {
    // OscProb/test/Utils.h SetTestPath: 3 x 1000 km at rho = 2, 4, 2
    static std::vector<opg::Segment<double>> p = {
        {1000, 2, 0.5, 0}, {1000, 4, 0.5, 0}, {1000, 2, 0.5, 0}};
    return p;
  }

  inline const std::vector<opg::Segment<double>>& vacuum_path()
  {
    static std::vector<opg::Segment<double>> p = {{1300, 0, 0.5, 0}};
    return p;
  }

  /// Fixed-path reference: file [2][nE][N][N]; engine out[a][b][iE].
  template <class Model>
  Result compare_path(opg::Propagator<Model>& prop, const std::string& file,
                      const std::vector<opg::Segment<double>>& path)
  {
    constexpr int N   = Model::N;
    auto          E   = npy::load<double>("grid_E_test.npy");
    auto          ref = npy::load<double>(file);
    Result        r;
    for (int nb = 0; nb < 2; nb++) {
      auto out = prop.prob_path(E.data, path, nb == 1);
      for (size_t ie = 0; ie < E.size(); ie++)
        for (int a = 0; a < N; a++)
          for (int b = 0; b < N; b++)
            accumulate(r, out[(a * N + b) * E.size() + ie],
                       ref[(((nb * E.size()) + ie) * N + a) * N + b]);
    }
    return r;
  }

  /// PREM grid reference: file [2][nC][nE][N][N]; engine probs[nb][a][b][iC][iE].
  template <class Model>
  Result compare_prem(opg::Propagator<Model>& prop, const std::string& file)
  {
    constexpr int N   = Model::N;
    auto          E   = npy::load<double>("grid_E_prem.npy");
    auto          C   = npy::load<double>("grid_cosZ.npy");
    auto          ref = npy::load<double>(file);
    const size_t  nE = E.size(), nC = C.size();
    prop.set_grid(E.data, C.data);
    prop.calculate(opg::Flavor::Both);
    const double* P = prop.probs();
    Result        r;
    for (int nb = 0; nb < 2; nb++)
      for (size_t ic = 0; ic < nC; ic++)
        for (size_t ie = 0; ie < nE; ie++)
          for (int a = 0; a < N; a++)
            for (int b = 0; b < N; b++)
              accumulate(
                  r, P[(((size_t(nb) * N + a) * N + b) * nC + ic) * nE + ie],
                  ref[((((size_t(nb) * nC) + ic) * nE + ie) * N + a) * N + b]);
    return r;
  }

  /// Event-list mode over the same PREM grid points (checks prob_points).
  template <class Model>
  Result compare_points(opg::Propagator<Model>& prop, const std::string& file,
                        size_t step = 7)
  {
    constexpr int        N   = Model::N;
    auto                 E   = npy::load<double>("grid_E_prem.npy");
    auto                 C   = npy::load<double>("grid_cosZ.npy");
    auto                 ref = npy::load<double>(file);
    const size_t         nE = E.size(), nC = C.size();
    std::vector<double>  e, c;
    std::vector<uint8_t> nb;
    std::vector<size_t>  idx;
    for (size_t k = 0; k < 2 * nC * nE; k += step) {
      size_t b = k / (nC * nE), ic = (k / nE) % nC, ie = k % nE;
      e.push_back(E[ie]);
      c.push_back(C[ic]);
      nb.push_back(uint8_t(b));
      idx.push_back(k);
    }
    auto   out = prop.prob_points(e, c, nb);
    Result r;
    for (size_t i = 0; i < idx.size(); i++)
      for (int a = 0; a < N; a++)
        for (int b = 0; b < N; b++)
          accumulate(r, out[(a * N + b) * idx.size() + i],
                     ref[(idx[i] * N + a) * N + b]);
    return r;
  }

} // namespace refcmp

#endif
