// Shared bin-averaging checks, run with any backend (CPU or GPU).

#ifndef OPG_TESTS_AVERAGING_H
#define OPG_TESTS_AVERAGING_H

#include <cmath>
#include <vector>

#include "doctest.h"

#include "npy.h"
#include "ref_compare.h"
#include "variants.h"

namespace avgtest {

  using Fast = opg::Fast<double>;

  /// Brute-force reference bins: edges [nbins][4] = (Elo, Ehi, Clo, Chi),
  /// values [nbins][2][3][3].
  struct BruteForce {
      npy::Array<double> edges, avg;
      BruteForce()
          : edges(npy::load<double>("avg_fast_bins.npy")),
            avg(npy::load<double>("avg_fast.npy"))
      {
      }
      size_t n() const { return edges.shape[0]; }
  };

  /// Max |GL average - brute force| over the reference bins, GL order n.
  inline double error_vs_brute(opg::Propagator<Fast>& prop,
                               const BruteForce& bf, int nE, int nC,
                               opg::CosZRule rule = opg::CosZRule::LayerAdapted)
  {
    double m = 0;
    for (size_t k = 0; k < bf.n(); k++) {
      const double* e = &bf.edges.data[k * 4];
      prop.set_bins({e[0], e[1]}, {e[2], e[3]}, nE, nC, opg::EMeasure::Linear, rule);
      prop.calculate_binned();
      for (int nb = 0; nb < 2; nb++)
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++) {
            double d = std::fabs(prop.binned(a, b, 0, 0, nb) -
                                 bf.avg[((k * 2 + nb) * 3 + a) * 3 + b]);
            if (!(d <= m)) m = d;
          }
    }
    return m;
  }

  /// Bin averages computed by the engine equal the explicit weighted sum of
  /// point probabilities at the GL nodes (checks the reduction and layout).
  inline void check_reduction(opg::Propagator<Fast>& prop, opg::EMeasure m)
  {
    std::vector<double> Ee = {0.5, 1.0, 2.0, 5.0, 10.0, 40.0};
    std::vector<double> Ce = {-1.0, -0.8, -0.45, -0.1, 0.2, 1.0};
    const int           nE = 4, nC = 3;
    prop.set_bins(Ee, Ce, nE, nC, m, opg::CosZRule::Plain);
    prop.calculate_binned();
    std::vector<double> got(prop.binned(),
                            prop.binned() + 2 * 9 * (Ee.size() - 1) * (Ce.size() - 1));

    opg::GaussLegendre gE(nE), gC(nC);
    double             maxd = 0;
    for (size_t ib = 0; ib + 1 < Ee.size(); ib++)
      for (size_t jb = 0; jb + 1 < Ce.size(); jb++) {
        std::vector<double>  xe, we, xc, wc, e, c, w;
        std::vector<uint8_t> nbv;
        double ulo = Ee[ib], uhi = Ee[ib + 1];
        if (m == opg::EMeasure::Log) { ulo = std::log(ulo); uhi = std::log(uhi); }
        if (m == opg::EMeasure::InvE) { double t = 1 / ulo; ulo = 1 / uhi; uhi = t; }
        gE.map(ulo, uhi, xe, we);
        gC.map(Ce[jb], Ce[jb + 1], xc, wc);
        for (int nb = 0; nb < 2; nb++)
          for (int j = 0; j < nC; j++)
            for (int i = 0; i < nE; i++) {
              double ee = m == opg::EMeasure::Linear ? xe[i]
                          : m == opg::EMeasure::Log  ? std::exp(xe[i])
                                                     : 1 / xe[i];
              e.push_back(ee);
              c.push_back(xc[j]);
              nbv.push_back(nb);
              w.push_back(we[i] / (uhi - ulo) * wc[j] / (Ce[jb + 1] - Ce[jb]));
            }
        auto   pts = prop.prob_points(e, c, nbv);
        size_t np  = e.size(), per = np / 2;
        for (int nb = 0; nb < 2; nb++)
          for (int ch = 0; ch < 9; ch++) {
            double s = 0;
            for (size_t q = 0; q < per; q++)
              s += w[nb * per + q] * pts[ch * np + nb * per + q];
            double g = got[((nb * 9 + ch) * (Ce.size() - 1) + jb) * (Ee.size() - 1) + ib];
            maxd     = std::max(maxd, std::fabs(g - s));
          }
      }
    INFO("measure " << int(m));
    CHECK(maxd < 1e-13);
  }

  inline opg::Propagator<Fast>& setup(opg::Propagator<Fast>& prop)
  {
    Fast::Params p;
    p.mix = variants::nominal_mix<3>();
    prop.set_params(p);
    return prop;
  }

} // namespace avgtest

#endif
