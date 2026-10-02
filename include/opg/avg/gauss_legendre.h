///////////////////////////////////////////////////////////////////////////////
/// \file gauss_legendre.h
///
/// \brief Gauss-Legendre nodes and weights (host side).
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_AVG_GAUSS_LEGENDRE_H
#define OPG_AVG_GAUSS_LEGENDRE_H

#include <cmath>
#include <stdexcept>
#include <vector>

namespace opg {

  struct GaussLegendre {
      std::vector<double> x;  ///< nodes on [-1, 1], increasing
      std::vector<double> w;  ///< weights, sum to 2

      /// n-point rule, exact for polynomials of degree <= 2n-1.
      explicit GaussLegendre(int n) : x(n), w(n)
      {
        if (n < 1) throw std::invalid_argument("GaussLegendre: n < 1");
        for (int i = 0; i < (n + 1) / 2; i++) {
          // Initial guess (Tricomi), then Newton on P_n
          double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));
          double dp = 0;
          for (int it = 0; it < 100; it++) {
            double p0 = 1, p1 = 0;
            for (int k = 1; k <= n; k++) {
              double p2 = p1;
              p1        = p0;
              p0        = ((2 * k - 1) * z * p1 - (k - 1) * p2) / k;
            }
            // p0 = P_n(z), p1 = P_{n-1}(z)
            dp        = n * (z * p0 - p1) / (z * z - 1);
            double dz = p0 / dp;
            z -= dz;
            if (std::fabs(dz) < 1e-16) break;
          }
          // recompute derivative at converged z
          double p0 = 1, p1 = 0;
          for (int k = 1; k <= n; k++) {
            double p2 = p1;
            p1        = p0;
            p0        = ((2 * k - 1) * z * p1 - (k - 1) * p2) / k;
          }
          dp = n * (z * p0 - p1) / (z * z - 1);

          x[i]         = -z;
          x[n - 1 - i] = z;
          w[i] = w[n - 1 - i] = 2 / ((1 - z * z) * dp * dp);
        }
        if (n % 2 == 1) x[n / 2] = 0;
      }

      /// Nodes mapped to [a, b] (weights scaled to sum to b-a).
      void map(double a, double b, std::vector<double>& xo,
               std::vector<double>& wo) const
      {
        xo.resize(x.size());
        wo.resize(w.size());
        double h = 0.5 * (b - a), c = 0.5 * (a + b);
        for (size_t i = 0; i < x.size(); i++) {
          xo[i] = c + h * x[i];
          wo[i] = h * w[i];
        }
      }
  };

} // namespace opg

#endif
