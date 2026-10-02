// Single-thread throughput of the original OscProb PMNS_Fast on a PREM grid,
// with and without its eigensystem cache, for comparison with OscProbGPU.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "PMNS_Fast.h"
#include "PremModel.h"

int main(int argc, char** argv)
{
  int nE = argc > 1 ? atoi(argv[1]) : 300, nC = argc > 2 ? atoi(argv[2]) : 300;
  for (int cache = 0; cache < 2; cache++) {
    OscProb::PMNS_Fast p;
    p.SetDm(2, 7.41e-5);
    p.SetDm(3, 2.507e-3);
    p.SetAngle(1, 2, asin(sqrt(0.303)));
    p.SetAngle(1, 3, asin(sqrt(0.02225)));
    p.SetAngle(2, 3, asin(sqrt(0.451)));
    p.SetDelta(1, 3, 232 * M_PI / 180);
    p.SetUseCache(cache);
    OscProb::PremModel prem;
    double sum = 0;
    auto   t0  = std::chrono::steady_clock::now();
    for (int nb = 0; nb < 2; nb++) {
      p.SetIsNuBar(nb);
      for (int ic = 0; ic < nC; ic++) {
        prem.FillPath(-1 + 1.0 * ic / (nC - 1.0));
        p.SetPath(prem.GetNuPath());
        for (int ie = 0; ie < nE; ie++) {
          double E = std::pow(10.0, -0.5 + 2.5 * ie / (nE - 1.0));
          sum += p.ProbMatrix(3, 3, E)[1][0];
        }
      }
    }
    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("OscProb PMNS_Fast ProbMatrix(3,3) cache=%d grid=%dx%d x2: %.3fs -> %.3g points/s (chk %g)\n",
           cache, nE, nC, t, 2.0 * nE * nC / t, sum);
  }
}
