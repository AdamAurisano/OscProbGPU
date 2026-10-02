// Single-thread throughput of the original OscProb on a PREM grid (nu and
// nubar, ProbMatrix for all flavours), with and without its eigensystem
// cache, for comparison with OscProbGPU's bench/oscillogram.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "PMNS_Decay.h"
#include "PMNS_Fast.h"
#include "PMNS_NSI.h"
#include "PMNS_NUNM.h"
#include "PMNS_Sterile.h"
#include "PremModel.h"

void nominal(OscProb::PMNS_Base& p)
{
  p.SetDm(2, 7.41e-5);
  p.SetDm(3, 2.507e-3);
  p.SetAngle(1, 2, asin(sqrt(0.303)));
  p.SetAngle(1, 3, asin(sqrt(0.02225)));
  p.SetAngle(2, 3, asin(sqrt(0.451)));
  p.SetDelta(1, 3, 232 * M_PI / 180);
}

void run(OscProb::PMNS_Base& p, int N, const char* name, int nE, int nC)
{
  for (int cache = 0; cache < 2; cache++) {
    p.SetUseCache(cache);
    p.ClearCache();
    OscProb::PremModel prem;
    double             sum = 0;
    auto               t0  = std::chrono::steady_clock::now();
    for (int nb = 0; nb < 2; nb++) {
      p.SetIsNuBar(nb);
      for (int ic = 0; ic < nC; ic++) {
        prem.FillPath(-1 + 1.0 * ic / (nC - 1.0));
        p.SetPath(prem.GetNuPath());
        for (int ie = 0; ie < nE; ie++) {
          double E = std::pow(10.0, -0.5 + 2.5 * ie / (nE - 1.0));
          sum += p.ProbMatrix(N, N, E)[1][0];
        }
      }
    }
    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("OscProb %-8s cache=%d grid=%dx%d x2: %.3fs -> %.3g points/s (chk %g)\n",
           name, cache, nE, nC, t, 2.0 * nE * nC / t, sum);
  }
}

int main(int argc, char** argv)
{
  int nE = argc > 1 ? atoi(argv[1]) : 200, nC = argc > 2 ? atoi(argv[2]) : 200;
  { OscProb::PMNS_Fast p; nominal(p); run(p, 3, "Fast", nE, nC); }
  { OscProb::PMNS_NSI p; nominal(p); p.SetEps(0, 1, 0.1, 0.3); p.SetEps(0, 2, 0.1, 0);
    run(p, 3, "NSI", nE, nC); }
  { OscProb::PMNS_NUNM p(0); nominal(p); p.SetAlpha(1, 0, 0.02, 0.1); run(p, 3, "NUNM", nE, nC); }
  { OscProb::PMNS_Sterile p(4); nominal(p); p.SetDm(4, 1.0); p.SetAngle(2, 4, 0.1);
    run(p, 4, "Sterile", nE, nC); }
  { OscProb::PMNS_Decay p; nominal(p); p.SetAlpha3(1e-4); run(p, 3, "Decay", nE, nC); }
}
