///////////////////////////////////////////////////////////////////////////////
//
// Dump reference oscillation probabilities from the original OscProb.
//
// For each model variant this writes NPY files (float64, C order):
//
//   <tag>_testpath.npy : [2 nubar][nE=100][N][N]    3-segment path 1000 km at
//                                                   rho = 2,4,2 (OscProb tests)
//   <tag>_vacuum.npy   : [2][nE=100][N][N]          1300 km, rho = 0
//   <tag>_prem.npy     : [2][nC=101][nE=120][N][N]  PremModel default
//   <tag>_hms.npy      : [N][N][2]                  fHms (re, im), upper tri.
//
// with P[..][a][b] = P(a -> b), from ProbMatrix(N, N).
//
// Shared grids:
//   grid_E_test.npy  : 100 log-spaced energies 0.1 .. 10 GeV
//   grid_E_prem.npy  : 120 log-spaced energies 0.5 .. 100 GeV
//   grid_cosZ.npy    : 101 linear cosZ -1 .. 1
//   prem_paths.npy   : [nC][128][4] FillPath segments (L, rho, zoa, layer),
//                      padded with zeros; prem_npaths.npy : [nC] counts
//   avg_*.npy        : brute-force 2D bin averages for Fast (see below)
//
// The eigensystem cache is disabled (OscProbGPU has none).
// Parameter sets follow OscProb/test/Utils.h (NuFIT 5.2 NO), plus some
// extra variants with complex phases.
//
///////////////////////////////////////////////////////////////////////////////

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "PMNS_Decay.h"
#include "PMNS_Fast.h"
#include "PMNS_LIV.h"
#include "PMNS_NSI.h"
#include "PMNS_NUNM.h"
#include "PMNS_SNSI.h"
#include "PMNS_Sterile.h"
#include "PremModel.h"

using namespace std;
using namespace OscProb;

//.............................................................................
// Minimal NPY writer (format version 1.0, little-endian float64 or int32)
//.............................................................................
template <class T>
void write_npy(const string& fname, const vector<T>& data,
               const vector<size_t>& shape)
{
  size_t n = 1;
  for (size_t s : shape) n *= s;
  if (n != data.size()) {
    cerr << "write_npy: shape mismatch for " << fname << endl;
    exit(1);
  }
  const char* descr = sizeof(T) == 8 ? "<f8" : "<i4";
  ostringstream h;
  h << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
  for (size_t i = 0; i < shape.size(); i++) {
    h << shape[i];
    if (shape.size() == 1 || i + 1 < shape.size()) h << ", ";
  }
  h << "), }";
  string header = h.str();
  size_t total  = 10 + header.size() + 1;
  header += string((64 - total % 64) % 64, ' ');
  header += '\n';

  ofstream f(fname, ios::binary);
  f.write("\x93NUMPY\x01\x00", 8);
  uint16_t hlen = header.size();
  f.write(reinterpret_cast<const char*>(&hlen), 2);
  f.write(header.data(), header.size());
  f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
}

//.............................................................................
vector<double> logspace(double a, double b, int n)
{
  vector<double> v(n);
  for (int i = 0; i < n; i++)
    v[i] = exp(log(a) + (log(b) - log(a)) * i / (n - 1.0));
  return v;
}

vector<double> linspace(double a, double b, int n)
{
  vector<double> v(n);
  for (int i = 0; i < n; i++) v[i] = a + (b - a) * i / (n - 1.0);
  return v;
}

//.............................................................................
// Expose the protected mass matrix for validation of the BuildHms port.
//.............................................................................
template <class Base> struct HmsPeek : public Base {
    using Base::Base;
    matrixC GetHms()
    {
      this->BuildHms();
      return this->fHms;
    }
};

//.............................................................................
void SetNominalPars(PMNS_Base* p)
{
  p->SetDm(2, 7.41e-5);
  p->SetDm(3, 2.507e-3);
  p->SetAngle(1, 2, asin(sqrt(0.303)));
  p->SetAngle(1, 3, asin(sqrt(0.02225)));
  p->SetAngle(2, 3, asin(sqrt(0.451)));
  p->SetDelta(1, 3, 232 * M_PI / 180);
}

struct Variant {
    string                          tag;
    int                             N;
    function<PMNS_Base*()>          make;
    function<matrixC(PMNS_Base*)>   hms;
    bool                            warmup = false;
};

template <class T> matrixC get_hms(PMNS_Base* p)
{
  return static_cast<HmsPeek<T>*>(p)->GetHms();
}

vector<Variant> GetVariants()
{
  vector<Variant> v;

  v.push_back({"fast", 3,
               [] {
                 auto p = new HmsPeek<PMNS_Fast>();
                 SetNominalPars(p);
                 return (PMNS_Base*)p;
               },
               get_hms<PMNS_Fast>});

  // Fast with inverted ordering and different delta for extra coverage
  v.push_back({"fast_io", 3,
               [] {
                 auto p = new HmsPeek<PMNS_Fast>();
                 SetNominalPars(p);
                 p->SetDm(3, -2.465e-3 + 7.41e-5);
                 p->SetDelta(1, 3, 1.1);
                 return (PMNS_Base*)p;
               },
               get_hms<PMNS_Fast>});

  v.push_back({"nsi", 3,
               [] {
                 auto p = new HmsPeek<PMNS_NSI>();
                 SetNominalPars(p);
                 p->SetEps(0, 0, 0.1, 0);
                 p->SetEps(0, 1, 0.2, 0);
                 p->SetEps(0, 2, 0.3, 0);
                 p->SetEps(1, 1, 0.4, 0);
                 p->SetEps(1, 2, 0.5, 0);
                 p->SetEps(2, 2, 0.6, 0);
                 return (PMNS_Base*)p;
               },
               get_hms<PMNS_NSI>});

  v.push_back({"nsi_phases", 3,
               [] {
                 auto p = new HmsPeek<PMNS_NSI>();
                 SetNominalPars(p);
                 p->SetEps(0, 0, -0.2, 0);
                 p->SetEps(0, 1, 0.05, 0.7);
                 p->SetEps(0, 2, 0.15, -1.3);
                 p->SetEps(1, 1, 0.02, 0);
                 p->SetEps(1, 2, 0.03, 2.1);
                 p->SetEps(2, 2, 0.1, 0);
                 p->SetFermCoup(0.5, 1.0, 0.8);
                 return (PMNS_Base*)p;
               },
               get_hms<PMNS_NSI>});

  v.push_back({"nunm", 3,
               [] {
                 auto p = new HmsPeek<PMNS_NUNM>(0);
                 SetNominalPars(p);
                 p->SetAlpha(0, 0, 0.05, 0);
                 p->SetAlpha(1, 0, 0.06, 0);
                 p->SetAlpha(2, 0, 0.07, 0);
                 p->SetAlpha(1, 1, 0.08, 0);
                 p->SetAlpha(2, 1, 0.09, 0);
                 p->SetAlpha(2, 2, 0.1, 0);
                 return (PMNS_Base*)p;
               },
               get_hms<PMNS_NUNM>});

  v.push_back({"nunm_phases", 3,
               [] {
                 auto p = new HmsPeek<PMNS_NUNM>(0);
                 SetNominalPars(p);
                 p->SetAlpha(0, 0, -0.03, 0);
                 p->SetAlpha(1, 0, 0.02, 0.4);
                 p->SetAlpha(2, 0, 0.04, -2.0);
                 p->SetAlpha(1, 1, -0.01, 0);
                 p->SetAlpha(2, 1, 0.05, 1.2);
                 p->SetAlpha(2, 2, -0.02, 0);
                 p->SetFracVnc(0.7);
                 return (PMNS_Base*)p;
               },
               get_hms<PMNS_NUNM>});

  // High-scale NUNM: OscProb normalises alpha inside PropagatePath, after
  // the initial state has already been rotated by the unnormalised alpha,
  // so the first call differs. We warm up once before dumping.
  v.push_back({"nunm_high", 3,
               [] {
                 auto p = new HmsPeek<PMNS_NUNM>(1);
                 SetNominalPars(p);
                 p->SetAlpha(0, 0, 0.05, 0);
                 p->SetAlpha(1, 0, 0.06, 0);
                 p->SetAlpha(2, 0, 0.07, 0);
                 p->SetAlpha(1, 1, 0.08, 0);
                 p->SetAlpha(2, 1, 0.09, 0);
                 p->SetAlpha(2, 2, 0.1, 0);
                 return (PMNS_Base*)p;
               },
               get_hms<PMNS_NUNM>, true});

  v.push_back({"sterile", 4,
               [] {
                 auto p = new HmsPeek<PMNS_Sterile>(4);
                 SetNominalPars(p);
                 p->SetDm(4, 0.1);
                 p->SetAngle(1, 4, 0.1);
                 p->SetAngle(2, 4, 0.1);
                 p->SetAngle(3, 4, 0.1);
                 return (PMNS_Base*)p;
               },
               get_hms<PMNS_Sterile>});

  v.push_back({"sterile_phases", 4,
               [] {
                 auto p = new HmsPeek<PMNS_Sterile>(4);
                 SetNominalPars(p);
                 p->SetDm(4, 1.3);
                 p->SetAngle(1, 4, 0.15);
                 p->SetAngle(2, 4, 0.2);
                 p->SetAngle(3, 4, 0.3);
                 p->SetDelta(1, 4, 0.9);
                 p->SetDelta(2, 4, -1.7);
                 return (PMNS_Base*)p;
               },
               get_hms<PMNS_Sterile>});

  v.push_back({"decay", 3,
               [] {
                 auto p = new HmsPeek<PMNS_Decay>();
                 SetNominalPars(p);
                 p->SetAlpha3(1e-4);
                 return (PMNS_Base*)p;
               },
               get_hms<PMNS_Decay>});

  v.push_back({"decay_both", 3,
               [] {
                 auto p = new HmsPeek<PMNS_Decay>();
                 SetNominalPars(p);
                 p->SetAlpha2(3e-5);
                 p->SetAlpha3(2e-4);
                 return (PMNS_Base*)p;
               },
               get_hms<PMNS_Decay>});

  // LIV: all dimensions, real coefficients
  v.push_back({"liv", 3,
               [] {
                 auto p = new HmsPeek<PMNS_LIV>();
                 SetNominalPars(p);
                 p->SetaT(0, 0, 3, 1e-21, 0);
                 p->SetaT(0, 1, 3, 2e-21, 0);
                 p->SetaT(1, 2, 3, -1e-21, 0);
                 p->SetcT(1, 1, 4, 5e-23, 0);
                 p->SetcT(0, 2, 4, 1e-22, 0);
                 p->SetaT(1, 2, 5, 1e-24, 0);
                 p->SetcT(2, 2, 6, 2e-26, 0);
                 p->SetaT(0, 1, 7, 1e-28, 0);
                 p->SetcT(1, 2, 8, 1e-30, 0);
                 return (PMNS_Base*)p;
               },
               get_hms<PMNS_LIV>});

  // LIV: complex coefficients (phases) in the minimal sector
  v.push_back({"liv_phases", 3,
               [] {
                 auto p = new HmsPeek<PMNS_LIV>();
                 SetNominalPars(p);
                 p->SetaT(0, 1, 3, 1.5e-21, 0.7);
                 p->SetaT(0, 2, 3, 8e-22, -1.9);
                 p->SetaT(2, 2, 3, -6e-22, 0);
                 p->SetcT(1, 2, 4, 1e-22, 2.3);
                 p->SetcT(0, 0, 4, -4e-23, 0);
                 p->SetcT(0, 1, 6, 3e-26, 1.1);
                 return (PMNS_Base*)p;
               },
               get_hms<PMNS_LIV>});

  // Scalar NSI with a non-zero lightest mass
  v.push_back({"snsi", 3,
               [] {
                 auto p = new HmsPeek<PMNS_SNSI>();
                 SetNominalPars(p);
                 p->SetLowestMass(0.05);
                 p->SetEps(0, 0, 0.5, 0);
                 p->SetEps(0, 1, 0.3, 0);
                 p->SetEps(1, 1, -0.2, 0);
                 p->SetEps(1, 2, 0.4, 0);
                 p->SetEps(2, 2, 0.1, 0);
                 return (PMNS_Base*)p;
               },
               get_hms<PMNS_SNSI>});

  // Scalar NSI, inverted ordering, massless lightest, phases and couplings
  v.push_back({"snsi_io", 3,
               [] {
                 auto p = new HmsPeek<PMNS_SNSI>();
                 SetNominalPars(p);
                 p->SetDm(3, -2.465e-3 + 7.41e-5);
                 p->SetLowestMass(0);
                 p->SetEps(0, 1, 0.4, 0.9);
                 p->SetEps(0, 2, 0.2, -2.1);
                 p->SetEps(1, 1, 0.3, 0);
                 p->SetFermCoup(0.5, 1.0, 0.8);
                 return (PMNS_Base*)p;
               },
               get_hms<PMNS_SNSI>});

  return v;
}

//.............................................................................
void append_matrix(vector<double>& out, const matrixD& m, int N)
{
  for (int a = 0; a < N; a++)
    for (int b = 0; b < N; b++) out.push_back(m[a][b]);
}

void dump_fixed_path(PMNS_Base* p, int N, const vector<NuPath>& path,
                     const vector<double>& E, const string& fname)
{
  vector<double> out;
  for (int nb = 0; nb < 2; nb++) {
    p->SetIsNuBar(nb);
    p->SetPath(path);
    for (double e : E) append_matrix(out, p->ProbMatrix(N, N, e), N);
  }
  write_npy(fname, out, {2, E.size(), (size_t)N, (size_t)N});
}

void dump_prem(PMNS_Base* p, int N, PremModel& prem, const vector<double>& C,
               const vector<double>& E, const string& fname)
{
  vector<double> out;
  for (int nb = 0; nb < 2; nb++) {
    p->SetIsNuBar(nb);
    for (double c : C) {
      prem.FillPath(c);
      p->SetPath(prem.GetNuPath());
      for (double e : E) append_matrix(out, p->ProbMatrix(N, N, e), N);
    }
  }
  write_npy(fname, out, {2, C.size(), E.size(), (size_t)N, (size_t)N});
}

void dump_hms(const matrixC& hms, int N, const string& fname)
{
  vector<double> out;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      out.push_back(hms[i][j].real());
      out.push_back(hms[i][j].imag());
    }
  write_npy(fname, out, {(size_t)N, (size_t)N, 2});
}

//.............................................................................
// Brute-force 2D bin averages (midpoint rule, uniform in E and cosZ) for
// PMNS_Fast on the default PREM model. Used to validate Gauss-Legendre
// averaging later on.
//.............................................................................
void dump_bin_averages(const string& dir, int nsub)
{
  auto p = new PMNS_Fast();
  SetNominalPars(p);
  p->SetUseCache(false);
  PremModel prem;

  // bins: {Elo, Ehi} x {Clo, Chi}
  vector<array<double, 2>> ebins = {{1.0, 1.5}, {2.5, 3.5}, {5.0, 7.0},
                                    {15, 25}};
  vector<array<double, 2>> cbins = {{-1.0, -0.9}, {-0.6, -0.5}, {-0.2, -0.1},
                                    {0.3, 0.5}};

  vector<double> edges_out, out;
  for (auto& eb : ebins)
    for (auto& cb : cbins) {
      edges_out.insert(edges_out.end(), {eb[0], eb[1], cb[0], cb[1]});
      for (int nb = 0; nb < 2; nb++) {
        p->SetIsNuBar(nb);
        matrixD acc(3, vectorD(3, 0));
        for (int ic = 0; ic < nsub; ic++) {
          double c = cb[0] + (cb[1] - cb[0]) * (ic + 0.5) / nsub;
          prem.FillPath(c);
          p->SetPath(prem.GetNuPath());
          for (int ie = 0; ie < nsub; ie++) {
            double  e = eb[0] + (eb[1] - eb[0]) * (ie + 0.5) / nsub;
            matrixD m = p->ProbMatrix(3, 3, e);
            for (int a = 0; a < 3; a++)
              for (int b = 0; b < 3; b++) acc[a][b] += m[a][b];
          }
        }
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++)
            out.push_back(acc[a][b] / (double(nsub) * nsub));
      }
    }
  size_t nbins = ebins.size() * cbins.size();
  write_npy(dir + "/avg_fast_bins.npy", edges_out, {nbins, 4});
  write_npy(dir + "/avg_fast.npy", out, {nbins, 2, 3, 3});
  delete p;
}

//.............................................................................
// OscProb AvgProb (1D energy-bin averages, default precision 1e-4) on the
// test path, for 50 log-spaced bins in 0.1 .. 10 GeV:
//   avg1d_fast_edges.npy : [51]
//   avg1d_fast.npy       : [2][50][3][3]
//.............................................................................
void dump_avgprob_1d(const string& dir)
{
  auto p = new PMNS_Fast();
  SetNominalPars(p);
  p->SetUseCache(false);
  p->SetPath({NuPath(1000, 2), NuPath(1000, 4), NuPath(1000, 2)});
  vector<double> edges = logspace(0.1, 10, 51);
  vector<double> out;
  for (int nb = 0; nb < 2; nb++) {
    p->SetIsNuBar(nb);
    for (size_t i = 0; i + 1 < edges.size(); i++) {
      double E = 0.5 * (edges[i] + edges[i + 1]), dE = edges[i + 1] - edges[i];
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) out.push_back(p->AvgProb(a, b, E, dE));
    }
  }
  write_npy(dir + "/avg1d_fast_edges.npy", edges, {edges.size()});
  write_npy(dir + "/avg1d_fast.npy", out, {2, edges.size() - 1, 3, 3});
  delete p;
}

//.............................................................................
int main(int argc, char** argv)
{
  string dir  = argc > 1 ? argv[1] : "tests/data";
  int    nsub = argc > 2 ? atoi(argv[2]) : 400;
  // optional comma-separated list of variant tags: dump only those (and
  // skip the shared averaging references)
  string only = argc > 3 ? string(",") + argv[3] + "," : "";

  vector<double> Etest = logspace(0.1, 10, 100);
  vector<double> Eprem = logspace(0.5, 100, 120);
  vector<double> C     = linspace(-1, 1, 101);

  write_npy(dir + "/grid_E_test.npy", Etest, {Etest.size()});
  write_npy(dir + "/grid_E_prem.npy", Eprem, {Eprem.size()});
  write_npy(dir + "/grid_cosZ.npy", C, {C.size()});

  // OscProb test path (test/Utils.h SetTestPath)
  vector<NuPath> testpath = {NuPath(1000, 2), NuPath(1000, 4), NuPath(1000, 2)};
  vector<NuPath> vacpath  = {NuPath(1300, 0)};

  // PREM paths
  PremModel prem;
  {
    const int      maxseg = 128;
    vector<double> segs;
    vector<int>    nseg;
    for (double c : C) {
      int n = prem.FillPath(c);
      if (n > maxseg) {
        cerr << "Too many segments" << endl;
        return 1;
      }
      vector<NuPath> path = prem.GetNuPath();
      nseg.push_back(path.size());
      for (int i = 0; i < maxseg; i++) {
        if (i < (int)path.size())
          segs.insert(segs.end(), {path[i].length, path[i].density,
                                   path[i].zoa, double(path[i].layer)});
        else
          segs.insert(segs.end(), {0, 0, 0, 0});
      }
    }
    write_npy(dir + "/prem_paths.npy", segs, {C.size(), (size_t)maxseg, 4});
    write_npy(dir + "/prem_npaths.npy", nseg, {C.size()});
  }

  for (auto& v : GetVariants()) {
    if (!only.empty() && only.find("," + v.tag + ",") == string::npos) continue;
    cout << "Dumping " << v.tag << endl;
    PMNS_Base* p = v.make();
    // OscProbGPU has no eigensystem cache; compare against uncached values.
    p->SetUseCache(false);
    if (v.warmup) p->ProbMatrix(v.N, v.N, 1.0);
    dump_hms(v.hms(p), v.N, dir + "/" + v.tag + "_hms.npy");
    dump_fixed_path(p, v.N, testpath, Etest, dir + "/" + v.tag + "_testpath.npy");
    dump_fixed_path(p, v.N, vacpath, Etest, dir + "/" + v.tag + "_vacuum.npy");
    dump_prem(p, v.N, prem, C, Eprem, dir + "/" + v.tag + "_prem.npy");
    delete p;
  }

  if (!only.empty()) return 0;

  dump_avgprob_1d(dir);

  if (nsub > 0) {
    cout << "Dumping brute-force bin averages with " << nsub << "^2 points"
         << endl;
    dump_bin_averages(dir, nsub);
  }

  return 0;
}
