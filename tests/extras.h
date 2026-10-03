// Per-event extra inputs (SiderealLIV azimuth and sidereal time) in event
// lists against the same values set per propagator (CPU and GPU backends).

#ifndef OPG_TESTS_EXTRAS_H
#define OPG_TESTS_EXTRAS_H

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "doctest.h"

#include "variants.h"

#include "opg/models/all.h"
#include "opg/propagator.h"

namespace extratest {

  using SLIV = opg::SiderealLIV<double>;

  /// Events with their own azimuth [deg] and sidereal time [h].
  struct Events {
      std::vector<double>  E, C, extra;  ///< extra[x * n + i]
      std::vector<uint8_t> nb;
      size_t size() const { return E.size(); }
      double azimuth(size_t i) const { return extra[i]; }
      double time(size_t i) const { return extra[size() + i]; }
  };

  inline Events make_events(size_t n = 48)
  {
    std::mt19937_64                        rng(7);
    std::uniform_real_distribution<double> u(0, 1);
    Events                                 ev;
    std::vector<double>                    azi, t;
    for (size_t i = 0; i < n; i++) {
      ev.E.push_back(std::pow(10.0, -0.3 + 2.0 * u(rng)));
      ev.C.push_back(-1 + 2 * u(rng));
      ev.nb.push_back(uint8_t(i % 2));
      azi.push_back(360 * u(rng));
      t.push_back(24 * u(rng));
    }
    ev.extra = azi;
    ev.extra.insert(ev.extra.end(), t.begin(), t.end());
    return ev;
  }

  /// The parameters of event i set per propagator: the event's azimuth
  /// (with the path zenith, or the fixed zenith) and sidereal time.
  inline SLIV::Params event_params(SLIV::Params p, const Events& ev, size_t i)
  {
    if (p.fixed_dir)
      p.SetNeutrinoDirection(p.zenith, ev.azimuth(i));
    else
      p.SetAzimuth(ev.azimuth(i));
    p.SetTimeHours(ev.time(i));
    return p;
  }

  /// Event lists with extras agree with one event per propagator setting:
  /// probabilities, gradients and the weighted modes. tol is absolute
  /// (0 = bit-identical, as on the CPU; the GPU evaluates the sidereal
  /// phase with the device sin/cos).
  inline void check_extras(opg::Propagator<SLIV>& prop, double tol)
  {
    static_assert(opg::Propagator<SLIV>::n_event_extra() == 2);
    static_assert(opg::Propagator<opg::Fast<double>>::n_event_extra() == 0);
    constexpr int NN = 9;
    const Events  ev = make_events();
    const size_t  n  = ev.size();
    const bool    grads = opg::Propagator<SLIV>::has_gradients();

    for (auto par : {variants::sidereal(), variants::sidereal_fixed()}) {
      const std::string tag = par.fixed_dir ? "fixed zenith" : "path zenith";
      INFO(tag);
      std::vector<std::string> names;
      if (grads) names = {"th23", "dm31", "aX_emu", "aZ_mutau", "cXX_ee", "cXZ_etau"};
      const size_t np = names.size();

      // reference: one event at a time with the extras set as parameters
      std::vector<double> Pref(NN * n), dPref(np * NN * n);
      for (size_t i = 0; i < n; i++) {
        prop.set_params(event_params(par, ev, i));
        std::vector<double>  e = {ev.E[i]}, c = {ev.C[i]};
        std::vector<uint8_t> b = {ev.nb[i]};
        std::vector<double>  P, dP;
        if (grads) {
          prop.set_gradient_params(names);
          prop.prob_points_grad(e, c, b, P, dP);
          for (size_t k = 0; k < np * NN; k++) dPref[k * n + i] = dP[k];
        }
        else
          P = prop.prob_points(e, c, b);
        for (int ab = 0; ab < NN; ab++) Pref[ab * n + i] = P[ab];
      }

      // the parameters' own azimuth and time are ignored with extras
      prop.set_params(par);
      auto   P = prop.prob_points(ev.E, ev.C, ev.nb, ev.extra);
      double d = 0;
      for (size_t k = 0; k < P.size(); k++) d = std::max(d, std::fabs(P[k] - Pref[k]));
      MESSAGE("SiderealLIV (" << tag << ") per-event extras vs per-propagator: max|dP| = "
                              << d);
      CHECK(d <= tol);
      // without extras: the parameters' values, which differ
      auto   P0 = prop.prob_points(ev.E, ev.C, ev.nb);
      double d0 = 0;
      for (size_t k = 0; k < P0.size(); k++) d0 = std::max(d0, std::fabs(P0[k] - P[k]));
      CHECK(d0 > 1e-6);

      // size checks
      std::vector<double> bad(ev.extra.begin(), ev.extra.begin() + n);
      CHECK_THROWS_AS(prop.prob_points(ev.E, ev.C, ev.nb, bad), std::invalid_argument);

      if (!grads) continue;
      prop.set_gradient_params(names);
      std::vector<double> Pg, dP;
      prop.prob_points_grad(ev.E, ev.C, ev.nb, Pg, dP, ev.extra);
      double dp = 0, dg = 0, gs = 0;
      for (size_t k = 0; k < Pg.size(); k++) dp = std::max(dp, std::fabs(Pg[k] - Pref[k]));
      for (size_t k = 0; k < dP.size(); k++) {
        dg = std::max(dg, std::fabs(dP[k] - dPref[k]));
        gs = std::max(gs, std::fabs(dPref[k]));
      }
      MESSAGE("  gradients: max|dP| = " << dp << ", max|d grad| = " << dg
                                        << " (max|grad| = " << gs << ")");
      CHECK(dp <= tol);
      CHECK(dg <= tol * std::max(gs, 1.0));

      // weighted and per-bin weighted gradients vs explicit contraction
      std::mt19937_64                        rng(11);
      std::uniform_real_distribution<double> u(-1, 1);
      std::vector<double>                    w(NN * n);
      for (auto& x : w) x = u(rng);
      std::vector<int> bins(n);
      const int        nbins = 5;
      for (size_t i = 0; i < n; i++) bins[i] = int(i % (nbins + 1)) - 1;  // -1: ignored
      auto g  = prop.weighted_gradient_points(ev.E, ev.C, ev.nb, w, ev.extra);
      auto Gb = prop.weighted_gradient_points_binned(ev.E, ev.C, ev.nb, w, bins, nbins,
                                                     ev.extra);
      std::vector<double> ge(np, 0), gbe(nbins * np, 0), scale(np, 0);
      for (size_t p = 0; p < np; p++)
        for (int ab = 0; ab < NN; ab++)
          for (size_t i = 0; i < n; i++) {
            double t = w[ab * n + i] * dPref[(p * NN + ab) * n + i];
            ge[p] += t;
            scale[p] += std::fabs(t);
            if (bins[i] >= 0) gbe[bins[i] * np + p] += t;
          }
      double rw = 0, rb = 0;
      for (size_t p = 0; p < np; p++) {
        const double s = std::max(scale[p], 1e-300);
        rw = std::max(rw, std::fabs(g[p] - ge[p]) / s);
        for (int b = 0; b < nbins; b++)
          rb = std::max(rb, std::fabs(Gb[b * np + p] - gbe[b * np + p]) / s);
      }
      MESSAGE("  weighted: rel " << rw << ", per-bin rel " << rb);
      CHECK(rw < 1e-12 + tol);
      CHECK(rb < 1e-12 + tol);
    }

    // models without extras reject them
    opg::Propagator<opg::Fast<double>> fast;
    opg::Fast<double>::Params          fp;
    fp.mix = variants::nominal_mix<3>();
    fast.set_params(fp);
    CHECK_THROWS_AS(fast.prob_points({1.0}, {-1.0}, {0}, {0.0}), std::invalid_argument);
  }

} // namespace extratest

#endif
