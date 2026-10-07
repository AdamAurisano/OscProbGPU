///////////////////////////////////////////////////////////////////////////////
/// \file all.h
///
/// \brief All models, plus an X-macro listing the explicit instantiations
///        compiled into the library (CUDA backend).
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_MODELS_ALL_H
#define OPG_MODELS_ALL_H

#include <type_traits>

#include "opg/models/decay.h"
#include "opg/models/deco.h"
#include "opg/models/fast.h"
#include "opg/models/liv.h"
#include "opg/models/nsi.h"
#include "opg/models/nunm.h"
#include "opg/models/oqs.h"
#include "opg/models/sidereal_liv.h"
#include "opg/models/snsi.h"
#include "opg/models/sterile.h"

/// X(ModelType) for every model compiled into the CUDA backend. Other
/// instantiations (e.g. long double) are available on the CPU backend only.
/// A build may define its own list first (fewer models compile faster),
/// e.g. with a header passed to the compilers by -include that contains
/// `#define OPG_FOR_EACH_MODEL(X) X(opg::Sterile<double>)`.
#ifndef OPG_FOR_EACH_MODEL
#define OPG_FOR_EACH_MODEL(X) \
  X(opg::Fast<double>)        \
  X(opg::Fast<float>)         \
  X(opg::NSI<double>)         \
  X(opg::NUNM<double>)        \
  X(opg::Sterile<double>)     \
  X(opg::Decay<double>)       \
  X(opg::LIV<double>)         \
  X(opg::SNSI<double>)        \
  X(opg::Deco<double>)        \
  X(opg::SiderealLIV<double>) \
  X(opg::OQS<double>)
#endif

namespace opg {
  template <class Model> struct has_cuda_engine : std::false_type {};
#define OPG_DECLARE_CUDA_MODEL(M) \
  template <> struct has_cuda_engine<M> : std::true_type {};
  OPG_FOR_EACH_MODEL(OPG_DECLARE_CUDA_MODEL)
#undef OPG_DECLARE_CUDA_MODEL
} // namespace opg

#endif
