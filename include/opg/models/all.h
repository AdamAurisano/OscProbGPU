///////////////////////////////////////////////////////////////////////////////
/// \file all.h
///
/// \brief All models, plus an X-macro listing the explicit instantiations
///        compiled into the library (CUDA backend).
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_MODELS_ALL_H
#define OPG_MODELS_ALL_H

#include "opg/models/fast.h"
#include "opg/models/nsi.h"
#include "opg/models/nunm.h"

/// X(ModelType) for every model compiled into the CUDA backend.
#define OPG_FOR_EACH_MODEL(X) \
  X(opg::Fast<double>)        \
  X(opg::Fast<float>)         \
  X(opg::NSI<double>)         \
  X(opg::NUNM<double>)

#endif
