#include "opg/propagator.h"

#ifdef OPG_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace opg {

  int cuda_device_count()
  {
#ifdef OPG_HAVE_CUDA
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
#else
    return 0;
#endif
  }

} // namespace opg
