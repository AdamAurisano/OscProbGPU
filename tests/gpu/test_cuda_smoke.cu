// Phase 0 smoke test: run OPG_HD code on the device and compare with host.

#include <cuda_runtime.h>

#include <vector>

#include "doctest.h"

#include "opg/core/matrix.h"

namespace {

  using M = opg::Mat<3, double>;

  __global__ void square_kernel(const M* in, M* out, int n)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = opg::matmul(in[i], in[i]);
  }

} // namespace

TEST_CASE("CUDA device is usable and OPG_HD code runs on it")
{
  int ndev = 0;
  REQUIRE(cudaGetDeviceCount(&ndev) == cudaSuccess);
  REQUIRE(ndev > 0);

  cudaDeviceProp prop;
  REQUIRE(cudaGetDeviceProperties(&prop, 0) == cudaSuccess);
  MESSAGE("Device 0: " << prop.name << " (sm_" << prop.major << prop.minor
                       << ")");

  const int      n = 1000;
  std::vector<M> h_in(n), h_out(n);
  for (int k = 0; k < n; k++)
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        h_in[k](i, j) = opg::Complex<double>(0.01 * k + i, 0.1 * j - k * 1e-3);

  M *d_in = nullptr, *d_out = nullptr;
  REQUIRE(cudaMalloc(&d_in, n * sizeof(M)) == cudaSuccess);
  REQUIRE(cudaMalloc(&d_out, n * sizeof(M)) == cudaSuccess);
  REQUIRE(cudaMemcpy(d_in, h_in.data(), n * sizeof(M),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  square_kernel<<<(n + 127) / 128, 128>>>(d_in, d_out, n);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(cudaMemcpy(h_out.data(), d_out, n * sizeof(M),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  cudaFree(d_in);
  cudaFree(d_out);

  double maxdiff = 0;
  for (int k = 0; k < n; k++) {
    M ref = opg::matmul(h_in[k], h_in[k]);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        maxdiff = std::max(maxdiff, opg::abs(ref(i, j) - h_out[k](i, j)));
  }
  CHECK(maxdiff < 1e-12);
}
