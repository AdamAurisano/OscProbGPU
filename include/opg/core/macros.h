///////////////////////////////////////////////////////////////////////////////
/// \file macros.h
///
/// \brief Portability macros so that the same physics code compiles for the
///        host (g++/clang) and for CUDA devices (nvcc).
///////////////////////////////////////////////////////////////////////////////

#ifndef OPG_CORE_MACROS_H
#define OPG_CORE_MACROS_H

#if defined(__CUDACC__)
#define OPG_HD __host__ __device__
#define OPG_DEVICE __device__
#define OPG_INLINE __forceinline__
#else
#define OPG_HD
#define OPG_DEVICE
#define OPG_INLINE inline
#endif

#if defined(__CUDA_ARCH__)
#define OPG_ON_DEVICE 1
#else
#define OPG_ON_DEVICE 0
#endif

#if defined(__CUDACC__) || defined(__clang__) || defined(__GNUC__)
#define OPG_UNROLL _Pragma("unroll")
#else
#define OPG_UNROLL
#endif

#endif
