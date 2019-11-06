#pragma once

#include <cufft.h>
#include <stdlib.h>
#include "common.h"

#ifdef _CUFFT_H_
// cuFFT API errors
static const char *_cudaGetErrorEnum(cufftResult error) {
  switch (error) {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";
    case CUFFT_INVALID_DEVICE:
      return "CUFFT_INVALID_DEVICE";
    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";
    case CUFFT_NO_WORKSPACE:
      return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:
      return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_PARSE_ERROR:
      return "CUFFT_PARSE_ERROR";
    case CUFFT_LICENSE_ERROR:
      return "CUFFT_LICENSE_ERROR";
    case CUFFT_NOT_SUPPORTED:
      return "CUFFT_NOT_SUPPORTED";
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return "CUFFT_INCOMPLETE_PARAMETER_LIST";
  }
}
#endif

#define CHECK_CUFFT_ERRORS(call)                           \
  {                                                        \
    cufftResult_t err;                                     \
    if ((err = (call)) != CUFFT_SUCCESS) {                 \
      fprintf(stderr, "cuFFT error %d:%s at %s:%d\n", err, \
              _cudaGetErrorEnum(err), __FILE__, __LINE__); \
      exit(1);                                             \
    }                                                      \
  }

namespace SCAMP {

class fft_precompute_helper {
 private:
  double *Q_reverse_pad;
  cuDoubleComplex *Qc, *Tc;
  cufftHandle fft_plan, ifft_plan;
  const size_t size;
  const size_t window_size;
  const size_t cufft_data_size;
  const bool double_precision;
  const int fft_work_size = 512;
  const int device;

 public:
  fft_precompute_helper(size_t sz, size_t window_sz, bool dp, int device_)
      : size(sz),
        window_size(window_sz),
        cufft_data_size(sz / 2 + 1),
        double_precision(dp),
        device(device_) {
    if (double_precision) {
      CHECK_CUFFT_ERRORS(cufftPlan1d(&fft_plan, size, CUFFT_D2Z, 1));
      CHECK_CUFFT_ERRORS(cufftPlan1d(&ifft_plan, size, CUFFT_Z2D, 1));
    } else {
      CHECK_CUFFT_ERRORS(cufftPlan1d(&fft_plan, size, CUFFT_R2C, 1));
      CHECK_CUFFT_ERRORS(cufftPlan1d(&ifft_plan, size, CUFFT_C2R, 1));
    }
    // cudaMalloc(&Q_reverse_pad, sizeof(double) * size);
    // gpuErrchk(cudaPeekAtLastError());
    Q_reverse_pad = (double *) malloc(sizeof(double) * size);
    // cudaMalloc(&Tc, sizeof(cuDoubleComplex) * cufft_data_size);
    // gpuErrchk(cudaPeekAtLastError());
    Tc = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * cufft_data_size);
    // cudaMalloc(&Qc, sizeof(cuDoubleComplex) * cufft_data_size);
    // gpuErrchk(cudaPeekAtLastError());
    Qc = (cuDoubleComplex *) malloc(sizeof(cuDoubleComplex) * cufft_data_size);
  }
  ~fft_precompute_helper() {
    // cudaFree(Q_reverse_pad);
    // gpuErrchk(cudaPeekAtLastError());
    free(Q_reverse_pad);
    // cudaFree(Tc);
    // gpuErrchk(cudaPeekAtLastError());
    free(Tc);
    // cudaFree(Qc);
    // gpuErrchk(cudaPeekAtLastError());
    free(Qc);
    CHECK_CUFFT_ERRORS(cufftDestroy(fft_plan));
    CHECK_CUFFT_ERRORS(cufftDestroy(ifft_plan));
  }
  SCAMPError_t compute_QT(double *QT, const double *T, const double *Q,
                          const double *qmeans, cudaStream_t s, size_t size_QT, 
                          size_t size_T, size_t size_Q, size_t size_qmeans);
};

}  // namespace SCAMP
