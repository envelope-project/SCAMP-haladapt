#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda.h>

#include "fft_helper.h"
#include "fft_kernels.h"
#include <dls.h>

namespace SCAMP {

SCAMPError_t fft_precompute_helper::compute_QT(double *QT, const double *T,
                                               const double *Q,
                                               const double *qmeans,
                                               cudaStream_t s,
                                               size_t size_QT, 
                                               size_t size_T, 
                                               size_t size_Q, 
                                               size_t size_qmeans
                                               ) {
  cufftResult cufftError;
  cudaError_t error;

  const int n = size - window_size + 1;

  double* T_cuda;
  cudaMalloc(&T_cuda, size_T);
  cudaMemcpy(T_cuda, T, size_T, cudaMemcpyHostToDevice);

  cuDoubleComplex* Tc_cuda;
  cudaMalloc(&Tc_cuda, sizeof(cuDoubleComplex) * cufft_data_size);

  // Compute the FFT of the time series
  // For some reason the input parameter to cufftExecD2Z is not held const by
  // cufft
  // I see nowhere in the documentation that the input vector is modified
  // using const_cast as a hack to get around this...
  // Since the cufft libary exepects a cuda adress and a cpu plan, HALadapt does not fit this needs.
  cufftError = cufftExecD2Z(fft_plan, T_cuda, Tc_cuda);  // NOLINT

  cudaMemcpy(Tc, Tc_cuda, sizeof(cuDoubleComplex) * cufft_data_size, cudaMemcpyDeviceToHost);

  cudaFree(Tc_cuda);

  cudaFree(T_cuda);

  if (cufftError != CUFFT_SUCCESS) {
    return SCAMP_CUFFT_EXEC_ERROR;
  }

  // Reverse and zero pad the query
  launch_populate_reverse_pad(Q, Q_reverse_pad, qmeans, window_size, size,
                              fft_work_size, size_Q, sizeof(double) * size , size_qmeans);

  error = cudaPeekAtLastError();
  if (error != cudaSuccess) {
    return SCAMP_CUDA_ERROR;
  }

  double* Q_reverse_pad_cuda;
  cudaMalloc(&Q_reverse_pad_cuda, sizeof(double) * size);
  cudaMemcpy(Q_reverse_pad_cuda, Q_reverse_pad, sizeof(double) * size, cudaMemcpyHostToDevice);

  cuDoubleComplex *Qc_cuda;
  cudaMalloc(&Qc_cuda, sizeof(cuDoubleComplex)  * cufft_data_size);

  cufftError = cufftExecD2Z(fft_plan, Q_reverse_pad_cuda, Qc_cuda);

  cudaMemcpy(Qc, Qc_cuda, sizeof(cuDoubleComplex) * cufft_data_size, cudaMemcpyDeviceToHost);

  cudaFree(Qc_cuda);

  cudaFree(Q_reverse_pad_cuda);

  if (cufftError != CUFFT_SUCCESS) {
    return SCAMP_CUFFT_EXEC_ERROR;
  }

  launch_elementwise_multiply_inplace(Tc, Qc, cufft_data_size, fft_work_size, sizeof(cuDoubleComplex) * cufft_data_size, sizeof(cuDoubleComplex) * cufft_data_size);
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) {
    return SCAMP_CUDA_ERROR;
  }

  cudaMalloc(&Q_reverse_pad_cuda, sizeof(double) * size);

  cudaMalloc(&Qc_cuda, sizeof(cuDoubleComplex)  * cufft_data_size);
  cudaMemcpy(Qc_cuda, Qc, sizeof(double) * size, cudaMemcpyHostToDevice);

  cufftError = cufftExecZ2D(ifft_plan, Qc_cuda, Q_reverse_pad_cuda);

  cudaMemcpy(Q_reverse_pad, Q_reverse_pad_cuda, sizeof(double) * size, cudaMemcpyDeviceToHost);

  cudaFree(Qc_cuda);
  cudaFree(Q_reverse_pad_cuda);

  if (cufftError != CUFFT_SUCCESS) {
    return SCAMP_CUFFT_EXEC_ERROR;
  }
  launch_normalized_aligned_dot_products(Q_reverse_pad, size, window_size, n,
                                         QT, fft_work_size, sizeof(double) * size, size_QT);
  error = cudaPeekAtLastError();

  if (error != cudaSuccess) {
    return SCAMP_CUDA_ERROR;
  }

  return SCAMP_NO_ERROR;
}

}  // namespace SCAMP
