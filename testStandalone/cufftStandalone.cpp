#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <stdio.h>

int main(int argc, char const *argv[])
{
  cufftResult cufftError;
  cufftHandle fft_plan;
  cufftPlan1d(&fft_plan, 10, CUFFT_D2Z, 1);

  double* T = (double *) malloc(10 * sizeof(double));

  for(int i = 0; i < 10; i++) {
    T[i] = 1.5 * i;
  }

  for(int i=0; i<10; i++) {
    printf("%f\n", T[i]);
  }

  double* T_cuda;
  cudaMalloc((void **)&T_cuda, 10* sizeof(double));
  cudaMemcpy(&T_cuda, &T, 10 * sizeof(double), cudaMemcpyHostToDevice);

  cuDoubleComplex * Tc_cuda;
  cudaMalloc((void **)&Tc_cuda, sizeof(cuDoubleComplex) * 10);

  cufftError = cufftExecD2Z(fft_plan, T_cuda, Tc_cuda);

  cuDoubleComplex *Tc;

  cudaMemcpy(&Tc, &Tc_cuda, 10, cudaMemcpyDeviceToHost);

  for(int i=0; i<10; i++) {
    printf("%f\n", Tc[i].y);
  }

  cudaError_t error;
  error = cudaPeekAtLastError();
  if (error != cudaSuccess) {
    printf("Got cuda Error %i\n", error);
  }

  if (cufftError != CUFFT_SUCCESS) {
    printf("Print the cufft error: %i\n", cufftError);
  }

  return 0;
}
