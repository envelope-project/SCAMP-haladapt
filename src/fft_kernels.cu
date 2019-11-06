#include "fft_kernels.h"
#include <dls.h>
#include <stdio.h>

__global__ void elementwise_multiply_inplace(const cuDoubleComplex *A,
                                             cuDoubleComplex *B,
                                             const int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    B[tid] = cuCmul(A[tid], B[tid]);
  }
}

// A is input unaligned sliding dot products produced by ifft
// out is the computed vector of distances
__global__ void normalized_aligned_dot_products(const double *A,
                                                const double divisor,
                                                const unsigned int m,
                                                const unsigned int n,
                                                double *QT) {
  int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a < n) {
    QT[a] = A[a + m - 1] / divisor;
  }
}

__global__ void populate_reverse_pad(const double *Q, double *Q_reverse_pad,
                                     const double *mean, const int window_size,
                                     const int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  double mu = *mean;
  if (tid < window_size) {
    Q_reverse_pad[tid] = Q[window_size - 1 - tid] - mu;
  } else if (tid < size) {
    Q_reverse_pad[tid] = 0;
  }
}

extern "C" {
  dls_decdef(launch_populate_reverse_pad_HALADAPT, void, const double *Q, double *Q_reverse_pad, const double *mean, const int window_size, const int size, int fft_work_size);

  void launch_populate_reverse_pad_HALADAPT_GPU_CUDA(const double *Q, double *Q_reverse_pad, const double *mean, const int window_size, const int size, int fft_work_size) {

    dls_get_addr((void **) &Q, "r");
    dls_get_addr((void **) &Q_reverse_pad, "w");
    dls_get_addr((void **) &mean, "r");
    dls_get_arg((void *)&window_size, sizeof(int));
    dls_get_arg((void *)&size, sizeof(int));
    dls_get_arg(&fft_work_size, sizeof(int));

    dim3 block(fft_work_size, 1, 1);
    populate_reverse_pad<<<dim3(ceil(size / (float)fft_work_size), 1, 1), block,
                           0, 0>>>(Q, Q_reverse_pad, mean, window_size, size);
  }
}

void launch_populate_reverse_pad(const double *Q, double *Q_reverse_pad,
                                 const double *mean, const int window_size,
                                 const int size, int fft_work_size, 
                                 size_t size_Q, size_t size_Q_reverse_pad, 
                                 size_t size_mean) {
  char *dls_modules = dls_get_module_info();

  if (!dls_is_in_list(dls_modules, "DLS_AUTOADD")) {
    printf("Manually register implementations...\n");
    dls_add_impl(launch_populate_reverse_pad_HALADAPT, "spm", "launch_populate_reverse_pad_HALADAPT_GPU_CUDA", &launch_populate_reverse_pad_HALADAPT_GPU_CUDA, PM_GPU | PM_CUDA);
  }

  dls_register_marea((void *)Q, size_Q , DLS_VT_D);
  dls_register_marea((void *)Q_reverse_pad, size_Q_reverse_pad, DLS_VT_D);
  dls_register_marea((void *)mean, size_mean, DLS_VT_D);

  dls_predict_call(launch_populate_reverse_pad_HALADAPT, "rwrvvvp", Q, Q_reverse_pad, mean, &window_size, &size, &fft_work_size, 1);

  dls_start_tgraph();

  dls_validate_marea((void *)Q);
  dls_validate_marea((void *)Q_reverse_pad);
  dls_validate_marea((void *)mean);

  dls_unregister_marea((void *)Q);
  dls_unregister_marea((void *)Q_reverse_pad);
  dls_unregister_marea((void *)mean); 
}

extern "C" {
  dls_decdef(launch_elementwise_multiply_inplace_HALADAPT, void, const cuDoubleComplex *A, cuDoubleComplex *B, const int size, int fft_work_size);

  void launch_elementwise_multiply_inplace_HALADAPT_GPU_CUDA(const cuDoubleComplex *A,
                                          cuDoubleComplex *B, const int size,
                                          int fft_work_size) {
    dls_get_addr((void **)&A, "r");
    dls_get_addr((void **)&B, "b");
    dls_get_arg((void *)&size, sizeof(const int));
    dls_get_arg(&fft_work_size, sizeof(int));

    dim3 block(fft_work_size, 1, 1);
    elementwise_multiply_inplace<<<dim3(ceil(size / (float)fft_work_size), 1, 1),
                                  block, 0, 0>>>(A, B, size);
  }
}

  void launch_elementwise_multiply_inplace(const cuDoubleComplex *A,
                                        cuDoubleComplex *B, const int size,
                                        int fft_work_size, 
                                        size_t size_A, size_t size_B) {
  char *dls_modules = dls_get_module_info();

  if (!dls_is_in_list(dls_modules, "DLS_AUTOADD")) {
    printf("Manually register implementations...\n");
    dls_add_impl(launch_elementwise_multiply_inplace_HALADAPT, "spm", "launch_elementwise_multiply_inplace_HALADAPT_GPU_CUDA", &launch_elementwise_multiply_inplace_HALADAPT_GPU_CUDA, PM_GPU | PM_CUDA);
  }

  dls_register_marea((void *)A, size_A, DLS_VT_CONST_PTR);
  dls_register_marea(B, size_B, DLS_VT_PTR);

  dls_predict_call(launch_elementwise_multiply_inplace_HALADAPT, "rbvvp", A, B, &size, &fft_work_size, 1);
  dls_start_tgraph();

  dls_validate_marea((void *)A);
  dls_validate_marea(B);

  dls_unregister_marea((void* )A);
  dls_unregister_marea(B);                                        
}

extern "C" {
  dls_decdef(launch_normalized_aligned_dot_products_HALADAPT, void, const double *A, const double divisor, const unsigned int m, const unsigned int n, double *QT, int fft_work_size);

  void launch_normalized_aligned_dot_products_HALADAPT_GPU_CUDA(
                                            const double *A,
                                            const double divisor,
                                            const unsigned int m,
                                            const unsigned int n, double *QT,
                                            int fft_work_size) {
    dls_get_addr((void **)&A, "r");
    dls_get_arg((void *)&divisor, sizeof(double));
    dls_get_arg((void *)&m, sizeof(const unsigned int));
    dls_get_arg((void *)&n, sizeof(const unsigned int));
    dls_get_addr((void **)&QT, "w");
    dls_get_arg((void **)&fft_work_size, sizeof(int));

    dim3 block(fft_work_size, 1, 1);
    normalized_aligned_dot_products<<<dim3(ceil(n / (float)fft_work_size), 1, 1),
                                      block, 0, 0>>>(A, divisor, m, n, QT);
  }
}
void launch_normalized_aligned_dot_products(const double *A,
                                            const double divisor,
                                            const unsigned int m,
                                            const unsigned int n, double *QT,
                                            int fft_work_size, 
                                            size_t size_A, size_t size_QT) {
  char *dls_modules = dls_get_module_info();

  if (!dls_is_in_list(dls_modules, "DLS_AUTOADD")) {
    printf("Manually register implementations...\n");
    dls_add_impl(launch_normalized_aligned_dot_products_HALADAPT, "spm", "launch_normalized_aligned_dot_products_HALADAPT_GPU_CUDA", &launch_normalized_aligned_dot_products_HALADAPT_GPU_CUDA, PM_GPU | PM_CUDA);
  }

  dls_register_marea((void *)A, size_A, DLS_VT_D);
  dls_register_marea(QT, size_QT, DLS_VT_D);

  dls_predict_call(launch_normalized_aligned_dot_products_HALADAPT, "rvvvwvp", A, &divisor, &m, &n, QT, &fft_work_size, 1);
  dls_start_tgraph();

  dls_validate_marea((void *)A);
  dls_validate_marea(QT);

  dls_unregister_marea((void* )A);
  dls_unregister_marea(QT);
}
