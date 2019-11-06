#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include "utils.h"
#include <dls.h>

namespace SCAMP {

// This kernel computes a sliding mean with specified window size and a
// corresponding prefix sum array (A)
__global__ void sliding_mean(double *pref_sum, size_t window, size_t size,
                             double *means) {
  const double coeff = 1.0 / (double)window;
  size_t a = blockIdx.x * blockDim.x + threadIdx.x;
  size_t b = blockIdx.x * blockDim.x + threadIdx.x + window;

  if (a == 0) {
    means[a] = pref_sum[window - 1] * coeff;
  }
  if (a < size - 1) {
    means[a + 1] = (pref_sum[b] - pref_sum[a]) * coeff;
  }
}

__global__ void sliding_norm(double *cumsumsqr, unsigned int window,
                             unsigned int size, double *norms) {
  int a = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.x * blockDim.x + threadIdx.x + window;
  if (a == 0) {
    norms[a] = 1 / sqrt(cumsumsqr[window - 1]);
  } else if (b < size + window) {
    norms[a] = 1 / sqrt(cumsumsqr[b - 1] - cumsumsqr[a - 1]);
  }
}

__global__ void sliding_dfdg(const double *T, const double *means, double *df,
                             double *dg, const int m, const int n) {
  const double half = 1.0 / (double)2.0;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n - 1) {
    df[tid] = (T[tid + m] - T[tid]) * half;
    dg[tid] = (T[tid + m] - means[tid + 1]) + (T[tid] - means[tid]);
  }
}

__global__ void __launch_bounds__(512, 4)
    fastinvnorm(double *norm, const double *mean, const double *T, int m,
                int n) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int jump = ceil(n / (double)(blockDim.x * gridDim.x));
  int start = jump * tid;
  int end = jump * (tid + 1);
  end = min(end, n);
  if (start >= n) {
    return;
  }
  double sum = 0;
  for (int i = 0; i < m; ++i) {
    double val = T[i + start] - mean[start];
    sum += val * val;
  }
  norm[start] = sum;

  for (int i = start + 1; i < end; ++i) {
    norm[i] =
        norm[i - 1] + ((T[i - 1] - mean[i - 1]) + (T[i + m - 1] - mean[i])) *
                          (T[i + m - 1] - T[i - 1]);
  }
  for (int i = start; i < end; ++i) {
    norm[i] = 1.0 / sqrt(norm[i]);
  }
}

__global__ void cross_correlation_to_ed(float *profile, unsigned int n,
                                        unsigned int m) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    profile[tid] = sqrt(max(2 * (1 - profile[tid]), 0.0)) * sqrt((double)m);
  }
}

__global__ void merge_mp_idx(float *mp, uint32_t *mpi, uint32_t n,
                             uint64_t *merged) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    mp_entry item;
    item.floats[0] = (float)mp[tid];
    item.ints[1] = mpi[tid];
    merged[tid] = item.ulong;
  }
}

void elementwise_max_with_index(std::vector<float> &mp_full,
                                std::vector<uint32_t> &mpi_full,
                                int64_t merge_start, int64_t tile_sz,
                                std::vector<uint64_t> *to_merge) {
  for (int i = 0; i < tile_sz; ++i) {
    mp_entry curr;
    curr.ulong = to_merge->at(i);
    if (mp_full[i + merge_start] < curr.floats[0]) {
      mp_full[i + merge_start] = curr.floats[0];
      mpi_full[i + merge_start] = curr.ints[1];
    }
  }
}

__global__ void elementwise_max_kernel(uint64_t *mp_full, uint64_t merge_start, uint64_t tile_sz,
                                       uint64_t *to_merge, uint64_t index_offset){                                           
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < tile_sz) {
    mp_entry e1, e2;
    e1.ulong = mp_full[tid + merge_start];
    e2.ulong = to_merge[tid];
    if (e1.floats[0] < e2.floats[0]) {
      e2.ints[1] += index_offset;
      mp_full[tid + merge_start] = e2.ulong;
    }
  }
}

void elementwise_max_device( uint64_t *mp_full, uint64_t merge_start, uint64_t tile_sz,
                             uint64_t *to_merge, uint64_t index_offset, cudaStream_t s){
  dim3 grid(ceil(tile_sz / (double)512), 1, 1);
  dim3 block(512, 1, 1);

  elementwise_max_kernel<<<grid, block, 0, s>>>(mp_full, merge_start, tile_sz, to_merge, index_offset);
  gpuErrchk(cudaPeekAtLastError());

}

extern "C" {
//   struct parameters_dls {
//     size_t n;
//     size_t m;
//     cudaStream_t s;
//   };

  dls_decdef(compute_statistics_HALADAPT,void, const double *T, double *input, double *scratch, size_t n, size_t m, size_t array_size);

  void compute_statistics_HALADAPT_GPU_CUDA(const double *T, double *input, double *scratch, size_t n, size_t m, size_t array_size) {
    dls_get_addr((void**) &T, "r");
    dls_get_addr((void**) &input, "b");
    dls_get_addr((void**) &scratch, "b");
    dls_get_arg(&n, sizeof(size_t));
    dls_get_arg(&m, sizeof(size_t));
    dls_get_arg(&array_size, sizeof(size_t));
    // dls_get_arg(&s, sizeof(cudaStream_t));
    //dls_get_arg(&parameters, sizeof(parameters_dls));

    double *norms = input;
    double *df = &input[array_size];
    double *dg = &input[2*array_size];
    double *means = &input[3*array_size];

    dim3 grid(ceil(n / (double)512), 1, 1);
    dim3 block(512, 1, 1);
  
    // gpuErrchk(cudaPeekAtLastError());
    // Use prefix sum to compute sliding mean
    sliding_mean<<<grid, block, 0, 0>>>(scratch, m, n, means);
    // gpuErrchk(cudaPeekAtLastError());

    // Compute differential values
    sliding_dfdg<<<grid, block, 0, 0>>>(T, means, df, dg, m, n);
    // gpuErrchk(cudaPeekAtLastError());

    // This will be kind of slow on the GPU, may cause latency between tiles
    int workers = n / m + 1;
    fastinvnorm<<<dim3(ceil(workers / (double)512), 1, 1), dim3(512, 1, 1), 0, 0>>>(norms, means, T, m, n);
    // gpuErrchk(cudaPeekAtLastError());
  }
}

void compute_statistics(const double *T, double *norms, double *df, double *dg,
                        double *means, size_t n, size_t m, cudaStream_t s,
                        double *scratch, size_t array_size, size_t tile_size) {
  // std::inclusive_scan(T, T + n + m -1, scratch); //TODO make that work, dont know why he cant find the inclusive scan in the headerfile numeric
  scratch[0] = T[0];
  int i;
  for (i=1 ; i< n + m - 1 ; i++) {
    scratch[i] = scratch[i-1] + T[i];
  }

  char *dls_modules = dls_get_module_info();

  if (!dls_is_in_list(dls_modules, "DLS_AUTOADD")) {
    printf("Manually register implementations...\n");
    dls_add_impl(compute_statistics_HALADAPT, "spm", "compute_statistics_HALADAPT_GPU_CUDA", &compute_statistics_HALADAPT_GPU_CUDA, PM_GPU | PM_CUDA);
  }

  double *input = (double*) malloc(sizeof(double) * array_size * 4);

  dls_register_marea((void*) T, tile_size*sizeof(double), DLS_VT_D);
  dls_register_marea(input, 4*array_size*sizeof(double), DLS_VT_D);
  dls_register_marea(scratch, tile_size*sizeof(double), DLS_VT_D);


  dls_predict_call(compute_statistics_HALADAPT, "rbbvvvp", T, input, scratch, &n, &m, &array_size, 1);

  dls_start_tgraph();

  dls_validate_marea((void*)T);
  dls_validate_marea((void*)scratch);
  dls_validate_marea((void*)input);

  dls_unregister_marea((void*)T);
  dls_unregister_marea((void*)scratch);
  dls_unregister_marea((void*)input);

  memcpy(norms, input, array_size*sizeof(double));
  memcpy(df, &input[array_size], array_size*sizeof(double));
  memcpy(dg, &input[2 * array_size], array_size*sizeof(double));
  memcpy(means, &input[3 * array_size], array_size*sizeof(double));

  free(input);
}

void launch_merge_mp_idx(float *mp, uint32_t *mpi, uint32_t n, uint64_t *merged,
                         cudaStream_t s) {
  merge_mp_idx<<<dim3(std::ceil(n / 1024.0), 1, 1), dim3(1024, 1, 1), 0, s>>>(
      mp, mpi, n, merged);
}

}  // namespace SCAMP
