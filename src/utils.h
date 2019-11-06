#pragma once
#include <stdint.h>
#include <vector>
#include "SCAMP.pb.h"
#include "common.h"

namespace SCAMP {

void compute_statistics(const double *T, double *norms, double *df, double *dg,
                        double *means, size_t n, size_t m, cudaStream_t s,
                        double *scratch, size_t array_size, size_t tile_size);

void launch_merge_mp_idx(float *mp, uint32_t *mpi, uint32_t n, uint64_t *merged,
                         cudaStream_t s);

void elementwise_max_device( uint64_t *mp_full, uint64_t merge_start, uint64_t tile_sz,
                             uint64_t *to_merge, uint64_t index_offset, cudaStream_t s);

}  // namespace SCAMP
