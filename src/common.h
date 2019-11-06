#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <cinttypes>
#include <unordered_map>
#include "SCAMP.pb.h"
namespace SCAMP {

typedef union {
  float floats[2];       // floats[0] = lowest
  unsigned int ints[2];  // ints[1] = lowIdx
  uint64_t ulong;        // for atomic update
} mp_entry;

template <unsigned int count>
struct reg_mem {
  float dist[count];
  double qt[count];
};

struct OptionalArgs {
  OptionalArgs(double threshold_) : threshold(threshold_) {}

  double threshold;
};

struct sizes_Array {
  size_t T_A_full_dev_size;
  size_t T_B_full_dev_size;
  size_t profile_a_full_dev_size;
  size_t profile_b_full_dev_size;
  size_t T_A_dev_size;
  size_t T_B_dev_size;
  size_t profile_a_tile_dev_size;
  size_t profile_b_tile_dev_size;
  size_t QT_dev_size;
  size_t means_A_size;
  size_t means_B_size;
  size_t norms_A_size;
  size_t norms_B_size;
  size_t df_A_size;
  size_t df_B_size;
  size_t dg_A_size;
  size_t dg_B_size;
  size_t scratchpad_size;
};

using DeviceProfile = std::unordered_map<int, void *>;

size_t GetProfileTypeSize(SCAMPProfileType t);

enum SCAMPError_t {
  SCAMP_NO_ERROR,
  SCAMP_FUNCTIONALITY_UNIMPLEMENTED,
  SCAMP_TILE_ILLEGAL_TYPE,
  SCAMP_CUDA_ERROR,
  SCAMP_CUFFT_ERROR,
  SCAMP_CUFFT_EXEC_ERROR,
  SCAMP_DIM_INCOMPATIBLE
};

enum SCAMPTileType {
  SELF_JOIN_FULL_TILE,
  SELF_JOIN_UPPER_TRIANGULAR,
  AB_JOIN_FULL_TILE,
  AB_FULL_JOIN_FULL_TILE
};

}  // namespace SCAMP

void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true);
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
