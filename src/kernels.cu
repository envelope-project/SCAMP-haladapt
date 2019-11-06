#include <unordered_map>
#include "kernels.h"
#include <dls.h>
#include <cstring>

namespace SCAMP {

constexpr int DIAGS_PER_THREAD = 4;
constexpr int BLOCKSZ_SP = 512;
constexpr int BLOCKSZ_DP = 256;
constexpr int BLOCKSPERSM_SELF = 2;
constexpr int BLOCKSPERSM_AB = 2;
constexpr int TILE_HEIGHT_SP = 200;
constexpr int TILE_HEIGHT_DP = 200;
constexpr float CC_MIN = -FLT_MAX;

template <typename T>
struct SCAMPKernelInputArgs {
  SCAMPKernelInputArgs(const T * __restrict__ cov_, const T * __restrict__ dfa_,
                       const T * __restrict__ dfb_, const T * __restrict__ dga_,
                       const T * __restrict__ dgb_,
                       const T * __restrict__ normsa_,
                       const T * __restrict__ normsb_, uint32_t n_x_,
                       uint32_t n_y_, int32_t exclusion_lower_,
                       int32_t exclusion_upper_, OptionalArgs opt_,
                       size_t size_cov_, size_t size_dfa_, size_t size_dfb_,
                       size_t size_dga_, size_t size_dgb_,
                       size_t size_normsa_, size_t size_normsb_)
      : cov(cov_),
        dfa(dfa_),
        dfb(dfb_),
        dga(dga_),
        dgb(dgb_),
        normsa(normsa_),
        normsb(normsb_),
        n_x(n_x_),
        n_y(n_y_),
        exclusion_lower(exclusion_lower_),
        exclusion_upper(exclusion_upper_),
        opt(opt_),
        size_cov(size_cov_),
        size_dfa(size_dfa_),
        size_dfb(size_dfb_),
        size_dga(size_dga_),
        size_dgb(size_dgb_),
        size_normsa(size_normsa_),
        size_normsb(size_normsb_) {}
  const T * __restrict__ cov;
  size_t size_cov;
  const T * __restrict__ dfa;
  size_t size_dfa;
  const T * __restrict__ dfb;
  size_t size_dfb;
  const T * __restrict__ dga;
  size_t size_dga;
  const T * __restrict__ dgb;
  size_t size_dgb;
  const T * __restrict__ normsa;
  size_t size_normsa;
  const T * __restrict__ normsb;
  size_t size_normsb;
  uint32_t n_x;
  uint32_t n_y;
  int32_t exclusion_lower;
  int32_t exclusion_upper;
  OptionalArgs opt;
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE>
struct SCAMPSmem {
  __device__ SCAMPSmem(char *smem, bool compute_rows, bool compute_columns,
                       int tile_width, int tile_height) {
    constexpr int data_size = sizeof(DATA_TYPE);
    constexpr int profile_size = sizeof(PROFILE_DATA_TYPE);
    int curr_byte = 0;
    df_col = (DATA_TYPE *)(smem);
    curr_byte += tile_width * data_size;
    dg_col = (DATA_TYPE *)(smem + curr_byte);
    curr_byte += tile_width * data_size;
    inorm_col = (DATA_TYPE *)(smem + curr_byte);
    curr_byte += tile_width * data_size;
    df_row = (DATA_TYPE *)(smem + curr_byte);
    curr_byte += tile_height * data_size;
    dg_row = (DATA_TYPE *)(smem + curr_byte);
    curr_byte += tile_height * data_size;
    inorm_row = (DATA_TYPE *)(smem + curr_byte);
    curr_byte += tile_height * data_size;
    if (compute_columns) {
      local_mp_col = (PROFILE_DATA_TYPE *)(smem + curr_byte);
      curr_byte += tile_width * profile_size;
    } else {
      local_mp_col = nullptr;
    }
    if (compute_rows) {
      local_mp_row = (PROFILE_DATA_TYPE *)(smem + curr_byte);
      curr_byte += tile_height * profile_size;
    } else {
      local_mp_row = nullptr;
    }
  }
  DATA_TYPE *__restrict__ df_col;
  DATA_TYPE *__restrict__ dg_col;
  DATA_TYPE *__restrict__ inorm_col;
  DATA_TYPE *__restrict__ df_row;
  DATA_TYPE *__restrict__ dg_row;
  DATA_TYPE *__restrict__ inorm_row;
  PROFILE_DATA_TYPE *__restrict__ local_mp_col;
  PROFILE_DATA_TYPE *__restrict__ local_mp_row;
};

template <typename ACCUM_TYPE>
struct SCAMPThreadInfo {
  ACCUM_TYPE cov1;
  ACCUM_TYPE cov2;
  ACCUM_TYPE cov3;
  ACCUM_TYPE cov4;
  uint32_t local_row;
  uint32_t local_col;
  uint32_t global_row;
  uint32_t global_col;
};

enum SCAMPAtomicType { ATOMIC_BLOCK, ATOMIC_GLOBAL, ATOMIC_SYSTEM };

#if __CUDA_ARCH__ < 600
// Double atomicAdd is not implemented in hardware before Pascal, providing a
// software implementation here
__device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

__device__ inline unsigned long long do_atomicCAS(unsigned long long *address,
                                                  unsigned long long v1,
                                                  unsigned long long v2) {
#if __CUDA_ARCH__ < 600
  return atomicCAS(address, v1, v2);
#else
  return atomicCAS_block(address, v1, v2);
#endif
}

template <typename T, SCAMPAtomicType type>
__device__ inline uint32_t do_atomicAdd(T *address, T amount) {
#if __CUDA_ARCH__ < 600
  return atomicAdd(address, amount);
#else
  switch (type) {
    case ATOMIC_BLOCK:
      return atomicAdd_block(address, amount);
    case ATOMIC_GLOBAL:
      return atomicAdd(address, amount);
    case ATOMIC_SYSTEM:
      return atomicAdd_system(address, amount);
  }
  // Should never happen
  return 0;
#endif
}

// Atomically updates the MP/idxs using a single 64-bit integer. We lose a small
// amount of precision in the output, if we do not do this we are unable
// to atomically update both the matrix profile and the indexes without using a
// critical section and dedicated locks.
__device__ inline void MPatomicMax(volatile uint64_t *address, float val,
                                   unsigned int idx) {
  mp_entry loc, loctest;
  loc.floats[0] = val;
  loc.ints[1] = idx;
  loctest.ulong = *address;
  while (loctest.floats[0] < val) {
    loctest.ulong =
        atomicCAS((unsigned long long int *)address, loctest.ulong, loc.ulong);
  }
}

// As above, but checks a previously read value before attempting another read
// This allows us to exploit vectorized loads of the matrix profile
__device__ inline void MPatomicMax_check(
    volatile uint64_t *__restrict__ address, float val, unsigned int idx,
    float curr_val) {
  if (val > curr_val) {
    mp_entry loc, loctest;
    loc.floats[0] = val;
    loc.ints[1] = idx;
    loctest.ulong = *address;
    while (loctest.floats[0] < val) {
      loctest.ulong = do_atomicCAS((unsigned long long int *)address,
                                   loctest.ulong, loc.ulong);
    }
  }
}

__device__ inline void MPMax(const float d1, const float d2,
                             const unsigned int i1, const unsigned int i2,
                             float &outd, unsigned int &outi) {
  if (d1 >= d2) {
    outd = d1;
    outi = i1;
  } else {
    outd = d2;
    outi = i2;
  }
}

// Computes max(a,b) with index and stores the result in a
__device__ inline void MPMax2(float &d1, const float &d2, unsigned int &i1,
                              const unsigned int &i2) {
  if (d2 > d1) {
    d1 = d2;
    i1 = i2;
  }
}
template <typename T>
__device__ inline T max4(T &d1, T &d2, T &d3, T &d4, const uint32_t init,
                         uint32_t &idx) {
  float ret = d1;
  idx = init;
  if (d2 > ret) {
    ret = d2;
    idx = init + 1;
  }
  if (d3 > ret) {
    ret = d3;
    idx = init + 2;
  }
  if (d4 > ret) {
    ret = d4;
    idx = init + 3;
  }
  return ret;
}

class SCAMPStrategy {
 public:
};

/////////////////////////////////////////////////////
//
//
// STRATEGIES FOR INITIALIZING SHARED MEMORY
//
//
//////////////////////////////////////////////////

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, int tile_width, int tile_height, int BLOCKSZ,
          SCAMPProfileType PROFILE_TYPE>
class InitMemStrategy : public SCAMPStrategy {
 public:
  __device__ void exec(SCAMPKernelInputArgs<double> &args,
                       SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                       PROFILE_DATA_TYPE *__restrict__ profile_a,
                       PROFILE_DATA_TYPE *__restrict__ profile_B,
                       uint32_t col_start, uint32_t row_start) {
    assert(false);
  }

 protected:
  __device__ InitMemStrategy() {}
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, int tile_width, int tile_height, int BLOCKSZ>
class InitMemStrategy<DATA_TYPE, PROFILE_DATA_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                      tile_width, tile_height, BLOCKSZ, PROFILE_TYPE_SUM_THRESH>
    : public SCAMPStrategy {
 public:
  __device__ InitMemStrategy() {}
  __device__ void exec(SCAMPKernelInputArgs<double> &args,
                       SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                       PROFILE_DATA_TYPE *__restrict__ profile_a,
                       PROFILE_DATA_TYPE *__restrict__ profile_B,
                       uint32_t col_start, uint32_t row_start) {
    int global_position = col_start + threadIdx.x;
    int local_position = threadIdx.x;
    while (local_position < tile_width && global_position < args.n_x) {
      smem.dg_col[local_position] = args.dga[global_position];
      smem.df_col[local_position] = args.dfa[global_position];
      smem.inorm_col[local_position] = args.normsa[global_position];
      if (COMPUTE_COLS) {
        smem.local_mp_col[local_position] = 0.0;
      }
      local_position += BLOCKSZ;
      global_position += BLOCKSZ;
    }

    global_position = row_start + threadIdx.x;
    local_position = threadIdx.x;
    while (local_position < tile_height && global_position < args.n_y) {
      smem.dg_row[local_position] = args.dgb[global_position];
      smem.df_row[local_position] = args.dfb[global_position];
      smem.inorm_row[local_position] = args.normsb[global_position];
      if (COMPUTE_ROWS) {
        smem.local_mp_row[local_position] = 0.0;
      }
      local_position += BLOCKSZ;
      global_position += BLOCKSZ;
    }
  }
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, int tile_width, int tile_height, int BLOCKSZ>
class InitMemStrategy<DATA_TYPE, PROFILE_DATA_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                      tile_width, tile_height, BLOCKSZ, PROFILE_TYPE_1NN_INDEX>
    : public SCAMPStrategy {
 public:
  __device__ InitMemStrategy() {}
  __device__ virtual void exec(SCAMPKernelInputArgs<double> &args,
                               SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                               PROFILE_DATA_TYPE *__restrict__ profile_a,
                               PROFILE_DATA_TYPE *__restrict__ profile_b,
                               uint32_t col_start, uint32_t row_start) {
    int global_position = col_start + threadIdx.x;
    int local_position = threadIdx.x;
    while (local_position < tile_width && global_position < args.n_x) {
      smem.dg_col[local_position] = args.dga[global_position];
      smem.df_col[local_position] = args.dfa[global_position];
      smem.inorm_col[local_position] = args.normsa[global_position];
      if (COMPUTE_COLS) {
        smem.local_mp_col[local_position] = profile_a[global_position];
      }
      local_position += BLOCKSZ;
      global_position += BLOCKSZ;
    }

    global_position = row_start + threadIdx.x;
    local_position = threadIdx.x;
    while (local_position < tile_height && global_position < args.n_y) {
      smem.dg_row[local_position] = args.dgb[global_position];
      smem.df_row[local_position] = args.dfb[global_position];
      smem.inorm_row[local_position] = args.normsb[global_position];
      if (COMPUTE_ROWS) {
        smem.local_mp_row[local_position] = profile_b[global_position];
      }
      local_position += BLOCKSZ;
      global_position += BLOCKSZ;
    }
  }
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE>
class DoRowOptStrategy : SCAMPStrategy {
 public:
  __device__ void exec(ACCUM_TYPE &cov1, ACCUM_TYPE &cov2, ACCUM_TYPE &cov3,
                       ACCUM_TYPE &cov4, DISTANCE_TYPE &distc1,
                       DISTANCE_TYPE &distc2, DISTANCE_TYPE &distc3,
                       DISTANCE_TYPE &distc4, const DATA_TYPE &inormcx,
                       const DATA_TYPE &inormcy, const DATA_TYPE &inormcz,
                       const DATA_TYPE &inormcw, const DATA_TYPE &inormr,
                       const DATA_TYPE &df_colx, const DATA_TYPE &df_coly,
                       const DATA_TYPE &df_colz, const DATA_TYPE &df_colw,
                       const DATA_TYPE &dg_colx, const DATA_TYPE &dg_coly,
                       const DATA_TYPE &dg_colz, const DATA_TYPE &dg_colw,
                       const DATA_TYPE &df_row, const DATA_TYPE &dg_row,
                       const int &row, const int &col, const int &global_row,
                       const int &global_col,
                       PROFILE_DATA_TYPE *__restrict__ mp_row,
                       OptionalArgs &args) {
    assert(false);
  }

 protected:
  __device__ DoRowOptStrategy() {}
};

/////////////////////////////////////////////////////
//
//
// STRATEGIES FOR COMPUTING A THREAD-ROW OF THE
// DISTANCE MATRIX (common, optimized case)
//
//
//////////////////////////////////////////////////

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS>
class DoRowOptStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                       COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE_SUM_THRESH>
    : public SCAMPStrategy {
 public:
  __device__ DoRowOptStrategy() {}
  __device__ inline __attribute__((always_inline)) void exec(
      SCAMPThreadInfo<ACCUM_TYPE> &info, DISTANCE_TYPE &distc1,
      DISTANCE_TYPE &distc2, DISTANCE_TYPE &distc3, DISTANCE_TYPE &distc4,
      const DATA_TYPE &inormcx, const DATA_TYPE &inormcy,
      const DATA_TYPE &inormcz, const DATA_TYPE &inormcw,
      const DATA_TYPE &inormr, const DATA_TYPE &df_colx,
      const DATA_TYPE &df_coly, const DATA_TYPE &df_colz,
      const DATA_TYPE &df_colw, const DATA_TYPE &dg_colx,
      const DATA_TYPE &dg_coly, const DATA_TYPE &dg_colz,
      const DATA_TYPE &dg_colw, const DATA_TYPE &df_row,
      const DATA_TYPE &dg_row, SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
      OptionalArgs &args) {
    DISTANCE_TYPE distx = info.cov1 * inormcx * inormr;
    DISTANCE_TYPE disty = info.cov2 * inormcy * inormr;
    DISTANCE_TYPE distz = info.cov3 * inormcz * inormr;
    DISTANCE_TYPE distw = info.cov4 * inormcw * inormr;
    DISTANCE_TYPE thresh = args.threshold;

    // Compute the next covariance values
    info.cov1 = info.cov1 + df_colx * dg_row + dg_colx * df_row;
    info.cov2 = info.cov2 + df_coly * dg_row + dg_coly * df_row;
    info.cov3 = info.cov3 + df_colz * dg_row + dg_colz * df_row;
    info.cov4 = info.cov4 + df_colw * dg_row + dg_colw * df_row;

    DISTANCE_TYPE count_row = 0;

    if (distx > thresh) {
      if (COMPUTE_ROWS) {
        count_row += distx;
      }
      if (COMPUTE_COLS) {
        distc1 += distx;
      }
    }
    if (disty > thresh) {
      if (COMPUTE_ROWS) {
        count_row += disty;
      }
      if (COMPUTE_COLS) {
        distc2 += disty;
      }
    }
    if (distz > thresh) {
      if (COMPUTE_ROWS) {
        count_row += distz;
      }
      if (COMPUTE_COLS) {
        distc3 += distz;
      }
    }
    if (distw > thresh) {
      if (COMPUTE_ROWS) {
        count_row += distw;
      }
      if (COMPUTE_COLS) {
        distc4 += distw;
      }
    }
    // coalesce all row updates to lane 0 of each warp and atomically update
    // This way is more efficient than atomics when we expect a lot of updates
    if (COMPUTE_ROWS) {
#pragma unroll
      for (int i = 16; i >= 1; i /= 2) {
        count_row += __shfl_down_sync(0xffffffff, count_row, i);
      }
      if ((threadIdx.x & 0x1f) == 0) {
        do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(
            smem.local_mp_row + info.local_row, count_row);
      }
    }
    info.local_row++;
    info.local_col++;
  }
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS>
class DoRowOptStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                       COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE_1NN_INDEX> {
 public:
  __device__ DoRowOptStrategy() {}
  __device__ inline __attribute__((always_inline)) void exec(
      SCAMPThreadInfo<ACCUM_TYPE> &info, DISTANCE_TYPE &distc1,
      DISTANCE_TYPE &distc2, DISTANCE_TYPE &distc3, DISTANCE_TYPE &distc4,
      uint32_t &idxc1, uint32_t &idxc2, uint32_t &idxc3, uint32_t &idxc4,
      const DATA_TYPE &inormcx, const DATA_TYPE &inormcy,
      const DATA_TYPE &inormcz, const DATA_TYPE &inormcw,
      const DATA_TYPE &inormr, const DATA_TYPE &df_colx,
      const DATA_TYPE &df_coly, const DATA_TYPE &df_colz,
      const DATA_TYPE &df_colw, const DATA_TYPE &dg_colx,
      const DATA_TYPE &dg_coly, const DATA_TYPE &dg_colz,
      const DATA_TYPE &dg_colw, const DATA_TYPE &df_row,
      const DATA_TYPE &dg_row, float &curr_mp_row_val,
      SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem, OptionalArgs &args) {
    DISTANCE_TYPE distx = info.cov1 * inormcx * inormr;
    DISTANCE_TYPE disty = info.cov2 * inormcy * inormr;
    DISTANCE_TYPE distz = info.cov3 * inormcz * inormr;
    DISTANCE_TYPE distw = info.cov4 * inormcw * inormr;

    // Compute the next covariance values
    info.cov1 = info.cov1 + df_colx * dg_row + dg_colx * df_row;
    info.cov2 = info.cov2 + df_coly * dg_row + dg_coly * df_row;
    info.cov3 = info.cov3 + df_colz * dg_row + dg_colz * df_row;
    info.cov4 = info.cov4 + df_colw * dg_row + dg_colw * df_row;
    // Update the column best-so-far values

    if (COMPUTE_COLS) {
      MPMax2(distc1, distx, idxc1, info.global_row);
      MPMax2(distc2, disty, idxc2, info.global_row);
      MPMax2(distc3, distz, idxc3, info.global_row);
      MPMax2(distc4, distw, idxc4, info.global_row);
    }

    if (COMPUTE_ROWS) {
      // We take the maximum of the columns we computed for the row
      // And use that value to check the matrix profile
      uint32_t idx;
      DISTANCE_TYPE d =
          max4<DISTANCE_TYPE>(distx, disty, distz, distw, info.global_col, idx);
      MPatomicMax_check(smem.local_mp_row + info.local_row, d,
                        idx, curr_mp_row_val);
    }

    info.local_row++;
    info.local_col++;
    info.global_row++;
    info.global_col++;
  }
};

/////////////////////////////////////////////////////
//
//
// STRATEGIES FOR COMPUTING A THREAD-ROW OF THE
// DISTANCE MATRIX (uncommon, edge case)
//
// Slow path (edge tiles)
// Does a single iteration of the inner loop on 4 diagonals per thread, not
// unrolled Checks for the boundary case where only 1, 2, or 3 diagonals can be
// updated
//
//////////////////////////////////////////////////

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE>
class DoRowEdgeStrategy : SCAMPStrategy {
 public:
  __device__ inline void exec(int i, int j, int x, int y, int n,
                              ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
                              ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag,
                              size_t num_diags,
                              SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                              OptionalArgs &args) {
    assert(false);
  }

 protected:
  __device__ DoRowEdgeStrategy() {}
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS>
// SCAMPProfileType PROFILE_TYPE, std::enable_if<PROFILE_TYPE ==
// PROFILE_TYPE_SUM_THRESH || PROFILE_TYPE ==
// PROFILE_TYPE_FREQUENCY_THRESH>::type>
class DoRowEdgeStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                        COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE_SUM_THRESH>
    : SCAMPStrategy {
 public:
  __device__ DoRowEdgeStrategy() {}
  __device__ inline void exec(int i, int j, int x, int y, int n,
                              ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
                              ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag,
                              size_t num_diags,
                              SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                              OptionalArgs &args) {
    DISTANCE_TYPE distr = 0;
    DISTANCE_TYPE distx, disty, distz, distw;
    DISTANCE_TYPE thresh = static_cast<DISTANCE_TYPE>(args.threshold);
    DATA_TYPE inormr = smem.inorm_row[i];
    DATA_TYPE dgr = smem.dg_row[i];
    DATA_TYPE dfr = smem.df_row[i];

    // Compute the next set of distances (row y)
    distx = cov1 * smem.inorm_col[j] * inormr;
    disty = cov2 * smem.inorm_col[j + 1] * inormr;
    distz = cov3 * smem.inorm_col[j + 2] * inormr;
    distw = cov4 * smem.inorm_col[j + 3] * inormr;
    // Update cov and compute the next distance values (row y)
    cov1 = cov1 + smem.df_col[j] * dgr + smem.dg_col[j] * dfr;
    cov2 = cov2 + smem.df_col[j + 1] * dgr + smem.dg_col[j + 1] * dfr;
    cov3 = cov3 + smem.df_col[j + 2] * dgr + smem.dg_col[j + 2] * dfr;
    cov4 = cov4 + smem.df_col[j + 3] * dgr + smem.dg_col[j + 3] * dfr;

    if (distx > thresh) {
      if (COMPUTE_ROWS) {
        distr += distx;
      }
      if (COMPUTE_COLS) {
        do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(smem.local_mp_col + j,
                                                      distx);
      }
    }
    if (x + 1 < n && diag + 1 < num_diags) {
      if (disty > thresh) {
        if (COMPUTE_ROWS) {
          distr += disty;
        }
        if (COMPUTE_COLS) {
          do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(
              smem.local_mp_col + j + 1, disty);
        }
      }
    }
    if (x + 2 < n && diag + 2 < num_diags) {
      if (distz > thresh) {
        if (COMPUTE_ROWS) {
          distr += distz;
        }
        if (COMPUTE_COLS) {
          do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(
              smem.local_mp_col + j + 2, distz);
        }
      }
    }
    if (x + 3 < n && diag + 3 < num_diags) {
      if (distw > thresh) {
        if (COMPUTE_ROWS) {
          distr += distw;
        }
        if (COMPUTE_COLS) {
          do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(
              smem.local_mp_col + j + 3, distw);
        }
      }
    }
    if (COMPUTE_ROWS) {
      do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(smem.local_mp_row + i,
                                                    distr);
    }
  }
};

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS>
// SCAMPProfileType PROFILE_TYPE, std::enable_if<PROFILE_TYPE ==
// PROFILE_TYPE_1NN_SUM, int>::value >
class DoRowEdgeStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                        COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE_1NN_INDEX>
    : SCAMPStrategy {
 public:
  __device__ DoRowEdgeStrategy() {}
  __device__ inline void exec(int i, int j, int x, int y, int n,
                              ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
                              ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag,
                              size_t num_diags,
                              SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                              OptionalArgs &args) {
    float dist_row;
    uint32_t idx_row;
    float distx;
    float disty;
    float distz;
    float distw;

    DATA_TYPE inormr = smem.inorm_row[i];
    DATA_TYPE dgr = smem.dg_row[i];
    DATA_TYPE dfr = smem.df_row[i];

    // Compute the next set of distances (row y)
    distx = cov1 * smem.inorm_col[j] * inormr;
    disty = cov2 * smem.inorm_col[j + 1] * inormr;
    distz = cov3 * smem.inorm_col[j + 2] * inormr;
    distw = cov4 * smem.inorm_col[j + 3] * inormr;

    // Update cov and compute the next distance values (row y)
    cov1 = cov1 + smem.df_col[j] * dgr + smem.dg_col[j] * dfr;
    cov2 = cov2 + smem.df_col[j + 1] * dgr + smem.dg_col[j + 1] * dfr;
    cov3 = cov3 + smem.df_col[j + 2] * dgr + smem.dg_col[j + 2] * dfr;
    cov4 = cov4 + smem.df_col[j + 3] * dgr + smem.dg_col[j + 3] * dfr;

    if (COMPUTE_COLS) {
      MPatomicMax((uint64_t *)(smem.local_mp_col + j), distx, y);
    }
    dist_row = distx;
    idx_row = x;
    if (x + 1 < n && diag + 1 < num_diags) {
      if (COMPUTE_ROWS) {
        MPMax(dist_row, disty, idx_row, x + 1, dist_row, idx_row);
      }
      if (COMPUTE_COLS) {
        MPatomicMax((uint64_t *)(smem.local_mp_col + j + 1), disty, y);
      }
    }
    if (x + 2 < n && diag + 2 < num_diags) {
      if (COMPUTE_ROWS) {
        MPMax(dist_row, distz, idx_row, x + 2, dist_row, idx_row);
      }
      if (COMPUTE_COLS) {
        MPatomicMax((uint64_t *)(smem.local_mp_col + j + 2), distz, y);
      }
    }
    if (x + 3 < n && diag + 3 < num_diags) {
      if (COMPUTE_ROWS) {
        MPMax(dist_row, distw, idx_row, x + 3, dist_row, idx_row);
      }
      if (COMPUTE_COLS) {
        MPatomicMax((uint64_t *)(smem.local_mp_col + j + 3), distw, y);
      }
    }
    if (COMPUTE_ROWS) {
      MPatomicMax((uint64_t *)(smem.local_mp_row + i), dist_row, idx_row);
    }
  }
};

//////////////////////////////////////////////////////////////////////
//
// STRATEGIES FOR UPDATING THE COLUMNS OF THE LOCAL MP VALUES IN
// THE OPTIMIZED CASE
//
//////////////////////////////////////////////////////////////////////

template <typename DISTANCE_TYPE, typename PROFILE_DATA_TYPE,
          SCAMPProfileType PROFILE_TYPE>
class UpdateColumnsStrategy : public SCAMPStrategy {
 public:
  __device__ void exec(DISTANCE_TYPE distc1, DISTANCE_TYPE distc2,
                       DISTANCE_TYPE distc3, DISTANCE_TYPE distc4,
                       DISTANCE_TYPE distc5, DISTANCE_TYPE distc6,
                       DISTANCE_TYPE distc7,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                       uint64_t col) {
    assert(false);
  }

 protected:
  __device__ UpdateColumnsStrategy() {}
};

template <typename DISTANCE_TYPE, typename PROFILE_DATA_TYPE>
class UpdateColumnsStrategy<DISTANCE_TYPE, PROFILE_DATA_TYPE,
                            PROFILE_TYPE_SUM_THRESH> : public SCAMPStrategy {
 public:
  __device__ UpdateColumnsStrategy() {}
  __device__ inline __attribute__((always_inline)) void exec(
      DISTANCE_TYPE distc1, DISTANCE_TYPE distc2, DISTANCE_TYPE distc3,
      DISTANCE_TYPE distc4, DISTANCE_TYPE distc5, DISTANCE_TYPE distc6,
      DISTANCE_TYPE distc7, PROFILE_DATA_TYPE *__restrict__ local_mp_col,
      uint64_t col) {
    int lane = threadIdx.x & 0x1f;
    DISTANCE_TYPE overlap_1, overlap_2, overlap_3;

    // Send the overlapping sums to the next thread
    overlap_1 = __shfl_up_sync(0xffffffff, distc5, 1);
    overlap_2 = __shfl_up_sync(0xffffffff, distc6, 1);
    overlap_3 = __shfl_up_sync(0xffffffff, distc7, 1);
    if (lane > 0) {
      distc1 += overlap_1;
      distc2 += overlap_2;
      distc3 += overlap_3;
    }
    // Update the shared memory sums
    do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(local_mp_col + col, distc1);
    do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(local_mp_col + col + 1,
                                                  distc2);
    do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(local_mp_col + col + 2,
                                                  distc3);
    do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(local_mp_col + col + 3,
                                                  distc4);
    // The last thread in the warp has to make additional updates to shared
    // memory as it had nowhere to send its overlapping sums
    if (lane == 31) {
      do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(local_mp_col + col + 4,
                                                    distc5);
      do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(local_mp_col + col + 5,
                                                    distc6);
      do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_BLOCK>(local_mp_col + col + 6,
                                                    distc7);
    }
  }
};

template <typename DISTANCE_TYPE, typename PROFILE_DATA_TYPE>
class UpdateColumnsStrategy<DISTANCE_TYPE, PROFILE_DATA_TYPE,
                            PROFILE_TYPE_1NN_INDEX> : public SCAMPStrategy {
 public:
  __device__ UpdateColumnsStrategy() {}
  __device__ void exec(DISTANCE_TYPE distc1, DISTANCE_TYPE distc2,
                       DISTANCE_TYPE distc3, DISTANCE_TYPE distc4,
                       DISTANCE_TYPE distc5, DISTANCE_TYPE distc6,
                       DISTANCE_TYPE distc7,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                       uint64_t col) {
    assert(false);
  }
};

///////////////////////////////////////////////////////////////////
//
// STRATEGIES FOR WRITING BACK THE LOCAL MATRIX PROFILE TO MEMORY
//
///////////////////////////////////////////////////////////////////

// Dummy (forces compilation failure when the wrong types are used)
template <typename PROFILE_DATA_TYPE, bool COMPUTE_COLS, bool COMPUTE_ROWS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ,
          SCAMPProfileType PROFILE_TYPE>
class WriteBackStrategy : public SCAMPStrategy {
 public:
  __device__ void exec(uint32_t tile_start_x, uint32_t tile_start_y,
                       uint32_t n_x, uint32_t n_y,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_row,
                       PROFILE_DATA_TYPE *__restrict__ profile_A,
                       PROFILE_DATA_TYPE *__restrict__ profile_B) {
    assert(false);
  }

 protected:
  __device__ WriteBackStrategy() {}
};

template <typename PROFILE_DATA_TYPE, bool COMPUTE_COLS, bool COMPUTE_ROWS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ>
class WriteBackStrategy<PROFILE_DATA_TYPE, COMPUTE_COLS, COMPUTE_ROWS,
                        TILE_WIDTH, TILE_HEIGHT, BLOCKSZ,
                        PROFILE_TYPE_SUM_THRESH> : public SCAMPStrategy {
 public:
  __device__ WriteBackStrategy() {}
  __device__ void exec(uint32_t tile_start_x, uint32_t tile_start_y,
                       uint32_t n_x, uint32_t n_y,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_row,
                       PROFILE_DATA_TYPE *__restrict__ profile_A,
                       PROFILE_DATA_TYPE *__restrict__ profile_B) {
    int global_position, local_position;
    if (COMPUTE_COLS) {
      global_position = tile_start_x + threadIdx.x;
      local_position = threadIdx.x;
      while (local_position < TILE_WIDTH && global_position < n_x) {
        do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_GLOBAL>(
            profile_A + global_position, local_mp_col[local_position]);
        global_position += BLOCKSZ;
        local_position += BLOCKSZ;
      }
    }
    if (COMPUTE_ROWS) {
      global_position = tile_start_y + threadIdx.x;
      local_position = threadIdx.x;
      while (local_position < TILE_HEIGHT && global_position < n_y) {
        do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_GLOBAL>(
            profile_B + global_position, local_mp_row[local_position]);
        global_position += BLOCKSZ;
        local_position += BLOCKSZ;
      }
    }
  }
};

template <typename PROFILE_DATA_TYPE, bool COMPUTE_COLS, bool COMPUTE_ROWS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ>
class WriteBackStrategy<PROFILE_DATA_TYPE, COMPUTE_COLS, COMPUTE_ROWS,
                        TILE_WIDTH, TILE_HEIGHT, BLOCKSZ,
                        PROFILE_TYPE_1NN_INDEX> : public SCAMPStrategy {
 public:
  __device__ WriteBackStrategy() {}
  __device__ void exec(uint32_t tile_start_x, uint32_t tile_start_y,
                       uint32_t n_x, uint32_t n_y,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                       PROFILE_DATA_TYPE *__restrict__ local_mp_row,
                       PROFILE_DATA_TYPE *__restrict__ profile_A,
                       PROFILE_DATA_TYPE *__restrict__ profile_B) {
    int global_position, local_position;
    if (COMPUTE_COLS) {
      global_position = tile_start_x + threadIdx.x;
      local_position = threadIdx.x;
      while (local_position < TILE_WIDTH && global_position < n_x) {
        mp_entry e;
        e.ulong = local_mp_col[local_position];
        MPatomicMax((uint64_t *)(profile_A + global_position), e.floats[0], e.ints[1]);
        global_position += BLOCKSZ;
        local_position += BLOCKSZ;
      }
    }
    if (COMPUTE_ROWS) {
      global_position = tile_start_y + threadIdx.x;
      local_position = threadIdx.x;
      while (local_position < TILE_HEIGHT && global_position < n_y) {
        mp_entry e;
        e.ulong = local_mp_row[local_position];
        MPatomicMax((uint64_t *)(profile_B + global_position), e.floats[0], e.ints[1]);
        global_position += BLOCKSZ;
        local_position += BLOCKSZ;
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// OPTIMIZED CODE PATH:
// do_unrolled_row4 is the optimized matrix profile code path which computes
// one row of work for a single thread. It is specialized for each profile type
// that is computed.
// do_iteration_unroll_4 computes a 4x4 block of the distance matrix by calling
// do_unrolled_row4 four separate times.
// We are computing a tile that looks like this:
// C:1 2 3 4 5 6 7
// R1 X X X X
// R2   X X X X
// R3     X X X X
// R4       X X X X
// For 4 diagonals unrolled 4 times we compute a total of 16 distances.
// These distances cover 4 possible rows and 7 possible columns.
///////////////////////////////////////////////////////////////////////////////
// Processes 4 iterations of the inner loop. Each thread computes 4 distances
// per iteration (x,y), (x+1,y), (x+2,y), and (x+3,y) This function assumes that
// the edge cases that occur on the edge of the distance matrix are not present.
// This is the faster path, with less conditional branching.
template <typename DATA_TYPE, typename VEC2_DATA_TYPE, typename VEC4_DATA_TYPE,
          typename ACCUM_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE>
class DoIterationStrategy : public SCAMPStrategy {
 public:
  __device__ inline void exec(SCAMPThreadInfo<ACCUM_TYPE> &info,
                              SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                              OptionalArgs &args) {
    assert(false);
  }

 protected:
  __device__ DoIterationStrategy() {}
};

template <typename DATA_TYPE, typename VEC2_DATA_TYPE, typename VEC4_DATA_TYPE,
          typename ACCUM_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS>
class DoIterationStrategy<DATA_TYPE, VEC2_DATA_TYPE, VEC4_DATA_TYPE, ACCUM_TYPE,
                          PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS,
                          COMPUTE_COLS, PROFILE_TYPE_SUM_THRESH>
    : public SCAMPStrategy {
 public:
  __device__ DoIterationStrategy() {}
  __device__ void exec(SCAMPThreadInfo<ACCUM_TYPE> &info,
                       SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                       OptionalArgs &args) {
    DISTANCE_TYPE distc1 = 0;
    DISTANCE_TYPE distc2 = 0;
    DISTANCE_TYPE distc3 = 0;
    DISTANCE_TYPE distc4 = 0;
    DISTANCE_TYPE distc5 = 0;
    DISTANCE_TYPE distc6 = 0;
    DISTANCE_TYPE distc7 = 0;

    // Load row values 2 at a time, load column values 4 at a time
    int r = info.local_row >> 1;
    int c = info.local_col >> 2;

    // Preload the shared memory values we will use into registers
    // We load 4 values per thread into a double4 vector type
    VEC4_DATA_TYPE dfc = reinterpret_cast<VEC4_DATA_TYPE *>(smem.df_col)[c];
    VEC4_DATA_TYPE dgc = reinterpret_cast<VEC4_DATA_TYPE *>(smem.dg_col)[c];
    VEC4_DATA_TYPE inormc =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.inorm_col)[c];
    VEC4_DATA_TYPE dfc2 =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.df_col)[c + 1];
    VEC4_DATA_TYPE dgc2 =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.dg_col)[c + 1];
    VEC4_DATA_TYPE inormc2 =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.inorm_col)[c + 1];

    // Due to a lack of registers on volta, we only load these row values 2 at a
    // time
    VEC2_DATA_TYPE dgr = reinterpret_cast<VEC2_DATA_TYPE *>(smem.dg_row)[r];
    VEC2_DATA_TYPE dfr = reinterpret_cast<VEC2_DATA_TYPE *>(smem.df_row)[r];
    VEC2_DATA_TYPE inormr =
        reinterpret_cast<VEC2_DATA_TYPE *>(smem.inorm_row)[r];

    // Do rows one at a time:
    _do_row.exec(info, distc1, distc2, distc3, distc4, inormc.x, inormc.y,
                 inormc.z, inormc.w, inormr.x, dfc.x, dfc.y, dfc.z, dfc.w,
                 dgc.x, dgc.y, dgc.z, dgc.w, dfr.x, dgr.x, smem, args);

    _do_row.exec(info, distc2, distc3, distc4, distc5, inormc.y, inormc.z,
                 inormc.w, inormc2.x, inormr.y, dfc.y, dfc.z, dfc.w, dfc2.x,
                 dgc.y, dgc.z, dgc.w, dgc2.x, dfr.y, dgr.y, smem, args);

    // Load the values for the next 2 rows
    dgr = reinterpret_cast<VEC2_DATA_TYPE *>(smem.dg_row)[r + 1];
    dfr = reinterpret_cast<VEC2_DATA_TYPE *>(smem.df_row)[r + 1];
    inormr = reinterpret_cast<VEC2_DATA_TYPE *>(smem.inorm_row)[r + 1];

    _do_row.exec(info, distc3, distc4, distc5, distc6, inormc.z, inormc.w,
                 inormc2.x, inormc2.y, inormr.x, dfc.z, dfc.w, dfc2.x, dfc2.y,
                 dgc.z, dgc.w, dgc2.x, dgc2.y, dfr.x, dgr.x, smem, args);

    _do_row.exec(info, distc4, distc5, distc6, distc7, inormc.w, inormc2.x,
                 inormc2.y, inormc2.z, inormr.y, dfc.w, dfc2.x, dfc2.y, dfc2.z,
                 dgc.w, dgc2.x, dgc2.y, dgc2.z, dfr.y, dgr.y, smem, args);

    if (COMPUTE_COLS) {
      _update_cols.exec(distc1, distc2, distc3, distc4, distc5, distc6, distc7,
                        smem.local_mp_col, info.local_col - DIAGS_PER_THREAD);
    }
    info.global_col += DIAGS_PER_THREAD;
    info.global_row += DIAGS_PER_THREAD;
  }

 private:
  DoRowOptStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                   COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE_SUM_THRESH>
      _do_row;
  UpdateColumnsStrategy<DISTANCE_TYPE, PROFILE_DATA_TYPE,
                        PROFILE_TYPE_SUM_THRESH>
      _update_cols;
};

template <typename DATA_TYPE, typename VEC2_DATA_TYPE, typename VEC4_DATA_TYPE,
          typename ACCUM_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS>
class DoIterationStrategy<DATA_TYPE, VEC2_DATA_TYPE, VEC4_DATA_TYPE, ACCUM_TYPE,
                          PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS,
                          COMPUTE_COLS, PROFILE_TYPE_1NN_INDEX>
    : public SCAMPStrategy {
 public:
  __device__ DoIterationStrategy() {}
  __device__ void exec(SCAMPThreadInfo<ACCUM_TYPE> &info,
                       SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                       OptionalArgs &args) {
    float4 distc = make_float4(CC_MIN, CC_MIN, CC_MIN, CC_MIN);
    float4 distc2 = make_float4(CC_MIN, CC_MIN, CC_MIN, CC_MIN);
    uint4 idxc, idxc2;

    // Load row values 2 at a time, load column values 4 at a time
    int r = info.local_row >> 2;
    int c = info.local_col >> 2;

    // Preload the shared memory values we will use into registers
    // We load 4 values per thread into a float4 vector type
    VEC4_DATA_TYPE dfc = reinterpret_cast<VEC4_DATA_TYPE *>(smem.df_col)[c];
    VEC4_DATA_TYPE dgc = reinterpret_cast<VEC4_DATA_TYPE *>(smem.dg_col)[c];
    VEC4_DATA_TYPE inormc =
        (reinterpret_cast<VEC4_DATA_TYPE *>(smem.inorm_col)[c]);
    VEC4_DATA_TYPE dfc2 =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.df_col)[c + 1];
    VEC4_DATA_TYPE dgc2 =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.dg_col)[c + 1];
    VEC4_DATA_TYPE inormc2 =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.inorm_col)[c + 1];
    ulonglong4 mp_row_check;

    // Copy the pieces of the cache we will use into registers with vectorized
    // loads
    if (COMPUTE_ROWS) {
      mp_row_check = reinterpret_cast<ulonglong4 *>(smem.local_mp_row)[r];
    }

    // Due to a lack of registers on volta, we only load these row values 2 at a
    // time
    VEC4_DATA_TYPE dgr = reinterpret_cast<VEC4_DATA_TYPE *>(smem.dg_row)[r];
    VEC4_DATA_TYPE dfr = reinterpret_cast<VEC4_DATA_TYPE *>(smem.df_row)[r];
    VEC4_DATA_TYPE inormr =
        reinterpret_cast<VEC4_DATA_TYPE *>(smem.inorm_row)[r];

    mp_entry e;
    e.ulong = mp_row_check.x;
    // Do rows one at a time:
    _do_row.exec(info, distc.x, distc.y, distc.z, distc.w, idxc.x, idxc.y,
                 idxc.z, idxc.w, inormc.x, inormc.y, inormc.z, inormc.w,
                 inormr.x, dfc.x, dfc.y, dfc.z, dfc.w, dgc.x, dgc.y, dgc.z,
                 dgc.w, dfr.x, dgr.x, e.floats[0], smem, args);

    e.ulong = mp_row_check.y;
    _do_row.exec(info, distc.y, distc.z, distc.w, distc2.x, idxc.y, idxc.z,
                 idxc.w, idxc2.x, inormc.y, inormc.z, inormc.w, inormc2.x,
                 inormr.y, dfc.y, dfc.z, dfc.w, dfc2.x, dgc.y, dgc.z, dgc.w,
                 dgc2.x, dfr.y, dgr.y, e.floats[0], smem, args);

    e.ulong = mp_row_check.z;
    _do_row.exec(info, distc.z, distc.w, distc2.x, distc2.y, idxc.z, idxc.w,
                 idxc2.x, idxc2.y, inormc.z, inormc.w, inormc2.x, inormc2.y,
                 inormr.z, dfc.z, dfc.w, dfc2.x, dfc2.y, dgc.z, dgc.w, dgc2.x,
                 dgc2.y, dfr.z, dgr.z, e.floats[0], smem, args);

    e.ulong = mp_row_check.w;
    _do_row.exec(info, distc.w, distc2.x, distc2.y, distc2.z, idxc.w, idxc2.x,
                 idxc2.y, idxc2.z, inormc.w, inormc2.x, inormc2.y, inormc2.z,
                 inormr.w, dfc.w, dfc2.x, dfc2.y, dfc2.z, dgc.w, dgc2.x, dgc2.y,
                 dgc2.z, dfr.w, dgr.w, e.floats[0], smem, args);

    // After the 4th row, we have completed columns 4, 5, 6, and 7
    if (COMPUTE_COLS) {
      ulonglong4 mp_col_check1, mp_col_check2;
      mp_col_check1 = reinterpret_cast<ulonglong4 *>(smem.local_mp_col)[c];
      mp_col_check2 = reinterpret_cast<ulonglong4 *>(smem.local_mp_col)[c + 1];
      e.ulong = mp_col_check1.x;
      MPatomicMax_check(smem.local_mp_col + info.local_col - 4, distc.x, idxc.x,
                        e.floats[0]);
      e.ulong = mp_col_check1.y;
      MPatomicMax_check(smem.local_mp_col + info.local_col - 3, distc.y, idxc.y,
                        e.floats[0]);
      e.ulong = mp_col_check1.z;
      MPatomicMax_check(smem.local_mp_col + info.local_col - 2, distc.z, idxc.z,
                        e.floats[0]);
      e.ulong = mp_col_check1.w;
      MPatomicMax_check(smem.local_mp_col + info.local_col - 1, distc.w, idxc.w,
                        e.floats[0]);
      e.ulong = mp_col_check2.x;
      MPatomicMax_check(smem.local_mp_col + info.local_col, distc2.x, idxc2.x,
                        e.floats[0]);
      e.ulong = mp_col_check2.y;
      MPatomicMax_check(smem.local_mp_col + info.local_col + 1, distc2.y,
                        idxc2.y, e.floats[0]);
      e.ulong = mp_col_check2.z;
      MPatomicMax_check(smem.local_mp_col + info.local_col + 2, distc2.z,
                        idxc2.z, e.floats[0]);
    }
  }

 private:
  DoRowOptStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, float,
                   COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE_1NN_INDEX>
      _do_row;
  // UpdateColumnsStrategy<DISTANCE_TYPE, PROFILE_DATA_TYPE,
  // PROFILE_TYPE_1NN_INDEX>
  //    _update_cols;
};

///////////////////////////////////////
// Slow path (edge tiles)
// Does a single iteration of the inner loop on 4 diagonals per thread, not
// unrolled Checks for the boundary case where only 1, 2, or 3 diagonals can be
// updated
//////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////
//
//  SCAMP TACTIC DESCRIBES STRATEGY FOR WHAT OPS TO EXECUTE IN THE KERNEL
//
/////////////////////////////////////////////////////////////////////////////////////

template <typename DATA_TYPE, typename VEC2_DATA_TYPE, typename VEC4_DATA_TYPE,
          typename PROFILE_DATA_TYPE, typename ACCUM_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ,
          SCAMPProfileType PROFILE_TYPE>
class SCAMPTactic {
 public:
  __device__ SCAMPTactic() {}
  __device__ void InitMem(SCAMPKernelInputArgs<double> &args,
                          SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                          PROFILE_DATA_TYPE *__restrict__ profile_a,
                          PROFILE_DATA_TYPE *__restrict__ profile_b,
                          uint32_t col_start, uint32_t row_start) {
    _init_mem.exec(args, smem, profile_a, profile_b, col_start, row_start);
  }
  __device__ inline __attribute__((always_inline)) void DoIteration(
      SCAMPThreadInfo<ACCUM_TYPE> &info,
      SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem, OptionalArgs &args) {
    _do_iteration.exec(info, smem, args);
  }
  __device__ inline void DoEdge(int i, int j, int x, int y, int n,
                                ACCUM_TYPE &cov1, ACCUM_TYPE &cov2,
                                ACCUM_TYPE &cov3, ACCUM_TYPE &cov4, size_t diag,
                                size_t num_diags,
                                SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> &smem,
                                OptionalArgs &args) {
    _do_edge.exec(i, j, x, y, n, cov1, cov2, cov3, cov4, diag, num_diags, smem,
                  args);
  }
  __device__ inline void WriteBack(uint32_t tile_start_x, uint32_t tile_start_y,
                                   uint32_t n_x, uint32_t n_y,
                                   PROFILE_DATA_TYPE *__restrict__ local_mp_col,
                                   PROFILE_DATA_TYPE *__restrict__ local_mp_row,
                                   PROFILE_DATA_TYPE *__restrict__ profile_A,
                                   PROFILE_DATA_TYPE *__restrict__ profile_B) {
    _do_writeback.exec(tile_start_x, tile_start_y, n_x, n_y, local_mp_col,
                       local_mp_row, profile_A, profile_B);
  }

 private:
  InitMemStrategy<DATA_TYPE, PROFILE_DATA_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                  TILE_WIDTH, TILE_HEIGHT, BLOCKSZ, PROFILE_TYPE>
      _init_mem;
  DoIterationStrategy<DATA_TYPE, VEC2_DATA_TYPE, VEC4_DATA_TYPE, ACCUM_TYPE,
                      PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS,
                      COMPUTE_COLS, PROFILE_TYPE>
      _do_iteration;
  DoRowEdgeStrategy<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                    COMPUTE_ROWS, COMPUTE_COLS, PROFILE_TYPE>
      _do_edge;
  WriteBackStrategy<PROFILE_DATA_TYPE, COMPUTE_COLS, COMPUTE_ROWS, TILE_WIDTH,
                    TILE_HEIGHT, BLOCKSZ, PROFILE_TYPE>
      _do_writeback;
};

// // Computes the matrix profile given the sliding dot products for the first
// // query and the precomputed data statisics
template <typename DATA_TYPE, typename VEC2_DATA_TYPE, typename VEC4_DATA_TYPE,
          typename ACCUM_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          SCAMPProfileType PROFILE_TYPE, int blocks_per_sm, int tile_height,
          int BLOCKSZ>
__global__ void __launch_bounds__(BLOCKSZ, blocks_per_sm)
    do_tile(SCAMPKernelInputArgs<double> args,
            PROFILE_DATA_TYPE *__restrict__ profile_A,
            PROFILE_DATA_TYPE *__restrict__ profile_B) {
  constexpr int diags_per_thread = 4;
  constexpr int tile_width = tile_height + BLOCKSZ * diags_per_thread;
  SCAMPTactic<DATA_TYPE, VEC2_DATA_TYPE, VEC4_DATA_TYPE, PROFILE_DATA_TYPE,
              ACCUM_TYPE, DISTANCE_TYPE, COMPUTE_ROWS, COMPUTE_COLS, tile_width,
              tile_height, BLOCKSZ, PROFILE_TYPE>
      tactic;
  SCAMPThreadInfo<ACCUM_TYPE> thread_info;

  extern __shared__ char smem_raw[];
  SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE> smem(
      smem_raw, COMPUTE_ROWS, COMPUTE_COLS, tile_width, tile_height);

  const unsigned int start_diag = (threadIdx.x * diags_per_thread) +
                                  blockIdx.x * (blockDim.x * diags_per_thread);

  // This is the index of the meta-diagonal that this thread block will work on
  const unsigned int meta_diagonal_idx = blockIdx.x;

  // The first threads are acutally computing the trivial match between the same
  // subsequence we exclude these from the calculation
  uint32_t tile_start_col =
      meta_diagonal_idx * (BLOCKSZ * diags_per_thread) + args.exclusion_lower;
  uint32_t tile_start_row = 0;

  // x is the global column of the distance matrix
  // y is the global row of the distance matrix
  // localX, localY are the local coordinates of the thread position in the tile
  // it is working on
  thread_info.global_col = tile_start_col + threadIdx.x * diags_per_thread;
  thread_info.global_row = 0;

  const unsigned int num_diags = args.n_x - args.exclusion_upper;

  // Load the first dot product values
  if (thread_info.global_col < args.n_x) {
    thread_info.cov1 = args.cov[thread_info.global_col];
  }

  if (thread_info.global_col + 1 < args.n_x) {
    thread_info.cov2 = args.cov[thread_info.global_col + 1];
  }

  if (thread_info.global_col + 2 < args.n_x) {
    thread_info.cov3 = args.cov[thread_info.global_col + 2];
  }

  if (thread_info.global_col + 3 < args.n_x) {
    thread_info.cov4 = args.cov[thread_info.global_col + 3];
  }

  /////////////////////////////////////
  // Main loop
  /////////////////////////////////////
  // Each threadblock finds all the distances on a 'metadiagonal'
  // We use a tiled approach for each thread block
  // The tiles are horizontal slices of the diagonal, think of a parallelogram
  // cut from a diagonal slice of the distance matrix Each thread starts on the
  // first row and works its way down-right towards right side of the distance
  // matrix
  while (tile_start_col < args.n_x && tile_start_row < args.n_y) {
    // Initialize the next tile's shared memory
    tactic.InitMem(args, smem, profile_A, profile_B, tile_start_col,
                   tile_start_row);
    thread_info.local_col = threadIdx.x * diags_per_thread;
    thread_info.local_row = 0;
    // Start of new tile, sync
    __syncthreads();

    // There are 2 pathways here, most of the time we take the fast path (top),
    // the last block (edge_tile) will take the slower path (bottom)
    if (tile_start_col + tile_width < args.n_x &&
        tile_start_row + tile_height < args.n_y &&
        start_diag + diags_per_thread - 1 < num_diags) {
      while (thread_info.local_row < tile_height) {
        tactic.DoIteration(thread_info, smem, args.opt);
      }
      //      thread_info.global_row += tile_height;
      //      thread_info.global_col += tile_height;

    } else if (start_diag < num_diags) {
      while (thread_info.global_col < args.n_x &&
             thread_info.global_row < args.n_y &&
             thread_info.local_row < tile_height) {
        tactic.DoEdge(thread_info.local_row, thread_info.local_col,
                      thread_info.global_col, thread_info.global_row, args.n_x,
                      thread_info.cov1, thread_info.cov2, thread_info.cov3,
                      thread_info.cov4, start_diag, num_diags, smem, args.opt);

        ++thread_info.global_col;
        ++thread_info.global_row;
        ++thread_info.local_col;
        ++thread_info.local_row;
      }
    }

    // After this sync, the caches will be updated with the best so far values
    // for this tile
    __syncthreads();

    tactic.WriteBack(tile_start_col, tile_start_row, args.n_x, args.n_y,
                     smem.local_mp_col, smem.local_mp_row, profile_A,
                     profile_B);

    // Update the tile position
    tile_start_col += tile_height;
    tile_start_row += tile_height;

    // Make sure our updates were committed before we pull in the next tile
    __threadfence_block();
  }
}

extern "C" {

  struct do_Tile_DP_HALADAPT_parameters {
    bool COMPUTE_ROWS;
    bool COMPUTE_COLS;
    int PROFILE_TYPE;
    uint64_t blocksz;
    uint64_t num_blocks;
    uint64_t smem;
    uint32_t n_x;
    uint32_t n_y;
    int32_t exclusion_lower;
    int32_t exclusion_upper;
    double threshold;
    size_t size_cov;
    size_t size_dfa;
    size_t size_dfb;
    size_t size_dga;
    size_t size_dgb;
    size_t size_normsa;
    size_t size_normsb;
  };

  do_Tile_DP_HALADAPT_parameters globalparameters;

  dls_decdef(do_Tile_DP_HALADAPT,void, double *profile_A_D, double *profile_B_D, uint64_t *profile_A_U, uint64_t *profile_B_U, double *input);

  void do_Tile_DP_HALADAPT_GPU_CUDA(
    double *profile_A_D,
    double *profile_B_D,
    uint64_t *profile_A_U,
    uint64_t *profile_B_U,
    double *input
  ) {
    // dls_get_addr((void **)&parameters, "r");
    dls_get_addr((void **)&profile_A_D, "b");
    dls_get_addr((void **)&profile_B_D, "b");
    dls_get_addr((void **)&profile_A_U, "b");
    dls_get_addr((void **)&profile_B_U, "b");
    dls_get_addr((void **)&input, "r");

    double *cov = input;
    double *dfa = &input[globalparameters.size_cov/sizeof(double)];
    double *dfb = &input[(globalparameters.size_cov + globalparameters.size_dfa)/sizeof(double)];
    double *dga = &input[(globalparameters.size_cov + globalparameters.size_dfa + globalparameters.size_dfb)/sizeof(double)];
    double *dgb = &input[(globalparameters.size_cov + globalparameters.size_dfa + globalparameters.size_dfb + globalparameters.size_dga)/sizeof(double)];
    double *normsa = &input[(globalparameters.size_cov + globalparameters.size_dfa + globalparameters.size_dfb + globalparameters.size_dga + globalparameters.size_dgb)/sizeof(double)];
    double *normsb = &input[(globalparameters.size_cov + globalparameters.size_dfa + globalparameters.size_dfb + globalparameters.size_dga + globalparameters.size_dgb + globalparameters.size_normsa)/sizeof(double)];



    SCAMPKernelInputArgs<double> args(cov, dfa, dfb, dga, dgb, normsa, normsb, 
                                                      globalparameters.n_x, globalparameters.n_y, globalparameters.exclusion_lower, 
                                                      globalparameters.exclusion_upper, OptionalArgs(globalparameters.threshold),
                                                      globalparameters.size_cov, globalparameters.size_dfa, globalparameters.size_dfb,
                                                      globalparameters.size_dga, globalparameters.size_dgb, globalparameters.size_normsa,
                                                      globalparameters.size_normsb);

    bool COMPUTE_ROWS = globalparameters.COMPUTE_ROWS;
    bool COMPUTE_COLS = globalparameters.COMPUTE_COLS;
    int PROFILE_TYPE = globalparameters.PROFILE_TYPE;
    uint64_t blocksz = globalparameters.blocksz;
    uint64_t num_blocks = globalparameters.num_blocks;
    uint64_t smem = globalparameters.smem;

    dim3 block(blocksz, 1, 1);
    dim3 grid(num_blocks, 1, 1);

    if(COMPUTE_ROWS && COMPUTE_COLS && PROFILE_TYPE  == PROFILE_TYPE_SUM_THRESH) {
      do_tile<double, double2, double4, double, double, double,
            true, true, PROFILE_TYPE_SUM_THRESH, 2,
            200, 256>
        <<<grid, block, smem, 0>>>(args, profile_A_D, profile_B_D);
    }
    
    if(COMPUTE_ROWS && COMPUTE_COLS && PROFILE_TYPE  == PROFILE_TYPE_1NN_INDEX) {
      do_tile<double, double2, double4, double, uint64_t, double,
              true, true, PROFILE_TYPE_1NN_INDEX, 2,
              200, 256>
          <<<grid, block, smem, 0>>>(args, (uint64_t *) profile_A_U, (uint64_t *) profile_B_U);
    }

    if(!COMPUTE_ROWS && COMPUTE_COLS && PROFILE_TYPE  == PROFILE_TYPE_SUM_THRESH) {
      do_tile<double, double2, double4, double, double, double,
            false, true, PROFILE_TYPE_SUM_THRESH, 2,
            200, 256>
        <<<grid, block, smem, 0>>>(args, (double *) profile_A_D, (double *) profile_B_D);
    }
    
    if(!COMPUTE_ROWS && COMPUTE_COLS && PROFILE_TYPE  == PROFILE_TYPE_1NN_INDEX) {
      do_tile<double, double2, double4, double, uint64_t, double,
              false, true, PROFILE_TYPE_1NN_INDEX, 2,
              200, 256>
          <<<grid, block, smem, 0>>>(args, (uint64_t *) profile_A_U, (uint64_t *) profile_B_U);
    }

    if(COMPUTE_ROWS && !COMPUTE_COLS && PROFILE_TYPE  == PROFILE_TYPE_SUM_THRESH) {
      do_tile<double, double2, double4, double, double, double,
            true, false, PROFILE_TYPE_SUM_THRESH, 2,
            200, 256>
        <<<grid, block, smem, 0>>>(args, (double *) profile_A_D, (double *) profile_B_D);
    }
    
    if(COMPUTE_ROWS && !COMPUTE_COLS && PROFILE_TYPE  == PROFILE_TYPE_1NN_INDEX) {
      do_tile<double, double2, double4, double, uint64_t, double,
              true, false, PROFILE_TYPE_1NN_INDEX, 2,
              200, 256>
          <<<grid, block, smem, 0>>>(args, (uint64_t *) profile_A_U, (uint64_t *) profile_B_U);
    }
  }
}

int get_diags_per_thread(bool fp64, const cudaDeviceProp &dev_prop) {
  return 4;
}

int get_blocksz(SCAMPPrecisionType t, const cudaDeviceProp &dev_prop) {
  if (t == PRECISION_DOUBLE) {
    return BLOCKSZ_DP;
  } else {
    return BLOCKSZ_SP;
  }
}

int get_exclusion(uint64_t window_size, uint64_t start_row,
                  uint64_t start_column) {
  int exclusion = window_size;
  if (start_column >= start_row && start_column <= start_row + exclusion) {
    return exclusion;
  }
  return 0;
}

std::pair<int, int> get_exclusion_for_ab_join(uint64_t window_size,
                                              uint64_t start_row,
                                              uint64_t start_column,
                                              bool upper_tile, int tile_dim) {
  int exclusion_lower = 0;
  int exclusion_upper = 0;
  if (upper_tile) {
    exclusion_lower = get_exclusion(window_size, start_row, start_column);
    if (start_row > start_column) {
      exclusion_upper =
          get_exclusion(window_size, start_row, start_column + tile_dim);
    } else {
      exclusion_upper = 0;
    }
    return std::make_pair(exclusion_lower, exclusion_upper);
  }
  exclusion_lower = get_exclusion(window_size, start_column, start_row);
  if (start_row >= start_column) {
    exclusion_upper = 0;
  } else {
    exclusion_upper =
        get_exclusion(window_size, start_column, start_row + tile_dim);
  }
  return std::make_pair(exclusion_lower, exclusion_upper);
}

constexpr int FPTypeSize(SCAMPPrecisionType dtype) {
  switch (dtype) {
    case PRECISION_DOUBLE:
      return sizeof(double);
    case PRECISION_MIXED:
    case PRECISION_SINGLE:
      return sizeof(float);
    case PRECISION_INVALID:
      return -1;
  }
  return -1;
}

int GetTileHeight(SCAMPPrecisionType dtype) {
  switch (dtype) {
    case PRECISION_DOUBLE:
      return TILE_HEIGHT_DP;
    case PRECISION_MIXED:
    case PRECISION_SINGLE:
      return TILE_HEIGHT_SP;
    case PRECISION_INVALID:
      return -1;
  }
  return -1;
}

int get_smem(bool computing_rows, bool computing_cols, int blocksz,
             SCAMPPrecisionType intermediate_data_type, int profile_data_size) {
  constexpr int diags_per_thread = 4;
  constexpr int num_shared_variables = 3;
  int intermediate_data_size = FPTypeSize(intermediate_data_type);
  int tile_height = GetTileHeight(intermediate_data_type);
  int tile_width = blocksz * diags_per_thread + tile_height;
  int smem = (tile_width + tile_height) * num_shared_variables *
             intermediate_data_size;
  if (computing_cols) {
    smem += tile_width * profile_data_size;
  }
  if (computing_rows) {
    smem += tile_height * profile_data_size;
  }
  return smem;
}

void copySCAMPKernelInputArgsToOneArray(SCAMPKernelInputArgs<double> args, double *input) {
  memcpy(input, args.cov, args.size_cov);
  memcpy(&input[args.size_cov/sizeof(double)], args.dfa, args.size_dfa);
  memcpy(&input[(args.size_cov + args.size_dfa)/sizeof(double)], args.dfb, args.size_dfb);
  memcpy(&input[(args.size_cov + args.size_dfa + args.size_dfb)/sizeof(double)], args.dga, args.size_dga);
  memcpy(&input[(args.size_cov + args.size_dfa + args.size_dfb + args.size_dga)/sizeof(double)], args.dgb, args.size_dgb);
  memcpy(&input[(args.size_cov + args.size_dfa + args.size_dfb + args.size_dga + args.size_dgb)/sizeof(double)], args.normsa, args.size_normsa);
  memcpy(&input[(args.size_cov + args.size_dfa + args.size_dfb + args.size_dga + args.size_dgb + args.size_normsa)/sizeof(double)], args.normsb, args.size_normsb);
}

template <typename PROFILE_DATA_TYPE, SCAMPProfileType PROFILE_TYPE,
          int BLOCKSPERSM>
SCAMPError_t LaunchDoTile(SCAMPKernelInputArgs<double> args,
                          PROFILE_DATA_TYPE *profile_A,
                          PROFILE_DATA_TYPE *profile_B,
                          SCAMPPrecisionType fp_type, bool computing_rows,
                          bool computing_cols, uint64_t blocksz,
                          uint64_t num_blocks, uint64_t smem, 
                          SCAMPProfileType profile_type,
                          size_t size_profile_A, size_t size_profile_B) {
  

  double *input = (double *) malloc(args.size_cov + args.size_dfa + args.size_dfb + args.size_dga + args.size_dgb + args.size_normsa + args.size_normsb);
  copySCAMPKernelInputArgsToOneArray(args, input);

  do_Tile_DP_HALADAPT_parameters parameters;
  parameters.PROFILE_TYPE = profile_type;
  parameters.blocksz = blocksz;
  parameters.num_blocks = num_blocks;
  parameters.smem = smem;
  parameters.n_x = args.n_x;
  parameters.n_y = args.n_y;
  parameters.exclusion_lower = args.exclusion_lower;
  parameters.exclusion_upper = args.exclusion_upper;
  parameters.threshold = args.opt.threshold;
  parameters.size_cov = args.size_cov;
  parameters.size_dfa = args.size_dfa;
  parameters.size_dfb = args.size_dfb;
  parameters.size_dga = args.size_dga;
  parameters.size_dgb = args.size_dgb;
  parameters.size_normsa = args.size_normsa;
  parameters.size_normsb = args.size_normsb;

  if (computing_rows && computing_cols) {
    parameters.COMPUTE_COLS = true;
    parameters.COMPUTE_ROWS = true;
  } else if (computing_cols) {
    parameters.COMPUTE_COLS = true;
    parameters.COMPUTE_ROWS = false;    
  } else if (computing_rows) {
    parameters.COMPUTE_COLS = false;
    parameters.COMPUTE_ROWS = true;
  }

  globalparameters = parameters;

  char *dls_modules = dls_get_module_info();

  if (!dls_is_in_list(dls_modules, "DLS_AUTOADD")) {
    printf("Manually register implementations...\n");
    dls_add_impl(do_Tile_DP_HALADAPT, "spm", "do_Tile_DP_HALADAPT_GPU_CUDA", &do_Tile_DP_HALADAPT_GPU_CUDA, PM_GPU | PM_CUDA);
  }

  double* profile_A_D = (double*) malloc(size_profile_A);
  double* profile_B_D = (double*) malloc(size_profile_B);

  memcpy(profile_A_D, profile_A, size_profile_A);
  memcpy(profile_B_D, profile_B, size_profile_B);

  uint64_t *profile_A_U = (uint64_t*) malloc(size_profile_A);
  uint64_t *profile_B_U = (uint64_t*) malloc(size_profile_B);

  memcpy(profile_A_U, profile_A, size_profile_A);
  memcpy(profile_B_U, profile_B, size_profile_B);

  // uint64_t *profile_A_U = (uint64_t *)profile_A;
  // uint64_t *profile_B_U = (uint64_t *)profile_B;


  // dls_register_marea(parameters, sizeof(double) * 18, DLS_VT_D);
  dls_register_marea(profile_A_D, size_profile_A, DLS_VT_D);
  dls_register_marea(profile_B_D, size_profile_B, DLS_VT_D);
  dls_register_marea(profile_A_U, size_profile_A, DLS_VT_UL);
  dls_register_marea(profile_B_U, size_profile_B, DLS_VT_UL);
  dls_register_marea((void *)input, args.size_cov + args.size_dfa + args.size_dfb + args.size_dga + args.size_dgb + args.size_normsa + args.size_normsb, DLS_VT_D);

  dls_predict_call(do_Tile_DP_HALADAPT, "bbbbrp", profile_A_D, profile_B_D, profile_A_U, profile_B_U, input, 1);
  dls_start_tgraph();

  // dls_validate_marea((void *)parameters);
  dls_validate_marea((void *)profile_A_D);
  dls_validate_marea((void *)profile_B_D);
  dls_validate_marea((void *)profile_A_U);
  dls_validate_marea((void *)profile_B_U);
  dls_validate_marea((void *)input);

  // dls_unregister_marea((void *) parameters);
  dls_unregister_marea((void *) profile_A_D);
  dls_unregister_marea((void *) profile_B_D);
  dls_unregister_marea((void *) profile_A_U);
  dls_unregister_marea((void *) profile_B_U);
  dls_unregister_marea((void *) input);
  if(profile_type == PROFILE_TYPE_1NN_INDEX) {
    memcpy(profile_A, profile_A_U, size_profile_A);
    memcpy(profile_B, profile_B_U, size_profile_B);
  } else {
    memcpy(profile_A, profile_A_D, size_profile_A);
    memcpy(profile_B, profile_B_D, size_profile_B);
  }

  free(input);
  free(profile_A_D);
  free(profile_A_U);
  free(profile_B_D);
  free(profile_B_U);

  return SCAMP_NO_ERROR;
}

SCAMPError_t kernel_self_join_upper(
    const double *__restrict__ QT, const double *__restrict__ df_A,
    const double *__restrict__ df_B, const double *__restrict__ dg_A,
    const double *__restrict__ dg_B, const double *__restrict__ norms_A,
    const double *__restrict__ norms_B, DeviceProfile *profile_A,
    DeviceProfile *profile_B, uint32_t window_size, uint32_t tile_width,
    uint32_t tile_height, uint64_t global_col, uint64_t global_row,
    const cudaDeviceProp &props, SCAMPPrecisionType t, const OptionalArgs &args,
    SCAMPProfileType profile_type, sizes_Array sizes) {
  constexpr int diags_per_thread = 4;
  uint64_t blocksz = get_blocksz(t, props);
  int32_t exclusion = get_exclusion(window_size, global_col, global_row);
  uint64_t num_workers =
      ceil((tile_width - exclusion) / (float)diags_per_thread);
  uint64_t num_blocks = ceil(num_workers / (double)blocksz);
  SCAMPKernelInputArgs<double> tile_args(QT, df_A, df_B, dg_A, dg_B, norms_A,
                                         norms_B, tile_width, tile_height,
                                         exclusion, 0, args, 
                                         sizes.QT_dev_size, sizes.df_A_size,
                                         sizes.df_B_size, sizes.dg_A_size,
                                         sizes.dg_B_size, sizes.norms_A_size,
                                         sizes.norms_B_size);
  uint64_t smem =
      get_smem(true, true, blocksz, t, GetProfileTypeSize(profile_type));
  if (exclusion < tile_width) {
    switch (profile_type) {
      case PROFILE_TYPE_SUM_THRESH:
        return LaunchDoTile<double, PROFILE_TYPE_SUM_THRESH, BLOCKSPERSM_SELF>(
            tile_args,
            reinterpret_cast<double *>(profile_A->at(PROFILE_TYPE_SUM_THRESH)),
            reinterpret_cast<double *>(profile_B->at(PROFILE_TYPE_SUM_THRESH)),
            t, true, true, blocksz, num_blocks, smem, PROFILE_TYPE_SUM_THRESH,
            sizes.profile_a_tile_dev_size, sizes.profile_b_tile_dev_size);
      case PROFILE_TYPE_1NN_INDEX:
        return LaunchDoTile<uint64_t, PROFILE_TYPE_1NN_INDEX, BLOCKSPERSM_SELF>(
            tile_args,
            reinterpret_cast<uint64_t *>(profile_A->at(PROFILE_TYPE_1NN_INDEX)),
            reinterpret_cast<uint64_t *>(profile_B->at(PROFILE_TYPE_1NN_INDEX)),
            t, true, true, blocksz, num_blocks, smem, PROFILE_TYPE_1NN_INDEX,
            sizes.profile_a_tile_dev_size, sizes.profile_b_tile_dev_size);
      default:
        return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
    }
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t kernel_self_join_lower(
    const double *QT, const double *df_A, const double *df_B,
    const double *dg_A, const double *dg_B, const double *norms_A,
    const double *norms_B, DeviceProfile *profile_A, DeviceProfile *profile_B,
    size_t window_size, size_t tile_width, size_t tile_height,
    size_t global_col, size_t global_row, const cudaDeviceProp &props,
    SCAMPPrecisionType t, const OptionalArgs &args,
    SCAMPProfileType profile_type, sizes_Array sizes) {
  constexpr int diags_per_thread = 4;
  uint64_t blocksz = get_blocksz(t, props);
  uint64_t exclusion =
      get_exclusion(window_size, global_col, global_row + tile_height);
  uint64_t num_workers =
      ceil((tile_height - exclusion) / (float)diags_per_thread);
  uint64_t num_blocks = ceil(num_workers / (double)blocksz);
  SCAMPKernelInputArgs<double> tile_args(QT, df_B, df_A, dg_B, dg_A, norms_B,
                                         norms_A, tile_height, tile_width, 0,
                                         exclusion, args,
                                         sizes.QT_dev_size, sizes.df_B_size,
                                         sizes.df_A_size, sizes.dg_B_size,
                                         sizes.dg_A_size, sizes.norms_B_size,
                                         sizes.norms_A_size);
  uint64_t smem =
      get_smem(true, true, blocksz, t, GetProfileTypeSize(profile_type));
  if (exclusion < tile_height) {
    switch (profile_type) {
      case PROFILE_TYPE_SUM_THRESH:
        return LaunchDoTile<double, PROFILE_TYPE_SUM_THRESH, BLOCKSPERSM_SELF>(
            tile_args,
            reinterpret_cast<double *>(profile_B->at(PROFILE_TYPE_SUM_THRESH)),
            reinterpret_cast<double *>(profile_A->at(PROFILE_TYPE_SUM_THRESH)),
            t, true, true, blocksz, num_blocks, smem, PROFILE_TYPE_SUM_THRESH,
            sizes.profile_b_tile_dev_size, sizes.profile_a_tile_dev_size);
      case PROFILE_TYPE_1NN_INDEX:
        return LaunchDoTile<uint64_t, PROFILE_TYPE_1NN_INDEX, BLOCKSPERSM_SELF>(
            tile_args,
            reinterpret_cast<uint64_t *>(profile_B->at(PROFILE_TYPE_1NN_INDEX)),
            reinterpret_cast<uint64_t *>(profile_A->at(PROFILE_TYPE_1NN_INDEX)),
            t, true, true, blocksz, num_blocks, smem, PROFILE_TYPE_1NN_INDEX,
            sizes.profile_b_tile_dev_size, sizes.profile_a_tile_dev_size);
      default:
        return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
    }
  }
  return SCAMP_NO_ERROR;
}
SCAMPError_t kernel_ab_join_upper(
    const double *__restrict__ QT, const double *__restrict__ df_A,
    const double *__restrict__ df_B, const double *__restrict__ dg_A,
    const double *__restrict__ dg_B, const double *__restrict__ norms_A,
    const double *__restrict__ norms_B, DeviceProfile *profile_A,
    DeviceProfile *profile_B, uint32_t window_size, uint32_t tile_width,
    uint32_t tile_height, uint64_t global_col, uint64_t global_row,
    int64_t distributed_col, int64_t distributed_row, bool aligned_ab_join,
    const cudaDeviceProp &props, SCAMPPrecisionType t, bool computing_rows,
    const OptionalArgs &args, SCAMPProfileType profile_type, sizes_Array sizes) {
  constexpr int diags_per_thread = 4;
  uint64_t blocksz = get_blocksz(t, props);
  std::pair<int, int> exclusion_pair(0, 0);
  if (aligned_ab_join) {
    int start_col = global_col;
    int start_row = global_row;
    if (distributed_col >= 0 && distributed_row >= 0) {
      start_col += distributed_col;
      start_row += distributed_row;
    }
    exclusion_pair = get_exclusion_for_ab_join(window_size, start_row,
                                               start_col, true, tile_width);
  }
  uint64_t num_workers =
      ceil((tile_width - (exclusion_pair.first + exclusion_pair.second)) /
           (float)diags_per_thread);
  uint64_t num_blocks = ceil(num_workers / (double)blocksz);
  SCAMPKernelInputArgs<double> tile_args(
      QT, df_A, df_B, dg_A, dg_B, norms_A, norms_B, tile_width, tile_height,
      exclusion_pair.first, exclusion_pair.second, args,
      sizes.QT_dev_size, sizes.df_A_size,
      sizes.df_B_size, sizes.dg_A_size,
      sizes.dg_B_size, sizes.norms_A_size,
      sizes.norms_B_size);
  uint64_t smem = get_smem(computing_rows, true, blocksz, t,
                           GetProfileTypeSize(profile_type));
  if ((exclusion_pair.first + exclusion_pair.second) < tile_width) {
    switch (profile_type) {
      case PROFILE_TYPE_SUM_THRESH:
        return LaunchDoTile<double, PROFILE_TYPE_SUM_THRESH, BLOCKSPERSM_AB>(
            tile_args,
            reinterpret_cast<double *>(profile_A->at(PROFILE_TYPE_SUM_THRESH)),
            reinterpret_cast<double *>(profile_B->at(PROFILE_TYPE_SUM_THRESH)),
            t, computing_rows, true, blocksz, num_blocks, smem, PROFILE_TYPE_SUM_THRESH,
            sizes.profile_a_tile_dev_size, sizes.profile_b_tile_dev_size);
      case PROFILE_TYPE_1NN_INDEX:
        return LaunchDoTile<uint64_t, PROFILE_TYPE_1NN_INDEX, BLOCKSPERSM_AB>(
            tile_args,
            reinterpret_cast<uint64_t *>(profile_A->at(PROFILE_TYPE_1NN_INDEX)),
            reinterpret_cast<uint64_t *>(profile_B->at(PROFILE_TYPE_1NN_INDEX)),
            t, computing_rows, true, blocksz, num_blocks, smem, PROFILE_TYPE_1NN_INDEX,
            sizes.profile_a_tile_dev_size, sizes.profile_b_tile_dev_size);
      default:
        return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
    }
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t kernel_ab_join_lower(
    const double *__restrict__ QT, const double *__restrict__ df_A,
    const double *__restrict__ df_B, const double *__restrict__ dg_A,
    const double *__restrict__ dg_B, const double *__restrict__ norms_A,
    const double *__restrict__ norms_B, DeviceProfile *profile_A,
    DeviceProfile *profile_B, uint32_t window_size, uint32_t tile_width,
    uint32_t tile_height, uint64_t global_col, uint64_t global_row,
    int64_t distributed_col, int64_t distributed_row, bool aligned_ab_join,
    const cudaDeviceProp &props, SCAMPPrecisionType t, bool computing_rows,
    const OptionalArgs &args, SCAMPProfileType profile_type, sizes_Array sizes) {
  constexpr int diags_per_thread = 4;
  uint64_t blocksz = get_blocksz(t, props);
  std::pair<int, int> exclusion_pair;
  if (aligned_ab_join) {
    int start_col = global_col;
    int start_row = global_row;
    if (distributed_col >= 0 && distributed_row >= 0) {
      start_col += distributed_col;
      start_row += distributed_row;
    }
    exclusion_pair = get_exclusion_for_ab_join(window_size, start_row,
                                               start_col, false, tile_height);
  }
  uint64_t num_workers =
      ceil((tile_height - (exclusion_pair.first + exclusion_pair.second)) /
           (float)diags_per_thread);
  uint64_t num_blocks = ceil(num_workers / (double)blocksz);
  SCAMPKernelInputArgs<double> tile_args(
      QT, df_B, df_A, dg_B, dg_A, norms_B, norms_A, tile_height, tile_width,
      exclusion_pair.first, exclusion_pair.second, args,
      sizes.QT_dev_size, sizes.df_B_size,
      sizes.df_A_size, sizes.dg_B_size,
      sizes.dg_A_size, sizes.norms_B_size,
      sizes.norms_A_size);
  uint64_t smem = get_smem(computing_rows, true, blocksz, t,
                           GetProfileTypeSize(profile_type));
  if (exclusion_pair.first + exclusion_pair.second < tile_height) {
    switch (profile_type) {
      case PROFILE_TYPE_SUM_THRESH:
        return LaunchDoTile<double, PROFILE_TYPE_SUM_THRESH, BLOCKSPERSM_AB>(
            tile_args,
            reinterpret_cast<double *>(profile_B->at(PROFILE_TYPE_SUM_THRESH)),
            reinterpret_cast<double *>(profile_A->at(PROFILE_TYPE_SUM_THRESH)),
            t, true, computing_rows, blocksz, num_blocks, smem, PROFILE_TYPE_SUM_THRESH,
            sizes.profile_b_tile_dev_size, sizes.profile_a_tile_dev_size);
      case PROFILE_TYPE_1NN_INDEX:
        return LaunchDoTile<uint64_t, PROFILE_TYPE_1NN_INDEX, BLOCKSPERSM_AB>(
            tile_args,
            reinterpret_cast<uint64_t *>(profile_B->at(PROFILE_TYPE_1NN_INDEX)),
            reinterpret_cast<uint64_t *>(profile_A->at(PROFILE_TYPE_1NN_INDEX)),
            t, true, computing_rows, blocksz, num_blocks, smem, PROFILE_TYPE_1NN_INDEX,
            sizes.profile_b_tile_dev_size, sizes.profile_a_tile_dev_size);
      default:
        return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
    }
  }
  return SCAMP_NO_ERROR;
}

}  // namespace SCAMP
