#pragma once
#include <memory>
#include "SCAMP.pb.h"
#include "common.h"
#include "fft_helper.h"
#include "kernels.h"

namespace SCAMP {

using DeviceProfile = std::unordered_map<int, void *>;

class SCAMP_Tile {
 private:
  SCAMPTileType type;
  const double *timeseries_A;
  const double *timeseries_B;

  const double *df_A;
  const double *df_B;
  const double *dg_A;
  const double *dg_B;
  const double *means_A;
  const double *means_B;
  const double *norms_A;
  const double *norms_B;
  std::shared_ptr<fft_precompute_helper> fft_info;
  double *QT_scratch;
  DeviceProfile *profile_A;
  DeviceProfile *profile_B;
  OptionalArgs opt_args;
  sizes_Array sizes;
  int64_t global_start_A;
  int64_t global_start_B;
  size_t tile_start_A;
  size_t tile_start_B;
  size_t tile_height;
  size_t tile_width;
  bool aligned_ab_join;
  const size_t window_size;
  bool full_join;
  double thresh;
  const SCAMPPrecisionType fp_type;
  const SCAMPProfileType profile_type;
  const cudaDeviceProp props;

 public:
  SCAMP_Tile(SCAMPTileType t, const double *ts_A, const double *ts_B,
             const double *dfA, const double *dfB, const double *dgA,
             const double *dgB, const double *normA, const double *normB,
             const double *meansA, const double *meansB, double *QT,
             DeviceProfile *profileA, DeviceProfile *profileB, size_t start_A,
             int64_t start_B, int64_t g_start_A, size_t g_start_B, bool aligned,
             size_t height, size_t width, size_t m,
             std::shared_ptr<fft_precompute_helper> scratch,
             const cudaDeviceProp &prop, SCAMPPrecisionType fp_t,
             SCAMPProfileType profile_type_, OptionalArgs opt_args_, sizes_Array sizes_)
      : type(t),
        timeseries_A(ts_A),
        timeseries_B(ts_B),
        df_A(dfA),
        df_B(dfB),
        means_A(meansA),
        means_B(meansB),
        dg_A(dgA),
        dg_B(dgB),
        norms_A(normA),
        norms_B(normB),
        QT_scratch(QT),
        profile_A(profileA),
        profile_B(profileB),
        tile_start_A(start_A),
        tile_start_B(start_B),
        global_start_A(g_start_A),
        global_start_B(g_start_B),
        tile_height(height),
        tile_width(width),
        fft_info(scratch),
        window_size(m),
        props(prop),
        fp_type(fp_t),
        full_join(false),
        aligned_ab_join(aligned),
        profile_type(profile_type_),
        opt_args(opt_args_),
        sizes(sizes_) {}
  SCAMPError_t do_self_join_full(cudaStream_t s);
  SCAMPError_t do_self_join_half(cudaStream_t s);
  SCAMPError_t do_ab_join_full(cudaStream_t s);
  SCAMPError_t execute(cudaStream_t s) {
    SCAMPError_t error;
    switch (type) {
      case SELF_JOIN_FULL_TILE:
        error = do_self_join_full(s);
        break;
      case SELF_JOIN_UPPER_TRIANGULAR:
        error = do_self_join_half(s);
        break;
      case AB_JOIN_FULL_TILE:
        error = do_ab_join_full(s);
        break;
      case AB_FULL_JOIN_FULL_TILE:
        full_join = true;
        error = do_ab_join_full(s);
        break;
      default:
        error = SCAMP_TILE_ILLEGAL_TYPE;
        break;
    }
    return error;
  }
};

}  // namespace SCAMP
