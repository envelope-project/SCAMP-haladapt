#pragma once
#include <cuda.h>
#include <future>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>
#include "SCAMP.pb.h"
#include "common.h"
#include "fft_helper.h"
using std::list;
using std::pair;
using std::unordered_map;
using std::vector;

namespace SCAMP {

void do_SCAMP(SCAMPArgs *args, const std::vector<int> &devices);

class SCAMP_Operation {
 private:
  unordered_map<int, double *> T_A_full_dev, T_B_full_dev;
  unordered_map<int, DeviceProfile> profile_a_full_dev, profile_b_full_dev;
  
  unordered_map<int, double *> T_A_dev, T_B_dev, QT_dev, means_A, means_B,
      norms_A, norms_B, df_A, df_B, dg_A, dg_B, scratchpad;
  unordered_map<int, DeviceProfile> profile_a_tile_dev, profile_b_tile_dev;
  unordered_map<int, cudaEvent_t> clocks_start, clocks_end, copy_to_host_done;
  unordered_map<int, cudaStream_t> streams;
  unordered_map<int, std::shared_ptr<fft_precompute_helper>> scratch;
  unordered_map<int, cudaDeviceProp> dev_props;
  Profile *_profile_a, *_profile_b;
  SCAMPProfileType _profile_type;
  size_t size_A;
  size_t size_B;
  size_t tile_size;
  size_t tile_n_x;
  size_t tile_n_y;
  size_t m;
  OptionalArgs opt_args;
  const bool self_join;
  const bool _computing_rows;
  const bool _computing_cols;
  const bool _is_aligned;
  const bool _keep_rows_separate;
  const size_t MAX_TILE_SIZE;
  const SCAMPPrecisionType fp_type;
  vector<int> devices;
  const int64_t tile_start_row_position;
  const int64_t tile_start_col_position;
  // Tile state variables
  list<pair<int, int>> tile_ordering; // AR: keep this for backward compatability with old with pure gpu
  list<pair<int, int>> tile_ordering_gpu; 
  list<pair<int, int>> tile_ordering_cpu; 
  int completed_tiles;
  size_t total_tiles;
  size_t total_tiles_cpu;
  size_t total_tiles_gpu;
  unordered_map<int, size_t> n_x;
  unordered_map<int, size_t> n_y;
  unordered_map<int, size_t> n_x_2;
  unordered_map<int, size_t> n_y_2;
  unordered_map<int, size_t> pos_x;
  unordered_map<int, size_t> pos_y;
  unordered_map<int, size_t> pos_x_2;
  unordered_map<int, size_t> pos_y_2;
  
  size_t n_x_cpu;
  size_t n_y_cpu;
  size_t n_x_2_cpu;
  size_t n_y_2_cpu;
  size_t pos_x_cpu;
  size_t pos_y_cpu;
  size_t pos_x_2_cpu;
  size_t pos_y_2_cpu;

  sizes_Array sizes;

  SCAMPError_t do_tile(SCAMPTileType t, int device,
                       const google::protobuf::RepeatedField<double> &Ta_h,
                       const google::protobuf::RepeatedField<double> &Tb_h,
                       int tile_row, int tile_col);

  bool pick_and_start_next_tile(
      int dev, const google::protobuf::RepeatedField<double> &timeseries_a,
      const google::protobuf::RepeatedField<double> &timeseries_b);
  int issue_and_merge_tiles(
      const google::protobuf::RepeatedField<double> &timeseries_a,
      const google::protobuf::RepeatedField<double> &timeseries_b,
      vector<Profile> *profile_a_tile, vector<Profile> *profile_b_tile,
      int last_device_idx);
  void get_tile_ordering();
  void CopyProfileToHost(Profile *destination_profile,
                         const DeviceProfile *device_tile_profile,
                         uint64_t length, cudaStream_t s);
  void MergeTileIntoFullProfile(Profile *tile_profile, uint64_t position,
                                uint64_t length, Profile *full_profile,
                                uint64_t index_start);
  Profile InitProfile(SCAMPProfileType t, uint64_t size);
  SCAMPError_t InitInputOnDevice(
      const google::protobuf::RepeatedField<double> &Ta_h,
      const google::protobuf::RepeatedField<double> &Tb_h, int device);

 public:
  SCAMP_Operation(size_t Asize, size_t Bsize, size_t window_sz,
                  size_t max_tile_size, const vector<int> &dev, bool selfjoin,
                  SCAMPPrecisionType t, bool do_full_join, int64_t start_row,
                  int64_t start_col, OptionalArgs args_,
                  SCAMPProfileType profile_type, Profile *pA, Profile *pB,
                  bool keep_rows, bool compute_rows, bool compute_cols,
                  bool is_aligned)
      : size_A(Asize),
        m(window_sz),
        MAX_TILE_SIZE(max_tile_size),
        devices(dev),
        self_join(selfjoin),
        completed_tiles(0),
        fp_type(t),
        tile_start_row_position(start_row),
        tile_start_col_position(start_col),
        opt_args(args_),
        _profile_type(profile_type),
        _profile_a(pA),
        _profile_b(pB),
        _keep_rows_separate(keep_rows),
        _computing_rows(compute_rows),
        _computing_cols(compute_cols),
        _is_aligned(is_aligned) {
    if (self_join) {
      size_B = size_A;
    } else {
      size_B = Bsize;
    }
    tile_size = Asize / (devices.size());
    if (tile_size > MAX_TILE_SIZE) {
      tile_size = MAX_TILE_SIZE;
    }
    for (auto device : devices) {
      n_x.emplace(device, 0);
      n_y.emplace(device, 0);
      n_x_2.emplace(device, 0);
      n_y_2.emplace(device, 0);
      pos_x.emplace(device, 0);
      pos_y.emplace(device, 0);
      pos_x_2.emplace(device, 0);
      pos_y_2.emplace(device, 0);
      cudaDeviceProp properties;
      cudaGetDeviceProperties(&properties, device);
      dev_props.emplace(device, properties);
    }
    tile_n_x = tile_size - m + 1;
    tile_n_y = tile_n_x;
  }
  SCAMPError_t do_join(
      const google::protobuf::RepeatedField<double> &timeseries_a,
      const google::protobuf::RepeatedField<double> &timeseries_b);
  SCAMPError_t init();
  SCAMPError_t destroy();

  SCAMPError_t CopySeriesToDevice(
    const google::protobuf::RepeatedField<double> &Ta_h,
    const google::protobuf::RepeatedField<double> &Tb_h, int device);

  SCAMPError_t InitInputDeviceToDevice(int device);
};

}  // namespace SCAMP
