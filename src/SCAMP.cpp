#include <cinttypes>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <unistd.h>
#include <omp.h>

#include "SCAMP.h"
#include "common.h"
#include "tile.h"
#include "utils.h"

#include <dls.h>

using std::cout;
using std::endl;
using std::vector;

#include "printer.cpp"

namespace SCAMP {

static const int ISSUED_ALL_DEVICES = -2;

template <typename T>
void elementwise_sum(T *mp_full, uint64_t merge_start, uint64_t tile_sz,
                     T *to_merge) {
  for (int i = 0; i < tile_sz; ++i) {
    mp_full[i + merge_start] += to_merge[i];
  }
}

template <typename T>
void elementwise_max(T *mp_full, uint64_t merge_start, uint64_t tile_sz,
                     T *to_merge, uint64_t index_offset) {
  for (int i = 0; i < tile_sz; ++i) {
    mp_entry e1, e2;
    e1.ulong = mp_full[i + merge_start];
    e2.ulong = to_merge[i];
    if (e1.floats[0] < e2.floats[0]) {
      e2.ints[1] += index_offset;
      mp_full[i + merge_start] = e2.ulong;
    }
  }
}

SCAMPError_t SCAMP_Operation::init() {
  // for all devices
  for (auto device : devices) {
    cudaSetDevice(device);
    T_A_full_dev.insert({device, nullptr});
    T_B_full_dev.insert({device, nullptr});
    T_A_dev.insert({device, nullptr});
    T_B_dev.insert({device, nullptr});
    QT_dev.insert({device, nullptr});
    means_A.insert({device, nullptr});
    means_B.insert({device, nullptr});
    norms_A.insert({device, nullptr});
    norms_B.insert({device, nullptr});
    df_A.insert({device, nullptr});
    df_B.insert({device, nullptr});
    dg_A.insert({device, nullptr});
    dg_B.insert({device, nullptr});
    DeviceProfile d;
    d[_profile_type] = nullptr;
    profile_a_tile_dev.insert({device, d});
    profile_b_tile_dev.insert({device, d});
    
    profile_a_full_dev.insert({device, d});
    profile_b_full_dev.insert({device, d});
    
    scratchpad.insert({device, nullptr});

    size_t profile_size = GetProfileTypeSize(_profile_type);

    sizes.T_A_full_dev_size = sizeof(double) * size_A;
    sizes.T_B_full_dev_size = sizeof(double) * size_B;
    sizes.profile_a_full_dev_size = profile_size * size_A;
    sizes.profile_b_full_dev_size = profile_size * size_B;
    sizes.T_A_dev_size = sizeof(double) * tile_size;
    sizes.T_B_dev_size = sizeof(double) * tile_size;
    sizes.profile_a_tile_dev_size = profile_size * tile_n_x;
    sizes.profile_b_tile_dev_size = profile_size * tile_n_y;
    sizes.QT_dev_size = sizeof(double) * tile_n_x;
    sizes.means_A_size = sizeof(double) * tile_n_x;
    sizes.means_B_size = sizeof(double) * tile_n_y;
    sizes.norms_A_size = sizeof(double) * tile_n_x;
    sizes.norms_B_size = sizeof(double) * tile_n_y;
    sizes.df_A_size = sizeof(double) * tile_n_x;
    sizes.df_B_size = sizeof(double) * tile_n_y;
    sizes.dg_A_size = sizeof(double) * tile_n_x;
    sizes.dg_B_size = sizeof(double) * tile_n_y;
    sizes.scratchpad_size = sizeof(double) * tile_size;

    // cudaMalloc(&T_A_full_dev.at(device), sizeof(double) * size_A);
    // gpuErrchk(cudaPeekAtLastError());
    T_A_full_dev.at(device) = (double *) malloc(sizeof(double) * size_A);
    // dls_register_marea(T_A_full_dev.at(device), sizeof(double) * size_A, DLS_VT_D);
    // cudaMalloc(&T_B_full_dev.at(device), sizeof(double) * size_B);
    // gpuErrchk(cudaPeekAtLastError());
    T_B_full_dev.at(device) = (double *) malloc(sizeof(double) * size_B);
    // dls_register_marea(T_B_full_dev.at(device), sizeof(double) * size_B, DLS_VT_D);
    // cudaMalloc(&profile_a_full_dev.at(device).at(_profile_type),
    //            profile_size * size_A);
    // gpuErrchk(cudaPeekAtLastError());
    profile_a_full_dev.at(device).at(_profile_type) = (void *) malloc(profile_size * size_A);
    // dls_register_marea(profile_a_full_dev.at(device).at(_profile_type), profile_size * size_A, DLS_VT_D);
    // cudaMalloc(&profile_b_full_dev.at(device).at(_profile_type),
    //            profile_size * size_B);
    profile_b_full_dev.at(device).at(_profile_type) = (void *) malloc(profile_size * size_B);
    // dls_register_marea(profile_b_full_dev.at(device).at(_profile_type), profile_size * size_B, DLS_VT_D);

    // cudaMalloc(&T_A_dev.at(device), sizeof(double) * tile_size);
    // gpuErrchk(cudaPeekAtLastError());
    T_A_dev.at(device) = (double *) malloc(sizeof(double) * tile_size);
    // dls_register_marea(T_A_dev.at(device), sizeof(double) * tile_size, DLS_VT_D);
    // cudaMalloc(&T_B_dev.at(device), sizeof(double) * tile_size);
    // gpuErrchk(cudaPeekAtLastError());
    T_B_dev.at(device) = (double *) malloc(sizeof(double) * tile_size);
    // dls_register_marea(T_B_dev.at(device), sizeof(double) * tile_size, DLS_VT_D);

    // cudaMalloc(&profile_a_tile_dev.at(device).at(_profile_type),
    //            profile_size * tile_n_x);
    // gpuErrchk(cudaPeekAtLastError());
    profile_a_tile_dev.at(device).at(_profile_type) = (void *) malloc(profile_size * tile_n_x);
    // dls_register_marea(profile_a_tile_dev.at(device).at(_profile_type), profile_size * tile_n_x, DLS_VT_D);
    // cudaMalloc(&profile_b_tile_dev.at(device).at(_profile_type),
    //            profile_size * tile_n_y);
    // gpuErrchk(cudaPeekAtLastError());
    profile_b_tile_dev.at(device).at(_profile_type) = (void *) malloc(profile_size * tile_n_y);
    // dls_register_marea(profile_b_tile_dev.at(device).at(_profile_type), profile_size * tile_n_y, DLS_VT_D);
    // cudaMalloc(&QT_dev.at(device), sizeof(double) * tile_n_x);
    // gpuErrchk(cudaPeekAtLastError());
    QT_dev.at(device) = (double *) malloc(sizeof(double) * tile_n_x);
    // dls_register_marea(QT_dev.at(device), sizeof(double) * tile_n_x, DLS_VT_D);
   
    // cudaMalloc(&means_A.at(device), sizeof(double) * tile_n_x);
    // gpuErrchk(cudaPeekAtLastError());
    means_A.at(device) = (double *) malloc(sizeof(double) * tile_n_x);
    //dls_register_marea(means_A.at(device), sizeof(double) * tile_n_x, DLS_VT_D);
    // cudaMalloc(&means_B.at(device), sizeof(double) * tile_n_y);
    // gpuErrchk(cudaPeekAtLastError());
    means_B.at(device) = (double *) malloc(sizeof(double) * tile_n_y);
    //dls_register_marea(means_B.at(device), sizeof(double) * tile_n_y, DLS_VT_D);
    // cudaMalloc(&norms_A.at(device), sizeof(double) * tile_n_x);
    // gpuErrchk(cudaPeekAtLastError());
    norms_A.at(device) = (double *) malloc(sizeof(double) * tile_n_x);
    //dls_register_marea(norms_A.at(device), sizeof(double) * tile_n_x, DLS_VT_D);
    // cudaMalloc(&norms_B.at(device), sizeof(double) * tile_n_y);
    // gpuErrchk(cudaPeekAtLastError());
    norms_B.at(device) = (double *) malloc(sizeof(double) * tile_n_y);
    //dls_register_marea(norms_B.at(device), sizeof(double) * tile_n_y, DLS_VT_D);
    // cudaMalloc(&df_A.at(device), sizeof(double) * tile_n_x);
    // gpuErrchk(cudaPeekAtLastError());
    df_A.at(device) = (double *) malloc(sizeof(double) * tile_n_x);
    //dls_register_marea(df_A.at(device), sizeof(double) * tile_n_x, DLS_VT_D);
    // cudaMalloc(&df_B.at(device), sizeof(double) * tile_n_y);
    // gpuErrchk(cudaPeekAtLastError());
    df_B.at(device) = (double *) malloc(sizeof(double) * tile_n_y);
    //dls_register_marea(df_B.at(device), sizeof(double) * tile_n_y, DLS_VT_D);
    // cudaMalloc(&dg_A.at(device), sizeof(double) * tile_n_x);
    // gpuErrchk(cudaPeekAtLastError());
    dg_A.at(device) = (double *) malloc(sizeof(double) * tile_n_x);
    //dls_register_marea(dg_A.at(device), sizeof(double) * tile_n_x, DLS_VT_D);
    // cudaMalloc(&dg_B.at(device), sizeof(double) * tile_n_y);
    // gpuErrchk(cudaPeekAtLastError());
    dg_B.at(device) = (double *) malloc(sizeof(double) * tile_n_y);
    //dls_register_marea(dg_B.at(device), sizeof(double) * tile_n_y, DLS_VT_D);

    // cudaMalloc(&scratchpad.at(device), sizeof(double) * tile_size);
    scratchpad.at(device) = (double *) malloc(sizeof(double) * tile_size);
    // dls_register_marea(scratchpad.at(device), sizeof(double) * tile_size, DLS_VT_D);

    scratch[device] =
        std::make_shared<fft_precompute_helper>(tile_size, m, true, device);
    cudaStream_t s;
    cudaStreamCreate(&s);
    gpuErrchk(cudaPeekAtLastError());
    streams.emplace(device, s);
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t SCAMP_Operation::destroy() {
  for (auto device : devices) {
    // cudaSetDevice(device);
    // gpuErrchk(cudaPeekAtLastError());
    
    // // cudaFree(T_A_full_dev[device]);
    // dls_unregister_marea(T_A_full_dev[device]);
    // // cudaFree(T_B_full_dev[device]);
    // dls_unregister_marea(T_B_full_dev[device]);

    // // cudaFree(T_A_dev[device]);
    // dls_unregister_marea(T_A_dev[device]);
    // // cudaFree(T_B_dev[device]);
    // dls_unregister_marea(T_B_dev[device]);
    // // cudaFree(QT_dev[device]);
    // dls_unregister_marea(QT_dev[device]);
    // // cudaFree(means_A[device]);
    // dls_unregister_marea(means_A[device]);
    // // cudaFree(means_B[device]);
    // dls_unregister_marea(means_B[device]);
    // // cudaFree(norms_A[device]);
    // dls_unregister_marea(norms_A[device]);
    // // cudaFree(norms_B[device]);
    // dls_unregister_marea(norms_B[device]);
    // // cudaFree(df_A[device]);
    // dls_unregister_marea(df_A[device]);
    // // cudaFree(df_B[device]);
    // dls_unregister_marea(df_B[device]);
    // // cudaFree(dg_A[device]);
    // dls_unregister_marea(dg_A[device]);
    // // cudaFree(dg_B[device]);
    // dls_unregister_marea(dg_B[device]);
    // // cudaFree(profile_a_tile_dev[device].at(_profile_type));
    // dls_unregister_marea(profile_a_tile_dev[device].at(_profile_type));
    // // cudaFree(profile_b_tile_dev[device].at(_profile_type));
    // dls_unregister_marea(profile_b_tile_dev[device].at(_profile_type));

    // // cudaFree(profile_a_full_dev[device].at(_profile_type));
    // dls_unregister_marea(profile_a_full_dev[device].at(_profile_type));
    // // cudaFree(profile_b_full_dev[device].at(_profile_type));
    // dls_unregister_marea(profile_b_full_dev[device].at(_profile_type));

    // // cudaFree(scratchpad.at(device));
    // dls_unregister_marea(scratchpad[device]);
    cudaStreamDestroy(streams.at(device));
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t SCAMP_Operation::InitInputOnDevice(
    const google::protobuf::RepeatedField<double> &Ta_h,
    const google::protobuf::RepeatedField<double> &Tb_h, int device) {
  int profile_size = GetProfileTypeSize(_profile_type);
  // cudaMemcpyAsync(T_A_dev[device], Ta_h.data() + pos_x[device],
  //                 sizeof(double) * n_x[device], cudaMemcpyHostToDevice,
  //                 streams.at(device));
  // gpuErrchk(cudaPeekAtLastError());
  memcpy(T_A_dev[device], Ta_h.data() + pos_x[device], sizeof(double) * n_x[device]);
  // cudaMemcpyAsync(T_B_dev[device], Tb_h.data() + pos_y[device],
  //                 sizeof(double) * n_y[device], cudaMemcpyHostToDevice,
  //                 streams.at(device));
  // gpuErrchk(cudaPeekAtLastError());
  memcpy(T_B_dev[device], Tb_h.data() + pos_y[device], sizeof(double) * n_y[device]);

  switch (_profile_type) {
    case PROFILE_TYPE_SUM_THRESH:
      // cudaMemsetAsync(profile_a_tile_dev.at(device).at(_profile_type), 0,
      //                 profile_size * (n_x[device] - m + 1), streams.at(device));
      // gpuErrchk(cudaPeekAtLastError());
      memset(profile_a_tile_dev.at(device).at(_profile_type), 0, profile_size * (n_x[device] - m + 1));
      // cudaMemsetAsync(profile_b_tile_dev.at(device).at(_profile_type), 0,
      //                 profile_size * (n_y[device] - m + 1), streams.at(device));
      // gpuErrchk(cudaPeekAtLastError());
      memset(profile_b_tile_dev.at(device).at(_profile_type), 0, profile_size * (n_y[device] - m + 1));
      break;
    case PROFILE_TYPE_1NN_INDEX: {
      const uint64_t *pA_ptr =
          _profile_a->data().Get(0).uint64_value().value().data();
      // cudaMemcpyAsync(profile_a_tile_dev.at(device).at(_profile_type),
      //                 pA_ptr + pos_x[device],
      //                 sizeof(uint64_t) * (n_x[device] - m + 1),
      //                 cudaMemcpyHostToDevice, streams.at(device));
      // gpuErrchk(cudaPeekAtLastError());
      memcpy(profile_a_tile_dev.at(device).at(_profile_type), pA_ptr + pos_x[device], sizeof(uint64_t) * (n_x[device] - m + 1));
      if (self_join) {
        // cudaMemcpyAsync(profile_b_tile_dev.at(device).at(_profile_type),
        //                 pA_ptr + pos_y[device],
        //                 sizeof(uint64_t) * (n_y[device] - m + 1),
        //                 cudaMemcpyHostToDevice, streams.at(device));
        // gpuErrchk(cudaPeekAtLastError());
        memcpy(profile_b_tile_dev.at(device).at(_profile_type), pA_ptr + pos_y[device], sizeof(uint64_t) * (n_y[device] - m + 1));
      } else if (_computing_rows && _keep_rows_separate) {
        const uint64_t *pB_ptr =
            _profile_b->data().Get(0).uint64_value().value().data();
        // cudaMemcpyAsync(profile_b_tile_dev.at(device).at(_profile_type),
        //                 pB_ptr + pos_y[device],
        //                 sizeof(uint64_t) * (n_y[device] - m + 1),
        //                 cudaMemcpyHostToDevice, streams.at(device));
        // gpuErrchk(cudaPeekAtLastError());
        memcpy(profile_b_tile_dev.at(device).at(_profile_type), pB_ptr + pos_y[device], sizeof(uint64_t) * (n_y[device] - m + 1));
      }
      break;
    }
    case PROFILE_TYPE_FREQUENCY_THRESH:
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    case PROFILE_TYPE_INVALID:
      break;
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t SCAMP_Operation::do_tile(
    SCAMPTileType t, int device,
    const google::protobuf::RepeatedField<double> &Ta_h,
    const google::protobuf::RepeatedField<double> &Tb_h,
    int tile_row, int tile_col) {
  size_t start_x = pos_x[device];
  size_t start_y = pos_y[device];
  SCAMPError_t err;
  size_t t_n_x = n_x[device] - m + 1;
  size_t t_n_y = n_y[device] - m + 1;
  InitInputOnDevice(Ta_h, Tb_h, device);

  // initPrinter(sizes, T_A_full_dev[device], T_B_full_dev[device], 
  //             profile_a_full_dev.at(device).at(_profile_type),
  //             profile_b_full_dev.at(device).at(_profile_type),
  //             T_A_dev[device], T_B_dev[device], profile_a_tile_dev.at(device).at(_profile_type),
  //             profile_b_tile_dev.at(device).at(_profile_type),
  //             QT_dev[device], means_A[device], means_B[device], 
  //             norms_A[device], norms_B[device],
  //             df_A[device], df_B[device], dg_A[device],
  //             dg_B[device], scratchpad[device],
  //             tile_row, tile_col);

  // FIXME?: Computing the sliding dot products & statistics for each tile is
  // overkill
  compute_statistics(T_A_dev[device], norms_A[device], df_A[device],
                     dg_A[device], means_A[device], t_n_x, m,
                     streams.at(device), scratchpad[device], tile_n_x, tile_size);
  compute_statistics(T_B_dev[device], norms_B[device], df_B[device],
                     dg_B[device], means_B[device], t_n_y, m,
                     streams.at(device), scratchpad[device], tile_n_y, tile_size);
  //printPointers("compute_statistics");
  SCAMP_Tile tile(t, T_A_dev[device], T_B_dev[device], df_A[device],
                  df_B[device], dg_A[device], dg_B[device], norms_A[device],
                  norms_B[device], means_A[device], means_B[device],
                  QT_dev[device], &profile_a_tile_dev[device],
                  &profile_b_tile_dev[device], start_x, start_y,
                  tile_start_col_position, tile_start_row_position, _is_aligned,
                  n_y[device], n_x[device], m, scratch[device],
                  dev_props.at(device), fp_type, _profile_type, opt_args, sizes);
  err = tile.execute(streams.at(device));
  //printPointers("tile_execute");
  return err;
}

void SCAMP_Operation::get_tile_ordering() {
  tile_ordering.clear();
  size_t num_tile_rows = ceil((size_B - m + 1) / static_cast<double>(tile_n_y));
  size_t num_tile_cols = ceil((size_A - m + 1) / static_cast<double>(tile_n_x));
  std::cout << "sizeB = " << size_B << "  sizeA = " << size_A << std::endl;
  std::cout << "tile_n_x = " << tile_n_x << "  tile_n_y = " << tile_n_y
            << std::endl;
  std::cout << "rows = " << num_tile_rows << "  cols = " << num_tile_cols
            << std::endl;
  if (self_join) {
    for (int offset = 0; offset < num_tile_rows - 1; ++offset) {
      for (int diag = 0; diag < num_tile_cols - 1 - offset; ++diag) {
        tile_ordering.emplace_back(diag, diag + offset);
        // std::cout  << " " << diag << " , " << diag + offset << std::endl;
      }
    }

    for (int i = 0; i < num_tile_rows; ++i) {
      tile_ordering.emplace_back(i, num_tile_cols - 1);
      // std::cout << " " << i << " , " << num_tile_cols - 1 << std::endl;
    }
  } else {
    // Add upper diagonals one at a time except for edge tiles
    for (int diag = 0; diag < num_tile_cols - 1; ++diag) {
      for (int offset = 0;
           offset + diag < num_tile_cols - 1 && offset < num_tile_rows - 1;
           ++offset) {
        tile_ordering.emplace_back(offset, diag + offset);
      }
    }

    // Add lower diagonals one at a time except for edge tiles
    for (int diag = 1; diag < num_tile_rows - 1; ++diag) {
      for (int offset = 0;
           offset + diag < num_tile_rows - 1 && offset < num_tile_cols - 1;
           ++offset) {
        tile_ordering.emplace_back(offset + diag, offset);
      }
    }

    // Add the corner edge tile
    tile_ordering.emplace_back(num_tile_rows - 1, num_tile_cols - 1);

    int x = 0;
    int y = 0;

    // Alternate between adding final row and final column edge tiles
    while (x < num_tile_cols - 1 && y < num_tile_rows - 1) {
      tile_ordering.emplace_back(y, num_tile_cols - 1);
      tile_ordering.emplace_back(num_tile_rows - 1, x);
      ++x;
      ++y;
    }

    // Add any remaining final row edge tiles
    while (x < num_tile_cols - 1) {
      tile_ordering.emplace_back(num_tile_rows - 1, x);
      ++x;
    }

    // Add any remaining final column edge tiles
    while (y < num_tile_rows - 1) {
      tile_ordering.emplace_back(y, num_tile_cols - 1);
      ++y;
    }
  }
  total_tiles = tile_ordering.size();
}


bool SCAMP_Operation::pick_and_start_next_tile(
    int dev, const google::protobuf::RepeatedField<double> &timeseries_a,
    const google::protobuf::RepeatedField<double> &timeseries_b) {
  bool done = false;

  // int tile_row = tile_ordering_gpu.front().first;
  // int tile_col = tile_ordering_gpu.front().second;

  int tile_row = tile_ordering.front().first;
  int tile_col = tile_ordering.front().second;

  // Get the position of the tile we will compute
  pos_x[dev] = tile_col * tile_n_x;
  pos_y[dev] = tile_row * tile_n_y;
  // Get the size of the tile we will compute
  n_x[dev] = std::min(tile_size, size_A - pos_x[dev]);
  n_y[dev] = std::min(tile_size, size_B - pos_y[dev]);
  /*
  std::cout << "Starting tile with starting row of " << pos_y[dev]
            << " starting column of " << pos_y[dev] << " with height "
            << n_y[dev] << " and width " << n_x[dev] << " on device: " << dev
            << std::endl;
  */
  std::cout << "GPU starting tile with row of " << tile_row
          << " column of " << tile_col << " with height "
          << n_y[dev] << " and width " << n_x[dev] << " on device: " << dev
          << std::endl;

  SCAMPError_t err;
    //cout << "Using aggregation on CPU" << endl;
  if (self_join) {
    if (tile_row == tile_col) {
      // partial tile on diagonal
      err =
          do_tile(SELF_JOIN_UPPER_TRIANGULAR, dev, timeseries_a, timeseries_b, tile_col, tile_row);
    } else {
      // full tile
      err = do_tile(SELF_JOIN_FULL_TILE, dev, timeseries_a, timeseries_b, tile_col, tile_row);
    }
  } else if (_computing_rows) {
    // BiDirectional AB-join
    err = do_tile(AB_FULL_JOIN_FULL_TILE, dev, timeseries_a, timeseries_b, tile_col, tile_row);
  } else {
    // Column AB-join
    err = do_tile(AB_JOIN_FULL_TILE, dev, timeseries_a, timeseries_b, tile_col, tile_row);
  }
  if (err != SCAMP_NO_ERROR) {
    printf("ERROR %d executing tile. \n", err);
  }

  // tile_ordering.pop_front(); // TODO AR remove
  
  tile_ordering.pop_front();
  if (tile_ordering.empty()) {
    done = true;
  }
  return done;
}

// TODO(zpzim): make generic on device
void SCAMP_Operation::CopyProfileToHost(
    Profile *destination_profile, const DeviceProfile *device_tile_profile,
    uint64_t length, cudaStream_t s) {
      
  switch (_profile_type) {
    case PROFILE_TYPE_SUM_THRESH:
      // cudaMemcpyAsync(destination_profile->mutable_data()
      //                     ->Mutable(0)
      //                     ->mutable_double_value()
      //                     ->mutable_value()
      //                     ->mutable_data(),
      //                 device_tile_profile->at(PROFILE_TYPE_SUM_THRESH),
      //                 length * sizeof(double), cudaMemcpyDeviceToHost, s);
      // gpuErrchk(cudaPeekAtLastError());
      memcpy(destination_profile->mutable_data()->Mutable(0)->mutable_double_value()->mutable_value()->mutable_data(), device_tile_profile->at(PROFILE_TYPE_SUM_THRESH), length* sizeof(double));
      break;
    case PROFILE_TYPE_1NN_INDEX:
      // cudaMemcpyAsync(destination_profile->mutable_data()
      //                     ->Mutable(0)
      //                     ->mutable_uint64_value()
      //                     ->mutable_value()
      //                     ->mutable_data(),
      //                 device_tile_profile->at(PROFILE_TYPE_1NN_INDEX),
      //                 length * sizeof(uint64_t), cudaMemcpyDeviceToHost, s);
      // gpuErrchk(cudaPeekAtLastError());
      memcpy(destination_profile->mutable_data()->Mutable(0)->mutable_uint64_value()->mutable_value()->mutable_data(), device_tile_profile->at(PROFILE_TYPE_1NN_INDEX), length* sizeof(uint64_t));
      break;
    case PROFILE_TYPE_FREQUENCY_THRESH:
      break;
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      break;
  }
}

void SCAMP_Operation::MergeTileIntoFullProfile(Profile *tile_profile,
                                               uint64_t position,
                                               uint64_t length,
                                               Profile *full_profile,
                                               uint64_t index_start = 0) {
  switch (_profile_type) {
    case PROFILE_TYPE_SUM_THRESH:
      elementwise_sum<double>(full_profile->mutable_data()
                                  ->Mutable(0)
                                  ->mutable_double_value()
                                  ->mutable_value()
                                  ->mutable_data(),
                              position, length,
                              tile_profile->mutable_data()
                                  ->Mutable(0)
                                  ->mutable_double_value()
                                  ->mutable_value()
                                  ->mutable_data());
      return;
    case PROFILE_TYPE_1NN_INDEX:
      elementwise_max<uint64_t>(full_profile->mutable_data()
                                    ->Mutable(0)
                                    ->mutable_uint64_value()
                                    ->mutable_value()
                                    ->mutable_data(),
                                position, length,
                                tile_profile->mutable_data()
                                    ->Mutable(0)
                                    ->mutable_uint64_value()
                                    ->mutable_value()
                                    ->mutable_data(),
                                index_start);
      // elementwise_max_with_index();
      return;
    case PROFILE_TYPE_FREQUENCY_THRESH:
      elementwise_sum<uint64_t>(full_profile->mutable_data()
                                    ->Mutable(0)
                                    ->mutable_uint64_value()
                                    ->mutable_value()
                                    ->mutable_data(),
                                position, length,
                                tile_profile->mutable_data()
                                    ->Mutable(0)
                                    ->mutable_uint64_value()
                                    ->mutable_value()
                                    ->mutable_data());
      return;
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      return;
  }
}

int SCAMP_Operation::issue_and_merge_tiles(
    const google::protobuf::RepeatedField<double> &timeseries_a,
    const google::protobuf::RepeatedField<double> &timeseries_b,
    vector<Profile> *profile_a_tile, vector<Profile> *profile_b_tile,
    int last_device_idx = ISSUED_ALL_DEVICES) {
  bool done = last_device_idx != ISSUED_ALL_DEVICES;
  int last_dev = ISSUED_ALL_DEVICES;
  if (last_device_idx == ISSUED_ALL_DEVICES) {
    last_device_idx = devices.size() - 1;
  }
  for (int i = 0; i <= last_device_idx; ++i) {
    int device = devices.at(i);
    CopyProfileToHost(&profile_a_tile->at(i), &profile_a_tile_dev[device],
                      n_x[device] - m + 1, streams[device]);
    if (_computing_rows) {
      CopyProfileToHost(&profile_b_tile->at(i), &profile_b_tile_dev[device],
                        n_y[device] - m + 1, streams[device]);
    }
    n_x_2[device] = n_x[device];
    n_y_2[device] = n_y[device];
    pos_x_2[device] = pos_x[device];
    pos_y_2[device] = pos_y[device];
    if (!done) {
      done = pick_and_start_next_tile(device, timeseries_a, timeseries_b);
      if (done) {
        last_dev = i;
      }
    }
  }

  for (int i = 0; i <= last_device_idx; ++i) {
    int device = devices.at(i);
    MergeTileIntoFullProfile(&profile_a_tile->at(i), pos_x_2[device],
                             n_x_2[device] - m + 1, _profile_a,
                             pos_y_2[device]);
    if (self_join) {
      MergeTileIntoFullProfile(&profile_b_tile->at(i), pos_y_2[device],
                               n_y_2[device] - m + 1, _profile_a,
                               pos_x_2[device]);

    } else if (_computing_rows && _keep_rows_separate) {
      MergeTileIntoFullProfile(&profile_b_tile->at(i), pos_y_2[device],
                               n_y_2[device] - m + 1, _profile_b,
                               pos_x_2[device]);
    }
    completed_tiles++;
  }

  std::cout << completed_tiles / static_cast<float>(total_tiles) * 100
            << " percent complete." << std::endl;
  return last_dev;
}

Profile SCAMP_Operation::InitProfile(SCAMPProfileType t, uint64_t size) {
  Profile p;
  p.set_type(t);
  switch (t) {
    case PROFILE_TYPE_SUM_THRESH:
      p.mutable_data()->Add()->mutable_double_value()->mutable_value()->Resize(
          size, 0);
      return p;
    case PROFILE_TYPE_1NN_INDEX:
      mp_entry e;
      e.ints[1] = -1u;
      e.floats[0] = std::numeric_limits<float>::lowest();
      p.mutable_data()->Add()->mutable_uint64_value()->mutable_value()->Resize(
          size, e.ulong);
      return p;
    case PROFILE_TYPE_FREQUENCY_THRESH:
      p.mutable_data()->Add()->mutable_uint64_value()->mutable_value()->Resize(
          size, 0);
      return p;
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_1NN_MULTIDIM:
    default:
      return p;
  }
}

SCAMPError_t SCAMP_Operation::do_join(
    const google::protobuf::RepeatedField<double> &timeseries_a,
    const google::protobuf::RepeatedField<double> &timeseries_b) {
  vector<Profile> profile_a_tile(devices.size(),
                                 InitProfile(_profile_type, tile_n_y));
  vector<Profile> profile_b_tile(devices.size(),
                                 InitProfile(_profile_type, tile_n_x));

  bool done = false;
  int last_dev = ISSUED_ALL_DEVICES;
  get_tile_ordering();

  std::cout << "Performing join with " << tile_ordering.size()
            << " tiles on GPU." << std::endl;

  for (int i = 0; i < devices.size(); ++i) {
    int device = devices.at(i);
    done = pick_and_start_next_tile(device, timeseries_a, timeseries_b);
    if (done) {
      last_dev = i;
      break;
    }
  }

  bool write_idx = false;

  while (last_dev == ISSUED_ALL_DEVICES) {
    last_dev = issue_and_merge_tiles(timeseries_a, timeseries_b,
                                     &profile_a_tile, &profile_b_tile);
    
  // TODO AR: remove printing it was only for debug
  //if(start_x == 0 && start_y == 0)
  if(write_idx == true)
  {
    std::cout << "writing gpu tile" << std::endl;
    std::ofstream outfile("tile_0_gpu.txt");
    auto arr = profile_a_tile[0].data().Get(0).uint64_value().value();
    for (int i = 0; i < arr.size(); ++i) {
      SCAMP::mp_entry e;
      e.ulong = arr.Get(i);
      outfile << std::setprecision(15) << e.floats[0] << std::endl;
    }
    outfile.close();
    write_idx = false;
  }
  }
  issue_and_merge_tiles(timeseries_a, timeseries_b, &profile_a_tile,
                        &profile_b_tile, last_dev);

  return SCAMP_NO_ERROR;
}

void do_SCAMP(SCAMPArgs *args, const std::vector<int> &devices) {
  if (devices.empty()) {
    printf("Error: no gpu provided\n");
    exit(0);
  }
  // Allocate and initialize memory
  clock_t start, end;
  OptionalArgs opt_args(args->distance_threshold());
  SCAMP_Operation op(
      args->timeseries_a().size(), args->timeseries_b().size(), args->window(),
      args->max_tile_size(), devices, !args->has_b(), args->precision_type(),
      args->computing_columns() && args->computing_rows(),
      args->distributed_start_row(), args->distributed_start_col(), opt_args,
      args->profile_type(), args->mutable_profile_a(),
      args->mutable_profile_b(), args->keep_rows_separate(),
      args->computing_rows(), args->computing_columns(), args->is_aligned());
  op.init();
  start = clock();

  if (args->has_b()) {
      op.do_join(args->timeseries_a(), args->timeseries_b());
  } else {
    op.do_join(args->timeseries_a(), args->timeseries_a());
  }

  end = clock();
  op.destroy();
  printf(
      "Finished SCAMP to generate  matrix profile in %f "
      "seconds on %lu devices:\n",
      (end - start) / static_cast<double>(CLOCKS_PER_SEC), devices.size());
}
}  // namespace SCAMP
