#include <iostream>
#include <fstream>
#include "common.h"
#include <sstream>
#include <string>
#include <cstdlib>


struct PrinterPointer {
  double* T_A_full_dev;
  double* T_B_full_dev;
  void* profile_a_full_dev;
  void* profile_b_full_dev;
  double* T_A_dev;
  double* T_B_dev;
  void* profile_a_tile_dev;
  void* profile_b_tile_dev;
  double* QT_dev;
  double* means_A;
  double* means_B;
  double* norms_A;
  double* norms_B;
  double* df_A;
  double* df_B;
  double* dg_A;
  double* dg_B;
  double* scratchpad;
};

static PrinterPointer pointers;
static SCAMP::sizes_Array sizes;
static int tile_row;
static int tile_col;

static void initPrinter(SCAMP::sizes_Array sizes_,
                        double* T_A_full_dev, 
                        double* T_B_full_dev,
                        void* profile_a_full_dev,
                        void* profile_b_full_dev,
                        double* T_A_dev,
                        double* T_B_dev,
                        void* profile_a_tile_dev,
                        void* profile_b_tile_dev,
                        double* QT_dev,
                        double* means_A,
                        double* means_B,
                        double* norms_A,
                        double* norms_B,
                        double* df_A,
                        double* df_B,
                        double* dg_A,
                        double* dg_B,
                        double* scratchpad,
                        int tile_row_,
                        int tile_col_
                      ) {
  sizes = sizes_;
  pointers.T_A_full_dev = T_A_full_dev;
  pointers.T_B_full_dev = T_B_full_dev;
  pointers.profile_a_full_dev = profile_a_full_dev;
  pointers.profile_b_full_dev = profile_b_full_dev;
  pointers.T_A_dev = T_A_dev;
  pointers.T_B_dev = T_B_dev;
  pointers.profile_a_tile_dev = profile_a_tile_dev;
  pointers.profile_b_tile_dev = profile_b_tile_dev;
  pointers.QT_dev = QT_dev;
  pointers.means_A = means_A;
  pointers.means_B = means_B;
  pointers.norms_A = norms_A;
  pointers.norms_B = norms_B;
  pointers.df_A = df_A;
  pointers.df_B = df_B;
  pointers.dg_A = dg_A;
  pointers.dg_B = dg_B;
  pointers.scratchpad = scratchpad;
  tile_row = tile_row_;
  tile_col = tile_col_;
}

static void printToFileDouble (std::string path, std::string filename, double* buffer, size_t size_buffer) {
  path += filename;
  char const* path_char = path.c_str();
  std::ofstream myfile (path_char);
  for(int i=0; i<size_buffer/sizeof(double); i++) {
    myfile << buffer[i] <<std::endl;
  }
  myfile.close();
}

static void printToFileUint (std::string path, std::string filename, uint64_t* buffer, size_t size_buffer) {
  path += filename;
  char const* path_char = path.c_str();
  std::ofstream myfile (path_char);
  for(int i=0; i<size_buffer/sizeof(uint64_t); i++) {
    myfile << buffer[i] <<std::endl;
  }
  myfile.close();
}

static void printPointers(char const* folderName) {
  printf("Start printing Buffers\n");
  std::string command = std::string("mkdir -p debugfiles/tile_") + std::to_string(tile_col) + std::string("_") + std::to_string(tile_row) + std::string("/") + std::string(folderName);
  printf("Print command: %s\n", command.c_str());
  const int dir_err = system(command.c_str());
  if (-1 == dir_err)
  {
      printf("Error creating directory!n");
  }

  std::stringstream ssPath;
  ssPath << "debugfiles/tile_" << tile_col << "_" << tile_row << "/"<< folderName << "/";
  std::string path = ssPath.str();

  printToFileDouble(path, "T_A_full_dev", pointers.T_A_full_dev, sizes.T_A_full_dev_size);
  printToFileDouble(path, "T_B_full_dev", pointers.T_B_full_dev, sizes.T_B_full_dev_size);
  printToFileDouble(path, "profile_a_full_dev", (double *)pointers.profile_a_full_dev, sizes.profile_a_full_dev_size);
  printToFileDouble(path, "profile_b_full_dev", (double *)pointers.profile_b_full_dev, sizes.profile_b_full_dev_size);
  printToFileDouble(path, "T_A_dev", pointers.T_A_dev, sizes.T_A_dev_size);
  printToFileDouble(path, "T_B_dev", pointers.T_B_dev, sizes.T_B_dev_size);
  printToFileDouble(path, "QT_dev", pointers.QT_dev, sizes.QT_dev_size);
  printToFileDouble(path, "profile_a_tile_dev", (double *)pointers.profile_a_tile_dev, sizes.profile_a_tile_dev_size);
  printToFileDouble(path, "profile_b_tile_dev", (double *)pointers.profile_b_tile_dev, sizes.profile_b_tile_dev_size);
  printToFileDouble(path, "means_A", pointers.means_A, sizes.means_A_size);
  printToFileDouble(path, "means_B", pointers.means_B, sizes.means_B_size);
  printToFileDouble(path, "norms_A", pointers.norms_A, sizes.norms_A_size);
  printToFileDouble(path, "norms_B", pointers.norms_B, sizes.norms_B_size);
  printToFileDouble(path, "df_A", pointers.df_A, sizes.df_A_size);
  printToFileDouble(path, "df_B", pointers.df_B, sizes.df_B_size);
  printToFileDouble(path, "dg_A", pointers.dg_A, sizes.dg_A_size);
  printToFileDouble(path, "dg_B", pointers.dg_B, sizes.dg_B_size);
  printToFileDouble(path, "scratchpad", pointers.scratchpad, sizes.scratchpad_size);
}