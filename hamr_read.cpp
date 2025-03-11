#include <iostream>
#include <string>
#include <cmath>
#include <unordered_map>
#include <filesystem>
#include <cstdio> // fread and fseek
#include <numeric> // accumulate
#include <omp.h>
#include "hamr_read.h"

#define MIN(X,Y) ((X) < (Y) ?  (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ?  (X) : (Y))

void kernel_read_grid(std::string dir, int axisym, int by_block, 
  int export_raytracing,
  const std::unordered_map<std::string, double>& amr,
  const std::unordered_map<std::string, std::vector<int>>& amr_info,
  const std::unordered_map<std::string, double>& block_info,
  const std::unordered_map<std::string, double>& params,
  const std::unordered_map<std::string, double>& options,
  float* x1, float* x2, float* x3, float* r, float* theta, float* phi,
  float* gcov, float* gcon, float* gdet, float* dxdxp) {
  
  int isize, jsize, filesize, read_file, read_data;
  int p1, p2, p3, coordx1, coordx2, coordx3, offsetx1, offsetx2, offsetx3;
  int dbl_size = sizeof(double);
  // variables (nvar): x1, x2, x3, 
  // r, theta, phi, i (global), j (global), k (global)
  // metrics (nmtc): gcov, gcon, 
  // dxdxp (Jacobian from x1, x2, x3 -> r, theta, phi)
  int nvar = 9, nmtc = 3;
  int upscalex1, upscalex2, upscalex3;
  int gridsize_2D, gridsize_3D, index_2D, index_3D;
  int iGlob, jGlob, kGlob; // global indices
  int n = 0; 
  // start/end global indices and step in x1,x2,x3 directions
  int iStrt, jStrt, kStrt, iEnd, jEnd, kEnd, iStp, jStp, kStp;     
  int iLoc,  jLoc,  kLoc, ii; // local indices on given block
  int bs1LR, bs2LR, bs3LR; // bs1,2,3 / low_resx1,2,3
  bool iInRange, jInRange, kInRange;
  FILE* fin;
  std::string filename;

  // maximum refinement level
  int max_levelx1 = int(amr.at("max_levelx1"));
  int max_levelx2 = int(amr.at("max_levelx2"));
  int max_levelx3 = int(amr.at("max_levelx3"));

  // information needed for AMR
  std::vector<int> n_ord       = amr_info.at("n_ord");
  // refinement level of block
  std::vector<int> amr_levelx1 = amr_info.at("levelx1");
  std::vector<int> amr_levelx2 = amr_info.at("levelx2");
  std::vector<int> amr_levelx3 = amr_info.at("levelx3");
  // coordinate of block
  std::vector<int> amr_coordx1 = amr_info.at("coordx1");
  std::vector<int> amr_coordx2 = amr_info.at("coordx2");
  std::vector<int> amr_coordx3 = amr_info.at("coordx3");

  // block indices
  int i_min = int(block_info.at("i_min"));
  int i_max = int(block_info.at("i_max"));
  int j_min = int(block_info.at("j_min"));
  int j_max = int(block_info.at("j_max"));
  int k_min = int(block_info.at("k_min"));
  int k_max = int(block_info.at("k_max"));

  // number of total active blocks
  int nb = int(params.at("n_active_total"));
  // block size
  int bs1   = int(params.at("bs1"));
  int bs2   = int(params.at("bs2"));
  int bs3   = int(params.at("bs3"));

  // downsampling factors
  int low_resx1 = int(options.at("low_resx1"));
  int low_resx2 = int(options.at("low_resx2"));
  int low_resx3 = int(options.at("low_resx3"));

  isize = (k_max - k_min) * (j_max - j_min);
  jsize = (k_max - k_min);

  #pragma omp parallel for private(n, p1, p2, p3, coordx1, coordx2, coordx3, \
  fin, filesize, read_file, read_data, upscalex1, upscalex2, upscalex3, \
  filename, iGlob, jGlob, kGlob, iLoc, jLoc, kLoc, \
  offsetx1, offsetx2, offsetx3, bs1LR, bs2LR, bs3LR, \
  iStrt, jStrt, kStrt, iEnd, jEnd, kEnd, iStp, jStp, kStp, \
  iInRange, jInRange, kInRange, ii, \
  index_3D, index_2D, gridsize_3D, gridsize_2D)
  for (n = 0; n < nb; n++) {
    gridsize_2D = (i_max - i_min) * (j_max - j_min);
    gridsize_3D = (i_max - i_min) * (j_max - j_min) * (k_max - k_min);
    std::vector<double> tempArray(58);
    std::vector<int> level_array(gridsize_2D);
    p1      = max_levelx1 - amr_levelx1[n_ord[n]];
    p2      = max_levelx2 - amr_levelx2[n_ord[n]];
    p3      = max_levelx3 - amr_levelx3[n_ord[n]];
    coordx1 = amr_coordx1[n_ord[n]];
    coordx2 = amr_coordx2[n_ord[n]];
    coordx3 = amr_coordx3[n_ord[n]];

    filename = dir + "/gdumps/gdump" + std::to_string(n_ord[n]);

    if (omp_get_thread_num() == 0) {
      std::cout << "Loading " + filename + " on rank 0." << std::endl;
    }

    if (std::filesystem::exists(filename)) {
      filesize = std::filesystem::file_size(filename);
      fin      = fopen(filename.c_str(), "rb");

      // 16 components to the metric variables + 1 for gdet (scalar)
      if (filesize >=
          (nvar * bs1 * bs2 * bs3 + (bs1 * bs2 * (nmtc * 16 + 1)) * (axisym) +
          (bs1 * bs2 * bs3 * (nmtc * 16 + 1)) * (!axisym)) * dbl_size) {

        read_file = 0;
        read_data = 0;

        // either 2^n or 1 (not used if by_block == 1)
        upscalex1 = pow(2, p1); 
        upscalex2 = pow(2, p2);
        upscalex3 = pow(2, p3);

        // allow data to be read into 4D arrays (organized by block) 
        // or as a 3D unigrid array
        if (by_block) {
          iStrt = 0;
          jStrt = 0;
          kStrt = 0;

          iEnd  = bs1 / low_resx1;
          jEnd  = bs2 / low_resx2;
          kEnd  = bs3 / low_resx3;

          iStp  = 1;
          jStp  = 1;
          kStp  = 1;
        } else {
          // offsets are always 0 unless low_resx1 etc. not a power of 2
          offsetx1 = ( (low_resx1 - (coordx1 * upscalex1 * bs1) % low_resx1) %
                        low_resx1 );
          offsetx2 = ( (low_resx2 - (coordx2 * upscalex2 * bs2) % low_resx2) %
                        low_resx2 );
          offsetx3 = ( (low_resx3 - (coordx3 * upscalex3 * bs3) % low_resx3) %
                        low_resx3 );

          iStrt = coordx1 * (upscalex1 * bs1) + offsetx1;
          jStrt = coordx2 * (upscalex2 * bs2) + offsetx2;
          kStrt = coordx3 * (upscalex3 * bs3) + offsetx3;

          iEnd  = (coordx1 + 1) * (upscalex1 * bs1);
          jEnd  = (coordx2 + 1) * (upscalex2 * bs2);
          kEnd  = (coordx3 + 1) * (upscalex3 * bs3);

          // if export_raytracing = 1: then you're averaging over the
          // blocks set by low_res 
          // if export_raytracing = 0: go up by low_res, which may be 1
          // the 1*flag + (1-flag) logic is good on GPUs since it avoids an "if" 
          iStp  = 1 * export_raytracing + low_resx1 * (1 - export_raytracing);
          jStp  = 1 * export_raytracing + low_resx2 * (1 - export_raytracing);
          kStp  = 1 * export_raytracing + low_resx3 * (1 - export_raytracing);
        }

        for (iGlob = iStrt; iGlob < iEnd; iGlob += iStp) {
          for (jGlob = jStrt; jGlob < jEnd; jGlob += jStp) {
            for (kGlob = kStrt; kGlob < kEnd; kGlob += kStp) {
              if (by_block) {
                read_data = 1;
                if (read_file == 0) {
                  read_file = 1;
                }
              } else {
                // if not by block, check data range
                iInRange = (iGlob / low_resx1 >= i_min && 
                iGlob / low_resx1 < i_max);
              
                jInRange = (jGlob / low_resx2 >= j_min && 
                jGlob / low_resx2 < j_max);
              
                kInRange = (kGlob / low_resx3 >= k_min && 
                kGlob / low_resx3 < k_max);

                if (iInRange && jInRange && kInRange) {
                  read_data = 1;
                  if (read_file == 0) {
                    read_file = 1;
                  }
                }
              }

              if (read_data) {
                if (by_block) {
                  // convert to 1D index to read in binary data
                  bs1LR = bs1 / low_resx1;
                  bs2LR = bs2 / low_resx2;
                  bs3LR = bs3 / low_resx3;

                  index_3D = ((n * bs1LR * bs2LR * bs3LR) + 
                              (iGlob * bs2LR * bs3LR) + 
                              (jGlob * bs3LR) + kGlob);
                  // gridsize is only modified if by block
                  gridsize_3D = nb * bs1LR * bs2LR * bs3LR;
                  
                  fseek(fin, 
                  ((iGlob * low_resx1) * bs2 * bs3 + 
                   (jGlob * low_resx2) * bs3 + 
                   (kGlob * low_resx3)) * nvar * dbl_size, SEEK_SET);
                  fread(&tempArray[0], dbl_size, nvar, fin);
                } else {
                  // indices on the given block
                  iLoc = (iGlob - (coordx1 * upscalex1 * bs1)) / upscalex1;            
                  jLoc = (jGlob - (coordx2 * upscalex2 * bs2)) / upscalex2;            
                  kLoc = (kGlob - (coordx3 * upscalex3 * bs3)) / upscalex3;

                  // convert to 1D index to read in binary data
                  index_3D = (((iGlob / low_resx1) - i_min) * isize +
                              ((jGlob / low_resx2) - j_min) * jsize +
                              ((kGlob / low_resx3) - k_min));
                                  
                  fseek(fin, ((iLoc * bs2 * bs3 + jLoc * bs3 + kLoc) * 
                  nvar * dbl_size), SEEK_SET);
                  fread(&tempArray[0], dbl_size, nvar, fin);
                } 
                // if export_raytracing = 1 then average by the volume
                // low_res is down-sampling (skipping data) NOT averaging
                // export_raytracing = 1: averaging while down-sampling
                if (export_raytracing) {
                  float volume = (float)(low_resx1 * low_resx2 * low_resx3);

                  x1[index_3D]    += tempArray[3] / volume;
                  x2[index_3D]    += tempArray[4] / volume;
                  x3[index_3D]    += tempArray[5] / volume;
                  r[index_3D]     += tempArray[6] / volume;
                  theta[index_3D] += tempArray[7] / volume;
                  phi[index_3D]   += tempArray[8] / volume;

                } else {
                  x1[index_3D]    = tempArray[3];
                  x2[index_3D]    = tempArray[4];
                  x3[index_3D]    = tempArray[5];
                  r[index_3D]     = tempArray[6];
                  theta[index_3D] = tempArray[7];
                  phi[index_3D]   = tempArray[8];
                }
                read_data = 0;
              } // read_data
            } // kGlob
          } // jGlob
        } // iGlob

        if (read_file) {
          if (axisym) {
            if (by_block) {
              iStrt = 0;
              jStrt = 0;
              
              iEnd  = bs1 / low_resx1;
              jEnd  = bs2 / low_resx2;
              
              iStp  = 1;
              jStp  = 1;
            } else {
              iStrt = coordx1 * (upscalex1 * bs1) + offsetx1;
              jStrt = coordx2 * (upscalex2 * bs2) + offsetx2;
            
              iEnd  = (coordx1 + 1) * (upscalex1 * bs1);
              jEnd  = (coordx2 + 1) * (upscalex2 * bs2);
            
              iStp  = low_resx1;
              jStp  = low_resx2;
            }

            for (iGlob = iStrt; iGlob < iEnd; iGlob += iStp) {
              for (jGlob = jStrt; jGlob < jEnd; jGlob += jStp) {
                if (by_block) {
                  read_data = 1;
                } else {
                  iInRange = (iGlob / low_resx1 >= i_min && 
                  iGlob / low_resx1 < i_max);
              
                  jInRange = (jGlob / low_resx2 >= j_min && 
                  jGlob / low_resx2 < j_max);

                  if (iInRange && jInRange) {
                    // convert to 1D index to read in binary data
                    index_2D = (
                      ((iGlob / low_resx1) - i_min) * (j_max - j_min) +
                      ((jGlob / low_resx2) - j_min));

                    // if you have an axisymmetric metric then take the
                    // coordinates from the highest refinement level
                    if (MAX(amr_levelx1[n_ord[n]],
                        MAX(amr_levelx2[n_ord[n]],
                            amr_levelx3[n_ord[n]])) >= level_array[index_2D]) {
                      read_data = 1;
                    }
                  }
                }

                if (read_data) {
                  if (by_block) {
                    // convert to 1D index to read in binary data
                    bs1LR = bs1 / low_resx1;
                    bs2LR = bs2 / low_resx2;
                    bs3LR = bs3 / low_resx3;

                    index_2D = ((n * bs1LR * bs2LR) + 
                                (iGlob * bs2LR) + jGlob);
                    // gridsize is only modified if by block
                    gridsize_2D = nb * bs1LR * bs2LR;

                    fseek(fin, 
                    (nvar * bs1 * bs2 * bs3 + 
                     (iGlob * low_resx1) * bs2 * (nmtc * 16 + 1) + 
                     (jGlob * low_resx2) * (nmtc * 16 + 1)) * dbl_size, 
                     SEEK_SET);
                    fread(&tempArray[0], dbl_size, (nmtc * 16 + 1), fin);
                  } else {                    
                    // indices on the given block
                    iLoc = (iGlob - (coordx1 * upscalex1 * bs1)) / upscalex1;
                    jLoc = (jGlob - (coordx2 * upscalex2 * bs2)) / upscalex2;

                    fseek(fin, 
                    (nvar * bs1 * bs2 * bs3 + 
                     iLoc * bs2 * (nmtc * 16 + 1) +
                     jLoc * (nmtc * 16 + 1)) * dbl_size, SEEK_SET);
                    fread(&tempArray[0], dbl_size, (nmtc * 16 + 1), fin);
                  }

                  for (ii = 0; ii < 16; ii++) {
                    gcov[ii * gridsize_2D + index_2D] = tempArray[ii];
                  }
                  for (ii = 0; ii < 16; ii++) {
                    gcon[ii * gridsize_2D + index_2D] = tempArray[16 + ii];
                  }
                  gdet[index_2D] = tempArray[32];
                  for (ii = 0; ii < 16; ii++) {
                    dxdxp[ii * gridsize_2D + index_2D] = 
                    tempArray[32 + 1 + ii];
                  }

                  if (!by_block) {
                    level_array[index_2D] = MAX(amr_levelx1[n_ord[n]], 
                                            MAX(amr_levelx2[n_ord[n]], 
                                                amr_levelx3[n_ord[n]]));
                  }
                  read_data = 0;
                } // read_data
              } // jGlob
            } // iGlob
          } else {
            if (by_block) {
              iStrt = 0;
              jStrt = 0;
              kStrt = 0;
              
              iEnd  = bs1 / low_resx1;
              jEnd  = bs2 / low_resx2;
              kEnd  = bs3 / low_resx3;
              
              iStp  = 1;
              jStp  = 1;
              kStp  = 1;
            } else {
              iStrt = coordx1 * (upscalex1 * bs1) + offsetx1;
              jStrt = coordx2 * (upscalex2 * bs2) + offsetx2;
              kStrt = coordx3 * (upscalex3 * bs3) + offsetx3;
            
              iEnd  = (coordx1 + 1) * (upscalex1 * bs1);
              jEnd  = (coordx2 + 1) * (upscalex2 * bs2);
              kEnd  = (coordx3 + 1) * (upscalex3 * bs3);
            
              iStp  = low_resx1;
              jStp  = low_resx2;
              kStp  = low_resx3;
            }

            for (iGlob = iStrt; iGlob < iEnd; iGlob += iStp) {
              for (jGlob = jStrt; jGlob < jEnd; jGlob += jStp) {
                for (kGlob = kStrt; kGlob < kEnd; kGlob += kStp) {
                  if (by_block) {
                    read_data = 1;
                  } else {
                    iInRange = (iGlob / low_resx1 >= i_min && 
                    iGlob / low_resx1 < i_max);
              
                    jInRange = (jGlob / low_resx2 >= j_min && 
                    jGlob / low_resx2 < j_max);
              
                    kInRange = (kGlob / low_resx3 >= k_min && 
                    kGlob / low_resx3 < k_max);

                    if (iInRange && jInRange && kInRange) {
                      read_data = 1;
                    }
                  }

                  if (read_data) {
                    if (by_block) {
                      // convert to 1D index to read in binary data
                      bs1LR = bs1 / low_resx1;
                      bs2LR = bs2 / low_resx2;
                      bs3LR = bs3 / low_resx3;

                      index_3D = ((n * bs1LR * bs2LR * bs3LR) + 
                                  (iGlob * bs2LR * bs3LR) + 
                                  (jGlob * bs3LR) + kGlob);
                      // gridsize is only modified if by block
                      gridsize_3D = nb * bs1LR * bs2LR * bs3LR;

                      fseek(fin, 
                      (nvar * bs1 * bs2 * bs3 + 
                       (iGlob * low_resx1) * bs2 * bs3 * (nmtc * 16 + 1) + 
                       (jGlob * low_resx2) * bs3 * (nmtc * 16 + 1) +
                       (kGlob * low_resx3) * (nmtc * 16 + 1)) * dbl_size, 
                       SEEK_SET);
                      fread(&tempArray[0], dbl_size, (nmtc * 16 + 1), fin);
                    } else {
                      // indices on the given block
                      iLoc = (
                        (iGlob - (coordx1 * upscalex1 * bs1)) / upscalex1);
                      jLoc = (
                        (jGlob - (coordx2 * upscalex2 * bs2)) / upscalex2);
                      kLoc = (
                        (kGlob - (coordx3 * upscalex3 * bs3)) / upscalex3);

                      // convert to 1D index to read in binary data
                      index_3D = (((iGlob / low_resx1) - i_min) * isize +
                                  ((jGlob / low_resx2) - j_min) * jsize +
                                  ((kGlob / low_resx3) - k_min));
                      fseek(fin, 
                      (nvar * bs1 * bs2 * bs3 +
                      iLoc * bs2 * bs3 * (nmtc * 16 + 1) +
                      jLoc * bs3 * (nmtc * 16 + 1) +
                      kLoc * (nmtc * 16 + 1)) * dbl_size, SEEK_SET);
                      fread(&tempArray[0], dbl_size, (nmtc * 16 + 1), fin);
                    }

                    for (ii = 0; ii < 16; ii++) {
                      gcov[ii * gridsize_3D + index_3D] = tempArray[ii];
                    }
                    for (ii = 0; ii < 16; ii++) {
                      gcon[ii * gridsize_3D + index_3D] = tempArray[16 + ii];
                    }
                    gdet[index_3D] = tempArray[32];
                    for (ii = 0; ii < 16; ii++) {
                      dxdxp[ii * gridsize_3D + index_3D] = 
                      tempArray[32 + 1 + ii];
                    }
                    read_data = 0;
                  } // read_data
                } // kGlob
              } // jGlob
            } // iGlob
          } // else axisymmetric
        } // if read_file    
      } else {
        std::cout << "Error: Possible data size error in file " <<
        filename << std::endl;
      }
      fclose(fin);
    } else {
      std::cout << "Error: Possible data corruption in file " << 
      filename << std::endl;
    } // if file exists
  } // loop over n
} // kernel_read_grid

void kernel_read_prims(std::string dir, int fnum, int by_block, 
  int export_raytracing,
  const std::unordered_map<std::string, double>& amr,
  const std::unordered_map<std::string, std::vector<int>>& amr_info,
  const std::unordered_map<std::string, double>& block_info,
  const std::unordered_map<std::string, double>& params,
  const std::unordered_map<std::string, int>& physics,
  const std::unordered_map<std::string, double>& options,
  float* rho, float* uGas, float* uu, float* B, float* E,
  float* uRad, float* uuRad, float* Ti, float* Te, float* phNum,
  float* Ye, float* xAlpha, float* xAtm, 
  float* uNu1, float* uNu2, float* uNu3, 
  float* uuNu1, float* uuNu2, float* uuNu3,
  float* nuNum1, float* nuNum2, float* nuNum3, float* tr1, float* tr2) {
  
  int isize, jsize, filesize, read_data;
  int p1, p2, p3, coordx1, coordx2, coordx3, offsetx1, offsetx2, offsetx3;
  int float_size = sizeof(float);

  // number of variables (nvar) and variable start index (vStrt)
  int nvar, vStrt;
  // nPrim: rho, uGas, uu (4-vector), B (3-vector)
  int nPrim = 9;
  // nRadM1: uRad, uuRad (4-vector)
  int nRadM1 = 5;
  // nRes: E (3-vector)
  int nRes = 3;
  // nTwoT: Ti, Te
  int nTwoT = 2;
  // nPnum: pNum
  int nPnum = 1;
  // nYe: Ye
  int nYe = 1;
  // nNUC: xAlpha, xAtm
  int nNUC = 2;
  // nNuM1: 3 species x (uNu, uuNu (4-vector), uNum)
  int nNuM1 = 18;
  // nTr: tr1, tr2
  int nTr = 2;

  int upscalex1, upscalex2, upscalex3;
  long int gridsize_3D, index_3D;
  int iGlob, jGlob, kGlob; // global indices
  int n, nStrt, nEnd, fileCount, iFile; 
  // start/end global indices and step in x1,x2,x3 directions
  int iStrt, jStrt, kStrt, iEnd, jEnd, kEnd, iStp, jStp, kStp;     
  int iLoc,  jLoc,  kLoc, ii; // local indices on given block
  int bs1LR, bs2LR, bs3LR; // bs1,2,3 / low_resx1,2,3
  bool iInRange, jInRange, kInRange;
  bool iNotInBlock, jNotInBlock, kNotInBlock;
  std::vector<int> blocksInFile;
  FILE* fin;
  std::string filename;

  // maximum refinement level
  int max_levelx1 = int(amr.at("max_levelx1"));
  int max_levelx2 = int(amr.at("max_levelx2"));
  int max_levelx3 = int(amr.at("max_levelx3"));

  // information needed for AMR
  std::vector<int> n_ord       = amr_info.at("n_ord");
  // refinement level of block
  std::vector<int> amr_levelx1 = amr_info.at("levelx1");
  std::vector<int> amr_levelx2 = amr_info.at("levelx2");
  std::vector<int> amr_levelx3 = amr_info.at("levelx3");
  // coordinate of block
  std::vector<int> amr_coordx1 = amr_info.at("coordx1");
  std::vector<int> amr_coordx2 = amr_info.at("coordx2");
  std::vector<int> amr_coordx3 = amr_info.at("coordx3");

  // block indices
  int i_min = int(block_info.at("i_min"));
  int i_max = int(block_info.at("i_max"));
  int j_min = int(block_info.at("j_min"));
  int j_max = int(block_info.at("j_max"));
  int k_min = int(block_info.at("k_min"));
  int k_max = int(block_info.at("k_max"));
  
  // number of total active blocks
  int nb    = int(params.at("n_active_total"));
  // block size
  int bs1   = int(params.at("bs1"));
  int bs2   = int(params.at("bs2"));
  int bs3   = int(params.at("bs3"));

  // downsampling factors
  int low_resx1 = int(options.at("low_resx1"));
  int low_resx2 = int(options.at("low_resx2"));
  int low_resx3 = int(options.at("low_resx3"));

  // physics flags
  int RADM1 = int(physics.at("RAD_M1"));
  int RES   = int(physics.at("RESISTIVE"));
  int TWOT  = int(physics.at("TWO_T"));
  int PNUM  = int(physics.at("P_NUM"));
  int DOYE  = int(physics.at("DO_YE"));
  int DONUC = int(physics.at("DO_NUCLEAR"));
  int NUM1  = int(physics.at("NEUTRINOS_M1"));
  int DOTR  = int(physics.at("DOTRACER"));

  nvar = nPrim + RADM1*nRadM1 + RES*nRes + TWOT*nTwoT + PNUM*nPnum + DOYE*nYe +
  DONUC*nNUC + NUM1*nNuM1 + DOTR*nTr;

  isize = (k_max - k_min) * (j_max - j_min);
  jsize = (k_max - k_min);

  // count the number of output files that need to be read (fileCount)
  // this corresponds to the number of GPUs used in computation
  blocksInFile.push_back(0.0);
  fileCount = 0;
  filename = dir + "/dumps" + std::to_string(fnum) + 
  "/new_dump" + std::to_string(fileCount);
  while (std::filesystem::exists(filename)) {
    filesize = std::filesystem::file_size(filename);
    if ((filesize % (nvar * bs1 * bs2 * bs3 * float_size)) != 0) {
      std::cout << "Error: Possible data size error in file " << 
      filename << std::endl;
    }
    blocksInFile.push_back(filesize / (nvar * bs1 * bs2 * bs3 * float_size));
    fileCount += 1;
    filename = dir + "/dumps" + std::to_string(fnum) + 
    "/new_dump" + std::to_string(fileCount);
  }

  #pragma omp parallel for private(iFile, n, nStrt, nEnd, vStrt, \
  p1, p2, p3, coordx1, coordx2, coordx3, \
  fin, filesize, read_data, upscalex1, upscalex2, upscalex3, \
  filename, iGlob, jGlob, kGlob, iLoc, jLoc, kLoc, \
  offsetx1, offsetx2, offsetx3, bs1LR, bs2LR, bs3LR, \
  iStrt, jStrt, kStrt, iEnd, jEnd, kEnd, iStp, jStp, kStp, \
  iInRange, jInRange, kInRange, ii, \
  iNotInBlock, jNotInBlock, kNotInBlock, \
  index_3D, gridsize_3D)
  for (iFile = 0; iFile < fileCount; iFile++) {
    gridsize_3D = (i_max - i_min) * (j_max - j_min) * (k_max - k_min);
    std::vector<float> tempArray(nvar);

    filename = dir + "/dumps" + std::to_string(fnum) + 
    "/new_dump" + std::to_string(iFile);
    
    if (std::filesystem::exists(filename)) {
      fin = fopen(filename.c_str(), "rb");
      if (omp_get_thread_num() == 0) {
        std::cout << "Loading " + filename + " on rank 0." << std::endl;
      }
      read_data = 0;

      nStrt = std::accumulate(blocksInFile.begin(), 
                              blocksInFile.begin() + iFile + 1, 0);
      nEnd  = nStrt + blocksInFile[iFile + 1];

      for (n = nStrt; n < nEnd; n++) {
        p1      = max_levelx1 - amr_levelx1[n_ord[n]];
        p2      = max_levelx2 - amr_levelx2[n_ord[n]];
        p3      = max_levelx3 - amr_levelx3[n_ord[n]];
        coordx1 = amr_coordx1[n_ord[n]];
        coordx2 = amr_coordx2[n_ord[n]];
        coordx3 = amr_coordx3[n_ord[n]];

        // either 2^n or 1 (not used if by_block == 1)
        upscalex1 = pow(2, p1); 
        upscalex2 = pow(2, p2);
        upscalex3 = pow(2, p3);
        
        // allow data to be read into 4D arrays (organized by block) 
        // or as a 3D unigrid array
        if (by_block) {
          iStrt = 0;
          jStrt = 0;
          kStrt = 0;

          iEnd  = bs1 / low_resx1;
          jEnd  = bs2 / low_resx2;
          kEnd  = bs3 / low_resx3;

          iStp  = 1;
          jStp  = 1;
          kStp  = 1;
        } else {
          // offsets are always 0 unless low_resx1 etc. not a power of 2
          offsetx1 = ( (low_resx1 - (coordx1 * upscalex1 * bs1) % low_resx1) % 
                        low_resx1 );
          offsetx2 = ( (low_resx2 - (coordx2 * upscalex2 * bs2) % low_resx2) %
                        low_resx2 );
          offsetx3 = ( (low_resx3 - (coordx3 * upscalex3 * bs3) % low_resx3) %
                        low_resx3 );

          iStrt = coordx1 * (upscalex1 * bs1) + offsetx1;
          jStrt = coordx2 * (upscalex2 * bs2) + offsetx2;
          kStrt = coordx3 * (upscalex3 * bs3) + offsetx3;

          iEnd  = (coordx1 + 1) * (upscalex1 * bs1);
          jEnd  = (coordx2 + 1) * (upscalex2 * bs2);
          kEnd  = (coordx3 + 1) * (upscalex3 * bs3);

          // if export_raytracing = 1: then you're averaging over the
          // blocks set by low_res 
          // if export_raytracing = 0: go up by low_res, which may be 1
          // the 1*flag + (1-flag) logic is good on GPUs since it avoids an "if" 
          iStp  = 1 * export_raytracing + low_resx1 * (1 - export_raytracing);
          jStp  = 1 * export_raytracing + low_resx2 * (1 - export_raytracing);
          kStp  = 1 * export_raytracing + low_resx3 * (1 - export_raytracing);
        }

        // do not loop through block if block not in data set
        // still loop through if low_res > bs
        iNotInBlock = iStrt / low_resx1 > i_max || iEnd / low_resx1 <= i_min;
        jNotInBlock = jStrt / low_resx2 > j_max || jEnd / low_resx2 <= j_min;
        kNotInBlock = kStrt / low_resx3 > k_max || kEnd / low_resx3 <= k_min;
       
        if ((iNotInBlock && low_resx1 < bs1) || 
            (jNotInBlock && low_resx2 < bs2) || 
            (kNotInBlock && low_resx3 < bs3)) {
          continue;
        } 
        
        for (iGlob = iStrt; iGlob < iEnd; iGlob += iStp) {
          for (jGlob = jStrt; jGlob < jEnd; jGlob += jStp) {
            for (kGlob = kStrt; kGlob < kEnd; kGlob += kStp) {
              if (by_block) {
                read_data = 1;
              } else {
                // if not by block, check data range
                iInRange = (iGlob / low_resx1 >= i_min && 
                iGlob / low_resx1 < i_max);
              
                jInRange = (jGlob / low_resx2 >= j_min && 
                jGlob / low_resx2 < j_max);
              
                kInRange = (kGlob / low_resx3 >= k_min && 
                kGlob / low_resx3 < k_max);

                if (iInRange && jInRange && kInRange) {
                  read_data = 1;
                }
              }
            
              if (read_data) {
                if (by_block) {
                  // convert to 1D index to read in binary data
                  bs1LR = bs1 / low_resx1;
                  bs2LR = bs2 / low_resx2;
                  bs3LR = bs3 / low_resx3;

                  index_3D = ((n * bs1LR * bs2LR * bs3LR) + 
                              (iGlob * bs2LR * bs3LR) + 
                              (jGlob * bs3LR) + kGlob);
                  // gridsize is only modified if by block
                  gridsize_3D = nb * bs1LR * bs2LR * bs3LR;
                  
                  fseek(fin, 
                  (((n - nStrt) * bs1 * bs2 * bs3) +
                   (iGlob * low_resx1) * bs2 * bs3 + 
                   (jGlob * low_resx2) * bs3 + 
                   (kGlob * low_resx3)) * nvar * float_size, SEEK_SET);
                  fread(&tempArray[0], float_size, nvar, fin);

                } else {
                  // indices on the given block
                  iLoc = (iGlob - (coordx1 * upscalex1 * bs1)) / upscalex1;            
                  jLoc = (jGlob - (coordx2 * upscalex2 * bs2)) / upscalex2;            
                  kLoc = (kGlob - (coordx3 * upscalex3 * bs3)) / upscalex3;

                  // convert to 1D index to read in binary data
                  index_3D = (((iGlob / low_resx1) - i_min) * isize +
                              ((jGlob / low_resx2) - j_min) * jsize +
                              ((kGlob / low_resx3) - k_min));

                  fseek(fin, ((n - nStrt) * bs1 * bs2 * bs3 + 
                  (iLoc * bs2 * bs3) + 
                  (jLoc * bs3) + kLoc) * nvar * float_size, SEEK_SET);
                  fread(&tempArray[0], float_size, nvar, fin);
                }
              
                if (export_raytracing) {
                  rho[index_3D]  = tempArray[0];
                  uGas[index_3D] = tempArray[1];
                  for (ii = 0; ii < 4; ii++) {
                    uu[ii * gridsize_3D + index_3D] = tempArray[2 + ii];
                  } 
                  for (ii = 0; ii < 3; ii++) {
                    B[(ii + 1) * gridsize_3D + index_3D] = tempArray[6 + ii];
                  }
                  
                  if (RADM1) {
                    uRad[index_3D] = tempArray[nPrim];
                    for(ii = 0; ii < 4; ii++) {
                      uuRad[ii * gridsize_3D + index_3D] = 
                      tempArray[nPrim + 1 + ii];
                    }
                  }
                  
                  if (RES) {
                    vStrt = nPrim + RADM1 * nRadM1;
                    for (ii = 0; ii < 3; ii++) {
                      E[ii * gridsize_3D + index_3D] = tempArray[vStrt + ii];
                    }
                  }

                  if (TWOT) {
                    vStrt = nPrim + RADM1 * nRadM1 + RES * nRes;
                    Te[index_3D] = tempArray[vStrt];
                    Ti[index_3D] = tempArray[vStrt + 1];
                  }

                  if (PNUM) {
                    vStrt = nPrim + RADM1 * nRadM1 + RES * nRes + TWOT * nTwoT;
                    phNum[index_3D] = tempArray[vStrt];
                  }
                  
                  if (DOYE) {
                    vStrt = nPrim + RADM1 * nRadM1 + RES * nRes + 
                    TWOT * nTwoT + PNUM * nPnum;
                    Ye[index_3D] = tempArray[vStrt];
                  }

                  if (DONUC) {
                    vStrt = nPrim + RADM1 * nRadM1 + RES * nRes + 
                    TWOT * nTwoT + PNUM * nPnum + DOYE * nYe;
                    xAlpha[index_3D] = tempArray[vStrt];
                    xAtm[index_3D]   = tempArray[vStrt + 1];
                  }

                  if (NUM1) {
                    vStrt = nPrim + RADM1 * nRadM1 + RES * nRes + 
                    TWOT * nTwoT + PNUM * nPnum + DOYE * nYe + 
                    DONUC * nNUC;
                    
                    uNu1[index_3D] = tempArray[vStrt];
                    for (ii = 0; ii < 4; ii++) {
                      uuNu1[ii * gridsize_3D + index_3D] = 
                      tempArray[vStrt + 1 + ii];
                    }
                    nuNum1[index_3D] = tempArray[vStrt + 5];
                    
                    uNu2[index_3D] = tempArray[vStrt + 6];
                    for (ii = 0; ii < 4; ii++) {
                      uuNu2[ii * gridsize_3D + index_3D] = 
                      tempArray[vStrt + 7 + ii];
                    }
                    nuNum2[index_3D] = tempArray[vStrt + 11];
                    
                    uNu3[index_3D] = tempArray[vStrt + 12];
                    for (ii = 0; ii < 4; ii++) {
                      uuNu3[ii * gridsize_3D + index_3D] = 
                      tempArray[vStrt + 13 + ii];
                    }
                    nuNum3[index_3D] = tempArray[vStrt + 17];
                  }

                  if (DOTR) {
                    vStrt = nPrim + RADM1 * nRadM1 + RES * nRes + 
                    TWOT * nTwoT + PNUM * nPnum + DOYE * nYe + 
                    DONUC * nNUC + NUM1 * nNuM1;
                    tr1[index_3D] = tempArray[vStrt];
                    tr2[index_3D] = tempArray[vStrt + 1];
                  }
                } else {
                  rho[index_3D]  = tempArray[0];
                  uGas[index_3D] = tempArray[1];
                  for (ii = 0; ii < 4; ii++) {
                    uu[ii * gridsize_3D + index_3D] = tempArray[2 + ii];
                  } 
                  for (ii = 0; ii < 3; ii++) {
                    B[(ii + 1) * gridsize_3D + index_3D] = tempArray[6 + ii];
                  }
                  
                  if (RADM1) {
                    uRad[index_3D] = tempArray[nPrim];
                    for(ii = 0; ii < 4; ii++) {
                      uuRad[ii * gridsize_3D + index_3D] = 
                      tempArray[nPrim + 1 + ii];
                    }
                  }
                  
                  if (RES) {
                    vStrt = nPrim + RADM1 * nRadM1;
                    for (ii = 0; ii < 3; ii++) {
                      E[ii * gridsize_3D + index_3D] = tempArray[vStrt + ii];
                    }
                  }

                  if (TWOT) {
                    vStrt = nPrim + RADM1 * nRadM1 + RES * nRes;
                    Te[index_3D] = tempArray[vStrt];
                    Ti[index_3D] = tempArray[vStrt + 1];
                  }

                  if (PNUM) {
                    vStrt = nPrim + RADM1 * nRadM1 + RES * nRes + TWOT * nTwoT;
                    phNum[index_3D] = tempArray[vStrt];
                  }
                  
                  if (DOYE) {
                    vStrt = nPrim + RADM1 * nRadM1 + RES * nRes + 
                    TWOT * nTwoT + PNUM * nPnum;
                    Ye[index_3D] = tempArray[vStrt];
                  }

                  if (DONUC) {
                    vStrt = nPrim + RADM1 * nRadM1 + RES * nRes + 
                    TWOT * nTwoT + PNUM * nPnum + DOYE * nYe;
                    xAlpha[index_3D] = tempArray[vStrt];
                    xAtm[index_3D]   = tempArray[vStrt + 1];
                  }

                  if (NUM1) {
                    vStrt = nPrim + RADM1 * nRadM1 + RES * nRes + 
                    TWOT * nTwoT + PNUM * nPnum + DOYE * nYe + 
                    DONUC * nNUC;
                    
                    uNu1[index_3D] = tempArray[vStrt];
                    for (ii = 0; ii < 4; ii++) {
                      uuNu1[ii * gridsize_3D + index_3D] = 
                      tempArray[vStrt + 1 + ii];
                    }
                    nuNum1[index_3D] = tempArray[vStrt + 5];
                    
                    uNu2[index_3D] = tempArray[vStrt + 6];
                    for (ii = 0; ii < 4; ii++) {
                      uuNu2[ii * gridsize_3D + index_3D] = 
                      tempArray[vStrt + 7 + ii];
                    }
                    nuNum2[index_3D] = tempArray[vStrt + 11];
                    
                    uNu3[index_3D] = tempArray[vStrt + 12];
                    for (ii = 0; ii < 4; ii++) {
                      uuNu3[ii * gridsize_3D + index_3D] = 
                      tempArray[vStrt + 13 + ii];
                    }
                    nuNum3[index_3D] = tempArray[vStrt + 17];
                  }

                  if (DOTR) {
                    vStrt = nPrim + RADM1 * nRadM1 + RES * nRes + 
                    TWOT * nTwoT + PNUM * nPnum + DOYE * nYe + 
                    DONUC * nNUC + NUM1 * nNuM1;
                    tr1[index_3D] = tempArray[vStrt];
                    tr2[index_3D] = tempArray[vStrt + 1];
                  }
                } // if export_ray_tracing
                read_data = 0;
              } // read_data
            } // kGlob
          } // jGlob
        } // iGlob 
      } // loop over n 
      fclose(fin); 
    } else {
      std::cout << "Error: Possible data corruption in file " << 
      filename << std::endl;
    } // if file exists
  } // parallel loop over number of files (GPUs)
}
 
void invert_4x4_cpp(std::vector<double> a, std::vector<double> b) {
/*
    Description: 

    Input 4x4 flattened matrix a; then, analytically do the inversion.
    Output b, the flattened + inverted 4x4 matrix.
    This assumes the matrix is invertile (det(a) != 0).

    Method is 'Laplace Expansion Theorem', 
    i.e. https://www.geometrictools.com/Documentation/LaplaceExpansionTheorem.pdf

*/

  double s0, s1, s2, s3, s4, s5;
  double c0, c1, c2, c3, c4, c5;
  double det_a;

  // Calculate determinants of submatrices
  s0 = a[0] * a[5] - a[4] * a[1];
  s1 = a[0] * a[6] - a[4] * a[2];
  s2 = a[0] * a[7] - a[4] * a[3];
  s3 = a[1] * a[6] - a[5] * a[2];
  s4 = a[1] * a[7] - a[5] * a[3];
  s5 = a[2] * a[7] - a[6] * a[3];

  c5 = a[10]*a[15] - a[14]*a[11];
  c4 = a[9]*a[15] - a[13]*a[11];
  c3 = a[9]*a[14] - a[13]*a[10];
  c2 = a[8]*a[15] - a[12]*a[11];
  c1 = a[8]*a[14] - a[12]*a[10];
  c0 = a[8]*a[13] - a[12]*a[9];

  // Calculate determinant of matrix a from determinants of its submatrices
  // Should include error check if det_a = 0
  det_a = s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0;

  // If det(a) != 0, then a^{-1} = adj(a)/det(a)
  // Calculate b = inv(a) = adj(a)/det(a)
  b[0] = ( a[5] * c5 - a[6] * c4 + a[7] * c3 ) / det_a;
  b[1] = (-a[1] * c5 + a[2] * c4 - a[3] * c3 ) / det_a;
  b[2] = (a[13] * s5 - a[14]* s4 + a[15]* s3 ) / det_a;
  b[3] = (-a[9] * s5 + a[10]* s4 - a[11]* s3 ) / det_a;
  b[4] = (-a[4] * c5 + a[6] * c2 - a[7] * c1 ) / det_a;
  b[5] = ( a[0] * c5 - a[2] * c2 + a[3] * c1 ) / det_a;
  b[6] = (-a[12]* s5 + a[14]* s2 - a[15]* s1 ) / det_a;
  b[7] = ( a[8] * s5 - a[10]* s2 + a[11]* s1 ) / det_a;
  b[8] = ( a[4] * c4 - a[5] * c2 + a[7] * c0 ) / det_a;
  b[9] = (-a[0] * c4 + a[1] * c2 - a[3] * c0 ) / det_a;
  b[10] = (a[12]* s4 - a[13]* s2 + a[15]* s0 ) / det_a;
  b[11] = (-a[8]* s4 + a[9] * s2 - a[11]* s0 ) / det_a;
  b[12] = (-a[4]* c3 + a[5] * c1 - a[6] * c0 ) / det_a;
  b[13] = ( a[0]* c3 - a[1] * c1 + a[2] * c0 ) / det_a;
  b[14] = (-a[12]*s3 + a[13]* s1 - a[14]* s0 ) / det_a;
  b[15] = (a[8] * s3 - a[9] * s1 + a[10]* s0 ) / det_a;

} // end invert_4x4()

void kernel_invert_4x4(float* A, float* B, int nb, int nx1, int nx2, int nx3) {
/*
    Description:
    
    Accepts array A that holds nb*bs1*bs2*bs3 4x4 matrices. Then, loops
    through each nb*bs1*bs2*bs3 index, gets 4x4 matrix at this coordinate,
    inverts it, and fills it into the array B

*/
  
  int n, i, j, k;
  int index_3D, gridsize_3D;
  int i1, j1;
  
  // Loop over each of nb * nx1 * nx2 * nx3 cells
  gridsize_3D = nb * nx1 * nx2 * nx3;
  for (n = 0; n < nb; n++) {
    #pragma omp parallel for private(i, j, k, index_3D, i1, j1)
    for (i = 0; i < nx1; i++) {
      // Flattened matrices a and b (inverse of a)
      std::vector<double> a_tmp(16), b_tmp(16);
      for (j = 0; j < nx2; j++) {
        for (k = 0; k < nx3; k++) {
          // Get index_3D and gridsize_3D for global indexing in A, B arrays
          index_3D = n * nx1 * nx2 * nx3 + i * nx3 * nx2 + j * nx3 + k;
          // First get flattened 4x4 matrix from A at this (n,i,j,k) cell
          for (i1 = 0; i1 < 4; i1++) {
            for (j1 = 0; j1 < 4; j1++) {
              a_tmp[i1 * 4 + j1] = A[(i1 * 4 + j1) * gridsize_3D + index_3D];
            }
          }
          // Then, invert this 4x4 matrix and load it into b_tmp
          invert_4x4_cpp(a_tmp, b_tmp);
          // Now take inverted 4x4 matrix, b_tmp, and put in B at this (n,i,j,k) cell
          for (i1 = 0; i1 < 4; i1++) { 
            for (j1 = 0; j1 < 4; j1++) {
              B[(i1 * 4 + j1) * gridsize_3D + index_3D] = b_tmp[i1 * 4 + j1];
            }
          }
        } // k
      } // j
    } // i
  } // n
} // end kernel_invert_4x4()

double det_4x4_cpp(const std::vector<double> a) {
  double s0, s1, s2, s3, s4, s5;
  double c0, c1, c2, c3, c4, c5;
  double det_a;

  // Calculate determinants of submatrices
  s0 = a[0] * a[5] - a[4] * a[1];
  s1 = a[0] * a[6] - a[4] * a[2];
  s2 = a[0] * a[7] - a[4] * a[3];
  s3 = a[1] * a[6] - a[5] * a[2];
  s4 = a[1] * a[7] - a[5] * a[3];
  s5 = a[2] * a[7] - a[6] * a[3];

  c5 = a[10]*a[15] - a[14]*a[11];
  c4 = a[9]*a[15] - a[13]*a[11];
  c3 = a[9]*a[14] - a[13]*a[10];
  c2 = a[8]*a[15] - a[12]*a[11];
  c1 = a[8]*a[14] - a[12]*a[10];
  c0 = a[8]*a[13] - a[12]*a[9];

  // Calculate determinant of matrix a from determinants of its submatrices
  // Should include error check if det_a = 0
  det_a = s0*c5 - s1*c4 + s2*c3 + s3*c2 - s4*c1 + s5*c0;

  return det_a;
}

void kernel_det_4x4(float* A, float* B, int nb, int nx1, int nx2, int nx3) {
  int n, i, j, k;
  int index_3D, gridsize_3D;
  int i1, j1;
  
  // Loop over each of nb * nx1 * nx2 * nx3 cells
  gridsize_3D = nb * nx1 * nx2 * nx3;
  for (n = 0; n < nb; n++) {
    #pragma omp parallel for private(i, j, k, index_3D, i1, j1)
    for (i = 0; i < nx1; i++) {
      // Flattened matrix a
      std::vector<double> a_tmp(16);
      for (j = 0; j < nx2; j++) {
        for (k = 0; k < nx3; k++) {
          // Get index_3D and gridsize_3D for global indexing in A, B arrays
          index_3D = n * nx1 * nx2 * nx3 + i * nx3 * nx2 + j * nx3 + k;
          // First get flattened 4x4 matrix from A at this (n,i,j,k) cell
          for (i1 = 0; i1 < 4; i1++) {
            for (j1 = 0; j1 < 4; j1++) {
              a_tmp[i1 * 4 + j1] = A[(i1 * 4 + j1) * gridsize_3D + index_3D];
            }
          }
          // Now take determinant of flattened matrix
          // and place in B at this (n,i,j,k) cell
          B[index_3D] = det_4x4_cpp(a_tmp);
        } // k
      } // j
    } // i
  } // n
} // end kernel_det_4x4()

int main() {
  std::cout << "Executing hamr_read.cpp" << std::endl;
  return 0;
}
