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

Read::Read() {};

Read::Read(std::string dir, int axisym, int by_block, int export_raytracing,
    const std::unordered_map<std::string, double>& amr,
    const std::unordered_map<std::string, std::vector<int>>& amr_info,
    const std::unordered_map<std::string, double>& block_info,
    const std::unordered_map<std::string, double>& options,
    const std::unordered_map<std::string, double>& params,
    const std::unordered_map<std::string, int>& physics) {
  this->dir               = dir;
  this->axisym            = axisym;
  this->by_block          = by_block;
  this->export_raytracing = export_raytracing;
  dbl_size = sizeof(double);
  float_size = sizeof(float);
  // nPrim: rho, uGas, uu (4-vector), B (3-vector)
  nPrim = 9;
  // nRadM1: uRad, uuRad (4-vector)
  nRadM1 = 5;
  // nRes: E (3-vector)
  nRes = 3;
  // nTwoT: Te, Ti
  nTwoT = 2;
  // nPnum: pNum
  nPnum = 1;
  // nYe: Ye
  nYe = 1;
  // nNUC: xAlpha, xAtm
  nNUC = 2;
  // nNuM1: 3 species x (uNu, uuNu (4-vector), uNum)
  nNuM1 = 18;
  // nTr: tr1, tr2
  nTr = 2; 
  // nDiss: diss, radSrc
  // Note that nvar is set by DISSTRK * nDiss and
  // DISSTRK = 1 or 2
  nDiss = 1; 
  
  // maximum refinement level
  maxLevel.x1 = int(amr.at("max_levelx1"));
  maxLevel.x2 = int(amr.at("max_levelx2"));
  maxLevel.x3 = int(amr.at("max_levelx3"));

  // information needed for AMR
  n_ord       = amr_info.at("n_ord");
      
  // refinement level of block
  amrLevel.x1 = amr_info.at("levelx1");
  amrLevel.x2 = amr_info.at("levelx2");
  amrLevel.x3 = amr_info.at("levelx3");
      
  // coordinate of block
  amrCoord.x1 = amr_info.at("coordx1");
  amrCoord.x2 = amr_info.at("coordx2");
  amrCoord.x3 = amr_info.at("coordx3");

  // block indices
  i_min = int(block_info.at("i_min"));
  i_max = int(block_info.at("i_max"));
  j_min = int(block_info.at("j_min"));
  j_max = int(block_info.at("j_max"));
  k_min = int(block_info.at("k_min"));
  k_max = int(block_info.at("k_max"));

  // downsampling factors
  lowRes.x1 = int(options.at("low_resx1"));
  lowRes.x2 = int(options.at("low_resx2"));
  lowRes.x3 = int(options.at("low_resx3"));

  // number of total active blocks
  nb  = int(params.at("n_active_total"));
      
  // block size
  bs1 = int(params.at("bs1"));
  bs2 = int(params.at("bs2"));
  bs3 = int(params.at("bs3"));

  // start/end blocks to select specific blocks
  blkStrt = int(block_info.at("start_block"));
  blkEnd  = int(block_info.at("end_block"));

  // physics flags
  RADM1   = int(physics.at("RAD_M1"));
  RES     = int(physics.at("RESISTIVE"));
  TWOT    = int(physics.at("TWO_T"));
  PNUM    = int(physics.at("P_NUM"));
  DOYE    = int(physics.at("DO_YE"));
  DONUC   = int(physics.at("DO_NUCLEAR"));
  NUM1    = int(physics.at("NEUTRINOS_M1"));
  DOTR    = int(physics.at("DOTRACER"));
  DISSTRK = int(physics.at("DUMP_DISS"));
};

void Read::readGrid(float* x1, float* x2, float* x3, 
    float* r, float* theta, float* phi, 
    float* gcov, float* gcon, float* gdet, float* dxdxp) {
  
  // variables (nvar): x1, x2, x3, 
  // r, theta, phi, i (global), j (global), k (global)
  // metrics (nmtc): gcov, gcon, 
  // dxdxp (Jacobian from x1, x2, x3 -> r, theta, phi)
  nvar = 9; 
  nmtc = 3;
      
  isize = (k_max - k_min) * (j_max - j_min);
  jsize = (k_max - k_min);

  ReadIn readIn; 
  
  std::cout << "Loading grid" << std::endl;

  #pragma omp parallel firstprivate(readIn)
  {
    long int gridsize_2D, gridsize_3D, index_2D, index_3D;
    int n; // index over blocks
    int iGlob, jGlob, kGlob; // global indices
    int ii; // index over metric components
    AMRVar iStrt, iEnd, iStp; // loop variables
    bool iInRange, jInRange, kInRange;
    FILE* fin;
    int read_data, read_file;
    std::string filename;

    gridsize_2D = (i_max - i_min) * (j_max - j_min);
    gridsize_3D = (i_max - i_min) * (j_max - j_min) * (k_max - k_min);

    double* tempArray = new double[58];
    int* levelArray = new int[gridsize_2D];
    #pragma omp for
    for (n = 0; n < nb; n++) {
      iInRange = false;
      jInRange = false;
      kInRange = false;
    
      filename = dir + "/gdumps/gdump" + std::to_string(n_ord[n]);

      checkFileExists(filename);
      checkGridFileSize(filename);
      fin = fopen(filename.c_str(), "rb");
    
      read_file = 0;
      read_data = 0;
   
      readIn.setLoops3D(n, iStrt, iEnd, iStp, *this);
    
      for (iGlob = iStrt.x1; iGlob < iEnd.x1; iGlob += iStp.x1) {
        for (jGlob = iStrt.x2; jGlob < iEnd.x2; jGlob += iStp.x2) {
          for (kGlob = iStrt.x3; kGlob < iEnd.x3; kGlob += iStp.x3) {
            read_data = checkIndexInRange(iGlob, jGlob, kGlob, 
              iInRange, jInRange, kInRange);
            if (read_file == 0 && read_data == 1) {
              read_file = 1;
            }
            if (read_data) {
              readIn.readGridData(fin, tempArray, index_3D, gridsize_3D,
                n, iGlob, jGlob, kGlob, *this);

              // if export_raytracing = 1 then average by the volume
              // low_res is down-sampling (skipping data) NOT averaging
              // export_raytracing = 1: averaging while down-sampling
              if (export_raytracing) {
                float volume = (float)(lowRes.x1 * lowRes.x2 * lowRes.x3);

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
            }
            read_data = 0;
          } // kGlob
        } // jGlob
      } // iGlob

      if (read_file) {
        if (axisym) {
          readIn.setMetricLoops2D(iStrt, iEnd, iStp, *this);

          for (iGlob = iStrt.x1; iGlob < iEnd.x1; iGlob += iStp.x1) {
            for (jGlob = iStrt.x2; jGlob < iEnd.x2; jGlob += iStp.x2) {
              read_data = checkIndexInRange(n, iGlob, jGlob, levelArray, 
                index_2D, iInRange, jInRange);
              if (read_data) {
                readIn.readMetricData2D(fin, tempArray, index_2D, gridsize_2D, 
                  n, iGlob, jGlob, *this);
                for (ii = 0; ii < 16; ii++) {
                  gcov[ii * gridsize_2D + index_2D] = tempArray[ii];
                }
                for (ii = 0; ii < 16; ii++) {
                  gcon[ii * gridsize_2D + index_2D] = tempArray[16 + ii];
                }
                gdet[index_2D] = tempArray[32];
                for (ii = 0; ii < 16; ii++) {
                  dxdxp[ii * gridsize_2D + index_2D] = tempArray[32 + 1 + ii];
                }
                if (!by_block) {
                  levelArray[index_2D] = MAX(amrLevel.x1[n_ord[n]], 
                                         MAX(amrLevel.x2[n_ord[n]], 
                                             amrLevel.x3[n_ord[n]]));
                }
                read_data = 0;
              } // read_data
            }  // jGlob
          } // iGlob
        } else {
          readIn.setMetricLoops2D(iStrt, iEnd, iStp, *this);

          for (iGlob = iStrt.x1; iGlob < iEnd.x1; iGlob += iStp.x1) {
            for (jGlob = iStrt.x2; jGlob < iEnd.x2; jGlob += iStp.x2) {
              for (kGlob = iStrt.x3; kGlob < iEnd.x3; kGlob += iStp.x3) {
                read_data = checkIndexInRange(iGlob, jGlob, kGlob, 
                  iInRange, jInRange, kInRange);
                if (read_data) {
                  readIn.readMetricData3D(fin, tempArray, index_3D, 
                    gridsize_3D, n, iGlob, jGlob, kGlob, *this);
                  for (ii = 0; ii < 16; ii++) {
                    gcov[ii * gridsize_3D + index_3D] = tempArray[ii];
                  }
                  for (ii = 0; ii < 16; ii++) {
                    gcon[ii * gridsize_3D + index_3D] = tempArray[16 + ii];
                  }
                  gdet[index_3D] = tempArray[32];
                  for (ii = 0; ii < 16; ii++) {
                    dxdxp[ii * gridsize_3D + index_3D] = tempArray[32+1 + ii];
                  }
                  read_data = 0;
                } // read_data
              } // kGlob
            } // jGlob
          } // iGlob
        } // else axisymmetric
      } // if read_file
      fclose(fin);
    } // loop over n
    delete[] tempArray;
    delete[] levelArray;
  }
}

void Read::readGridCoords(float* x1, float* x2, float* x3, 
  float* r, float* theta, float* phi) {

  // variables (nvar): x1, x2, x3, 
  // r, theta, phi, i (global), j (global), k (global)
  // metrics (nmtc): gcov, gcon, 
  // dxdxp (Jacobian from x1, x2, x3 -> r, theta, phi)
  nvar = 9; 
  nmtc = 3;
    
  isize = (k_max - k_min) * (j_max - j_min);
  jsize = (k_max - k_min);

  ReadIn readIn; 
  
  std::cout << "Loading grid coordinates" << std::endl;

  #pragma omp parallel firstprivate(readIn)
  {
    long int gridsize_3D, index_3D;
    int n; // index over blocks
    int iGlob, jGlob, kGlob; // global indices
    AMRVar iStrt, iEnd, iStp; // loop variables
    bool iInRange, jInRange, kInRange;
    FILE* fin;
    int read_data;
    std::string filename;

    gridsize_3D = (i_max - i_min) * (j_max - j_min) * (k_max - k_min);

    double* tempArray = new double[58];

    #pragma omp for
    for (n = 0; n < nb; n++) {
      iInRange = false;
      jInRange = false;
      kInRange = false;
  
      filename = dir + "/gdumps/gdump" + std::to_string(n_ord[n]);

      checkFileExists(filename);
      checkGridFileSize(filename);
      fin = fopen(filename.c_str(), "rb");
  
      read_data = 0;
 
      readIn.setLoops3D(n, iStrt, iEnd, iStp, *this);
  
      for (iGlob = iStrt.x1; iGlob < iEnd.x1; iGlob += iStp.x1) {
        for (jGlob = iStrt.x2; jGlob < iEnd.x2; jGlob += iStp.x2) {
          for (kGlob = iStrt.x3; kGlob < iEnd.x3; kGlob += iStp.x3) {
            read_data = checkIndexInRange(iGlob, jGlob, kGlob, 
              iInRange, jInRange, kInRange);
            if (read_data) {
              readIn.readGridData(fin, tempArray, index_3D, gridsize_3D,
                n, iGlob, jGlob, kGlob, *this);

              // if export_raytracing = 1 then average by the volume
              // low_res is down-sampling (skipping data) NOT averaging
              // export_raytracing = 1: averaging while down-sampling
              if (export_raytracing) {
                float volume = (float)(lowRes.x1 * lowRes.x2 * lowRes.x3);

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
            }
            read_data = 0;
          } // kGlob
        } // jGlob
      } // iGlob
      fclose(fin);
    } // loop over n
    delete[] tempArray;
  }
}

void Read::readGridResolution(int* res_level_x1, int* res_level_x2, 
    int* res_level_x3) {
    
  isize = (k_max - k_min) * (j_max - j_min);
  jsize = (k_max - k_min);

  long int gridsize_3D, index_3D;
  int n; // index over blocks
  int iGlob, jGlob, kGlob; // global indices
  AMRVar iStrt, iEnd, iStp; // loop variables
  bool iInRange, jInRange, kInRange;
  int read_data;
  ReadIn readIn; 

  #pragma omp parallel for private(n, iGlob, jGlob, kGlob, \
    iStrt, iEnd, iStp, index_3D, gridsize_3D, iInRange, jInRange, kInRange, \
    read_data) firstprivate(readIn)
  for (n = 0; n < nb; n++) {
    gridsize_3D = (i_max - i_min) * (j_max - j_min) * (k_max - k_min);

    iInRange = false;
    jInRange = false;
    kInRange = false;
  
    read_data = 0;
 
    readIn.setLoops3D(n, iStrt, iEnd, iStp, *this);
  
    for (iGlob = iStrt.x1; iGlob < iEnd.x1; iGlob += iStp.x1) {
      for (jGlob = iStrt.x2; jGlob < iEnd.x2; jGlob += iStp.x2) {
        for (kGlob = iStrt.x3; kGlob < iEnd.x3; kGlob += iStp.x3) {
          read_data = checkIndexInRange(iGlob, jGlob, kGlob, 
            iInRange, jInRange, kInRange);
          if (read_data) {
            readIn.getIndex3D(index_3D, gridsize_3D, n, 
              iGlob, jGlob, kGlob, *this);
            res_level_x1[index_3D] = amrLevel.x1[n_ord[n]];
            res_level_x2[index_3D] = amrLevel.x2[n_ord[n]];
            res_level_x3[index_3D] = amrLevel.x3[n_ord[n]];
          }
          read_data = 0;
        } // kGlob
      } // jGlob
    } // iGlob
  } // loop over n
}

void Read::readPrims(int fnum, float* rho, float* uGas, float* uu, float* B, 
    float* E, float* uRad, float* uuRad, 
    float* Te, float* Ti, float* phNum,
    float* Ye, float* xAlpha, float* xAtm, 
    float* uNu1, float* uNu2, float* uNu3, 
    float* uuNu1, float* uuNu2, float* uuNu3,
    float* nuNum1, float* nuNum2, float* nuNum3, 
    float* tr1, float* tr2, float* diss, float* radSrc) {
  
  nvar = nPrim + RADM1 * nRadM1 + RES * nRes + TWOT * nTwoT + 
    PNUM * nPnum + DOYE * nYe + DONUC * nNUC + NUM1 * nNuM1 + 
    DOTR * nTr   + DISSTRK * nDiss;

  isize = (k_max - k_min) * (j_max - j_min);
  jsize = (k_max - k_min);

  std::string filename;
  ReadIn readIn; 

  std::cout << "Loading primitive variables for " << 
  dir + "/dumps" + std::to_string(fnum) << std::endl;

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
    blocksInFile.push_back(filesize / (
      nvar * bs1 * bs2 * bs3 * float_size));
    fileCount += 1;
    filename = dir + "/dumps" + std::to_string(fnum) + 
    "/new_dump" + std::to_string(fileCount);
  }

  // check if the number of blocks in all files equals nb
  nbInFiles = std::accumulate(blocksInFile.begin(),
                              blocksInFile.begin() + fileCount + 1, 0);
  if (nbInFiles != nb) {
    std::cout << "Error: number of total blocks != nb" << std::endl;
    std::cout << "Number of blocks: " << nbInFiles << std::endl;
    std::cout << "Expected number of blocks: " << nb << std::endl;
  }

  #pragma omp parallel firstprivate(readIn)
  {
    long int gridsize_3D, index_3D;
    int n, nStrt, nEnd; // variables for looping over blocks
    int iFile, vStrt, ii; // file index and variable start index (vStrt)
    int iGlob, jGlob, kGlob; // global indices
    AMRVar iStrt, iEnd, iStp; // loop variables
    bool iInRange, jInRange, kInRange;
    BoolVar iNotInBlock; // is data in selection box/on block?
    FILE* fin;
    int read_data;
    std::string local_filename;
    float* tempArray = new float[nvar];
    #pragma omp for
    for (iFile = 0; iFile < fileCount; iFile++) {
      gridsize_3D = (i_max - i_min) * (j_max - j_min) * (k_max - k_min);

      iInRange = false;
      jInRange = false;
      kInRange = false;

      local_filename = dir + "/dumps" + std::to_string(fnum) + 
      "/new_dump" + std::to_string(iFile);
      
      checkFileExists(local_filename);
      fin = fopen(local_filename.c_str(), "rb");

      read_data = 0;
        
      nStrt = std::accumulate(blocksInFile.begin(), 
        blocksInFile.begin() + iFile + 1, 0);
      nEnd  = nStrt + blocksInFile[iFile + 1];

      nStrt = std::max(nStrt, blkStrt);
      nEnd  = std::min(nEnd,  blkEnd);

      for (n = nStrt; n < nEnd; n++) {
        readIn.setLoops3D(n, iStrt, iEnd, iStp, *this);

        // do not loop through block if block not in data set
        // still loop through if low_res > bs
        if (checkNotInBlock(iStrt, iEnd, iNotInBlock)) {
          continue;
        }

        for (iGlob = iStrt.x1; iGlob < iEnd.x1; iGlob += iStp.x1) {
          for (jGlob = iStrt.x2; jGlob < iEnd.x2; jGlob += iStp.x2) {
            for (kGlob = iStrt.x3; kGlob < iEnd.x3; kGlob += iStp.x3) {
              read_data = checkIndexInRange(iGlob, jGlob, kGlob, 
                iInRange, jInRange, kInRange);
              if (read_data) {
                readIn.readPrimsData(fin, tempArray, index_3D, gridsize_3D, 
                  n, nStrt, iGlob, jGlob, kGlob, *this);

                if (export_raytracing) {
                  std::cout << "Raytracing not implemented." << std::endl;
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
                  
                  if (DISSTRK) {
                    vStrt = nPrim + RADM1 * nRadM1 + RES * nRes + 
                      TWOT * nTwoT + PNUM * nPnum + DOYE * nYe + 
                      DONUC * nNUC + NUM1 * nNuM1 + DOTR * nTr;
                    diss[index_3D] = tempArray[vStrt];
                    if (DISSTRK == 2) {
                      radSrc[index_3D] = tempArray[vStrt + 1];
                    }
                  }
                }
              } // read_data
              read_data = 0;
            } // kGlob
          } // jGlob
        } // iGlob
      } // loop over n
      fclose(fin); 
    } // parallel loop over number of files (GPUs)
    delete[] tempArray;
  }
}

void Read::checkFileExists(std::string filename) {
  if (!std::filesystem::exists(filename)) {
    std::runtime_error(filename + " does not exist.");
  }
}

void Read::checkGridFileSize(std::string filename) {
  int expected_filesize;
  expected_filesize = (nvar * bs1 * bs2 * bs3 + 
    (bs1 * bs2 * (nmtc * 16 + 1)) * (axisym)  + 
    (bs1 * bs2 * bs3 * (nmtc * 16 + 1)) * (!axisym)) * dbl_size;

  filesize = std::filesystem::file_size(filename);
  if (filesize < expected_filesize) {
    std::runtime_error(filename + " does not exist."); 
  }
}

bool Read::checkNotInBlock(AMRVar iStrt, AMRVar iEnd, BoolVar iNotInBlock) {
  // do not loop through block if block not in data set
  // still loop through if low_res > bs
  iNotInBlock.x1 = iStrt.x1/ lowRes.x1 > i_max || iEnd.x1/ lowRes.x1 <= i_min;
  iNotInBlock.x2 = iStrt.x2/ lowRes.x2 > j_max || iEnd.x2/ lowRes.x2 <= j_min;
  iNotInBlock.x3 = iStrt.x3/ lowRes.x3 > k_max || iEnd.x3/ lowRes.x3 <= k_min;
  if ((iNotInBlock.x1 && lowRes.x1 < bs1) || 
      (iNotInBlock.x2 && lowRes.x2 < bs2) || 
      (iNotInBlock.x3 && lowRes.x3 < bs3)) {
    return true;
  } else {
    return false;
  }
}

int Read::checkIndexInRange(int iGlob, int jGlob, int kGlob,
    bool iInRange, bool jInRange, bool kInRange) {
  if (by_block) {
    return 1;
  } else {
    // if not by block, check data range
    iInRange = (iGlob / lowRes.x1 >= i_min && iGlob / lowRes.x1 < i_max);
    jInRange = (jGlob / lowRes.x2 >= j_min && jGlob / lowRes.x2 < j_max);
    kInRange = (kGlob / lowRes.x3 >= k_min && kGlob / lowRes.x3 < k_max);
    if (iInRange && jInRange && kInRange) {
      return 1;
    } else {
      return 0;
    }
  }
}

int Read::checkIndexInRange(int n, int iGlob, int jGlob, 
    const int* levelArray, int index_2D, 
    bool iInRange, bool jInRange) {
  if (by_block) {
    return 1;
  } else {
    // if not by block, check data range
    iInRange = (iGlob / lowRes.x1 >= i_min && iGlob / lowRes.x1 < i_max);
    jInRange = (jGlob / lowRes.x2 >= j_min && jGlob / lowRes.x2 < j_max);

    if (iInRange && jInRange) {
      // convert to 1D index to read in binary data
      index_2D = (((iGlob / lowRes.x1) - i_min) * (j_max - j_min) + 
                  ((jGlob / lowRes.x2) - j_min));

      // if you have an axisymmetric metric then take the
      // coordinates from the highest refinement level
      if (MAX(amrLevel.x1[n_ord[n]],
          MAX(amrLevel.x2[n_ord[n]],
              amrLevel.x3[n_ord[n]])) >= levelArray[index_2D]) {
        return 1;
      } else {
        return 0;
      }
    } else {
      return 0;
    }
  }
}

Read::~Read() {};

ReadIn::ReadIn() {};

ReadIn::ReadIn (ReadIn& readIn) {
  iLoc = readIn.iLoc;
  jLoc = readIn.jLoc;
  kLoc = readIn.kLoc;

  bsLR     = readIn.bsLR;
  coord    = readIn.coord;
  exponent = readIn.exponent;
  offset   = readIn.offset;
  upscale  = readIn.upscale;

};

void ReadIn::setLoops(int &iStrt, int &iEnd, int &iStp, int &offset, 
    int bs, int coord, int upscale, int low_res, int by_block, 
    int export_raytracing) {
  if (by_block) {
    iStrt = 0;
    iEnd  = bs / low_res;
    iStp  = 1;
  } else {
    // offsets are always 0 unless low_resx1 etc. not a power of 2
    offset = (low_res - (coord * upscale * bs) % low_res) % low_res;
    iStrt  = coord * (upscale * bs) + offset;
    iEnd   = (coord + 1) * (upscale * bs);
    iStp   = 1 * export_raytracing + low_res * (1 - export_raytracing);
  }
}

void ReadIn::setLoops3D(int n, AMRVar &iStrt, AMRVar &iEnd, AMRVar &iStp, 
    const Read &read) {
  exponent.x1 = read.maxLevel.x1 - read.amrLevel.x1[read.n_ord[n]];
  exponent.x2 = read.maxLevel.x2 - read.amrLevel.x2[read.n_ord[n]];
  exponent.x3 = read.maxLevel.x3 - read.amrLevel.x3[read.n_ord[n]];
  coord.x1    = read.amrCoord.x1[read.n_ord[n]];
  coord.x2    = read.amrCoord.x2[read.n_ord[n]];
  coord.x3    = read.amrCoord.x3[read.n_ord[n]];

  // either 2^n or 1 (not used if by_block == 1)
  upscale.x1 = pow(2, exponent.x1); 
  upscale.x2 = pow(2, exponent.x2);
  upscale.x3 = pow(2, exponent.x3);

  // allow data to be read into 4D arrays (organized by block) 
  // or as a 3D unigrid array
  setLoops(iStrt.x1, iEnd.x1, iStp.x1, offset.x1, read.bs1, coord.x1, 
    upscale.x1, read.lowRes.x1, read.by_block, read.export_raytracing);
  setLoops(iStrt.x2, iEnd.x2, iStp.x2, offset.x2, read.bs2, coord.x2, 
    upscale.x2, read.lowRes.x2, read.by_block, read.export_raytracing);
  setLoops(iStrt.x3, iEnd.x3, iStp.x3, offset.x3, read.bs3, coord.x3, 
    upscale.x3, read.lowRes.x3, read.by_block, read.export_raytracing);
}

void ReadIn::setMetricLoops2D(AMRVar &iStrt, AMRVar &iEnd, AMRVar &iStp, 
    const Read &read) {
  setMetricLoops(iStrt.x1, iEnd.x1, iStp.x1, offset.x1, 
    read.bs1, coord.x1, upscale.x1, read.lowRes.x1, read.by_block);
  setMetricLoops(iStrt.x2, iEnd.x2, iStp.x2, offset.x2, 
    read.bs2, coord.x2, upscale.x2, read.lowRes.x2, read.by_block);
}

void ReadIn::setMetricLoops3D(AMRVar &iStrt, AMRVar &iEnd, AMRVar &iStp, 
  const Read &read) {
  setMetricLoops(iStrt.x1, iEnd.x1, iStp.x1, offset.x1, 
    read.bs1, coord.x1, upscale.x1, read.lowRes.x1, read.by_block);
  setMetricLoops(iStrt.x2, iEnd.x2, iStp.x2, offset.x2, 
    read.bs2, coord.x2, upscale.x2, read.lowRes.x2, read.by_block);
  setMetricLoops(iStrt.x3, iEnd.x3, iStp.x3, offset.x3, 
    read.bs3, coord.x3, upscale.x3, read.lowRes.x3, read.by_block);
}

void ReadIn::setMetricLoops(int &iStrt, int &iEnd, int &iStp, 
    int &offset, int bs, int coord, int upscale, int low_res, int by_block) {
  if (by_block) {
    iStrt = 0;
    iEnd  = bs / low_res;
    iStp  = 1;
  } else {
    // offsets are always 0 unless low_resx1 etc. not a power of 2
    offset = (low_res - (coord * upscale * bs) % low_res) % low_res;
    iStrt  = coord * (upscale * bs) + offset;
    iEnd   = (coord + 1) * (upscale * bs);
    iStp  = low_res;
  }
}

void ReadIn::getIndex3D(long int &index_3D, long int &gridsize_3D, 
    int n, int iGlob, int jGlob, int kGlob, const Read &read) {
  if (read.by_block) {
    // convert to 1D index to read in binary data
    bsLR.x1 = read.bs1 / read.lowRes.x1;
    bsLR.x2 = read.bs2 / read.lowRes.x2;
    bsLR.x3 = read.bs3 / read.lowRes.x3;
    
    index_3D = ((n * bsLR.x1 * bsLR.x2 * bsLR.x3) + 
      (iGlob * bsLR.x2 * bsLR.x3) + (jGlob * bsLR.x3) + kGlob);
      
    // gridsize is only modified if by block
    gridsize_3D = read.nb * bsLR.x1 * bsLR.x2 * bsLR.x3;
  } else {
    // indices on the given block
    iLoc = (iGlob - (coord.x1 * upscale.x1 * read.bs1)) / upscale.x1;            
    jLoc = (jGlob - (coord.x2 * upscale.x2 * read.bs2)) / upscale.x2;            
    kLoc = (kGlob - (coord.x3 * upscale.x3 * read.bs3)) / upscale.x3;
    
    // convert to 1D index to read in binary data
    index_3D = (((iGlob / read.lowRes.x1) - read.i_min) * read.isize +
                ((jGlob / read.lowRes.x2) - read.j_min) * read.jsize +
                ((kGlob / read.lowRes.x3) - read.k_min));
  }  
}

void ReadIn::readGridData(FILE* fin, double* tempArray,
    long int &index_3D, long int &gridsize_3D, 
    int n, int iGlob, int jGlob, int kGlob, const Read &read) {
  if (read.by_block) {
    // convert to 1D index to read in binary data
    bsLR.x1 = read.bs1 / read.lowRes.x1;
    bsLR.x2 = read.bs2 / read.lowRes.x2;
    bsLR.x3 = read.bs3 / read.lowRes.x3;
    
    index_3D = ((n * bsLR.x1 * bsLR.x2 * bsLR.x3) + 
      (iGlob * bsLR.x2 * bsLR.x3) + (jGlob * bsLR.x3) + kGlob);
      
    // gridsize is only modified if by block
    gridsize_3D = read.nb * bsLR.x1 * bsLR.x2 * bsLR.x3;
                      
    fseek(fin, ((iGlob * read.lowRes.x1) * read.bs2 * read.bs3 + 
                (jGlob * read.lowRes.x2) * read.bs3 + 
                (kGlob * read.lowRes.x3)) * read.nvar * read.dbl_size, 
                SEEK_SET);
    fread(&tempArray[0], read.dbl_size, read.nvar, fin);
  } else {
    // indices on the given block
    iLoc = (iGlob - (coord.x1 * upscale.x1 * read.bs1)) / upscale.x1;            
    jLoc = (jGlob - (coord.x2 * upscale.x2 * read.bs2)) / upscale.x2;            
    kLoc = (kGlob - (coord.x3 * upscale.x3 * read.bs3)) / upscale.x3;
    
    // convert to 1D index to read in binary data
    index_3D = (((iGlob / read.lowRes.x1) - read.i_min) * read.isize +
                ((jGlob / read.lowRes.x2) - read.j_min) * read.jsize +
                ((kGlob / read.lowRes.x3) - read.k_min));
                                      
    fseek(fin, ((iLoc * read.bs2 * read.bs3 + 
      jLoc * read.bs3 + kLoc) * read.nvar * read.dbl_size), SEEK_SET);
    fread(&tempArray[0], read.dbl_size, read.nvar, fin);
  }  
}

void ReadIn::readMetricData2D(FILE* fin, double* tempArray,
    long int &index_2D, long int &gridsize_2D, 
    int n, int iGlob, int jGlob, const Read &read) {
  if (read.by_block) {
    // convert to 1D index to read in binary data
    bsLR.x1 = read.bs1 / read.lowRes.x1;
    bsLR.x2 = read.bs2 / read.lowRes.x2;
    bsLR.x3 = read.bs3 / read.lowRes.x3;

    index_2D = ((n * bsLR.x1 * bsLR.x2) + 
      ((iGlob / read.lowRes.x1) * bsLR.x2) + (jGlob / read.lowRes.x2));
  
    // gridsize is only modified if by block
    gridsize_2D = read.nb * bsLR.x1 * bsLR.x2;

    fseek(fin, (read.nvar * read.bs1 * read.bs2 * read.bs3 + 
      iGlob * read.bs2 * (read.nmtc * 16 + 1) + 
      jGlob * (read.nmtc * 16+1)) * read.dbl_size, SEEK_SET);
    fread(&tempArray[0], read.dbl_size, (read.nmtc * 16 + 1), fin);
  } else {
    // indices on the given block
    iLoc = (iGlob - (coord.x1 * upscale.x1 * read.bs1)) / upscale.x1;            
    jLoc = (jGlob - (coord.x2 * upscale.x2 * read.bs2)) / upscale.x2; 

    index_2D = (
      ((iGlob / read.lowRes.x1) - read.i_min) *  (read.j_max - read.j_min) + 
      ((jGlob / read.lowRes.x2) - read.j_min));           

    fseek(fin, (read.nvar * read.bs1 * read.bs2 * read.bs3 + 
      iLoc * read.bs2 * (read.nmtc * 16 + 1) +
      jLoc * (read.nmtc * 16 + 1)) * read.dbl_size, SEEK_SET);
    fread(&tempArray[0], read.dbl_size, (read.nmtc * 16 + 1), fin);
  }
}

void ReadIn::readMetricData3D(FILE* fin, double* tempArray,
    long int &index_3D, long int &gridsize_3D, 
    int n, int iGlob, int jGlob, int kGlob, const Read &read) {
  if (read.by_block) {
    // convert to 1D index to read in binary data
    bsLR.x1 = read.bs1 / read.lowRes.x1;
    bsLR.x2 = read.bs2 / read.lowRes.x2;
    bsLR.x3 = read.bs3 / read.lowRes.x3;

    index_3D = ((n * bsLR.x1 * bsLR.x2 * bsLR.x3) + 
                (iGlob * bsLR.x2 * bsLR.x3) + 
                (jGlob * bsLR.x3) + kGlob);
    // gridsize is only modified if by block
    gridsize_3D = read.nb * bsLR.x1 * bsLR.x2 * bsLR.x3;

    fseek(fin, (read.nvar * read.bs1 * read.bs2 * read.bs3 + 
      (iGlob * read.lowRes.x1) * read.bs2 * read.bs3 * (read.nmtc * 16 + 1) + 
      (jGlob * read.lowRes.x2) * read.bs3 * (read.nmtc * 16 + 1) +
      (kGlob * read.lowRes.x3) * (read.nmtc * 16+1)) * read.dbl_size, 
      SEEK_SET);
    fread(&tempArray[0], read.dbl_size, (read.nmtc * 16 + 1), fin);
  } else {
    // indices on the given block
    iLoc = (iGlob - (coord.x1 * upscale.x1 * read.bs1)) / upscale.x1;            
    jLoc = (jGlob - (coord.x2 * upscale.x2 * read.bs2)) / upscale.x2;            
    kLoc = (kGlob - (coord.x3 * upscale.x3 * read.bs3)) / upscale.x3;
  
    // convert to 1D index to read in binary data
    index_3D = (((iGlob / read.lowRes.x1) - read.i_min) * read.isize +
                ((jGlob / read.lowRes.x2) - read.j_min) * read.jsize +
                ((kGlob / read.lowRes.x3) - read.k_min));
    fseek(fin, (read.nvar * read.bs1 * read.bs2 * read.bs3 +
      iLoc * read.bs2 * read.bs3 * (read.nmtc * 16 + 1) +
      jLoc * read.bs3 * (read.nmtc * 16 + 1) +
      kLoc * (read.nmtc * 16 + 1)) * read.dbl_size, SEEK_SET);
    fread(&tempArray[0], read.dbl_size, (read.nmtc * 16 + 1), fin);
  }
}

void ReadIn::readPrimsData(FILE* fin, float* tempArray,
    long int &index_3D, long int &gridsize_3D, int n, int nStrt, 
    int iGlob, int jGlob, int kGlob, const Read &read) {
  if (read.by_block) {
    // convert to 1D index to read in binary data
    bsLR.x1 = read.bs1 / read.lowRes.x1;
    bsLR.x2 = read.bs2 / read.lowRes.x2;
    bsLR.x3 = read.bs3 / read.lowRes.x3;

    index_3D = ((n * bsLR.x1 * bsLR.x2 * bsLR.x3) + 
      (iGlob * bsLR.x2 * bsLR.x3) + (jGlob * bsLR.x3) + kGlob);

    // gridsize is only modified if by block
    gridsize_3D = read.nb * bsLR.x1 * bsLR.x2 * bsLR.x3;
                
    fseek(fin, (((n - nStrt) * read.bs1 * read.bs2 * read.bs3) +
      (iGlob * read.lowRes.x1) * read.bs2 * read.bs3 + 
      (jGlob * read.lowRes.x2) * read.bs3 + 
      (kGlob * read.lowRes.x3)) * read.nvar * read.float_size, 
      SEEK_SET);
    fread(tempArray, read.float_size, read.nvar, fin);
  } else {
    // indices on the given block
    iLoc = (iGlob - (coord.x1 * upscale.x1 * read.bs1)) / upscale.x1;            
    jLoc = (jGlob - (coord.x2 * upscale.x2 * read.bs2)) / upscale.x2;            
    kLoc = (kGlob - (coord.x3 * upscale.x3 * read.bs3)) / upscale.x3;

    // convert to 1D index to read in binary data
    index_3D = (((iGlob / read.lowRes.x1) - read.i_min) * read.isize +
                ((jGlob / read.lowRes.x2) - read.j_min) * read.jsize +
                ((kGlob / read.lowRes.x3) - read.k_min));                
    fseek(fin, ((n - nStrt) * read.bs1 * read.bs2 * read.bs3 + 
      (iLoc * read.bs2 * read.bs3) + (jLoc * read.bs3) + kLoc) * (
      read.nvar * read.float_size), SEEK_SET);
    fread(tempArray, read.float_size, read.nvar, fin);  
  }
}

ReadIn::~ReadIn() {};

// default constructor
Tools::Tools() {};

void Tools::kernelInvert4x4(const std::array<double, 16>& a, 
    std::array<double, 16>& b) {
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

void Tools::invert4x4(float* A, float* B, int nb, int nx1, int nx2, int nx3) {
/*
    Description:
    
    Accepts array A that holds nb*bs1*bs2*bs3 4x4 matrices. Then, loops
    through each nb*bs1*bs2*bs3 index, gets 4x4 matrix at this coordinate,
    inverts it, and fills it into the array B

*/
  
  int gridsize_3D;
  
  // Loop over each of nb * nx1 * nx2 * nx3 cells
  gridsize_3D = nb * nx1 * nx2 * nx3;
  #pragma omp parallel 
  {
    // Flattened matrices a and b (inverse of a)
    std::array<double, 16> a_tmp, b_tmp;

    #pragma omp for
    for (int n = 0; n < nb; n++) {
      // First get flattened 4x4 matrix from A at this (n,i,j,k) cell
      for (int i1 = 0; i1 < 4; i1++) {
        for (int j1 = 0; j1 < 4; j1++) {
          a_tmp[i1 * 4 + j1] = A[(i1 * 4 + j1) * gridsize_3D + n];
        }
      }
      // Then, invert this 4x4 matrix and load it into b_tmp
      kernelInvert4x4(a_tmp, b_tmp);
      // Now take inverted 4x4 matrix, b_tmp, and put in B at this (n,i,j,k) cell
      for (int i1 = 0; i1 < 4; i1++) { 
        for (int j1 = 0; j1 < 4; j1++) {
          B[(i1 * 4 + j1) * gridsize_3D + n] = b_tmp[i1 * 4 + j1];
        }
      }
    }
  }
} // end invert_4x4()

double Tools::kernelDet4x4(const std::array<double, 16>& a) {
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

void Tools::det4x4(float* A, float* B, int nb, int nx1, int nx2, int nx3) {
  int gridsize_3D;
  
  // Loop over each of nb * nx1 * nx2 * nx3 cells
  gridsize_3D = nb * nx1 * nx2 * nx3;

  #pragma omp parallel
  {
    // Flattened matrix a
    std::array<double, 16> a_tmp;
    #pragma omp for
    for (int n = 0; n < gridsize_3D; n++) {
      for (int i1 = 0; i1 < 4; i1++) {
        for (int j1 = 0; j1 < 4; j1++) {
          a_tmp[i1 * 4 + j1] = A[(i1 * 4 + j1) * gridsize_3D + n];
        }
      }
      // Now take determinant of flattened matrix
      // and place in B at this (n,i,j,k) cell
      B[n] = kernelDet4x4(a_tmp);
    }
  }
} // end det4x4()

void Tools::lowerIndex(float* u, float* u_mu, float* gcov,
    int nb, int nx1, int nx2, int nx3, int axisym) {
  // compute u_mu = g_mu,nu u^nu for all mu
  // computes (4, nb, nx1, nx2, nx3) dimensional array
  long int gridsize_3D, gridsize_2D;

  // Loop over each of nb * nx1 * nx2 * nx3 cells
  gridsize_3D = nb * nx1 * nx2 * nx3;
  if (axisym) {
    gridsize_2D = nb * nx1 * nx2;
  } else {
    gridsize_2D = gridsize_3D;
  }

  #pragma omp parallel
  {
    std::array<float, 4> uu, ud;
    std::array<float, 16> gcovLoc;
    #pragma omp for
    for (int n = 0; n < gridsize_3D; n++) {
      for (int l = 0; l < 4; l++) {
        for (int m = 0; m < 4; m++) {
          gcovLoc[4 * l + m] = gcov[(4 * l + m) * gridsize_2D + 
            axisym * (n / nx3) + (1 - axisym) * n];
          uu[m] = u[m * gridsize_3D + n];
        }
        ud[l] = 0.0;
      }
      lowerVec(uu, ud, gcovLoc);
      for (int l = 0; l < 4; l++) {
        u_mu[l * gridsize_3D + n] = ud[l];
      }
    }
  }
}

void Tools::lowerIndex(float* u, float* u_mu, float* gcov,
    int nb, int nx1, int nx2, int nx3, int axisym, int l) {
  // compute u_mu = g_mu,nu u^nu for all mu
  // computes (4, nb, nx1, nx2, nx3) dimensional array
  long int gridsize_3D, gridsize_2D;

  // Loop over each of nb * nx1 * nx2 * nx3 cells
  gridsize_3D = nb * nx1 * nx2 * nx3;
  if (axisym) {
    gridsize_2D = nb * nx1 * nx2;
  } else {
    gridsize_2D = gridsize_3D;
  }

  #pragma omp parallel
  {
    std::array<float, 4> uu, ud;
    std::array<float, 16> gcovLoc;
    #pragma omp for
    for (int n = 0; n < gridsize_3D; n++) {
      for (int m = 0; m < 4; m++) {
        gcovLoc[4 * l + m] = gcov[(4 * l + m) * gridsize_2D + 
          axisym * (n / nx3) + (1 - axisym) * n];
        uu[m] = u[m * gridsize_3D + n];
      }
      for (int i = 0; i < 4; i++) {
        ud[i] = 0.0;
      }
      lowerVec(uu, ud, gcovLoc);
      u_mu[n] = ud[l];
    }
  }
}

void Tools::lowerVec(const std::array<float, 4>& uu,
    std::array<float, 4>& ud, const std::array<float, 16>& gcov) {
  for (int i = 0 ; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      ud[i] += gcov[4 * i + j] * uu[j];
    }
  }
}

void Tools::boostVector(float* u, float* u_boost, float* uu_c,
    int nb, int nx1, int nx2, int nx3, int inverse) {
  // compute u_boost = Lambda^mu,nu u^nu for all mu
  // computes (4, nb, nx1, nx2, nx3) dimensional array
  long int gridsize_3D;
  // Loop over each of nb * nx1 * nx2 * nx3 cells
  gridsize_3D = nb * nx1 * nx2 * nx3;
  #pragma omp parallel
  {
    std::array<float, 4> uu, ub, uuLoc; 
    std::array<float, 16> lambdaLoc;
    #pragma omp for
    for (int n = 0; n < gridsize_3D; n++) {
      for (int m = 0; m < 4; m++) {
        uu[m]    = u[   m * gridsize_3D + n];
        uuLoc[m] = uu_c[m * gridsize_3D + n];
        ub[m]    = 0.0;
        for (int i = 0; i < 4; i++) {
          lambdaLoc[4 * m + i] = 0.0;
        }
      } 
      calcLorentzBoostVec(uuLoc, lambdaLoc, inverse);
      lowerVec(uu, ub, lambdaLoc);
      for (int m = 0; m < 4; m++) {
        u_boost[m * gridsize_3D + n] = ub[m];
      }
    }
  }
}

void Tools::calcLorentzBoostVec(const std::array<float, 4>& uu,
    std::array<float, 16>& lambda, int inverse) {
  float vsq, pm;
  std::array<float, 4> v;
  pm = 1.0 * (1 - inverse) - 1.0 * (inverse);
  v[0] = 0.0;
  v[1] = pm * uu[1] / uu[0];
  v[2] = pm * uu[2] / uu[0];
  v[3] = pm * uu[3] / uu[0];
  vsq  = v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
  lambda[0] = uu[0];
  for (int i = 0; i < 3; i++) {
    lambda[4 * (i + 1)] = -uu[i+1];
    lambda[i + 1]       = -uu[i+1];
  }
  for (int i = 0 ; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      lambda[4 * (i + 1) + (j + 1)] = ((uu[0] - 1.0) / vsq) * (
       v[i + 1] * v[j + 1] + (i == j) * 1.0);
    }
  }
}

// default destructor
Tools::~Tools() {};

int main() {
  std::cout << "Executing hamr_read.cpp" << std::endl;
  return 0;
}
