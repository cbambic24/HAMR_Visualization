#ifndef HAMR_H
#define HAMR_H

#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <cstdio> // fread and fseek
#include <omp.h> // omp parallel for
#include <numeric> // accumulate

struct AMRVar {
  int x1;
  int x2;
  int x3;
};

struct AMRVec {
  std::vector<int> x1;
  std::vector<int> x2;
  std::vector<int> x3;
};

struct BoolVar {
  bool x1;
  bool x2;
  bool x3;
};

struct GridArrays {
  float* x1;
  float* x2;
  float* x3;
  float* r;
  float* theta;
  float* phi;
  float* gcov;
  float* gcon;
  float* gdet;
  float* dxdxp;
};

struct GridCoords {
  float* x1;
  float* x2;
  float* x3;
  float* r;
  float* theta;
  float* phi;
};

struct GridResolution {
  int* x1;
  int* x2;
  int* x3;
};

class Read {
  public:
    // number of variables (nvar), metric variables (nmtc)
    int nvar, nmtc;
    int isize, jsize, dbl_size, float_size; 

    // block indices
    int i_min, i_max, j_min, j_max, k_min, k_max;
    // block start and end
    int blkStrt, blkEnd;
    // number of total active blocks and block sizes
    int nb, bs1, bs2, bs3;
    // number of blocks in all files
    int nbInFiles;
    int axisym, by_block, export_raytracing; 

    // downsampling (lowRes) parameter and max refinement level 
    AMRVar lowRes, maxLevel;
    // refinement level of block
    AMRVec amrLevel, amrCoord;

    // information needed for AMR
    std::vector<int> n_ord;

    // Local Block IDs to read, in the order the caller wants them back.
    // Always supplied: the whole dump (0..nb-1), a start_block/end_block
    // range, or the scattered set that overlaps a selection box. readGrid
    // iterates this instead of a contiguous [blkStrt, blkEnd) range, so a
    // box selection costs one call rather than one call per contiguous run.
    std::vector<int> blockList;

    // default constructor
    Read();

    // parameterized constructor
    Read(std::string dir, int axisym, int by_block, int export_raytracing,
        const std::unordered_map<std::string, double>& amr,
        const std::unordered_map<std::string, std::vector<int>>& amr_info,
        const std::unordered_map<std::string, double>& block_info,
        const std::unordered_map<std::string, double>& options,
        const std::unordered_map<std::string, double>& params,
        const std::unordered_map<std::string, int>& physics);

    void readGrid(const GridArrays& grid);

    void readGridCoords(const GridCoords& coords);

    void readGridResolution(const GridResolution& res_level);

    void readPrims(int fnum, const std::unordered_map<std::string, float*>& prims);

    // Look up one array. Absent + required throws naming the key, rather
    // than unordered_map::at's opaque "_Map_base::at". Absent + optional
    // yields nullptr, which is safe only because every optional array is
    // guarded by its physics flag at the point of use.
    float* primPtr(const std::unordered_map<std::string, float*>& prims,
        const std::string& key, bool required = true) const;

    // default destructor
    ~Read();

  private:
    std::string dir;
    int filesize;
    int fileCount; // number of prim files (GPUs for computation)   
    
    // physics flags
    int RADM1, RES, TWOT, PNUM, DOYE, DONUC, NUM1, DOTR, DISSTRK;
    // number of primitives for each physics flag
    int nPrim, nRadM1, nRes, nTwoT, nPnum, nYe, nNUC, nNuM1, nTr, nDiss;
    std::vector<int> blocksInFile;
    
    // Open a gdump and validate its length in one step, returning a handle
    // positioned at byte 0. Throws (naming the file) if it cannot be opened
    // or is shorter than the block layout requires; on the short-file path
    // it closes the handle before throwing, so the caller never has to.
    //
    // This replaces a separate exists() + file_size() pair. Both were
    // path-based stats, i.e. two extra metadata round-trips per block on
    // Lustre -- ~44k of them on a full THIN_RAD_MAD_HIRES_TRACER read --
    // to learn two things the fopen and its file descriptor already know.
    FILE* openGridFile(const std::string& filename);
    bool checkNotInBlock(AMRVar iStrt, AMRVar iEnd, BoolVar iNotInBlock);
    int checkIndexInRange(int iGlob, int jGlob, int kGlob, 
        bool iInRange, bool jInRange, bool kInRange);
    int checkIndexInRange(int n, int iGlob, int jGlob, 
        const int* levelArray, int index_2D, 
        bool iInRange, bool jInRange);
    
};

class ReadIn {
  public:
    ReadIn();

    ReadIn(ReadIn& readIn);

    void setLoops3D(int n, AMRVar &iStrt, AMRVar &iEnd, AMRVar &iStp, 
        const Read &read);
    
    void setMetricLoops2D(AMRVar &iStrt, AMRVar &iEnd, AMRVar &iStp, 
        const Read &read);

    void setMetricLoops3D(AMRVar &iStrt, AMRVar &iEnd, AMRVar &iStp, 
        const Read &read);

    void getIndex3D(long int &index_3D, long int &gridsize_3D, 
        int n, int iGlob, int jGlob, int kGlob, const Read &read);

    void readGridData(FILE* fin, double* tempArray,
        long int &index_3D, long int &gridsize_3D, 
        int n, int iGlob, int jGlob, int kGlob, const Read &read);

    void readMetricData2D(FILE* fin, double* tempArray,
        long int &index_2D, long int &gridsize_2D, 
        int n, int iGlob, int jGlob, const Read &read);

    void readMetricData3D(FILE* fin, double* tempArray,
        long int &index_3D, long int &gridsize_3D, 
        int n, int iGlob, int jGlob, int kGlob, const Read &read);

    // fileStart is the global index of the first block in the open file.
    // The destination offset uses blkStrt.
    // Three different block indices meet here and must not be conflated:
    //   n         -- Local Block ID; picks the AMR metadata, and
    //                (n - fileStart) locates the block inside the open file
    //   fileStart -- Local Block ID of the first block in that file, never
    //                clamped to the selection
    //   dest      -- position within blockList, i.e. which slot of the
    //                caller's arrays this block is written to. Not derivable
    //                from n once a box makes the selection non-contiguous.
    void readPrimsData(FILE* fin, float* tempArray,
        long int &index_3D, long int &gridsize_3D, int n, int fileStart,
        int dest, int iGlob, int jGlob, int kGlob, const Read &read);

    ~ReadIn();

  private:
    int iLoc, jLoc, kLoc; // local indices    
    AMRVar bsLR, coord, exponent, offset, upscale;

    void setLoops(int &iStrt, int &iEnd, int &iStp, int &offset, int bs, 
      int coord, int upscale, int low_res, int by_block, 
      int export_raytracing);

    void setMetricLoops(int &iStrt, int &iEnd, int &iStp, int &offset, 
        int bs, int coord, int upscale, int low_res, int by_block);
    
};

class Tools {
  public:
    // default constructor
    Tools();

    void invert4x4(float* A, float* B, int nb, int nx1, int nx2, int nx3);
    void det4x4(float* A, float* B, int nb, int nx1, int nx2, int nx3);
    void lowerIndex(float* u, float* u_mu, float* gcov, 
        int nb, int nx1, int nx2, int nx3, int axisym);
    void lowerIndex(float* u, float* u_mu, float* gcov,
        int nb, int nx1, int nx2, int nx3, int axisym, int l);
    void boostVector(float* u, float* u_boost, float* uu_c,
        int nb, int nx1, int nx2, int nx3, int inverse);
    void contractTensor(float* tnsr, float* tnsr_out, float* M,
        int nb, int nx1, int nx2, int nx3, int axisym_tnsr, int axisym_M);

    // default destructor
    ~Tools();

  private:
    void kernelInvert4x4(const std::array<double, 16>& a,
        std::array<double, 16>& b);
    double kernelDet4x4(const std::array<double, 16>& a);
    void lowerVec(const std::array<float, 4>& uu,
        std::array<float, 4>& ud, const std::array<float, 16>& gcov);
    void calcLorentzBoostVec(const std::array<float, 4>& uu,
        std::array<float, 16>& lambda, int inverse);
    void contractTnsr(const std::array<float, 16>& T,
        std::array<float, 16>& Tp, const std::array<float, 16>& M);
    // AxisymTnsr/AxisymM resolved at compile time per instantiation, so
    // the hot loop has no per-iteration branch on them.
    template<bool AxisymTnsr, bool AxisymM>
    void contractTensorImpl(float* tnsr, float* tnsr_out, float* M,
        int nb, int nx1, int nx2, int nx3);
};

#endif
