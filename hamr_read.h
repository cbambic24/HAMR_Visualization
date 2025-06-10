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

// Fluid class
class Fluid {
  public:
    Fluid();

    Fluid(const std::unordered_map<std::string, double>& params,
        const std::unordered_map<std::string, double>& CONSTANTS);

    void calcQie(double* qie, float *rho, float *Te, float *Ti,
        int nb, int nx1, int nx2, int nx3);

    ~Fluid();
  
  private:
    double rho0, c, kB, mu_i, mu_e, mp, mi, me, sigT, ThetaMin, xAB, yAB;
    double cksigmaT, me_mi;
    int index_3D, gridsize_3D;

};

// Two-temperature class
class TwoTemperature {
  public:
    TwoTemperature();
    
    TwoTemperature(double ThetaMin);
    
    TwoTemperature(TwoTemperature& twoT);
    
    double CoulombLog(double ne, float Te);
    double BesselTerms(double Thetae, double Thetai);
    
    ~TwoTemperature();
  
  private:
    double ThetaMin, Thetam, Thetasum, k0, k1, k2k2;  

    double coldEcoldI(double Thetae, double Thetai);
    double hotEcoldI(double Thetae, double Thetai);
    double coldEhotI(double Thetae, double Thetai);
    double hotEhotI(double Thetae, double Thetai);

};

// Opacity class
class Opacity {
  public:
    Opacity();

    Opacity(const std::unordered_map<std::string, double>& params,
        const std::unordered_map<std::string, double>& CONSTANTS);
    
    Opacity (Opacity& opac);
    
    double kappaAbsTotal(float rho, float Te, float Tr, float b2, int XRB); 
    
    double kappaEmiTotal(float rho, float Te, float Tr, float b2, int XRB); 

    void kappaAbsTotal(double* kappa, float* rho, float* Te, 
      float* Tr, float *b2, int XRB, int nb, int nx1, int nx2, int nx3);

    void kappaEmiTotal(double* kappa, float* rho, float* Te, 
      float* Tr, float *b2, int XRB, int nb, int nx1, int nx2, int nx3);
    
    void kappaSynchrotron(double* kappa, float* rho, float* Te, 
        float* Tr, float* b2, int emi, int XRB, 
        int nb, int nx1, int nx2, int nx3);
    
    ~Opacity();

  private:
    double rho0, c, kB, aRAD, mu_i, mu_e, mp, mi, me, sigT; 
    double xAB, yAB, zAB, eStatC, h, kappaMol;
    double e_2pimc, ne, Thetae, nu_mu, phi, g, kappaGInv;
    double async, bsync, csync, dsync, esync;
    int index_3D, gridsize_3D;

    double kappaBoundFreeAbs(float rho, float Te, float Tr);
    double kappaBoundFreeEmi(float rho, float Te, float Tr);
    double kappaCHIANTIAbs(float rho, float Te, float Tr);
    double kappaCHIANTIEmi(float rho, float Te, float Tr);
    double kappaFreeFreeAbs(float rho, float Te, float Tr);
    double kappaFreeFreeEmi(float rho, float Te, float Tr);
    double kappaH(float rho, float Te, float Tr);
    double kappaFe(float rho, float Te, float Tr);
    double kappaSyncAbs(float rho, float Te, float Tr, float b2, 
        double async, double bsync, double csync, double dsync, double esync);
    double kappaSyncEmi(float rho, float Te, float Tr, float b2, 
        double async, double bsync, double csync, double dsync, double esync);
    void getABCDESynch(int XRB, double& a, double& b, double& c, 
        double& d, double& e);
    
};

// Radiation class
class Radiation {
  public:
    Radiation();

    Radiation(const std::unordered_map<std::string, double>& params,
        const std::unordered_map<std::string, double>& CONSTANTS);
    
    void qAbs(double* qabs, float* rho, float* Te, float* Tr, float *b2, 
        int XRB, int nb, int nx1, int nx2, int nx3);

    void qEmi(double* qemi, float* rho, float* Te, float* Tr, float *b2, 
        int XRB, int nb, int nx1, int nx2, int nx3);

    ~Radiation();

  private:
    double rho0, c, kB, aRAD, mu_i, mu_e, mp, mi, me, sigT; 
    double xAB, yAB, zAB, eStatC, kappaMol;
    double async, bsync, csync, dsync, esync;
    int index_3D, gridsize_3D;
    std::unordered_map<std::string, double> params, CONSTANTS; 

    double kappaAbsTotal(float rho, float Te, float Tr, float b2, 
        double async, double bsync, double csync, double dsync, double esync); 
    double kappaEmiTotal(float rho, float Te, float Tr, float b2, 
        double async, double bysnc, double csync, double dsync, double esync);

};

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

class Read {
  public:
    // number of variables (nvar), metric variables (nmtc)
    int nvar, nmtc;
    int isize, jsize, dbl_size, float_size; 

    // block indices
    int i_min, i_max, j_min, j_max, k_min, k_max;
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

    void readGrid(float* x1, float* x2, float* x3, 
        float* r, float* theta, float* phi, 
        float* gcov, float* gcon, float* gdet, float* dxdxp);

    void readGridCoords(float* x1, float* x2, float* x3, 
        float* r, float* theta, float* phi);

    void readPrims(int fnum, float* rho, float* uGas, float* uu, float* B, 
        float* E, float* uRad, float* uuRad, 
        float* Te, float* Ti, float* phNum,
        float* Ye, float* xAlpha, float* xAtm, 
        float* uNu1, float* uNu2, float* uNu3, 
        float* uuNu1, float* uuNu2, float* uuNu3,
        float* nuNum1, float* nuNum2, float* nuNum3, float* tr1, float* tr2);

    // default destructor
    ~Read();

  private:
    std::string dir;
    int filesize;
    int fileCount; // number of prim files (GPUs for computation)   
    
    // physics flags
    int RADM1, RES, TWOT, PNUM, DOYE, DONUC, NUM1, DOTR;
    // number of primitives for each physics flag
    int nPrim, nRadM1, nRes, nTwoT, nPnum, nYe, nNUC, nNuM1, nTr;
    std::vector<int> blocksInFile;
    
    void checkFileExists(std::string filename);
    void checkGridFileSize(std::string filename);
    bool checkNotInBlock(AMRVar iStrt, AMRVar iEnd, BoolVar iNotInBlock);
    int checkIndexInRange(int iGlob, int jGlob, int kGlob, 
        bool iInRange, bool jInRange, bool kInRange);
    int checkIndexInRange(int n, int iGlob, int jGlob, 
        const std::vector<int>& levelArray, int index_2D, 
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

    void readGridData(FILE* fin, std::vector<double>& tempArray,
        long int &index_3D, long int &gridsize_3D, 
        int n, int iGlob, int jGlob, int kGlob, const Read &read);

    void readMetricData2D(FILE* fin, std::vector<double>& tempArray,
        long int &index_2D, long int &gridsize_2D, 
        int n, int iGlob, int jGlob, const Read &read);

    void readMetricData3D(FILE* fin, std::vector<double>& tempArray,
        long int &index_3D, long int &gridsize_3D, 
        int n, int iGlob, int jGlob, int kGlob, const Read &read);

    void readPrimsData(FILE* fin, std::vector<float>& tempArray,
        long int &index_3D, long int &gridsize_3D, int n, int nStrt, 
        int iGlob, int jGlob, int kGlob, const Read &read);

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

    // default destructor
    ~Tools();
  
  private: 
    void kernelInvert4x4(const std::vector<double>, std::vector<double>& b); 
    double kernelDet4x4(const std::vector<double> a);
    void lowerVec(const std::vector<float> uu,
        std::vector<float>& ud, const std::vector<float> gcov);
};

#endif
