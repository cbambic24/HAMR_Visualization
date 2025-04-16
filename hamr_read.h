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

class Fluid {
  public:
    // default constructor
    Fluid();

    // parameterized constructor
    Fluid(const std::unordered_map<std::string, double>& params,
        const std::unordered_map<std::string, double>& CONSTANTS);

    void calcQie(double* qie, float *rho, float *Te, float *Ti,
        int nb, int nx1, int nx2, int nx3);

    // default destructor
    ~Fluid();
  
  private:
    double rho0, c, kB, mu_i, mu_e, mp, mi, me, sigT, ThetaMin, xAB, yAB;
    double Thetae, Thetai, Thetam, nAvg, ne, Thetasum, cksigmaT, me_mi;
    double k2k2, k1, k0;
    int n, index_3D, gridsize_3D;

    double CoulombLog(double ne, float Te);
    double BesselTerms(double Thetae, double Thetai, 
        double Thetam, double Thetasum, double k2k2, double k1, double k0);
    double coldEcoldI(double Thetae, double Thetai, 
        double Thetam, double Thetasum, double k2k2, double k1, double k0);
    double hotEcoldI(double Thetae, double Thetai, 
        double Thetam, double Thetasum, double k2k2, double k1, double k0);
    double coldEhotI(double Thetae, double Thetai, 
        double Thetam, double Thetasum, double k2k2, double k1, double k0);
    double hotEhotI(double Thetae, double Thetai, 
        double Thetam, double Thetasum, double k2k2, double k1, double k0);

};

class Opacity {
  public:
    Opacity();

    Opacity(const std::unordered_map<std::string, double>& params,
        const std::unordered_map<std::string, double>& CONSTANTS);

    void kappaSynchrotron(double* kappa, float* rho, float* Te, 
        float* Tr, float* b2, int emi, int XRB, 
        int nb, int nx1, int nx2, int nx3);
    
    void kappaAbsTotal(double* kappa, float* rho, float* Te, 
      float* Tr, float *b2, int XRB, int nb, int nx1, int nx2, int nx3);

    void kappaEmiTotal(double* kappa, float* rho, float* Te, 
      float* Tr, float *b2, int XRB, int nb, int nx1, int nx2, int nx3);

    double kappaAbsTotal(float rho, float Te, float Tr, float b2, int XRB); 
    
    double kappaEmiTotal(float rho, float Te, float Tr, float b2, int XRB); 
    
    ~Opacity();

  private:
    double rho0, c, kB, aRAD, mu_i, mu_e, mp, mi, me, sigT; 
    double xAB, yAB, zAB, eStatC, kappaMol;
    double ne, e_2pimc, h, nu_mu, Thetae, phi, g, kappaGInv;
    double async, bsync, csync, dsync, esync;
    int n, index_3D, gridsize_3D;

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
    double async, bsync, csync, dsync, esync, kappaAbs, kappaEmi;
    int n, index_3D, gridsize_3D;

    Opacity Opac;
    double kappaAbsTotal(float rho, float Te, float Tr, float b2, 
        double async, double bsync, double csync, double dsync, double esync); 
    double kappaEmiTotal(float rho, float Te, float Tr, float b2, 
        double async, double bysnc, double csync, double dsync, double esync);
};


class Read {
  public: 
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
    int isize, jsize, filesize, read_file, read_data;
    int axisym, by_block, export_raytracing; 
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
    AMRVar coord, bsLR, exponent, lowRes, maxLevel, offset, upscale, 
      iStrt, iEnd, iStp;
    // refinement level of block
    AMRVec amrLevel, amrCoord;
    BoolVar iNotInBlock; // is data in selection box/ on block?
    bool iInRange, jInRange, kInRange;
    int dbl_size, float_size; 
    long int gridsize_2D, gridsize_3D, index_2D, index_3D;
    int iGlob, jGlob, kGlob; // global indices
    int n, nStrt, nEnd, fileCount, iFile; // looping through prim files   
    int iLoc,  jLoc,  kLoc, ii; // local indices on given block
    
    // number of variables (nvar), metric variables (nmtc), and 
    // variable start index (vStrt)
    int nvar, nmtc, vStrt;
    // physics flags
    int RADM1, RES, TWOT, PNUM, DOYE, DONUC, NUM1, DOTR;
    // number of primitives for each physics flag
    int nPrim, nRadM1, nRes, nTwoT, nPnum, nYe, nNUC, nNuM1, nTr;
    std::vector<int> blocksInFile;
    FILE* fin;
    std::string filename, dir;

    // information needed for AMR
    std::vector<int> n_ord;
    // block indices
    int i_min, i_max, j_min, j_max, k_min, k_max;
    // number of total active blocks and block sizes
    int nb, bs1, bs2, bs3;

    void checkFileExists(std::string filename);
    void checkGridFileSize(std::string filename, int nvar, int nmtc);
    void setLoops(int &iStrt, int &iEnd, int &iStp, int &offset, 
        int bs, int coord, int upscale, int low_res);
    void setMetricLoops(int &iStrt, int &iEnd, int &iStp, int &offset, 
        int bs, int coord, int upscale, int low_res);
    bool checkNotInBlock(AMRVar iStrt, AMRVar iEnd);
      /*
      int checkIndexInRange(int n, int iGlob, int jGlob, 
          const std::vector<int>& levelArray, AMRVec amrLevel,
          const std::vector<int> n_ord);
      int checkIndexInRange(int iGlob, int jGlob, int kGlob);
      void readGridData(std::vector<double>& tempArray,
          long int &index_3D, long int &gridsize_3D,
          int n, int iGlob, int jGlob, int kGlob,
          AMRVar coord, AMRVar upscale, FILE* fin);
      void readMetricData2D(std::vector<double>& tempArray, 
          long int &index_2D, long int &gridsize_2D, 
          int n, int iGlob, int jGlob, 
          AMRVar coord, AMRVar upscale, FILE* fin);
      void readMetricData3D(std::vector<double>& tempArray,
          long int &index_3D, long int &gridsize_3D, 
          int n, int iGlob, int jGlob, int kGlob,
          AMRVar coord, AMRVar upscale, FILE* fin);
      void readPrimsData(std::vector<float>& tempArray, 
          long int &index_3D, long int &gridsize_3D,
          int n, int nStrt, int iGlob, int jGlob, int kGlob,
          AMRVar coord, AMRVar upscale, FILE* fin);
      */
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
