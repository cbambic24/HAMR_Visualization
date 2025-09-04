#ifndef HAMR_ANALYSIS_H
#define HAMR_ANALYSIS_H

#include <iostream>
#include <string>
#include <cmath>
#include <unordered_map>
#include <numeric> // accumulate
#include <omp.h>

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

#endif
