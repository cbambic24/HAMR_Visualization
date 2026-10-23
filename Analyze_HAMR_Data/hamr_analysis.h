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
    int gridsize_3D;

};

// Two-temperature class
class TwoTemperature {
  public:
    TwoTemperature();
    
    TwoTemperature(double ThetaMin);
    
    // '= default' asks the compiler to generate the copy constructor:
    // it copies EVERY member, and stays correct when members are added.
    // (Required by OpenMP firstprivate(); '= delete' would remove it.)
    TwoTemperature(const TwoTemperature& twoT) = default;
    
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
    
    // Compiler-generated: copies every member, including the params /
    // CONSTANTS maps that the hand-written version silently dropped.
    Opacity(const Opacity& opac) = default;
    
    virtual double kappaAbsTotal(float rho, float Te, float Tr, float b2, int XRB); 
    
    virtual double kappaEmiTotal(float rho, float Te, float Tr, float b2, int XRB); 

    virtual void kappaAbsTotal(double* kappa, float* rho, float* Te, 
      float* Tr, float *b2, int XRB, int nb, int nx1, int nx2, int nx3);

    virtual void kappaEmiTotal(double* kappa, float* rho, float* Te, 
      float* Tr, float *b2, int XRB, int nb, int nx1, int nx2, int nx3);
    
    virtual void kappaSynchrotron(double* kappa, float* rho, float* Te, 
        float* Tr, float* b2, int emi, int XRB, 
        int nb, int nx1, int nx2, int nx3);
    
    virtual ~Opacity();

  protected:
    double rho0, c, kB, aRAD, mu_i, mu_e, mp, mi, me, sigT;
    double xAB, yAB, zAB, eStatC, h, kappaMol;
    double ne;
    int gridsize_3D;
    std::unordered_map<std::string, double> params, CONSTANTS;
    
    virtual double kappaSyncAbs(float rho, float Te, float Tr, float b2, 
        double async, double bsync, double csync, double dsync, double esync);
    virtual double kappaSyncEmi(float rho, float Te, float Tr, float b2, 
        double async, double bsync, double csync, double dsync, double esync);
    
    double kappaBoundFreeAbs(float rho, float Te, float Tr);
    double kappaBoundFreeEmi(float rho, float Te, float Tr);
    double kappaCHIANTIAbs(float rho, float Te, float Tr);
    double kappaCHIANTIEmi(float rho, float Te, float Tr);
    double kappaFreeFreeAbs(float rho, float Te, float Tr);
    double kappaFreeFreeEmi(float rho, float Te, float Tr);
    double kappaH(float rho, float Te, float Tr);
    double kappaFe(float rho, float Te, float Tr);

  private:
    double e_2pimc, Thetae, nu_mu, phi, g, kappaGInv;
    double async, bsync, csync, dsync, esync;

    void getABCDESynch(int XRB, double& a, double& b, double& c, 
        double& d, double& e);
    
};

// RIAF Opacity class
// Use synchrotron opacities from Sadowski+ (2017) instead of McKinney+ (2017)
// Appropriate for low-Mdot radiatively inefficient accretion flows
class RIAF_Opacity : public Opacity {
  public:
    RIAF_Opacity();
    
    RIAF_Opacity(const std::unordered_map<std::string, double>& params,
        const std::unordered_map<std::string, double>& CONSTANTS);
    
    RIAF_Opacity(const RIAF_Opacity& opac) = default;
   
    double kappaAbsTotal(float rho, float Te, float Tr, float b2, int XRB); 
    
    double kappaEmiTotal(float rho, float Te, float Tr, float b2, int XRB); 

    void kappaAbsTotal(double* kappa, float* rho, float* Te, 
        float* Tr, float *b2, int XRB, int nb, int nx1, int nx2, int nx3);

    void kappaEmiTotal(double* kappa, float* rho, float* Te, 
        float* Tr, float *b2, int XRB, int nb, int nx1, int nx2, int nx3);
    
    void kappaSynchrotron(double* kappa, float* rho, float* Te, 
        float* Tr, float* b2, int emi, int XRB, 
        int nb, int nx1, int nx2, int nx3);
    
   ~RIAF_Opacity();

  private:
    double xi, kappa_emi, pime3c5_ekh, ksi;
    double kappaSyncAbs(float rho, float Te, float Tr, float b2);
    double kappaSyncEmi(float rho, float Te, float Tr, float b2);
};

// Radiation class
class Radiation {
  public:
    Radiation();

    Radiation(const std::unordered_map<std::string, double>& params,
        const std::unordered_map<std::string, double>& CONSTANTS);
    
    void qAbs(double* qabs, float* rho, float* Te, float* Tr, float* b2, 
        int XRB, int RIAF, int nb, int nx1, int nx2, int nx3);

    void qEmi(double* qemi, float* rho, float* Te, float* Tr, float* b2, 
        int XRB, int RIAF, int nb, int nx1, int nx2, int nx3);

    ~Radiation();

  private:
    double rho0, c, kB, aRAD, mu_i, mu_e, mp, mi, me, sigT; 
    double xAB, yAB, zAB, eStatC, kappaMol;
    double async, bsync, csync, dsync, esync;
    int gridsize_3D;
    std::unordered_map<std::string, double> params, CONSTANTS; 

};

#endif
