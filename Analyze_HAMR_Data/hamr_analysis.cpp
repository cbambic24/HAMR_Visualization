#include <iostream>
#include <string>
#include <cmath>
#include <unordered_map>
#include <numeric> // accumulate
#include <omp.h>
#include "hamr_analysis.h"

// Fluid class 
Fluid::Fluid() {};

Fluid::Fluid(const std::unordered_map<std::string, double>& params,
    const std::unordered_map<std::string, double>& CONSTANTS) {
  rho0 = params.at("density_scale");
  c    = CONSTANTS.at("c");
  kB   = CONSTANTS.at("kB");
  xAB  = CONSTANTS.at("X_AB");
  yAB  = CONSTANTS.at("Y_AB");
  mu_i = 4.0 / (4.0 * xAB + yAB);
  mu_e = 2.0 / (1.0 + xAB);
  mp   = CONSTANTS.at("mp");
  mi   = (xAB + yAB) * mp;
  me   = CONSTANTS.at("me");
  sigT = CONSTANTS.at("sigmaT");
  ThetaMin = 1.0e-2;
};

void Fluid::calcQie(double* qie, float* rho, float* Te, float* Ti, 
    int nb, int nx1, int nx2, int nx3) {

  cksigmaT = c * kB * sigT;
  me_mi    = me / mi;

  // Loop over each of nb * nx1 * nx2 * nx3 cells
  int n;
  double ne, nAvg, Thetae, Thetai; 
  gridsize_3D = nb * nx1 * nx2 * nx3;
  TwoTemperature twoT(ThetaMin); 
  #pragma omp parallel for private(n, ne, nAvg, \
    Thetae, Thetai) firstprivate(twoT)
  for (n = 0; n < gridsize_3D; n++) {
    ne     = rho[n] / (mu_e * mp);
    nAvg   = (xAB + yAB) * rho[n] / mp;
    Thetae = kB * Te[n] / (me * c * c);
    Thetai = kB * Ti[n] / (mi * c * c);
    qie[n] = ((3./2.) * me_mi * cksigmaT * ne * nAvg * (Ti[n] - Te[n]) * (
      twoT.CoulombLog(ne, Te[n]) * twoT.BesselTerms(Thetae, Thetai)));
  }
}

Fluid::~Fluid() {};

// Two-temperature class 
TwoTemperature::TwoTemperature() {};

TwoTemperature::TwoTemperature(double ThetaMin) {
  this->ThetaMin = ThetaMin;
  Thetam   = 0.0;
  Thetasum = 0.0;
  k0       = 0.0;
  k1       = 0.0;
  k2k2     = 0.0;
};

double TwoTemperature::CoulombLog(double ne, float Te) {
  return 35.4 + std::log((Te / 1.e7) * std::sqrt(1.0e-3 / ne));
}

double TwoTemperature::BesselTerms(double Thetae, double Thetai) {
  if ((Thetae < ThetaMin) && (Thetai < ThetaMin)) {
    return coldEcoldI(Thetae, Thetai);
  } else if ((Thetae >= ThetaMin) && (Thetai < ThetaMin)) {
    return hotEcoldI(Thetae, Thetai);
  } else if ((Thetae < ThetaMin) && (Thetai >= ThetaMin)) {
    return coldEhotI(Thetae, Thetai);
  } else {
    return hotEhotI(Thetae, Thetai);
  }
}

double TwoTemperature::coldEcoldI(double Thetae, double Thetai) {
  Thetasum = Thetae + Thetai;
  return (2. * Thetasum * Thetasum + 2.0 * Thetasum + 1.0)
  // sqrt(0.5 pi Ts^3) == sqrt(0.5 pi) * Ts * sqrt(Ts); avoids a pow()
       / (std::sqrt(0.5 * M_PI) * Thetasum * std::sqrt(Thetasum));
}

double TwoTemperature::hotEcoldI(double Thetae, double Thetai) { 
  Thetasum  = Thetae + Thetai;
  double numer = (2. * (Thetasum * Thetasum) + 2.0 * Thetasum + 1.0)
               * std::sqrt(Thetae);
  double denom = std::cyl_bessel_k(2, 1.0 / Thetae)
               * (std::exp(1.0 / Thetae) * Thetasum * std::sqrt(Thetasum));
  return numer / denom;
}

double TwoTemperature::coldEhotI(double Thetae, double Thetai) {
  Thetasum  = Thetae + Thetai;
  double numer = (2. * (Thetasum * Thetasum) + 2.0 * Thetasum + 1.0)
               * std::sqrt(Thetai);
  double denom = std::cyl_bessel_k(2, 1.0 / Thetai)
               * (std::exp(1.0 / Thetai) * Thetasum * std::sqrt(Thetasum));
  return numer / denom;
}

double TwoTemperature::hotEhotI(double Thetae, double Thetai) {
  Thetasum = Thetae + Thetai;
  Thetam   = (Thetae * Thetai) / Thetasum;
  k2k2     = ((1.0 / std::cyl_bessel_k(2, 1.0 / Thetai)) * 
              (1.0 / std::cyl_bessel_k(2, 1.0 / Thetae)));
  k1       = ((2.0 * Thetasum * Thetasum + 1.0) / Thetasum) * (
              std::cyl_bessel_k(1, 1.0 / Thetam));
  k0       = 2.0 * std::cyl_bessel_k(0, 1.0 / Thetam);
  return k2k2 * (k1 + k0);
}

TwoTemperature::~TwoTemperature() {};

// Opacity class
Opacity::Opacity() {};

Opacity::Opacity(const std::unordered_map<std::string, double>& params,
    const std::unordered_map<std::string, double>& CONSTANTS) {
  rho0 = params.at("density_scale");
  c    = CONSTANTS.at("c");
  kB   = CONSTANTS.at("kB");
  aRAD = CONSTANTS.at("aRAD");
  xAB  = CONSTANTS.at("X_AB");
  yAB  = CONSTANTS.at("Y_AB");
  zAB  = CONSTANTS.at("Z_AB");
  mu_i = 4.0 / (4.0 * xAB + yAB);
  mu_e = 2.0 / (1.0 + xAB);
  mp   = CONSTANTS.at("mp");
  mi   = (xAB + yAB) * mp;
  me   = CONSTANTS.at("me");
  sigT = CONSTANTS.at("sigmaT");
  eStatC = CONSTANTS.at("e");
  h    = CONSTANTS.at("h");
  kappaMol = 3.0 * zAB;
  // Loop-invariant: depends only on physical constants
  e_2pimc  = eStatC / (2.0 * M_PI * me * c);

  this->params = params;
  this->CONSTANTS = CONSTANTS;
};

// Bound-free opacities
double Opacity::kappaBoundFreeAbs(float rho, float Te, float Tr) {
  return (9.0e26 * (zAB * (1.0 + xAB + 0.75 * yAB) * (
    rho * (1.0 / std::sqrt(Te)) * (1.0 / (Tr * Tr * Tr)) * (
    std::log(1.0 + 1.6 * (Tr / Te))))));
}

double Opacity::kappaBoundFreeEmi(float rho, float Te, float Tr) {
  // Te^-3.5 == 1 / (Te^3 sqrt(Te)), ~9x cheaper than pow(). Promoted to
  // double first so the cubing keeps the precision pow() would have given.
  double T = Te;
  return (9.0e26 * (zAB * (1.0 + xAB + 0.75 * yAB) * (
    rho * (1.0 / (T * T * T * std::sqrt(T))) * std::log(1.0 + 1.6))));
}

// CHIANTI opacities
double Opacity::kappaCHIANTIAbs(float rho, float Te, float Tr) {
  return (3.0e34 * rho * (0.1 + (zAB / 0.02)) * ( 
    (std::pow(Te, -1.7) * (1.0 / (Tr * Tr * Tr)))));
}

double Opacity::kappaCHIANTIEmi(float rho, float Te, float Tr) {
  return (3.0e34 * rho * ((0.1 + (zAB / 0.02)) * std::pow(Te, -4.7)));
}

// Free-free opacities
double Opacity::kappaFreeFreeAbs(float rho, float Te, float Tr) {
  return (1.2e24 * (1.0 + xAB) * (1.0 - zAB) * rho * ( 
    (1.0 / std::sqrt(Te)) * (1.0 / (Tr * Tr * Tr)) * (
    std::log(1.0 + 1.6 * (Tr / Te)) * (1.0 + 4.4e-10 * Te))));
}

double Opacity::kappaFreeFreeEmi(float rho, float Te, float Tr) {
  // Te^-3.5 == 1 / (Te^3 sqrt(Te)); see kappaBoundFreeEmi
  double T = Te;
  return (1.2e24 * (1.0 + xAB) * (1.0 - zAB) * rho * (
    (1.0 / (T * T * T * std::sqrt(T))) * std::log(1.0 + 1.6)
    * (1.0 + 4.4e-10 * Te)));
}

// Hydrogen opacity
double Opacity::kappaH(float rho, float Te, float Tr) {
  return 33.0 * 1.e-25 * std::sqrt(zAB) * std::sqrt(rho) * std::pow(Te, 7.7);
}

// Iron-line opacity
double Opacity::kappaFe(float rho, float Te, float Tr) {
  return (0.3 * (zAB / 0.02) * (
    std::exp(-6.0 * (std::pow((-12.0 + std::log(Te)), 2)))));
}

// Synchrotron opacities
double Opacity::kappaSyncAbs(float rho, float Te, 
    float Tr, float b2, double async, double bsync,
    double csync, double dsync, double esync) {
  ne     = rho / (mu_e * mp);
  Thetae = (kB * Te) / (me * c * c);
  nu_mu  = (3.0/2.0) * e_2pimc * std::sqrt(b2) * Thetae * Thetae;
  phi    = kB * Tr / (h * nu_mu);
  g = 5.85374e-14 * ne * phi * (1.0 / (Thetae * Thetae * Thetae * Tr));
  kappaGInv = (
    1.0 / (async * std::pow(phi, -bsync) * std::log(1.0 + csync * phi)) + (
    1.0 / (dsync * std::pow(phi, -esync))));
  return (g / kappaGInv) / rho;
}

double Opacity::kappaSyncEmi(float rho, float Te, 
    float Tr, float b2, double async, double bysnc, 
    double csync, double dsync, double esync) {
  ne     = rho / (mu_e * mp);
  Thetae = (kB * Te) / (me * c * c);
  nu_mu  = (3.0/2.0) * e_2pimc * std::sqrt(b2) * Thetae * Thetae;
  phi    = kB * Te / (h * nu_mu);
  g = 5.85374e-14 * ne * phi * (1.0 / (Thetae * Thetae * Thetae * Te));
  kappaGInv = (
    1.0 / (async * std::pow(phi, -bsync) * std::log(1.0 + csync * phi)) + (
    1.0 / (dsync * std::pow(phi, -esync))));
  return (g / kappaGInv) / rho;
}

void Opacity::getABCDESynch(int XRB, double& a, double& b, double& c, 
    double& d, double& e) {
  if (XRB) { // XRB synchrotron opacity coefficients
    a = 1.27; 
    b = 1.03; 
    c = 0.000763;
    d = 0.616;
    e = 2.91;
  } else { // AGN synchrotron opacity coefficients
    a = 0.206; 
    b = 2.59;
    c = 3.44;
    d = 9.33;
    e = 3.09;
  }
}

// Total absorption and emission opacities
double Opacity::kappaAbsTotal(float rho, float Te, float Tr, float b2,
    int XRB) {
  getABCDESynch(XRB, async, bsync, csync, dsync, esync);
  double kappa_sync  = kappaSyncAbs(rho, Te, Tr, b2,
                                    async, bsync, csync, dsync, esync);
  double kappa_lowT  = kappaH(rho, Te, Tr) + kappaMol;
  double kappa_highT = kappaBoundFreeAbs(rho, Te, Tr)
                     + kappaFreeFreeAbs(rho, Te, Tr)
                     + kappaCHIANTIAbs(rho, Te, Tr);
  // Low-T and high-T channels add harmonically -- the smaller one dominates
  return kappa_sync + 1.0 / (1.0 / kappa_lowT + 1.0 / kappa_highT);
}

double Opacity::kappaEmiTotal(float rho, float Te, float Tr, float b2,
    int XRB) {
  getABCDESynch(XRB, async, bsync, csync, dsync, esync);
  double kappa_sync  = kappaSyncEmi(rho, Te, Tr, b2,
                                    async, bsync, csync, dsync, esync);
  double kappa_lowT  = kappaH(rho, Te, Tr) + kappaMol;
  double kappa_highT = kappaBoundFreeEmi(rho, Te, Tr)
                     + kappaFreeFreeEmi(rho, Te, Tr)
                     + kappaCHIANTIEmi(rho, Te, Tr);
  // Low-T and high-T channels add harmonically -- the smaller one dominates
  return kappa_sync + 1.0 / (1.0 / kappa_lowT + 1.0 / kappa_highT);
}

// Opacities exposed to Cython/Python
// Total absorption and emission opacities
void Opacity::kappaAbsTotal(double* kappa, float* rho, float* Te, float* Tr,
  float* b2, int XRB, int nb, int nx1, int nx2, int nx3) {
  // Loop over each of nb * nx1 * nx2 * nx3 cells
  gridsize_3D = nb * nx1 * nx2 * nx3;
  int n;
  getABCDESynch(XRB, async, bsync, csync, dsync, esync);
  Opacity opac(this->params, this->CONSTANTS);
  #pragma omp parallel for private(n) firstprivate(opac)
  for (n = 0; n < gridsize_3D; n++) {
    double kappa_sync  = opac.kappaSyncAbs(rho[n], Te[n], Tr[n], b2[n],
                                   async, bsync, csync, dsync, esync);
    double kappa_lowT  = opac.kappaH(rho[n], Te[n], Tr[n]) + kappaMol;
    double kappa_highT = opac.kappaBoundFreeAbs(rho[n], Te[n], Tr[n])
                       + opac.kappaFreeFreeAbs(rho[n], Te[n], Tr[n])
                       + opac.kappaCHIANTIAbs(rho[n], Te[n], Tr[n]);
    kappa[n] = kappa_sync + 1.0 / (1.0 / kappa_lowT + 1.0 / kappa_highT);
  }
}

void Opacity::kappaEmiTotal(double* kappa, float* rho, float* Te, float* Tr,
  float* b2, int XRB, int nb, int nx1, int nx2, int nx3) {
  // Loop over each of nb * nx1 * nx2 * nx3 cells
  gridsize_3D = nb * nx1 * nx2 * nx3;
  int n;
  getABCDESynch(XRB, async, bsync, csync, dsync, esync);
  Opacity opac(this->params, this->CONSTANTS);
  #pragma omp parallel for private(n) firstprivate(opac)
  for (n = 0; n < gridsize_3D; n++) {
    double kappa_sync  = opac.kappaSyncEmi(rho[n], Te[n], Tr[n], b2[n],
                                   async, bsync, csync, dsync, esync);
    double kappa_lowT  = opac.kappaH(rho[n], Te[n], Tr[n]) + kappaMol;
    double kappa_highT = opac.kappaBoundFreeEmi(rho[n], Te[n], Tr[n])
                       + opac.kappaFreeFreeEmi(rho[n], Te[n], Tr[n])
                       + opac.kappaCHIANTIEmi(rho[n], Te[n], Tr[n]);
    kappa[n] = kappa_sync + 1.0 / (1.0 / kappa_lowT + 1.0 / kappa_highT);
  }
}

// Synchrotron opacity exposed to Cython/Python
void Opacity::kappaSynchrotron(double* kappa, float* rho, float* Te, 
    float* Tr, float* b2, int emi, int XRB, 
    int nb, int nx1, int nx2, int nx3) {
  // Loop over each of nb * nx1 * nx2 * nx3 cells
  int n;
  // Private (P) variables in OMP threads
  double neP, ThetaeP, nu_muP, phiP, gP, kappaGInvP;
  gridsize_3D = nb * nx1 * nx2 * nx3;
  getABCDESynch(XRB, async, bsync, csync, dsync, esync);
  if (emi) {
    #pragma omp parallel for private(n, \
      neP, ThetaeP, nu_muP, phiP, gP, kappaGInvP) 
    for (n = 0; n < gridsize_3D; n++) {
      neP     = rho[n] / (mu_e * mp);
      ThetaeP = (kB * Te[n]) / (me * c * c);
      nu_muP  = (3.0/2.0) * e_2pimc * std::sqrt(b2[n]) * ThetaeP * ThetaeP;
      phiP    = kB * Te[n] / (h * nu_muP);
      gP      = (5.85374e-14 * neP * phiP * (
        1.0 / (ThetaeP * ThetaeP * ThetaeP * Te[n])));
      kappaGInvP = (
        1.0 / (async * std::pow(phiP, -bsync) * std::log(1.0 + csync * phiP)) 
        + (1.0 / (dsync * std::pow(phiP, -esync))));
      kappa[n] = (gP / kappaGInvP) / rho[n];
    }
  } else {
    #pragma omp parallel for private(n, \
      neP, ThetaeP, nu_muP, phiP, gP, kappaGInvP) 
    for (n = 0; n < gridsize_3D; n++) {
      neP     = rho[n] / (mu_e * mp);
      ThetaeP = (kB * Te[n]) / (me * c * c);
      nu_muP  = (3.0/2.0) * e_2pimc * std::sqrt(b2[n]) * ThetaeP * ThetaeP;
      phiP    = kB * Tr[n] / (h * nu_muP);
      gP      = (5.85374e-14 * neP * phiP * (
        1.0 / (ThetaeP * ThetaeP * ThetaeP * Tr[n])));
      kappaGInvP = (
        1.0 / (async * std::pow(phiP, -bsync) * std::log(1.0 + csync * phiP)) 
        + (1.0 / (dsync * std::pow(phiP, -esync))));
      kappa[n] = (gP / kappaGInvP) / rho[n];
    }
  }
}

Opacity::~Opacity() {};

// RIAF Opacity class
// Use synchrotron opacities from Sadowski+ (2017) instead of McKinney+ (2017)
// Appropriate for low-Mdot radiatively inefficient accretion flows
RIAF_Opacity::RIAF_Opacity() {};

RIAF_Opacity::RIAF_Opacity(const std::unordered_map<std::string, double>& params,
    const std::unordered_map<std::string, double>& CONSTANTS) 
    : Opacity(params, CONSTANTS) {
  // Sadowski+ (2017) Equation 42 -- built only from physical constants,
  // so compute it once here instead of once per grid cell.
  pime3c5_ekh = (M_PI * me * me * me * std::pow(c, 5)) / (eStatC * kB * h);
};

// Synchrotron opacities
// Override inherited functions
double RIAF_Opacity::kappaSyncAbs(float rho, float Te, float Tr, float b2) {
  ne          = rho / (mu_e * mp);
  xi          = Tr / Te;
  // Sadowski+ (2017) Equation 41
  kappa_emi   = 1.59e-30 * ne * b2 / (Te * Te * rho);
  ksi         = (4.0 / 3.0) * pime3c5_ekh * Tr / (std::sqrt(b2) * Te * Te);
  // Sadowski+ (2017) Equation 43
  // ksi^-2/3 via cbrt, and ksi^-4/3 is its square: ~4x cheaper than 2 pow()
  double ksi23 = 1.0 / std::cbrt(ksi * ksi);
  return kappa_emi * (1.0 / (xi * xi * xi)) * (1.0 / (
    1.0 + 5.444 * ksi23 + 7.218 * ksi23 * ksi23));
} 

double RIAF_Opacity::kappaSyncEmi(float rho, float Te, float Tr, float b2) {
  ne = rho / (mu_e * mp);
  // Sadowski+ (2017) Equation 41
  return 1.59e-30 * ne * b2 / (Te * Te * rho);
} 
  
// Total absorption and emission opacities
double RIAF_Opacity::kappaAbsTotal(float rho, float Te, float Tr, 
    float b2, int XRB) {
  double kappa_sync  = kappaSyncAbs(rho, Te, Tr, b2);
  double kappa_lowT  = kappaH(rho, Te, Tr) + kappaMol;
  double kappa_highT = kappaBoundFreeAbs(rho, Te, Tr)
                     + kappaFreeFreeAbs(rho, Te, Tr)
                     + kappaCHIANTIAbs(rho, Te, Tr);
  // Low-T and high-T channels add harmonically -- the smaller one dominates
  return kappa_sync + 1.0 / (1.0 / kappa_lowT + 1.0 / kappa_highT);
}

double RIAF_Opacity::kappaEmiTotal(float rho, float Te, float Tr, 
    float b2, int XRB) {
  double kappa_sync  = kappaSyncEmi(rho, Te, Tr, b2);
  double kappa_lowT  = kappaH(rho, Te, Tr) + kappaMol;
  double kappa_highT = kappaBoundFreeEmi(rho, Te, Tr)
                     + kappaFreeFreeEmi(rho, Te, Tr)
                     + kappaCHIANTIEmi(rho, Te, Tr);
  // Low-T and high-T channels add harmonically -- the smaller one dominates
  return kappa_sync + 1.0 / (1.0 / kappa_lowT + 1.0 / kappa_highT);
}
  
// Opacities exposed to Cython/Python
// Total absorption and emission opacities
void RIAF_Opacity::kappaAbsTotal(double* kappa, float* rho, float* Te, float* Tr,
  float* b2, int XRB, int nb, int nx1, int nx2, int nx3) {
  // Loop over each of nb * nx1 * nx2 * nx3 cells
  gridsize_3D = nb * nx1 * nx2 * nx3;
  int n;
  RIAF_Opacity opac(this->params, this->CONSTANTS);
  #pragma omp parallel for private(n) firstprivate(opac)
  for (n = 0; n < gridsize_3D; n++) {
    double kappa_sync  = opac.kappaSyncAbs(rho[n], Te[n], Tr[n], b2[n]);
    double kappa_lowT  = opac.kappaH(rho[n], Te[n], Tr[n]) + kappaMol;
    double kappa_highT = opac.kappaBoundFreeAbs(rho[n], Te[n], Tr[n])
                       + opac.kappaFreeFreeAbs(rho[n], Te[n], Tr[n])
                       + opac.kappaCHIANTIAbs(rho[n], Te[n], Tr[n]);
    kappa[n] = kappa_sync + 1.0 / (1.0 / kappa_lowT + 1.0 / kappa_highT);
  }
}

void RIAF_Opacity::kappaEmiTotal(double* kappa, float* rho, float* Te, float* Tr,
  float* b2, int XRB, int nb, int nx1, int nx2, int nx3) {
  // Loop over each of nb * nx1 * nx2 * nx3 cells
  gridsize_3D = nb * nx1 * nx2 * nx3;
  int n;
  RIAF_Opacity opac(this->params, this->CONSTANTS);
  #pragma omp parallel for private(n) firstprivate(opac)
  for (n = 0; n < gridsize_3D; n++) {
    double kappa_sync  = opac.kappaSyncEmi(rho[n], Te[n], Tr[n], b2[n]);
    double kappa_lowT  = opac.kappaH(rho[n], Te[n], Tr[n]) + kappaMol;
    double kappa_highT = opac.kappaBoundFreeEmi(rho[n], Te[n], Tr[n])
                       + opac.kappaFreeFreeEmi(rho[n], Te[n], Tr[n])
                       + opac.kappaCHIANTIEmi(rho[n], Te[n], Tr[n]);
    kappa[n] = kappa_sync + 1.0 / (1.0 / kappa_lowT + 1.0 / kappa_highT);
  }
}
  
// Synchrotron opacity exposed to Cython/Python
void RIAF_Opacity::kappaSynchrotron(double* kappa, float* rho, float* Te, 
    float* Tr, float* b2, int emi, int XRB, 
    int nb, int nx1, int nx2, int nx3) {
  // Loop over each of nb * nx1 * nx2 * nx3 cells
  int n;
  // Private (P) variables in OMP threads
  double neP, ksiP, xiP, kappa_emiP;
  gridsize_3D = nb * nx1 * nx2 * nx3;
  if (emi) {
    #pragma omp parallel for private(n, neP)
    for (n = 0; n < gridsize_3D; n++) {
      neP = rho[n] / (mu_e * mp);
      // Sadowski+ (2017) Equation 41
      kappa[n] = 1.59e-30 * neP * b2[n] / (Te[n] * Te[n] * rho[n]);
    }
  } else {
    #pragma omp parallel for private(n, neP, ksiP, xiP, kappa_emiP)
    for (n = 0; n < gridsize_3D; n++) {
      neP = rho[n] / (mu_e * mp);
      ksiP= (4.0/3.0) * pime3c5_ekh * Tr[n]/(std::sqrt(b2[n]) * Te[n] * Te[n]);
      xiP = Tr[n] / Te[n];
      // Sadowski+ (2017) Equation 41
      kappa_emiP = 1.59e-30 * neP * b2[n] / (Te[n] * Te[n] * rho[n]);
      double ksi23P = 1.0 / std::cbrt(ksiP * ksiP);
      kappa[n]   = kappa_emiP * (1.0 / (xiP * xiP * xiP)) * (1.0 / (1.0 +
        5.444 * ksi23P + 7.218 * ksi23P * ksi23P));
    }
  }
}

RIAF_Opacity::~RIAF_Opacity() {};

// Radiation class
Radiation::Radiation() {};

Radiation::Radiation(const std::unordered_map<std::string, double>& params,
    const std::unordered_map<std::string, double>& CONSTANTS) 
{
  rho0 = params.at("density_scale");
  c    = CONSTANTS.at("c");
  kB   = CONSTANTS.at("kB");
  aRAD = CONSTANTS.at("aRAD");
  xAB  = CONSTANTS.at("X_AB");
  yAB  = CONSTANTS.at("Y_AB");
  zAB  = CONSTANTS.at("Z_AB");
  mu_i = 4.0 / (4.0 * xAB + yAB);
  mu_e = 2.0 / (1.0 + xAB);
  mp   = CONSTANTS.at("mp");
  mi   = (xAB + yAB) * mp;
  me   = CONSTANTS.at("me");
  sigT = CONSTANTS.at("sigmaT");
  eStatC = CONSTANTS.at("e");
  kappaMol = 3.0 * zAB;
  this->params = params;
  this->CONSTANTS = CONSTANTS;
};

// Total absorption and emission rates
void Radiation::qAbs(double* qabs, float* rho, float* Te, float* Tr,
  float* b2, int XRB, int RIAF, int nb, int nx1, int nx2, int nx3) {
  // Loop over each of nb * nx1 * nx2 * nx3 cells
  gridsize_3D = nb * nx1 * nx2 * nx3;
  int n;
  double kappaAbs;

  if (!RIAF) {
    Opacity opac(this->params, this->CONSTANTS);
    #pragma omp parallel for private(n, kappaAbs) firstprivate(opac)
    for (n = 0; n < gridsize_3D; n++) {
      kappaAbs = opac.kappaAbsTotal(rho[n], Te[n], Tr[n], b2[n], XRB);
      qabs[n]  = kappaAbs * rho[n] * aRAD * Tr[n] * Tr[n] * Tr[n] * Tr[n] * c;
    }
  } else {
    RIAF_Opacity opac(this->params, this->CONSTANTS);
    #pragma omp parallel for private(n, kappaAbs) firstprivate(opac)
    for (n = 0; n < gridsize_3D; n++) {
      kappaAbs = opac.kappaAbsTotal(rho[n], Te[n], Tr[n], b2[n], XRB);
      qabs[n]  = kappaAbs * rho[n] * aRAD * Tr[n] * Tr[n] * Tr[n] * Tr[n] * c;
    }
  }
}

void Radiation::qEmi(double* qemi, float* rho, float* Te, float* Tr,
  float* b2, int XRB, int RIAF, int nb, int nx1, int nx2, int nx3) {
  // Loop over each of nb * nx1 * nx2 * nx3 cells
  gridsize_3D = nb * nx1 * nx2 * nx3;
  int n;
  double kappaEmi;

  if (!RIAF) {
    Opacity opac(this->params, this->CONSTANTS);
    #pragma omp parallel for private(n, kappaEmi) firstprivate(opac)
    for (n = 0; n < gridsize_3D; n++) {
      kappaEmi = opac.kappaEmiTotal(rho[n], Te[n], Tr[n], b2[n], XRB);
      qemi[n]  = kappaEmi * rho[n] * aRAD * Te[n] * Te[n] * Te[n] * Te[n] * c;
    }
  } else {
    RIAF_Opacity opac(this->params, this->CONSTANTS);
    #pragma omp parallel for private(n, kappaEmi) firstprivate(opac)
    for (n = 0; n < gridsize_3D; n++) {
      kappaEmi = opac.kappaEmiTotal(rho[n], Te[n], Tr[n], b2[n], XRB);
      qemi[n]  = kappaEmi * rho[n] * aRAD * Te[n] * Te[n] * Te[n] * Te[n] * c;
    }
  }
}

Radiation::~Radiation() {};