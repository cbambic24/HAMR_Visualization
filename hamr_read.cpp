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

TwoTemperature::TwoTemperature (TwoTemperature& twoT) {
  ThetaMin = twoT.ThetaMin;
  Thetam   = twoT.Thetam;
  Thetasum = twoT.Thetasum;
  k0       = twoT.k0;
  k1       = twoT.k1;
  k2k2     = twoT.k2k2;
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
  return ((2. * Thetasum * Thetasum + 2.0 * Thetasum + 1.0) / (
    std::sqrt(0.5 * M_PI * std::pow(Thetasum, 3.0))));
}

double TwoTemperature::hotEcoldI(double Thetae, double Thetai) { 
  Thetasum  = Thetae + Thetai;
  return (((2. * (Thetasum * Thetasum) + 2.0 * Thetasum + 1.0) * (
    std::sqrt(Thetae))) / (std::cyl_bessel_k(2, 1.0 / Thetae) * (
    std::exp(1.0 / Thetae) * std::sqrt(std::pow(Thetasum, 3.0)))));
}

double TwoTemperature::coldEhotI(double Thetae, double Thetai) {
  Thetasum  = Thetae + Thetai;
  return (((2. * (Thetasum * Thetasum) + 2.0 * Thetasum + 1.0) * (
    std::sqrt(Thetai))) / (std::cyl_bessel_k(2, 1.0 / Thetai) * (
    std::exp(1.0 / Thetai) * std::sqrt(std::pow(Thetasum, 3.0)))));
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
};

Opacity::Opacity (Opacity& opac) {
  rho0 = opac.rho0;
  c    = opac.c;
  kB   = opac.kB;
  aRAD = opac.aRAD;
  xAB  = opac.xAB;
  yAB  = opac.yAB;
  zAB  = opac.zAB;
  mu_i = opac.mu_i;
  mu_e = opac.mu_e;
  mp   = opac.mp;
  mi   = opac.mi;
  me   = opac.me;
  sigT = opac.sigT;
  eStatC = opac.eStatC;
  h    = opac.h; 
  kappaMol = opac.kappaMol;
  
  // copy variables used within member functions
  e_2pimc   = opac.e_2pimc;
  ne        = opac.ne;
  Thetae    = opac.Thetae;
  nu_mu     = opac.nu_mu;
  phi       = opac.phi;
  g         = opac.g;
  kappaGInv = opac.kappaGInv;
};

// Bound-free opacities
double Opacity::kappaBoundFreeAbs(float rho, float Te, float Tr) {
  return (9.0e26 * (zAB * (1.0 + xAB + 0.75 * yAB) * (
    rho * (1.0 / std::sqrt(Te)) * (1.0 / (Tr * Tr * Tr)) * (
    std::log(1.0 + 1.6 * (Tr / Te))))));
}

double Opacity::kappaBoundFreeEmi(float rho, float Te, float Tr) {
  return (9.0e26 * (zAB * (1.0 + xAB + 0.75 * yAB) * (
    rho * std::pow(Te, -3.5) * std::log(1.0 + 1.6))));
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
  return (1.2e24 * (1.0 + xAB) * (1.0 - zAB) * rho * (
    std::pow(Te, -3.5) * std::log(1.0 + 1.6) * (1.0 + 4.4e-10 * Te)));
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
  e_2pimc = eStatC / (2.0 * M_PI * me * c);
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
  e_2pimc = eStatC / (2.0 * M_PI * me * c);
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
double Opacity::kappaAbsTotal(float rho, float Te, float Tr, 
    float b2, int XRB) {
  getABCDESynch(XRB, async, bsync, csync, dsync, esync);
  return kappaSyncAbs(rho, Te, Tr, b2, 
    async, bsync, csync, dsync, esync) + ( 
    (1.0 / ((1.0 / (kappaH(rho, Te, Tr) + kappaMol)) + 
    (1.0 / (kappaBoundFreeAbs(rho, Te, Tr) + 
	     kappaFreeFreeAbs(rho, Te, Tr) + 
             kappaCHIANTIAbs( rho, Te, Tr)))))); 
}

double Opacity::kappaEmiTotal(float rho, float Te, float Tr, 
    float b2, int XRB) {
  getABCDESynch(XRB, async, bsync, csync, dsync, esync);
  return kappaSyncEmi(rho, Te, Tr, b2, 
    async, bsync, csync, dsync, esync) + ( 
    (1.0 / ((1.0 / (kappaH(rho, Te, Tr) + kappaMol)) + 
    (1.0 / (kappaBoundFreeEmi(rho, Te, Tr) + 
	     kappaFreeFreeEmi(rho, Te, Tr) + 
             kappaCHIANTIEmi( rho, Te, Tr)))))); 
}

// Opacities exposed to Cython/Python
// Total absorption and emission opacities
void Opacity::kappaAbsTotal(double* kappa, float* rho, float* Te, float* Tr,
  float *b2, int XRB, int nb, int nx1, int nx2, int nx3) {
  // Loop over each of nb * nx1 * nx2 * nx3 cells
  gridsize_3D = nb * nx1 * nx2 * nx3;
  int n;
  getABCDESynch(XRB, async, bsync, csync, dsync, esync);
  #pragma omp parallel for private(n)
  for (n = 0; n < gridsize_3D; n++) {
    kappa[n] = kappaSyncAbs(rho[n], Te[n], Tr[n], b2[n], 
      async, bsync, csync, dsync, esync) + ( 
      (1.0 / ((1.0 / (kappaH(rho[n], Te[n], Tr[n]) + kappaMol)) + 
      (1.0 / (kappaBoundFreeAbs(rho[n], Te[n], Tr[n]) + 
	      kappaFreeFreeAbs( rho[n], Te[n], Tr[n]) + 
      kappaCHIANTIAbs(rho[n], Te[n], Tr[n])))))); 
  }
}

void Opacity::kappaEmiTotal(double* kappa, float* rho, float* Te, float* Tr,
  float *b2, int XRB, int nb, int nx1, int nx2, int nx3) {
  // Loop over each of nb * nx1 * nx2 * nx3 cells
  gridsize_3D = nb * nx1 * nx2 * nx3;
  int n;
  getABCDESynch(XRB, async, bsync, csync, dsync, esync);
  #pragma omp parallel for private(n)
  for (n = 0; n < gridsize_3D; n++) {
    kappa[n] = kappaSyncEmi(rho[n], Te[n], Tr[n], b2[n], 
      async, bsync, csync, dsync, esync) + ( 
      (1.0 / ((1.0 / (kappaH(rho[n], Te[n], Tr[n]) + kappaMol)) + 
      (1.0 / (kappaBoundFreeEmi(rho[n], Te[n], Tr[n]) + 
	      kappaFreeFreeEmi( rho[n], Te[n], Tr[n]) + 
      kappaCHIANTIEmi(rho[n], Te[n], Tr[n])))))); 
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
  e_2pimc = eStatC / (2.0 * M_PI * me * c);
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
  float *b2, int XRB, int nb, int nx1, int nx2, int nx3) {
  // Loop over each of nb * nx1 * nx2 * nx3 cells
  gridsize_3D = nb * nx1 * nx2 * nx3;
  Opacity opac(this->params, this->CONSTANTS);
  int n;
  double kappaAbs;
  #pragma omp parallel for private(n, kappaAbs) firstprivate(opac)
  for (n = 0; n < gridsize_3D; n++) {
    kappaAbs = opac.kappaAbsTotal(rho[n], Te[n], Tr[n], b2[n], XRB); 
    qabs[n]  = kappaAbs * rho[n] * aRAD * Tr[n] * Tr[n] * Tr[n] * Tr[n] * c;  
  }
}

void Radiation::qEmi(double* qemi, float* rho, float* Te, float* Tr,
  float *b2, int XRB, int nb, int nx1, int nx2, int nx3) {
  // Loop over each of nb * nx1 * nx2 * nx3 cells
  gridsize_3D = nb * nx1 * nx2 * nx3;
  Opacity opac(this->params, this->CONSTANTS);
  int n;
  double kappaEmi;
  #pragma omp parallel for private(n, kappaEmi) firstprivate(opac)
  for (n = 0; n < gridsize_3D; n++) {
    kappaEmi = opac.kappaEmiTotal(rho[n], Te[n], Tr[n], b2[n], XRB); 
    qemi[n]  = kappaEmi * rho[n] * aRAD * Te[n] * Te[n] * Te[n] * Te[n] * c;  
  }
}

Radiation::~Radiation() {};

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

  // physics flags
  RADM1 = int(physics.at("RAD_M1"));
  RES   = int(physics.at("RESISTIVE"));
  TWOT  = int(physics.at("TWO_T"));
  PNUM  = int(physics.at("P_NUM"));
  DOYE  = int(physics.at("DO_YE"));
  DONUC = int(physics.at("DO_NUCLEAR"));
  NUM1  = int(physics.at("NEUTRINOS_M1"));
  DOTR  = int(physics.at("DOTRACER"));

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

  long int gridsize_2D, gridsize_3D, index_2D, index_3D;
  int n; // index over blocks
  int iGlob, jGlob, kGlob; // global indices
  int ii; // index over metric components
  AMRVar iStrt, iEnd, iStp; // loop variables
  bool iInRange, jInRange, kInRange;
  std::string filename;
  FILE* fin;
  int read_data, read_file;
  ReadIn readIn; 

  #pragma omp parallel for private(n, iGlob, jGlob, kGlob, ii, \
    iStrt, iEnd, iStp, index_3D, \
    index_2D, gridsize_3D, gridsize_2D, \
    iInRange, jInRange, kInRange, \
    filename, fin, read_data, read_file) firstprivate(readIn)
  for (n = 0; n < nb; n++) {
    gridsize_2D = (i_max - i_min) * (j_max - j_min);
    gridsize_3D = (i_max - i_min) * (j_max - j_min) * (k_max - k_min);
    std::vector<double> tempArray(58);
    std::vector<int> levelArray(gridsize_2D);

    iInRange = false;
    jInRange = false;
    kInRange = false;
    
    filename = dir + "/gdumps/gdump" + std::to_string(n_ord[n]);

    if (omp_get_thread_num() == 0) {
      std::cout << "Loading " + filename + " on rank 0." << std::endl;
    }

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
                readIn.readMetricData3D(fin, tempArray, index_3D, gridsize_3D, 
                  n, iGlob, jGlob, kGlob, *this);
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
    fclose(fin);
  } // loop over n
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

  long int gridsize_3D, index_3D;
  int n; // index over blocks
  int iGlob, jGlob, kGlob; // global indices
  AMRVar iStrt, iEnd, iStp; // loop variables
  bool iInRange, jInRange, kInRange;
  std::string filename;
  FILE* fin;
  int read_data, read_file;
  ReadIn readIn; 

  #pragma omp parallel for private(n, iGlob, jGlob, kGlob, \
    iStrt, iEnd, iStp, index_3D, gridsize_3D, iInRange, jInRange, kInRange, \
    filename, fin, read_data, read_file) firstprivate(readIn)
  for (n = 0; n < nb; n++) {
    gridsize_3D = (i_max - i_min) * (j_max - j_min) * (k_max - k_min);
    std::vector<double> tempArray(58);

    iInRange = false;
    jInRange = false;
    kInRange = false;
  
    filename = dir + "/gdumps/gdump" + std::to_string(n_ord[n]);

    if (omp_get_thread_num() == 0) {
      std::cout << "Loading " + filename + " on rank 0." << std::endl;
    }

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
    fclose(fin);
  } // loop over n
}

void Read::readPrims(int fnum, float* rho, float* uGas, float* uu, float* B, 
    float* E, float* uRad, float* uuRad, 
    float* Te, float* Ti, float* phNum,
    float* Ye, float* xAlpha, float* xAtm, 
    float* uNu1, float* uNu2, float* uNu3, 
    float* uuNu1, float* uuNu2, float* uuNu3,
    float* nuNum1, float* nuNum2, float* nuNum3, float* tr1, float* tr2) {
  
  nvar = nPrim + RADM1 * nRadM1 + RES * nRes + TWOT * nTwoT + 
    PNUM * nPnum + DOYE * nYe + DONUC * nNUC + NUM1 * nNuM1 + DOTR * nTr;

  isize = (k_max - k_min) * (j_max - j_min);
  jsize = (k_max - k_min);

  long int gridsize_3D, index_3D;
  int n, nStrt, nEnd; // variables for looping over blocks
  int iFile, vStrt, ii; // file index and variable start index (vStrt)
  int iGlob, jGlob, kGlob; // global indices
  AMRVar iStrt, iEnd, iStp; // loop variables
  bool iInRange, jInRange, kInRange;
  BoolVar iNotInBlock; // is data in selection box/ on block?
  std::string filename;
  FILE* fin;
  int read_data;
  ReadIn readIn; 

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

  #pragma omp parallel for private(iFile, n, nStrt, nEnd, vStrt, \
    iGlob, jGlob, kGlob, ii, iStrt, iEnd, iStp, index_3D, gridsize_3D, \
    iInRange, jInRange, kInRange, iNotInBlock, \
    filename, fin, read_data) firstprivate(readIn)
  for (iFile = 0; iFile < fileCount; iFile++) {
    gridsize_3D = (i_max - i_min) * (j_max - j_min) * (k_max - k_min);
    std::vector<float> tempArray(nvar);

    iInRange = false;
    jInRange = false;
    kInRange = false;

    filename = dir + "/dumps" + std::to_string(fnum) + 
    "/new_dump" + std::to_string(iFile);
    
    checkFileExists(filename);
    fin = fopen(filename.c_str(), "rb");
    if (omp_get_thread_num() == 0) {
      std::cout << "Loading " + filename + " on rank 0." << std::endl;
    }
    read_data = 0;

    nStrt = std::accumulate(blocksInFile.begin(), 
                            blocksInFile.begin() + iFile + 1, 0);
    nEnd = nStrt + blocksInFile[iFile + 1];

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
              }
            } // read_data
            read_data = 0;
          } // kGlob
        } // jGlob
      } // iGlob
    } // loop over n
    fclose(fin); 
  } // parallel loop over number of files (GPUs)
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
    const std::vector<int>& levelArray, int index_2D, 
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

void ReadIn::readGridData(FILE* fin, std::vector<double>& tempArray,
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

void ReadIn::readMetricData2D(FILE* fin, std::vector<double>& tempArray,
    long int &index_2D, long int &gridsize_2D, 
    int n, int iGlob, int jGlob, const Read &read) {
  if (read.by_block) {
    // convert to 1D index to read in binary data
    bsLR.x1 = read.bs1 / read.lowRes.x1;
    bsLR.x2 = read.bs2 / read.lowRes.x2;
    bsLR.x3 = read.bs3 / read.lowRes.x3;

    index_2D = ((n * bsLR.x1 * bsLR.x2) + (iGlob * bsLR.x2) + jGlob);
  
    // gridsize is only modified if by block
    gridsize_2D = read.nb * bsLR.x1 * bsLR.x2;

    fseek(fin, (read.nvar * read.bs1 * read.bs2 * read.bs3 + 
      (iGlob * read.lowRes.x1) * read.bs2 * (read.nmtc * 16 + 1) + 
      (jGlob * read.lowRes.x2) * (read.nmtc * 16+1)) * read.dbl_size, 
      SEEK_SET);
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

void ReadIn::readMetricData3D(FILE* fin, std::vector<double>& tempArray,
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

void ReadIn::readPrimsData(FILE* fin, std::vector<float>& tempArray,
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
    fread(&tempArray[0], read.float_size, read.nvar, fin);
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
    fread(&tempArray[0], read.float_size, read.nvar, fin);  
  }
}

ReadIn::~ReadIn() {};

// default constructor
Tools::Tools() {};

void Tools::kernelInvert4x4(const std::vector<double> a, 
    std::vector<double>& b) {
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
          kernelInvert4x4(a_tmp, b_tmp);
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

double Tools::kernelDet4x4(const std::vector<double> a) {
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
          B[index_3D] = kernelDet4x4(a_tmp);
        } // k
      } // j
    } // i
  } // n
} // end kernel_det_4x4()

void Tools::lowerIndex(float* u, float* u_mu, float* gcov,
    int nb, int nx1, int nx2, int nx3, int axisym) {
  // compute u_mu = g_mu,nu u^nu for all mu
  // computes (4, nb, nx1, nx2, nx3) dimensional array
  long int gridsize_3D, gridsize_2D;
  int n, l, m;

  // Loop over each of nb * nx1 * nx2 * nx3 cells
  gridsize_3D = nb * nx1 * nx2 * nx3;
  if (axisym) {
    gridsize_2D = nb * nx1 * nx2;
  } else {
    gridsize_2D = gridsize_3D;
  }

  #pragma omp parallel for private(n, l, m)
  for (n = 0; n < gridsize_3D; n++) {
    std::vector<float> uu(4), ud(4), gcovLoc(16);
    for (l = 0; l < 4; l++) {
      for (m = 0; m < 4; m++) {
        gcovLoc[4 * l + m] = gcov[(4 * l + m) * gridsize_2D + 
          axisym * (n / nx3) + (1 - axisym) * n];
        uu[m] = u[m * gridsize_3D + n];
      }
      ud[l] = 0.0;
    }
    lowerVec(uu, ud, gcovLoc);
    for (l = 0; l < 4; l++) {
      u_mu[l * gridsize_3D + n] = ud[l];
    }
  }
}

void Tools::lowerIndex(float* u, float* u_mu, float* gcov,
    int nb, int nx1, int nx2, int nx3, int axisym, int l) {
  // compute u_mu = g_mu,nu u^nu for all mu
  // computes (4, nb, nx1, nx2, nx3) dimensional array
  long int gridsize_3D, gridsize_2D;
  int n, m, i;

  // Loop over each of nb * nx1 * nx2 * nx3 cells
  gridsize_3D = nb * nx1 * nx2 * nx3;
  if (axisym) {
    gridsize_2D = nb * nx1 * nx2;
  } else {
    gridsize_2D = gridsize_3D;
  }

  #pragma omp parallel for private(n, m, i)
  for (n = 0; n < gridsize_3D; n++) {
    std::vector<float> uu(4), ud(4), gcovLoc(16);
    for (m = 0; m < 4; m++) {
      gcovLoc[4 * l + m] = gcov[(4 * l + m) * gridsize_2D + 
        axisym * (n / nx3) + (1 - axisym) * n];
      uu[m] = u[m * gridsize_3D + n];
    }
    for (i = 0; i < 4; i++) {
      ud[i] = 0.0;
    }
    lowerVec(uu, ud, gcovLoc);
    u_mu[n] = ud[l];
  }
}

void Tools::lowerVec(const std::vector<float> uu,
    std::vector<float>& ud, const std::vector<float> gcov) {
  for (int i = 0 ; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      ud[i] += gcov[4 * i + j] * uu[j];
    }
  }
}

// default destructor
Tools::~Tools() {};

int main() {
  std::cout << "Executing hamr_read.cpp" << std::endl;
  return 0;
}