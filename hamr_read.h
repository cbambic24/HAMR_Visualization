#ifndef HAMR_H
#define HAMR_H

#include <iostream>
#include <string>
#include <cmath>
#include <unordered_map>
#include <filesystem>
#include <cstdio> // fread and fseek
#include <omp.h> // omp parallel for
#include <numeric> // accumulate

void kernel_read_grid(std::string dir, int axisym, int by_block, 
  int export_raytracing,
  const std::unordered_map<std::string, double>& amr,
  const std::unordered_map<std::string, std::vector<int>>& amr_info,
  const std::unordered_map<std::string, double>& block_info,
  const std::unordered_map<std::string, double>& params,
  const std::unordered_map<std::string, double>& options,
  float* x1, float* x2, float* x3, float* r, float* theta, float* phi,
  float* gcov, float* gcon, float* gdet, float* dxdxp);

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
  float* nuNum1, float* nuNum2, float* nuNum3, float* tr1, float* tr2);

void invert_4x4_cpp(std::vector<double>, std::vector<double> b); 

void kernel_invert_4x4(float* A, float* B, int nb, int nx1, int nx2, int nx3);

double det_4x4_cpp(const std::vector<double> a);

void kernel_det_4x4(float* A, float* B, int nb, int nx1, int nx2, int nx3);

#endif
