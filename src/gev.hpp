#ifndef SRC_GEV_HPP_
#define SRC_GEV_HPP_

#include <RcppArmadillo.h>

double rgev(double mu, double sigma, double xi);
double dgev(double x, double mu, double sigma, double xi, bool returnLog);
double estimatePwm(Rcpp::NumericVector x, int r);

#endif  // SRC_GEV_HPP_
