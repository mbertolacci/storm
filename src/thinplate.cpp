#include <RcppArmadillo.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <Eigen/Core>
#pragma GCC diagnostic pop

#include <SymEigsSolver.h>

#include "logging.hpp"

using Rcpp::as;
using Rcpp::List;
using Rcpp::NumericMatrix;

using arma::colvec;
using arma::diagmat;
using arma::eig_sym;
using arma::mat;
using arma::rowvec;

inline double xLogXApprox(double x) {
    double result = x * log(x);
    if (!std::isfinite(result)) {
        return 0;
    }
    return result;
}

inline double square(double x) { return x * x; }

// [[Rcpp::export(name="thinplate_basis_2d")]]
List thinplateBasis2d(NumericMatrix X, unsigned int nBases) {
    ptsm::Logger logger("ptsm.thinplate");

    unsigned int n = X.nrow();
    mat designMatrix = as<mat>(X);
    rowvec minX = arma::min(designMatrix, 0);
    rowvec maxX = arma::max(designMatrix, 0);

    // Constrain each row to be from 0 to 1
    designMatrix.each_row() -= minX;
    designMatrix.each_row() /= (maxX - minX);

    logger.debug("Calculating %dx%d covariance matrix", n, n);
    mat omega(n, n, arma::fill::zeros);
    #pragma omp parallel for if (n > 100)
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = i; j < n; ++j) {
            double rij = sqrt(square(designMatrix(i, 0) - designMatrix(j, 0)) + square(designMatrix(i, 1) - designMatrix(j, 1)));

            double p1i = 1.0 - 2.0 * designMatrix(i, 0) - 2.0 * designMatrix(i, 1);
            double p1j = 1.0 - 2.0 * designMatrix(j, 0) - 2.0 * designMatrix(j, 1);
            double p2i = 2.0 * designMatrix(i, 0);
            double p2j = 2.0 * designMatrix(j, 0);
            double p3i = 2.0 * designMatrix(i, 1);
            double p3j = 2.0 * designMatrix(j, 1);

            double rs1i = sqrt(square(0.00 - designMatrix(i, 0)) + square(0.00 - designMatrix(i, 1)));
            double rs1j = sqrt(square(0.00 - designMatrix(j, 0)) + square(0.00 - designMatrix(j, 1)));
            double rs2i = sqrt(square(0.50 - designMatrix(i, 0)) + square(0.00 - designMatrix(i, 1)));
            double rs2j = sqrt(square(0.50 - designMatrix(j, 0)) + square(0.00 - designMatrix(j, 1)));
            double rs3i = sqrt(square(0.00 - designMatrix(i, 0)) + square(0.50 - designMatrix(i, 1)));
            double rs3j = sqrt(square(0.00 - designMatrix(j, 0)) + square(0.50 - designMatrix(j, 1)));
            double rs1s2 = sqrt(square(0.00 - 0.50) + square(0.00 - 0.00));
            double rs2s3 = sqrt(square(0.50 - 0.00) + square(0.00 - 0.50));
            double rs1s3 = sqrt(square(0.00 - 0.00) + square(0.00 - 0.50));

            double Aij = xLogXApprox(rij);
            double As1i = xLogXApprox(rs1i);
            double As1j = xLogXApprox(rs1j);
            double As2i = xLogXApprox(rs2i);
            double As2j = xLogXApprox(rs2j);
            double As3i = xLogXApprox(rs3i);
            double As3j = xLogXApprox(rs3j);
            double As1s2 = xLogXApprox(rs1s2);
            double As2s3 = xLogXApprox(rs2s3);
            double As1s3 = xLogXApprox(rs1s3);

            omega(i, j) = (
                Aij
                - p1j * As1i - p2j * As2i - p3j * As3i - p1i * As1j - p2i * As2j - p3i * As3j
                + p1i * p2j * As1s2 + p1i * p3j * As1s3 + p2i * p1j * As1s2 + p2i * p3j * As2s3
                + p3i * p1j * As1s3 + p3i * p2j * As2s3
            );
        }
    }

    if (!omega.is_finite()) {
        Rcpp::stop("Calculation of covariance matrix failed: some members are non-finite");
    }
    // NOTE(mgnb): the two slow bits are the above double loop, and the eigenvector call below. So give the user a
    // chance to jump out in-between.
    Rcpp::checkUserInterrupt();

    // Prepend a column of ones and make room for the chosen bases
    designMatrix.resize(n, 3 + nBases);
    designMatrix.col(2) = designMatrix.col(1);
    designMatrix.col(1) = designMatrix.col(0);
    designMatrix.col(0).fill(1.0);

    omega = arma::symmatu(omega);

    logger.debug("Computing %d eigenvalues and eigenvectors", nBases);
    // NOTE(mgnb): Map interfaces Eigen into the armadillo matrix
    Eigen::Map<Eigen::MatrixXd> omegaEigen(omega.memptr(), n, n);
    Spectra::DenseSymMatProd<double> op(omegaEigen);
    Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, Spectra::DenseSymMatProd<double> > solver(
        &op, nBases, std::min(n, 2 * nBases)
    );
    solver.init();
    solver.compute();
    if (solver.info() != Spectra::SUCCESSFUL) {
        Rcpp::stop("Calculation of eigenvectors failed");
    }

    colvec eigenvalues(nBases);
    mat eigenvectors(n, nBases);
    Eigen::Map<Eigen::VectorXd> eigenvaluesMap(eigenvalues.memptr(), nBases);
    Eigen::Map<Eigen::MatrixXd> eigenvectorsMap(eigenvectors.memptr(), n, nBases);
    eigenvaluesMap = solver.eigenvalues();
    eigenvectorsMap = solver.eigenvectors();

    logger.debug("Filling out design matrix");

    designMatrix.cols(3, 3 + nBases - 1) = eigenvectors * diagmat(sqrt(eigenvalues));

    List output;
    output["covariance"] = Rcpp::wrap(omega);
    output["design_matrix"] = Rcpp::wrap(designMatrix);
    output["eigenvalues"] = Rcpp::wrap(eigenvalues);

    return output;
}
