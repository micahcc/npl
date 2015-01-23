// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Micah Chambers <micahc.vt@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// @file lanczossvd_test4.cpp Test the SVD based on a band lanczos algorithm of
// eigenvector computation using large non-square matrices.

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/IterativeSolvers>

using namespace std;
using namespace Eigen;

/**
 * @brief Generates a random unitary matrix, so
 *
 * UU^H = I
 *
 * @param rows Number of rows in output
 * @param cols Number of columns in output
 * @param U U matrix, which is unitary
 * @param S Vector of singular values
 * @param V V matrix, which is unitary
 *
 * @return U diag(S) V^T
 */
template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> createRandomUnitary(size_t rows, size_t cols)
{
    Matrix<Scalar, Dynamic, Dynamic> out(rows, cols);
    for(size_t cc = 0; cc<cols; cc++) {
        out.col(cc).setRandom();

        // orthogonalize
        for(size_t ii = 0; ii<cc; ii++) {
            double v = out.col(cc).dot(out.col(ii));
            out.col(cc) -= out.col(ii)*v;
        }

        out.col(cc).normalize();
    }

    return out;
}

/**
 * @brief Generates a random unitary matrix U, another random unitary matrix V,
 * and a random vector S.
 *
 * UU^H = I
 *
 * So given a random vector in column i of U (U_i), the corresponding row of
 * U^H must be chosen so that dot(U_i, conj(U_i)) = 1, dot(U_i, conj(U_j)) = 0
 * if i != j. Further the columns of U should form an orthonormal basis.
 *
 * @param rows Number of rows in output
 * @param cols Number of columns in output
 * @param U U matrix, which is unitary
 * @param S Vector of singular values
 * @param V V matrix, which is unitary
 *
 * @return U diag(S) V^T
 */
template <typename Scalar>
Matrix<Scalar, Dynamic, Dynamic> createRandomSVD(size_t rows, size_t cols,
        size_t rank, Matrix<Scalar,Dynamic,Dynamic>& U,
        Matrix<Scalar,Dynamic,1>& S, Matrix<Scalar,Dynamic,Dynamic>& V)
{
    U = createRandomUnitary<Scalar>(rows, rank);
    V = createRandomUnitary<Scalar>(cols, rank);
    S.resize(rank);
    S.setRandom();
    for(size_t rr=0; rr<rank; rr++)
        S[rr] = abs(S[rr]);
    S.normalize();
    std::sort(S.array().data(), S.array().data()+rank);
    std::reverse(S.array().data(), S.array().data()+rank);

    return U*S.asDiagonal()*V.transpose();
}

int test(size_t rows, size_t cols, size_t rank, size_t nbasis)
{
    MatrixXd true_U, true_V;
    VectorXd true_S;
    Matrix<double,Dynamic,Dynamic> A = createRandomSVD<double>(
            rows, cols, rank, true_U, true_S, true_V);

    cerr << "Computing with Eigen::JacobiSVD";
    clock_t t = clock();
    Eigen::JacobiSVD<MatrixXd> jacsvd(A, ComputeThinV|ComputeThinU);
    t = clock()-t;
    const VectorXd& jvals = jacsvd.singularValues();
    const MatrixXd& jU = jacsvd.matrixU();
    const MatrixXd& jV = jacsvd.matrixV();
    cerr << "\nDone ("<<t<<")"<<endl;

    Eigen::TruncatedLanczosSVD<MatrixXd> lsvd;
    lsvd.setLanczosBasis(nbasis);
    cerr << "Computing with TruncatedLanczosSVD";
    t = clock();
    lsvd.compute(A, ComputeThinV|ComputeThinU);
    t = clock()-t;

    if(lsvd.info() == Eigen::NoConvergence) {
        cerr << "Non-Convergence!" << endl;
        return -1;
    }

    const VectorXd& lvals = lsvd.singularValues();
    const MatrixXd& lU = lsvd.matrixU();
    const MatrixXd& lV = lsvd.matrixV();
    cerr << "\nDone ("<<t<<")"<<endl;

    int64_t srows = min(lvals.rows(), jvals.rows());

    cerr << "Comparing Singular Values"<<endl;
    for(int64_t ii=0; ii<srows; ii++) {
        if(std::abs(jvals[ii]) > .01) {
            if(std::abs(jvals[ii] - lvals[ii]) > 0.05) {
                cerr << "Difference in singular values" << endl;
                cerr << jvals[ii] << " vs. " << lvals[ii] << endl;
                return -1;
            }
        }
    }

    cerr << "Comparing U Matrix" << endl;
    for(int64_t ii=0; ii<srows; ii++) {
        if(std::abs(jvals[ii]) > .01) {
            double v = std::abs(lU.col(ii).dot(jU.col(ii)));
            if(std::abs(v) < .95) {
                cerr << "Difference in eigenvector " << ii << endl;
                cerr << lU.col(ii) << endl << "vs. " << endl
                    << lV.col(ii) << endl;
                return -1;
            }
        }
    }

    cerr << "Comparing V Matrix" << endl;
    for(int64_t ii=0; ii<srows; ii++) {
        if(std::abs(jvals[ii]) > .01) {
            double v = std::abs(lV.col(ii).dot(jV.col(ii)));
            if(std::abs(v) < .95) {
                cerr << "Difference in eigenvector " << ii << endl;
                return -1;
            }
        }
    }

    cerr << "Success!"<<endl;
    return 0;
}

int main(int argc, char** argv)
{
    // Size of Matrix to Compute Eigenvalues of
    size_t matrows = 150;
    size_t matcols = 200;

    // Number of orthogonal vectors to start with
    size_t nbasis = 200;

    // Number of nonzero singular values
    size_t rank = 100;

    if(argc == 2) {
        matrows = atoi(argv[1]);
    } else if(argc == 3) {
        matrows = atoi(argv[1]);
        matcols = atoi(argv[2]);
    } else if(argc == 4) {
        matrows = atoi(argv[1]);
        matcols = atoi(argv[2]);
        rank = atoi(argv[3]);
    } else if(argc == 5) {
        matrows = atoi(argv[1]);
        matcols = atoi(argv[2]);
        rank = atoi(argv[3]);
        nbasis = atoi(argv[4]);
    } else {
        cerr << "Using default size, basis size, and rank (to set use "
            "arguments: " << argv[0] << " [matrows] [matcols] [nbasis] [rank]"
            << endl;
    }

    cerr << "Matrix (rank "<<rank<<"): " << matrows << "x" << matcols << endl;
    cerr << "Initializing with " << nbasis << " basis vectors" << endl;

    cerr << "Test " << matrows << "x" << matcols << endl;
    if(test(matrows, matcols, rank, nbasis))
        return -1;

    cerr << "Test " << matcols << "x" << matrows << endl;
    if(test(matcols, matrows, rank, nbasis))
        return -1;

    cerr << "Test " << matcols << "x" << matcols << endl;
    if(test(matcols, matcols, rank, nbasis))
        return -1;

    cerr << "Test " << matrows << "x" << matrows << endl;
    if(test(matrows, matrows, rank, nbasis))
        return -1;

}


