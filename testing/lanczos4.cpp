// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Micah Chambers <micahc.vt@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// @file lanczos_test4.cpp Create known eigenvalues in low rank matrix then
// find them using Band Lanczos. Non verbose for very large system testing

#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/IterativeSolvers>

using namespace std;

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::BandLanczosSelfAdjointEigenSolver;

/**
 * @brief Fills a matrix with random (but symmetric, positive definite) values
 *
 * @param tgt output target matrix
 * @param evals output eigenvalues that have been used to generate tgt
 * @param evecs output eigenvectors that have been used to generate tgt
 * @param rank input rank of tgt matrix
 */
void createRandom(MatrixXd& tgt, VectorXd& evals, MatrixXd& evecs, size_t rank)
{
    assert(tgt.rows() == tgt.cols());

    // Create Outputs to match tgt size
    evecs.resize(tgt.rows(), rank);
    evals.resize(rank);

    // Zero Output Matrices
    tgt.setZero();
    evecs.setZero();
    evals.setZero();

    // Create Random EigenVectors/EigenValues the add using Hotellings
    // 'deflation', although we actually are inflating
    for(size_t ii=0; ii<rank; ii++) {
        // Create Eigenvector at random, then orthogonalize and normalize
        evecs.col(ii).setRandom();
        for(size_t jj=0; jj<ii; jj++) {
            double proj = evecs.col(ii).dot(evecs.col(jj));
            evecs.col(ii) -= proj*evecs.col(jj);
        }
        evecs.col(ii).normalize();

        // Create Random EigenValue 0 to 1
        double v = rand()/(double)RAND_MAX;
        evals[ii] = ii == 0 ? v : v + evals[ii-1];

        tgt += evals[ii]*evecs.col(ii)*evecs.col(ii).transpose();
    }
}

int main(int argc, char** argv)
{
    // Size of Matrix to Compute Eigenvalues of
    size_t matsize = 1000;

    // Number of orthogonal vectors to start with
    size_t nbasis = 25;

    // Rank of matrix to construct
    size_t nrank = 10;
    if(argc == 2) {
        matsize = atoi(argv[1]);
    } else if(argc == 3) {
        matsize = atoi(argv[1]);
        nbasis = atoi(argv[2]);
    } else if(argc == 4) {
        matsize = atoi(argv[1]);
        nbasis = atoi(argv[2]);
        nrank = atoi(argv[3]);
    } else {
        cerr << "Using default matsize, nbasis, rank (set with: " << argv[0]
            << " [matsize] [nbasis] [rank]" << endl;
    }

    clock_t t;
    MatrixXd A(matsize, matsize);
    MatrixXd evecs;
    VectorXd evals;
    createRandom(A, evals, evecs, nrank);

    double trace = A.trace();
    cerr << "Trace=" << trace << endl;
    cerr << "Sum of Eigenvals: " << evals.sum() << endl;
    //cerr << "True Eigenvals: " << evals.transpose() << endl;
    //cerr << "True Eigenvectors: " << endl << evecs << endl;
    //
    //cerr << "Computing with Eigen::SelfAdjointEigenSolver";
    //t = clock();
    //Eigen::SelfAdjointEigenSolver<MatrixXd> egsolver(A);
    //t = clock()-t;
    //evecs = egsolver.eigenvectors();
    //evals = egsolver.eigenvalues();
    //cerr << "Done ("<<t<<")"<<endl;
    //cerr << evals.transpose() << endl;

    BandLanczosSelfAdjointEigenSolver<MatrixXd> blsolver;
    cerr << "Computing with BandLanczosHermitian";
    t = clock();
    blsolver.compute(A, nbasis);
    t = clock()-t;

    MatrixXd bvecs = blsolver.eigenvectors();
    VectorXd bvals = blsolver.eigenvalues();
    cerr << "Done ("<<t<<")"<<endl;
    //	cerr << "My Solution (" << t << "): " << endl << bvals.transpose() << endl;

    int egrank = evals.rows();
    int blrank = bvals.rows();

    cerr << "Comparing"<<endl;
    for(int ii=1; ii<=std::min(bvals.rows(), evals.rows()); ii++) {
        if(fabs(bvals[blrank-ii])/trace > .01) {
            if(fabs(bvals[blrank-ii] - evals[egrank-ii])/trace > 0.05) {
                cerr << "Difference in eigenvalues" << endl;
                cerr << bvals[blrank-ii] << " vs. " << evals[egrank-ii] << endl;
                return -1;
            }
        }
    }

    for(int ii=1; ii<=std::min(bvals.rows(), evals.rows()); ii++) {
        if(fabs(bvals[blrank-ii])/trace > .01) {
            double v = fabs(bvecs.col(blrank-ii).dot(evecs.col(egrank-ii)));
            cerr << ii << " dot prod = " << v << endl;
            if(fabs(v) < .95) {
                cerr << "Difference in eigenvector " << ii << endl;
                return -1;
            }
        }
    }
    cerr << "Success!"<<endl;

    return 0;
}

