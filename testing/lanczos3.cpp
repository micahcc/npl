/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 ******************************************************************************/

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
    size_t matsize = 8;

    // Number of orthogonal vectors to start with
    size_t nbasis = 5;

    // Rank of matrix to construct
    size_t nrank = 5;
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

    MatrixXd A(matsize, matsize);
    MatrixXd evecs;
    VectorXd evals;
    createRandom(A, evals, evecs, nrank);

    double trace = A.trace();
    cerr << "Trace=" << trace << endl;
    cerr << "Sum of Eigenvals: " << evals.sum() << endl;
    cerr << "True Eigenvals: " << evals.transpose() << endl;
    cerr << "True Eigenvectors: " << endl << evecs << endl;

    BandLanczosSelfAdjointEigenSolver<MatrixXd> blsolver;
    cerr << "Computing with BandLanczosHermitian";
    clock_t t = clock();
    blsolver.compute(A, nbasis);
    t = clock()-t;

    MatrixXd bvecs = blsolver.eigenvectors();
    VectorXd bvals = blsolver.eigenvalues();
    cerr << "Done ("<<t<<")"<<endl;
    cerr << "My Solution (" << t << "): " << endl << bvecs << endl << endl
        << bvals << endl;

    size_t egrank = evals.rows();
    size_t blrank = bvals.rows();

    cerr << "Comparing"<<endl;
    for(int64_t ii=1; ii<=std::min(bvals.rows(), evals.rows()); ii++) {
        if(fabs(bvals[blrank-ii])/trace > .01) {
            if(fabs(bvals[blrank-ii] - evals[egrank-ii])/trace > 0.05) {
                cerr << "Difference in eigenvalues" << endl;
                cerr << bvals[blrank-ii] << " vs. " << evals[egrank-ii] << endl;
                return -1;
            }
        }
    }

    for(int64_t ii=1; ii<=std::min(bvals.rows(), evals.rows()); ii++) {
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


