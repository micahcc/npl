/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file statistics.cpp Tools for analyzing data, including PCA, ICA and
 * general linear modeling.
 *
 *****************************************************************************/

#include <Eigen/SVD>
#include "statistics.h"

#include <random>
#include <cmath>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXdXd;

namespace npl {

//Hyperbolic Tangent and its derivative is one potential optimization function
double fastICA_g1(double u)
{
	return tanh(u);
}


double fastICA_dg1(double u)
{
	return 1./(cosh(u)*cosh(u));
}

//exponential and its derivative is another optimization function
double fastICA_g2(double u)
{
	return u*exp(-u*u/2);
}

double fastICA_dg2(double u)
{
	return (1-u*u)*exp(-u*u/2);
}

void apply(double* dst, const double* src,  double(*func)(double), size_t sz)
{
	for(unsigned int ii = 0 ; ii < sz; ii++)
		dst[ii] = func(src[ii]);
}

/**
 * @brief Computes the Principal Components of input matrix X
 *
 * Outputs reduced dimension (fewer cols) in output. Note that prio to this,
 * the columns of X should be 0 mean. 
 *
 * @param X 	RxC matrix where each column row is a sample, each column a
 *              dimension (or feature). The number of columns in the output
 *              will be fewer because there will be fewer features
 * @param varth Variance threshold. Don't include dimensions after this percent
 *              of the variance has been explained.
 *
 * @return 		RxP matrix, where P is the number of principal components
 */
MatrixXd pca(const MatrixXd& X, double varth)
{
	const double VARTHRESH = .05;
    double totalv = 0; // total variance
    int outdim = 0;

#ifndef NDEBUG
    std::cout << "Computing SVD" << std::endl;
#endif //DEBUG
    Eigen::JacobiSVD<MatrixXd> svd(X, Eigen::ComputeThinU);
#ifndef NDEBUG
    std::cout << "Done" << std::endl;
#endif //DEBUG
	
    const VectorXd& W = svd.singularValues();
    const MatrixXd& U = svd.matrixU();
    //only keep dimensions with variance passing the threshold
    for(size_t ii=0; ii<W.rows(); ii++)
        totalv = W[ii]*W[ii];

    double sum = 0;
    for(outdim = 0; outdim < W.rows(); outdim++) {
        sum += W[outdim]*W[outdim];
        if(sum > totalv*(1-VARTHRESH))
            break;
    }
#ifndef NDEBUG
    std::cout << "Output Dimensions: " << outdim 
            << "\nCreating Reduced MatrixXd..." << std::endl;
#endif //DEBUG

    MatrixXd Xr(X.rows(), outdim);
	for(int rr=0; rr<X.rows(); rr++) {
		for(int cc=0; cc<outdim ; cc++) {
			Xr(rr,cc) = U(rr, cc)*W[cc];
		}
	}
#ifndef NDEBUG
	std::cout  << "  Done" << std::endl;
#endif 

    return Xr;
}

/**
 * @brief Computes the Independent Components of input matrix X. Note that
 * you should run PCA on X before running ICA.
 *
 * Outputs reduced dimension (fewer cols) in output
 *
 * Note that this whole problem is the transpose of the version listed on
 * wikipedia.
 *
 * In: I number of components
 * In: X RxC matrix with rows representing C-D samples
 * for p in 1 to I:
 *   wp = random weight
 *   while wp changes:
 *     wp = g(X wp)X/R - wp SUM(g'(X wp))/R
 *     wp = wp - SUM wp^T wj wj
 *     normalize(wp)
 *
 * Output: W = [w0 w1 w2 ... ]
 * Output: S = XW, where each column is a dimension, each row a sample
 *
 * @param X 	RxC matrix where each column row is a sample, each column a
 *              dimension (or feature). The number of columns in the output
 *              will be fewer because there will be fewer features
 * @param varth Variance threshold. Don't include dimensions after this percent
 *              of the variance has been explained.
 *
 * @return 		RxP matrix, where P is the number of independent components
 */
MatrixXd ica(const MatrixXd& X, double varth)
{
    const size_t ITERS = 100;
    const double MAGTHRESH = 0.0001;

    // Seed with a real random value, if available
    std::random_device rd;
    std::default_random_engine rng(rd());
	std::uniform_real_distribution<double> unif(0, 1);

	int samples = X.rows();
	int dims = X.cols();
    int ncomp = std::min(samples, dims)-1; // -1 is for debugging purposes...
	
	double mag = 1;
	VectorXd proj(samples);
    VectorXd nonlin1(samples);
    VectorXd nonlin2(samples);
    VectorXd wprev(dims);
	
	MatrixXd W(dims, ncomp);

	for(int pp = 0 ; pp < ncomp ; pp++) {
		//randomize weights
		for(unsigned int ii = 0; ii < dims ; ii++) 
			W.col(pp)[ii] = unif(rng);
			
		//GramSchmidt Decorrelate
		//sum(w^t_p w_j w_j) for j < p
		//cache w_p for wt_wj mutlication
		for(int jj = 0 ; jj < pp; jj++){
			//w^t_p w_j
			double wt_wj = -W.col(pp).dot(W.col(jj));

			//w_p -= (w^t_p w_j) w_j
			W.col(pp) -= wt_wj*W.col(jj);
		}
		W.col(pp).normalize();
#ifndef NDEBUG
        std::cout << "Peforming Fast ICA: " << pp << std::endl;
#endif// NDEBUG
		mag = 1;
		for(int ii = 0 ; mag > MAGTHRESH && ii < ITERS; ii++) {
			
			//move to working
            wprev = W.col(pp);

			/* 
             * g(X wp) X^T/R - wp SUM(g'(X wp)))/R
             */

			//w^tx
            proj = X*W.col(pp);
			
            //- wp SUM(g'(X wp)))/R
            double sum = 0;
            for(size_t jj=0; jj<samples; jj++)
                sum += fastICA_dg2(proj[jj]);
            W.col(pp) = -W.col(pp)*sum/samples;
	
            // g(X wp) X^T/R
            for(size_t jj=0; jj<samples; jj++)
                proj[jj] = fastICA_g2(proj[jj]);
            W.col(pp) += proj*X.transpose()/samples;
		
            //GramSchmidt Decorrelate
            //sum(w^t_p w_j w_j) for j < p
            //cache w_p for wt_wj mutlication
            for(int jj = 0 ; jj < pp; jj++){
                //w^t_p w_j
                double wt_wj = -W.col(pp).dot(W.col(jj));

                //w_p -= (w^t_p w_j) w_j
                W.col(pp) -= wt_wj*W.col(jj);
            }
            W.col(pp).normalize();
            double mag = (W.col(pp)-wprev).norm();
		}

#ifndef NDEBUG
        std::cout << "Final (" << pp << "): ";
		for(unsigned int cc = 0 ; cc < dims ; cc++)
			std::cout << W(pp, cc) << ' ';
        std::cout << std::endl;
#endif// NDEBUG
	}
	
    // TODO sort by variance
    return X*W;
	
}


}
