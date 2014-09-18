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
#include "basic_functions.h"
#include "macros.h"

#include <random>
#include <cmath>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::JacobiSVD;

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
    varth=1-varth; 

    double totalv = 0; // total variance
    int outdim = 0;

#ifndef NDEBUG
    std::cout << "Computing SVD" << std::endl;
#endif //DEBUG
    JacobiSVD<MatrixXd> svd(X, Eigen::ComputeThinU);
#ifndef NDEBUG
    std::cout << "Done" << std::endl;
#endif //DEBUG
	
    const VectorXd& W = svd.singularValues();
    const MatrixXd& U = svd.matrixU();
    //only keep dimensions with variance passing the threshold
    for(size_t ii=0; ii<W.rows(); ii++)
        totalv += W[ii]*W[ii];

    double sum = 0;
    for(outdim = 0; outdim < W.rows() && sum < totalv*varth; outdim++) 
        sum += W[outdim]*W[outdim];
#ifndef NDEBUG
    std::cout << "Output Dimensions: " << outdim 
            << "\nCreating Reduced MatrixXd..." << std::endl;
#endif //DEBUG

    MatrixXd Xr(X.rows(), outdim);
	for(int rr=0; rr<X.rows(); rr++) {
		for(int cc=0; cc<outdim ; cc++) {
			Xr(rr,cc) = U(rr, cc);
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
 * @param Xin 	RxC matrix where each column row is a sample, each column a
 *              dimension (or feature). The number of columns in the output
 *              will be fewer because there will be fewer features
 * @param varth Variance threshold. Don't include dimensions after this percent
 *              of the variance has been explained.
 *
 * @return 		RxP matrix, where P is the number of independent components
 */
MatrixXd ica(const MatrixXd& Xin, double varth)
{
    (void)varth;

    // remove mean/variance
    MatrixXd X(Xin.rows(), Xin.cols());
    for(size_t cc=0; cc<X.cols(); cc++)  {
        double sum = 0;
        double sumsq = 0;
        for(size_t rr=0; rr<X.rows(); rr++)  {
            sum += Xin(rr,cc);
            sumsq += Xin(rr,cc)*Xin(rr,cc);
        }
        double sigma = sqrt(sample_var(X.rows(), sum, sumsq));
        double mean = sum/X.rows();

        for(size_t rr=0; rr<X.rows(); rr++)  
            X(rr,cc) = (Xin(rr,cc)-mean)/sigma;
    }

    const size_t ITERS = 10000;
    const double MAGTHRESH = 0.0001;

    // Seed with a real random value, if available
    std::random_device rd;
    std::default_random_engine rng(rd());
	std::uniform_real_distribution<double> unif(0, 1);

	int samples = X.rows();
	int dims = X.cols();
    int ncomp = std::min(samples, dims); 
	
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
			double wt_wj = W.col(pp).dot(W.col(jj));

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
	
            // X^Tg(X wp)/R
            for(size_t jj=0; jj<samples; jj++)
                proj[jj] = fastICA_g2(proj[jj]);
            W.col(pp) += X.transpose()*proj/samples;
		
            //GramSchmidt Decorrelate
            //sum(w^t_p w_j w_j) for j < p
            //cache w_p for wt_wj mutlication
            for(int jj = 0 ; jj < pp; jj++){
                //w^t_p w_j
                double wt_wj = W.col(pp).dot(W.col(jj));

                //w_p -= (w^t_p w_j) w_j
                W.col(pp) -= wt_wj*W.col(jj);
            }
            W.col(pp).normalize();
            mag = (W.col(pp)-wprev).norm();
		}

#ifndef NDEBUG
        std::cout << "Final (" << pp << "):\n";
        std::cout << W.col(pp).transpose() << std::endl;
#endif// NDEBUG
	}
	
    // TODO sort by variance
    return X*W;
	
}

/**
 * @brief Computes cdf at a particular number of degrees of freedom. 
 * Note, this only computes +t values, for negative values invert then use.
 *
 * @param nu
 * @param x
 *
 * @return 
 */
std::vector<double> students_t_cdf(double nu, double dt, double maxt)
{
    std::vector<double> out;
    out.reserve(ceil(maxt/dt));

    double sum = 0.5;
    double dp = tgamma((nu+1)/2)/(tgamma(nu/2)*sqrt(nu*M_PI));
    for(size_t ii = 0; ii*dt < maxt; ii++) {
        double t = ii*dt;
        out.push_back(sum);
        sum += dt*dp/pow(1+t*t/nu, (nu+1)/2);
    }

    return out;
}
    
// need to compute the CDF for students_t_cdf
const double MAX_T = 100;
const double STEP_T = 0.1;

/**
 * @brief Computes the Ordinary Least Square predictors, beta for 
 *
 * \f$ y = \hat \beta X \f$
 *
 * Returning beta. This is the same as the other regress function, but allows
 * for cacheing of pseudoinverse of X
 *
 * @param y response variables
 * @param X independent variables
 * @param covInv Inverse of covariance matrix, to compute us pseudoinverse(X^TX)
 * @param Xinv Pseudo inverse of X. Compute with pseudoInverse(X)
 * @param student_cdf Pre-computed students' T distribution. Example:
 * auto v = students_t_cdf(X.rows()-1, .1, 1000);
 *
 * @return Struct with Regression Results. 
 */
RegrResult regress(const VectorXd& y, const MatrixXd& X, const MatrixXd& covInv,
        const MatrixXd& Xinv, std::vector<double>& student_cdf)
{
    if(y.rows() != X.rows()) 
        throw INVALID_ARGUMENT("y and X matrices row mismatch");
    if(X.rows() != Xinv.cols()) 
        throw INVALID_ARGUMENT("X and pseudo inverse of X row mismatch");
    
    RegrResult out;
    out.bhat = Xinv*y;
    out.yhat = out.bhat*X;
    out.ssres = (out.yhat - y).squaredNorm();

    // compute total sum of squares
    double mean = y.mean();
    out.sstot = 0;
    for(size_t rr=0; rr<y.rows(); rr++)
        out.sstot += (y[rr]-mean)*(y[rr]-mean);
    out.rsqr = 1-out.ssres/out.sstot;
    out.adj_rsqr = out.rsqr - (1-out.rsqr)*X.cols()/(X.cols()-X.rows()-1);
    
    double sigmahat = out.ssres/(X.rows()-X.cols()+2);
    out.std_err.resize(X.cols());
    out.t.resize(X.cols());
    out.p.resize(X.cols());
    out.dof = X.rows()-1;

    for(size_t ii=0; ii<X.cols(); ii++) {
        out.std_err[ii] = sqrt(sigmahat*covInv(ii,ii)/X.cols());
        double t = out.bhat[ii]/out.std_err[ii];
        out.t[ii] = t;
        
        t = fabs(t);
        size_t t_index = round(t/STEP_T);
        if(t_index >= student_cdf.size())
            out.p[ii] = 0;
        else
            out.p[ii] = student_cdf[t_index];
    }

    return out;
}

/**
 * @brief Computes the Ordinary Least Square predictors, beta for 
 *
 * \f$ y = \hat \beta X \f$
 *
 * Returning beta. This is the same as the other regress function, but allows
 * for cacheing of pseudoinverse of X
 *
 * @param y response variables
 * @param X independent variables
 *
 * @return Struct with Regression Results. 
 */
RegrResult regress(const VectorXd& y, const MatrixXd& X)
{
    if(y.rows() != X.rows()) 
        throw INVALID_ARGUMENT("y and X matrices row mismatch");
  
    auto Xinv = pseudoInverse(X);
    auto covInv = pseudoInverse(X.transpose()*X);

    RegrResult out;
    out.bhat = Xinv*y;
    out.yhat = out.bhat*X;
    out.ssres = (out.yhat - y).squaredNorm();

    // compute total sum of squares
    double mean = y.mean();
    out.sstot = 0;
    for(size_t rr=0; rr<y.rows(); rr++)
        out.sstot += (y[rr]-mean)*(y[rr]-mean);
    out.rsqr = 1-out.ssres/out.sstot;
    out.adj_rsqr = out.rsqr - (1-out.rsqr)*X.cols()/(X.cols()-X.rows()-1);
    
    // estimate the standard deviation of the error term
    double sigmahat = out.ssres/(X.rows()-X.cols()+2);

    out.std_err.resize(X.cols());
    out.t.resize(X.cols());
    out.p.resize(X.cols());
    out.dof = X.rows()-1;

    auto student_cdf = students_t_cdf(out.dof, STEP_T, MAX_T);

    for(size_t ii=0; ii<X.cols(); ii++) {
        out.std_err[ii] = sqrt(sigmahat*covInv(ii,ii)/X.cols());

        double t = out.bhat[ii]/out.std_err[ii];
        out.t[ii] = t;
        
        t = fabs(t);
        size_t t_index = round(t/STEP_T);
        if(t_index >= student_cdf.size())
            out.p[ii] = 0;
        else
            out.p[ii] = student_cdf[t_index];
    }

    return out;
}


/**
 * @brief Computes the pseudoinverse of the input matrix
 *
 * \f$ P = UE^-1V^* \f$
 *
 * @return Psueodinverse 
 */
MatrixXd pseudoInverse(const MatrixXd& X)
{
    double THRESH = 0.000001;
    JacobiSVD<MatrixXd> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
    VectorXd singular_values = svd.singularValues();

    for(size_t ii=0; ii<svd.singularValues().rows(); ii++) {
        if(singular_values[ii] > THRESH)
            singular_values[ii] = 1./singular_values[ii];
        else
            singular_values[ii] = 0;
    }
    return svd.matrixV()*singular_values.asDiagonal()*
            svd.matrixU().transpose();
}

} // NPL
