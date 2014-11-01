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
#include <Eigen/QR>
#include "slicer.h"
#include "statistics.h"
#include "basic_functions.h"
#include "macros.h"

#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>
#include <map>
#include <iomanip>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixXf;
using Eigen::VectorXf;
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
 * @brief Computes the Principal Components of input matrix X using the
 * covariance matrix.
 *
 * Outputs projections in the columns of a matrix. 
 *
 * @param cov 	Covariance matrix of XtX
 * @param varth Variance threshold. Don't include dimensions after this percent
 *              of the variance has been explained.
 *
 * @return 		RxP matrix, where P is the number of principal components
 */
MatrixXd pcacov(const MatrixXd& cov, double varth)
{
	assert(cov.rows() == cov.cols());
    int outdim = 0;

#ifndef NDEBUG
    std::cout << "Computing ..." << std::endl;
#endif //DEBUG
	Eigen::SelfAdjointEigenSolver<MatrixXd> solver(cov);
#ifndef NDEBUG
    std::cout << "Done" << std::endl;
#endif //DEBUG
	
	double total = 0;
	for(int64_t ii=solver.eigenvalues().rows()-1; ii>=0; ii--) {
		assert(solver.eigenvalues()[ii] >= 0);
		total += solver.eigenvalues()[ii];
	}

	double sum = 0;
	size_t ndim = 0;
	for(int64_t ii=solver.eigenvalues().rows()-1; ii>=0; ii--) {
		sum += solver.eigenvalues()[ii];
		if(sum / total < varth)
			ndim++;
		else 
			break;
	}

#ifndef NDEBUG
    std::cout << "Output Dimensions: " << ndim 
            << "\nCreating Reduced MatrixXd..." << std::endl;
#endif //DEBUG

	MatrixXd out(cov.rows(), ndim);
	for(int64_t ii=solver.eigenvalues().rows()-1, jj=0; ii>=0; ii--, jj++) {
		for(size_t rr=0; rr<cov.rows(); rr++) {
			out(rr, jj) = solver.eigenvectors()(rr, ii)
		}
	}

#ifndef NDEBUG
	std::cout  << "  Done" << std::endl;
#endif 

    return out;
}

/**
 * @brief Computes the Principal Components of input matrix X
 *
 * Outputs reduced dimension (fewer cols) in output. Note that prio to this,
 * the columns of X should be 0 mean. 
 *
 * TODO put math here
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

StudentsT::StudentsT(int dof, double dt, double tmax) : 
            m_dt(dt), m_tmax(tmax), m_dof(dof)
{
    init();
};

void StudentsT::setDOF(double dof)
{
    m_dof = dof;
    init();
};
    
void StudentsT::setStepT(double dt)
{
    m_dt = dt;
    init();
};

void StudentsT::setMaxT(double tmax)
{
    m_tmax = tmax;
    init();
};

double StudentsT::cumulative(double t) const
{
    bool negative = false;
    if(t < 0) {
        negative = true;
        t = fabs(t);
    }

    double out = 0;
    vector<double>::const_iterator it = 
        std::lower_bound(m_tvals.begin(), m_tvals.end(), t);

    if(it == m_tvals.end()) {
        cerr << "Warning, effectively 0 p-value returned!" << endl;
        return 0;
    }

    int ii = distance(m_tvals.begin(), it);
    if(ii > 0) {
        double tp = m_tvals[ii-1];
        double tn = m_tvals[ii];

        double prev = m_cdf[ii-1];
        double next = m_cdf[ii];
        out = prev*(tn-t)/(tn-tp) + next*(t-tp)/(tn-tp);
    } else {
        assert(m_cdf[ii] == 0.5);
        out = m_cdf[ii];
    }

    if(negative)
        return 1-out;
    else
        return out;
};

double StudentsT::density(double t) const
{
    bool negative = false;
    if(t < 0) {
        negative = true;
        t = fabs(t);
    }

    double out = 0;
    vector<double>::const_iterator it = std::lower_bound(m_tvals.begin(),
            m_tvals.end(), t);
    if(it == m_tvals.end()) {
#ifndef NDEBUG
        cerr << "Warning, effectively 0 p-value returned!" << endl;
#endif
        return 0;
    }

    int ii = distance(m_tvals.begin(), it);
    if(ii > 0) {
        double tp = m_tvals[ii-1];
        double tn = m_tvals[ii];

        double prev = m_pdf[ii-1];
        double next = m_pdf[ii];
        out = prev*(tn-t)/(tn-tp) + next*(t-tp)/(tn-tp);
    } else {
        assert(m_pdf[ii] == 0.5);
        out = m_pdf[ii];
    }

    if(negative)
        return 1-out;
    else
        return out;
};

void StudentsT::init()
{
    m_cdf.resize(m_tmax/m_dt);
    m_pdf.resize(m_tmax/m_dt);
    m_tvals.resize(m_tmax/m_dt);

    double sum = 0.5;
    double coeff;
    if(m_dof%2 == 0) {
        coeff = 1./(2*sqrt((double)m_dof));
        for(int ii = m_dof-1; ii >= 3; ii-=2) 
            coeff *= ((double)ii)/(ii-1.);
    } else {
        coeff = 1./(M_PI*sqrt((double)m_dof));
        for(int ii = m_dof-1; ii >= 2; ii-=2) 
            coeff *= ((double)ii)/(ii-1.);
    }

    for(size_t ii = 0; ii*m_dt < m_tmax; ii++) {
        double t = ii*m_dt;
        m_tvals[ii] = t;
        m_pdf[ii] = coeff*pow(1+t*t/m_dof, -(m_dof+1)/2);
        m_cdf[ii] = sum;
        sum += m_dt*m_pdf[ii];
    }
};

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
        const MatrixXd& Xinv, const StudentsT& distrib)
{
    if(y.rows() != X.rows()) 
        throw INVALID_ARGUMENT("y and X matrices row mismatch");
    if(X.rows() != Xinv.cols()) 
        throw INVALID_ARGUMENT("X and pseudo inverse of X row mismatch");
    
    RegrResult out;
    out.bhat = Xinv*y;
    out.yhat = X*out.bhat;
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
        out.p[ii] = distrib.cumulative(t);
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

    // need to compute the CDF for students_t_cdf
    const double MAX_T = 100;
    const double STEP_T = 0.1;
    StudentsT distrib(out.dof, STEP_T, MAX_T);

    for(size_t ii=0; ii<X.cols(); ii++) {
        out.std_err[ii] = sqrt(sigmahat*covInv(ii,ii)/X.cols());

        double t = out.bhat[ii]/out.std_err[ii];
        out.t[ii] = t;
        out.p[ii] = distrib.cdf(t);
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

/******************************************************
 * Classifiers
 *****************************************************/

/**
 * @brief Approximates k-means using the algorithm of:
 *
 * 'Fast Approximate k-Means via Cluster Closures' by Wang et al
 *
 * @param samples Matrix of samples, one sample per row, 
 * @param nclass Number of classes to break samples up into
 * @param extimated groupings
 */
void approxKMeans(const MatrixXd& samples, size_t nclass, Eigen::VectorXi& labels)
{
	DBG1(cerr << "Approximating K-Means" << endl);
	size_t ndim = samples.cols();
	size_t npoints = samples.rows();
	double norm = 0;

	MatrixXd means(nclass, ndim);
    std::vector<double> dists(npoints);
    std::vector<int> indices(npoints);
	
	// Select Point
    std::default_random_engine rng;
    std::uniform_int_distribution<int> randi(0, npoints-1);
    std::uniform_real_distribution<double> randf(0, 1);
    int tmp = randi(rng);
    size_t pp;
	means.row(0) = samples.row(tmp);

	//set the rest of the centers
	for(int cc = 1; cc < nclass; cc++) { 
		norm = 0;

		//create list of distances
		for(pp = 0 ; pp < npoints ; pp++) {
			dists[pp] = INFINITY;
			for(int tt = 0 ; tt < cc; tt++) {
                double v = (samples.row(pp)-means.row(tt)).squaredNorm();
                dists[pp] = std::min(dists[pp], v);
			}

			//keep normalization factor for later
			norm += dists[pp];
		}

		//set target pp^th greatest distance
		double pct = norm*randf(rng);

        // fill indices
        for(size_t ii=0; ii<npoints; ii++)
            indices[ii] = ii;

        // sort, while keeping indices
        std::sort(indices.begin(), indices.end(), 
                [&dists](size_t i, size_t j){
                    return dists[i] < dists[j]; 
                });

		//go through sorted list to find matching location in CDF
        for(pp = 0; pp < npoints && pct > 0 ; pp++) {
            double d = dists[indices[pp]]; 
            pct -= d;
        }

		//copy randomly selected  point into middle
		means.row(cc) = samples.row(pp);
	}

	labels.resize(samples.rows());
	for(size_t rr=0; rr<samples.rows(); rr++) {
		double bestdist = INFINITY;
		int bestlabel = -1;
		for(size_t cc=0; cc<nclass; cc++) {
			double distsq = (samples.row(rr) - means.row(cc)).squaredNorm();
			if(distsq < bestdist) {
				bestdist = distsq;
				bestlabel = cc;
			}
		}
		labels[rr] = bestlabel;
	}
}

/**
 * @brief Approximates k-means using the algorithm of:
 *
 * 'Fast Approximate k-Means via Cluster Closures' by Wang et al
 *
 * @param samples Matrix of samples, one sample per row, 
 * @param nclass Number of classes to break samples up into
 * @param means Estimated mid-points/means
 */
void approxKMeans(const MatrixXd& samples, size_t nclass, MatrixXd& means)
{
	DBG1(cerr << "Approximating K-Means" << endl);
	size_t ndim = samples.cols();
	size_t npoints = samples.rows();
	double norm = 0;

	means.resize(nclass, ndim);
    std::vector<double> dists(npoints);
    std::vector<int> indices(npoints);
	
	// Select Point
    std::default_random_engine rng;
    std::uniform_int_distribution<int> randi(0, npoints-1);
    std::uniform_real_distribution<double> randf(0, 1);
    int tmp = randi(rng);
    size_t pp;
	means.row(0) = samples.row(tmp);

	//set the rest of the centers
	for(int cc = 1; cc < nclass; cc++) { 
		norm = 0;

		//create list of distances
		for(pp = 0 ; pp < npoints ; pp++) {
			dists[pp] = INFINITY;
			for(int tt = 0 ; tt < cc; tt++) {
                double v = (samples.row(pp)-means.row(tt)).squaredNorm();
                dists[pp] = std::min(dists[pp], v);
			}

			//keep normalization factor for later
			norm += dists[pp];
		}

		//set target pp^th greatest distance
		double pct = norm*randf(rng);

        // fill indices
        for(size_t ii=0; ii<npoints; ii++)
            indices[ii] = ii;

        // sort, while keeping indices
        std::sort(indices.begin(), indices.end(), 
                [&dists](size_t i, size_t j){
                    return dists[i] < dists[j]; 
                });

		//go through sorted list to find matching location in CDF
        for(pp = 0; pp < npoints && pct > 0 ; pp++) {
            double d = dists[indices[pp]]; 
            pct -= d;
        }

		//copy randomly selected  point into middle
		means.row(cc) = samples.row(pp);
	}
}


/**
 * @brief Constructor for k-means class
 *
 * @param rank Number of dimensions in input samples.
 * @param k Number of groups to classify samples into
 */
KMeans::KMeans(size_t rank, size_t k) : Classifier(rank), m_k(k), m_mu(k, ndim)
{ }

void KMeans::setk(size_t ngroups) 
{
    m_k = ngroups;
    m_mu.resize(m_k, ndim);
    m_valid = false;
}

/**
 * @brief Sets the mean matrix. Each row of the matrix is a ND-mean, where
 * N is the number of columns.
 *
 * @param newmeans Matrix with new mean
 */
void KMeans::updateMeans(const MatrixXd& newmeans)
{
    if(newmeans.rows() != m_mu.rows() || newmeans.cols() != m_mu.cols()) {
        throw RUNTIME_ERROR("new mean must have matching size with old!");
    }
    m_mu = newmeans;
    m_valid = true;
}

/**
 * @brief Updates the mean coordinates by providing a set of labeled samples.
 *
 * @param samples Matrix of samples, where each row is an ND-sample.
 * @param classes Classes, where rows match the rows of the samples matrix.
 * Classes should be integers 0 <= c < K where K is the number of classes
 * in this.
 */
void KMeans::updateMeans(const MatrixXd samples, const Eigen::VectorXi classes)
{
    if(classes.rows() != samples.rows()){
        throw RUNTIME_ERROR("Rows in sample and group membership vectors "
                "must match, but do not!");
    }
    if(ndim != samples.cols()){
        throw RUNTIME_ERROR("Columns in sample vector must match number of "
                "dimensions, but do not!");
    }
    for(size_t ii=0; ii<classes.rows(); ii++) {
        if(classes[ii] < 0 || classes[ii] >= m_k) {
            throw RUNTIME_ERROR("Invalid class: "+to_string(classes[ii])+
                    " class must be > 0 and < "+to_string(m_k));
        }
    }

    m_mu.setZero();
    vector<size_t> counts(m_k, 0);

    // sum up samples by group
    for(size_t rr = 0; rr < samples.rows(); rr++ ){
        assert(classes[rr] < m_k);
        m_mu.row(classes[rr]) += samples.row(rr);
        counts[classes[rr]]++;
    }

    // normalize
    for(size_t cc=0; cc<m_k; cc++) {
        m_mu.row(cc) /= counts[cc];
    }
    m_valid = true;
}

/**
 * @brief Given a matrix of samples (Samples x Dims, sample on each row),
 * apply the classifier to each sample and return a vector of the classes.
 *
 * @param samples Set of samples, 1 per row
 *
 * @return Vector of classes, rows match up with input sample rows
 */
Eigen::VectorXi KMeans::classify(const MatrixXd& samples)
{   
    Eigen::VectorXi out;
    classify(samples, out);
    return out;
}

/**
 * @brief Given a matrix of samples (Samples x Dims, sample on each row),
 * apply the classifier to each sample and return a vector of the classes.
 *
 * @param samples Set of samples, 1 per row
 * @param classes input/output samples. Returned value indicates number that
 * changed.
 *
 * @return Number of classes that changed
 */
size_t KMeans::classify(const MatrixXd& samples, Eigen::VectorXi& classes)
{ 
    if(!m_valid) {
        throw RUNTIME_ERROR("Error, cannot classify samples because "
                "classifier has not been run on any samples yet. Call "
                "compute on a samples matrix first!");
    }
    if(samples.cols() != ndim) {
        throw RUNTIME_ERROR("Number of columns does in samples matrix should "
                "match KMeans classifier, but doesn't");
    }
    classes.resize(samples.rows());

    size_t change = 0;
    for(size_t rr=0; rr<samples.rows(); rr++) {

        // check all the means, to find the minimum distance
        double bestdist = INFINITY;
        int bestc = -1;
        for(size_t kk=0; kk<m_k; kk++) {
            double dist = (samples.row(rr)-m_mu.row(kk)).squaredNorm();
            if(dist < bestdist) {
                bestdist = dist;
                bestc = kk;
            }
        }
        
        if(classes[rr] != bestc)
            change++;

        // assign the min squared distance
        classes[rr] = bestc;
    }

    return change;
}

/**
 * @brief Updates the classifier with new samples, if reinit is true then
 * no prior information will be used. If reinit is false then any existing
 * information will be left intact. In Kmeans that would mean that the
 * means will be left at their previous state.
 * 
 * @param samples Samples, S x D matrix with S is the number of samples and
 * D is the dimensionality. This must match the internal dimension count.
 *
 * @return -1 if maximum iterations hit, 0 otherwise
 */
int KMeans::update(const MatrixXd& samples, bool reinit)
{
    Eigen::VectorXi classes(samples.rows());

    // initialize with approximate k-means
    if(reinit || !m_valid) 
        approxKMeans(samples, m_k, m_mu);
    m_valid = true;

    // now for the 'real' k-means
    size_t change = SIZE_MAX;
    int ii = 0;
    for(ii=0; ii != maxit && change > 0; maxit++, ii++) {
        change = classify(samples, classes);
        updateMeans(samples, classes);
		cerr << "iter: " << ii << ", " << change << " changed" << endl;
    }

    if(ii == maxit) {
        cerr << "K-Means Failed to Converge" << endl;
        return -1;
    } else 
        return 0;
}

/**
 * @brief Constructor for k-means class
 *
 * @param rank Number of dimensions in input samples.
 * @param k Number of groups to classify samples into
 */
ExpMax::ExpMax(size_t rank, size_t k) : Classifier(rank), m_k(k), m_mu(k, ndim),
    m_cov(k*ndim, ndim), m_tau(k)
{ }

void ExpMax::setk(size_t ngroups) 
{
    m_k = ngroups;
    m_mu.resize(m_k, ndim);
    m_cov.resize(ndim*m_k, ndim);
    m_tau.resize(m_k);
    m_valid = false;
}

/**
 * @brief Sets the mean matrix. Each row of the matrix is a ND-mean, where
 * N is the number of columns.
 *
 * @param newmeans Matrix with new mean
 */
void ExpMax::updateMeanCovTau(const MatrixXd& newmeans, const MatrixXd& newcov,
        const VectorXd& tau)
{
    if(newmeans.rows() != m_mu.rows() || newmeans.cols() != m_mu.cols()) {
        throw RUNTIME_ERROR("new mean must have matching size with old!"
                " Expected: " + to_string(m_mu.rows())+"x" +
                to_string(m_mu.cols()) + ", but got "+
                to_string(newmeans.rows()) + "x" + to_string(newmeans.cols()));
    }
    if(newcov.rows() != m_cov.rows() || newcov.cols() != m_cov.cols()) {
        throw RUNTIME_ERROR("new covariance must have matching size with old!"
                " Expected: " + to_string(m_cov.rows())+"x" +
                to_string(m_cov.cols()) + ", but got "+
                to_string(newcov.rows()) + "x" + to_string(newcov.cols()));
    }
    m_mu = newmeans;
    m_cov = newcov;
    m_tau = tau;
    m_valid = true;
}

/**
 * @brief Updates the mean coordinates by providing a set of labeled samples.
 *
 * @param samples Matrix of samples, where each row is an ND-sample.
 * @param classes Classes, where rows match the rows of the samples matrix.
 * Classes should be integers 0 <= c < K where K is the number of classes
 * in this.
 */
void ExpMax::updateMeanCovTau(const MatrixXd samples, const Eigen::VectorXi classes)
{
    if(classes.rows() != samples.rows()){
        throw RUNTIME_ERROR("Rows in sample and group membership vectors "
                "must match, but do not!");
    }
    if(ndim != samples.cols()){
        throw RUNTIME_ERROR("Columns in sample vector must match number of "
                "dimensions, but do not!");
    }
    for(size_t ii=0; ii<classes.rows(); ii++) {
        if(classes[ii] < 0 || classes[ii] >= m_k) {
            throw RUNTIME_ERROR("Invalid class: "+to_string(classes[ii])+
                    " class must be > 0 and < "+to_string(m_k));
        }
    }

#ifdef VERYDEBUG
    for(size_t cc=0; cc<m_k; cc++) {
        cerr << "Cluster " << cc << endl;
        cerr << "[";
        for(size_t rr = 0; rr < samples.rows(); rr++ ){
            if(classes[rr] == cc) 
                cerr << "[" << samples.row(rr) << "];\n";
        }
        cerr << "]\n";
    }
#endif 

    // compute mean, store counts in tau
    m_mu.setZero();
    m_tau.setZero();
    size_t total = 0;
    for(size_t rr = 0; rr < samples.rows(); rr++ ){
        size_t c = classes[rr]; 
        m_mu.row(c) += samples.row(rr);
        m_tau[c]++;
        total++;
    }
    for(size_t cc=0; cc<m_k; cc++)
        m_mu.row(cc) /= m_tau[cc];

    // compute covariance
    VectorXd x(ndim);
    m_cov.setZero();
    for(size_t rr = 0; rr < samples.rows(); rr++) {
        assert(classes[rr] < m_k);
        size_t c = classes[rr]; 
        x = samples.row(rr)-m_mu.row(c);
        m_cov.block(c*ndim,0, ndim, ndim) += (x*x.transpose());
    }
    // normalize covariance
    for(size_t cc = 0; cc < m_k; cc++)
        m_cov.block(cc*ndim, 0, ndim, ndim) /= (m_tau[cc]-1);

    // convert tau from count to ratio
    for(size_t cc = 0; cc < m_k; cc++)
        m_tau[cc] = m_tau[cc]/total;

    m_valid = true;
}

/**
 * @brief Given a matrix of samples (Samples x Dims, sample on each row),
 * apply the classifier to each sample and return a vector of the classes.
 *
 * @param samples Set of samples, 1 per row
 *
 * @return Vector of classes, rows match up with input sample rows
 */
Eigen::VectorXi ExpMax::classify(const MatrixXd& samples)
{   
    Eigen::VectorXi out;
    classify(samples, out);
    return out;
}

/**
 * @brief Given a matrix of samples (Samples x Dims, sample on each row),
 * apply the classifier to each sample and return a vector of the classes.
 *
 * @param samples Set of samples, 1 per row
 * @param classes input/output samples. Returned value indicates number that
 * changed.
 *
 * @return Number of classes that changed
 */
size_t ExpMax::classify(const MatrixXd& samples, Eigen::VectorXi& classes)
{ 
    if(!m_valid) {
        throw RUNTIME_ERROR("Error, cannot classify samples because "
                "classifier has not been run on any samples yet. Call "
                "compute on a samples matrix first!");
    }
    if(samples.cols() != ndim) {
        throw RUNTIME_ERROR("Number of columns does in samples matrix should "
                "match ExpMax classifier, but doesn't");
    }
    classes.resize(samples.rows());

    static std::default_random_engine rng;

    Eigen::FullPivHouseholderQR<MatrixXd> qr(ndim, ndim);
    MatrixXd prob(samples.rows(), m_k);
    VectorXd x(ndim);
    MatrixXd Cinv(ndim, ndim);
    double det = 0;
    size_t change = 0;
	vector<int64_t> zero_tau;
	zero_tau.reserve(m_k);
    
    //compute Cholesky decomp, then determinant and inverse covariance matrix
	for(int cc = 0; cc < m_k; cc++) {
        cerr << "Covariance:\n" << m_cov.block(cc*ndim, 0, ndim, ndim) << endl;
		if(m_tau[cc] > 0) {
			if(ndim == 1) {
				det = m_cov(0,0);
				Cinv(0,0) = 1./m_cov(cc*ndim,0);
			} else {
                qr.compute(m_cov.block(cc*ndim,0,ndim,ndim));
                Cinv = qr.inverse();
                det = qr.absDeterminant();
			}
		} else {
			//no points in this sample, make inverse covariance matrix infinite
			//dist will be nan or inf, prob will be nan, (dist > max) -> false
            Cinv.fill(INFINITY);
			det = 1;
			zero_tau.push_back(cc);
		}
#ifndef  NDEBUG
        cerr << "Covariance Det:\n" << det << endl;
        cerr << "Inverse Covariance:\n" << Cinv << endl;
#endif 

		//calculate probable location of each point
		double cval = log(m_tau[cc])- .5*log(det)-ndim/2.*log(2*M_PI);
		for(int pp = 0; pp < samples.rows(); pp++) {
            x = samples.row(pp) - m_mu.row(cc);
			
            //log likelihood = (note that last part is ignored because it is
            // constant for all points)
            //log(tau) - log(sigma)/2 - (x-mu)^Tsigma^-1(x-mu) - dlog(2pi)/2 
            double llike = cval - .5*(x.dot(Cinv*x));

			llike = (std::isinf(llike) || std::isnan(llike)) ? -INFINITY : llike;
			prob(pp, cc) = llike;
		}
	}


	double RANDFACTOR = 10;
	double reassigned = 0;
	//place every point in its most probable group
	std::uniform_int_distribution<int> randi(0, zero_tau.size()-1);
    std::uniform_real_distribution<double> randf(0, 1);
	if(zero_tau.size() > 0) {
		cerr << "Zero Tau, Randomly Assigning Based on Probabilities" << endl;
		for(int pp = 0 ; pp < samples.rows(); pp++) {
			double max = -INFINITY;
			int max_class = -1;
			for(int cc = 0 ; cc < m_k; cc++) {
				if(prob(pp, cc) > max) {
					max = prob(pp,cc);
					max_class = cc;
				}
			}

			double p = pow(1-exp(prob(pp, max_class)),RANDFACTOR);
			bool reassign = randf(rng) < p;
			if(reassign) {
				reassigned++;
//				cerr << "Original p: " << exp(prob(pp, max_class)) << endl;
//				cerr << "p=" << p << ", " << "Reassigned? " << reassign;
				max_class = zero_tau[randi(rng)];
//				cerr << " to " << max_class << endl;
			}

			if(classes[pp] != max_class)
				change++;
			classes[pp] = max_class;
		}
		cerr << "Reassigned: " << 100*reassigned/samples.rows() <<"%" << endl;
	} else {
		for(int pp = 0 ; pp < samples.rows(); pp++) {
			double max = -INFINITY;
			int max_class = -1;
			for(int cc = 0 ; cc < m_k; cc++) {
				if(prob(pp, cc) > max) {
					max = prob(pp,cc);
					max_class = cc;
				}
			}

			if(classes[pp] != max_class)
				change++;
			classes[pp] = max_class;
		}
	}
    return change;
}

/**
 * @brief Updates the classifier with new samples, if reinit is true then
 * no prior information will be used. If reinit is false then any existing
 * information will be left intact. In Kmeans that would mean that the
 * means will be left at their previous state.
 * 
 * @param samples Samples, S x D matrix with S is the number of samples and
 * D is the dimensionality. This must match the internal dimension count.
 *
 * @return -1 if maximum iterations hit, 0 otherwise
 */
int ExpMax::update(const MatrixXd& samples, bool reinit)
{
    Eigen::VectorXi classes(samples.rows());

    // initialize with approximate k-means
    if(reinit || !m_valid) {
		approxKMeans(samples, m_k, classes);
//		cerr << "Randomly Assigning Groups" << endl;
//        // In real world data random works as well as anything else
//        for(size_t ii=0; ii < samples.rows(); ii++) 
//            classes[ii] = ii%m_k;
		cerr << "Updating Mean/Cov/Tau" << endl;
        updateMeanCovTau(samples, classes);
#ifndef NDEBUG
        cout << "==========================================" << endl;
        cout << "Init Distributions: " << endl;
        for(size_t cc=0; cc<m_k; cc++ ) {
            cout << "Cluster " << cc << ", prob: " << m_tau[cc] << endl;
            cout << "Mean:\n" << m_mu.row(cc) << endl;
            cout << "Covariance:\n" << 
                m_cov.block(cc*ndim, 0, ndim, ndim) << endl << endl;
        }
        cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << endl;
#endif
    }
    m_valid = true;

    // now for the 'real' k-means
    size_t change = SIZE_MAX;
    int ii = 0;
	DBG1(auto c = clock());
    for(ii=0; ii != maxit && change > 0; ++ii, ++maxit) {
		DBG1(c = clock());
        change = classify(samples, classes);
		DBG1(c = clock() - c);
		DBG1(cerr << "Classify Time: " << c << endl);
		
		DBG1(c = clock());
        updateMeanCovTau(samples, classes);
		DBG1(c = clock() -c );
		DBG1(cerr << "Mean/Cov Time: " << c << endl);

#ifndef NDEBUG
        cout << "==========================================" << endl;
        cout << "Changed Labels: " << change << endl;
        cout << "Current Distributions: " << endl;
        for(size_t cc=0; cc<m_k; cc++ ) {
            cout << "Cluster " << cc << ", prob: " << m_tau[cc] << endl;
            cout << "Mean:\n" << m_mu.row(cc) << endl;
            cout << "Covariance:\n" << 
                m_cov.block(cc*ndim, 0, ndim, ndim) << endl << endl;
        }
        cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << endl;
#endif
		cerr << "iter: " << ii << ", " << change << " changed" << endl;
    }
    
    if(ii == maxit) {
        cerr << "Expectation Maximization of Gaussian Mixture Model Failed "
            "to Converge" << endl;
        return -1;
    } else 
        return 0;
}

/*****************************************************************************
 * Fast Search and Find Density Peaks Clustering Algorithm
 ****************************************************************************/

/**
 * @brief Bin data structure for storing information about nearby points.
 */
struct BinT
{
	double max_rho;
	vector<size_t> neighbors;
	vector<int> members;
	bool visited;
};


/**
 * @brief Computes Density and Peak computation for Fast Search and Find of
 * Density Peaks algorithm. This is a slower, non-bin based version
 *
 * @param samples Samples, S x D matrix with S is the number of samples and
 * D is the dimensionality. This must match the internal dimension count.
 * @param thresh Threshold for density calculation
 * @param rho Point densities
 * @param delta Distance to nearest peak
 * @param parent Index (point) that is the nearest peak
 *
 * @return 0 if successful
 */
int findDensityPeaks_brute(const MatrixXf& samples, double thresh,
		Eigen::VectorXf& rho, VectorXf& delta,
		Eigen::VectorXi& parent)
{
	size_t nsamp = samples.rows();
	rho.resize(nsamp);
	delta.resize(nsamp);
	parent.resize(nsamp);

	const double thresh_sq = thresh*thresh;

	/*************************************************************************
	 * Compute Local Density (rho), by computing distance from every 
	 * other point and summing the number of points within thresh distance. 
	 *************************************************************************/
	for(size_t ii=0; ii<nsamp; ii++)
		rho[ii] = 0;

	cerr << "Computing Rho" << endl;
	double dsq;
	for(size_t ii=0; ii<nsamp; ii++) {
		for(size_t jj=ii+1; jj<nsamp; jj++) {
			dsq = (samples.row(ii) - samples.row(jj)).squaredNorm();
			if(dsq < thresh_sq) {
				rho[ii]++;
				rho[jj]++;
			}
		}
	}

	for(size_t ii=0; ii<nsamp; ii++) {
		rho[ii] += (double)ii/nsamp;
	}

	/************************************************************************
	 * Compute Delta (distance to nearest point with higher density than this
	 ***********************************************************************/
	cerr << "Delta" << endl;
	double maxd = 0;
	for(size_t ii=0; ii<nsamp; ii++) {
		delta[ii] = INFINITY;
		parent[ii] = ii;
		for(size_t jj=0; jj<nsamp; jj++) {
			if(rho[jj] > rho[ii]) {
				dsq = (samples.row(ii) - samples.row(jj)).squaredNorm();
				if(dsq < delta[ii]) {
					delta[ii] = min<double>(dsq, delta[ii]);
					parent[ii] = jj;
				}
			}
		}

		if(!std::isinf(delta[ii])) 
			maxd = max<double>(maxd, delta[ii]);
	}

	for(size_t ii=0; ii<nsamp; ii++) {
		if(std::isinf(delta[ii]))
			delta[ii] = maxd;
	}

	return 0;
}

/**
 * @brief Computes Density and Peak computation for Fast Search and Find of
 * Density Peaks algorithm.
 *
 * Sketch of Algorithm: 
 *
 * Instead of computing distance from ALL points to all points, we compute
 * distance of nearby points. So we construct bins of nearby points. To begin
 * we compute the bin location of all points and save a reference to the point
 * Bin sizes are equal to the threshold distance in the algorithm so that no 
 * point is more than 1 bin away from every point within the threshold. 
 *
 * To compute rho, we
 * then go through all points and compute the distance from every point within
 * the center and neighboring bins, summing rho for every point within the 
 * distance threshold. This should be order N^2/B instead of N^2
 *
 * To compute delta (the distance of a point to the nearest higher rho point),
 * you go to every point and search for bins that have rho greater than rho
 * for the point. This is sped up by caching the maximum rho in every bin
 * and therefore the actual number of distances computed is roughly N*N/B. 
 *
 * @param samples Samples, S x D matrix with S is the number of samples and
 * D is the dimensionality. This must match the internal dimension count.
 * @param thresh Threshold for density calculation
 * @param rho Point densities
 * @param delta Distance to nearest peak
 * @param parent Index (point) that is the nearest peak
 *
 * @return 0 if successful
 */
int findDensityPeaks(const MatrixXf& samples, double thresh,
		Eigen::VectorXf& rho, VectorXf& delta,
		Eigen::VectorXi& parent)
{
	size_t ndim = samples.cols();
	size_t nsamp = samples.rows();
	rho.resize(nsamp);
	delta.resize(nsamp);
	parent.resize(nsamp);

	const double thresh_sq = thresh*thresh;
	const double binwidth = thresh;

	/*************************************************************************
	 * Construct Bins, with diameter thresh so that points within thresh 
	 * are limited to center and immediate neighbor bins
	 *************************************************************************/

	cerr << "Filling Bins" << endl;
	// First Determine Size of Bins in each Dimension
	vector<size_t> sizes(ndim);
	vector<size_t> strides(ndim);
	vector<pair<double,double>> range(ndim); //min/max in each dim
	size_t totalbins = 1;
	for(size_t cc=0; cc<ndim; cc++) {

		// compute range
		range[cc].first = INFINITY;
		range[cc].second = -INFINITY;
		for(size_t rr=0; rr<nsamp; rr++) {
			range[cc].first = std::min<double>(range[cc].first, samples(rr,cc));
			range[cc].second = std::max<double>(range[cc].second, samples(rr,cc));
		}

		// break bins up into thresh+episolon chunks. The extra bin is to
		// preven the maximum point from overflowing. This would happen if
		// MAX-MIN = K*thresh for integer K. If K = 1 then the maximum value
		// would make exactly to the top of the bin
		sizes[cc] = 1+(range[cc].second-range[cc].first)/binwidth;
		totalbins *= sizes[cc];
	}

	// compute strides (row major order, highest dimension fastest)
	strides[ndim-1] = 1;
	for(int64_t ii=ndim-2; ii>=0; ii--)
		strides[ii] = sizes[ii+1]*strides[ii+1];
	
	// Now Initialize Bins
	KSlicer slicer(ndim, sizes.data());
	slicer.setRadius(1);
	vector<BinT> bins(totalbins);
	for(slicer.goBegin(); !slicer.eof(); ++slicer) {
		bins[*slicer].max_rho = 0;
		bins[*slicer].neighbors.clear();
		bins[*slicer].members.clear();
		bins[*slicer].visited = true;

		for(size_t kk=0; kk<slicer.ksize(); kk++) {
			if(slicer.insideK(kk) && slicer.getK(kk) != slicer.getC()) 
				bins[*slicer].neighbors.push_back(slicer.getK(kk));
		}

	}

	// fill bins with member points
	for(size_t rr=0; rr<nsamp; rr++) {
		// determine bin
		// determine bin
		size_t bin = 0;
		for(size_t cc=0; cc<ndim; cc++) {
			bin += strides[cc]*floor((samples(rr,cc)-range[cc].first)/binwidth);
		}
	
		// place this sample into bin's membership
		bins[bin].members.push_back(rr);
	}

	/*************************************************************************
	 * Compute Local Density (rho), by creating a list of points in the
	 * vicinity of each bin, then computing the distance between all pairs
	 * locally
	 *************************************************************************/
	double distsq;
	double max_rho = 0; // overall maximum rho, so we don't search for it later
	cerr << "Computing Rho" << endl;
	for(size_t bb=0; bb<bins.size(); bb++) {
		if(bb % 1024 == 0)
			cerr << bb << "/" << bins.size() << endl;

		// for every member of this bin, check 1) this bin 2) neighboring bins
		for(const auto& xi : bins[bb].members) {
			rho[xi] = 0;
			// check others in this bin
			for(const auto& xj : bins[bb].members) {
				if(xi != xj) {
					distsq = (samples.row(xj)-samples.row(xi)).squaredNorm();
					if(distsq < thresh_sq) {
						rho[xi]++;
					}
				}
			}
	
			// neigboring/adjacent bins
			for(auto adjbin: bins[bb].neighbors) {
				for(const auto& xj : bins[adjbin].members) {
					double distsq = (samples.row(xj)-samples.row(xi)).squaredNorm();
					if(distsq < thresh_sq) {
						rho[xi]++;
					}
				}
			}

			rho[xi] += (double)xi/nsamp; 
			bins[bb].max_rho = std::max<double>(bins[bb].max_rho, rho[xi]);
			max_rho = std::max<double>(max_rho, rho[xi]);
		}
	}


	/************************************************************************
	 * Compute Delta (distance to nearest point with higher density than this
	 ***********************************************************************/
	cerr << "Computing Delta" << endl;
	list<int> unresolved;
	std::list<pair<size_t, size_t>> queue; // queue of bins (by index)
	double dsq; // distance squared

	// circle enclosing the hypercube
	double enc_radsq= 0;
	double enc_rad= 0;
	for(size_t dd=0; dd<ndim; dd++){
		enc_radsq += binwidth*binwidth/4;
		enc_rad += sqrt(enc_radsq);
	}

	double maxdelta = 0;
	for(size_t ii=0; ii<nsamp; ii++) {
		if(ii % 1024 == 0)
			cerr << ii << "/" << nsamp << endl;

		parent[ii] = ii;
		delta[ii] = INFINITY;
		if(rho[ii] == max_rho) 
			continue;

		// determine bin
		size_t bin = 0;
		for(size_t cc=0; cc<ndim; cc++) {
			bin += strides[cc]*floor((samples(ii,cc)-range[cc].first)/binwidth);
		}

		// set visited to false in all bins
		for(size_t bb=0; bb<bins.size(); bb++)
			bins[bb].visited = false;

		double dmin = INFINITY;

		// push center and neighbors
		queue.clear();
		queue.push_back(make_pair(bin, 0));
		bins[bin].visited = true;
		for(auto bn : bins[bin].neighbors) {
			queue.push_back(make_pair(bn, 0));
			bins[bin].visited = true;
		}


		while(!queue.empty() && queue.front().second*binwidth < dmin) {
			size_t b = queue.front().first;
			size_t priority = queue.front().second;
			queue.pop_front();

			// this bin contains at least 1 point that satisfies the rho 
			// criteria. Find that point (or a closer one) and update dmin
			if(bins[b].max_rho > rho[ii]) {
				for(auto jj : bins[b].members) {
					if(rho[jj] > rho[ii]) {
						dsq = (samples.row(ii)-samples.row(jj)).squaredNorm();
						if (dsq < dmin*dmin) {
							dmin = sqrt(dsq);
							parent[ii] = jj;
							delta[ii] = dsq;
						}
					}
				}
			}

			for(auto bn : bins[b].neighbors) {
				if(!bins[bn].visited) {
					queue.push_back(make_pair(bn, priority+1));
					bins[bn].visited = true;
				}
			}
		}
		
		if(!std::isinf(delta[ii]))
			maxdelta = max<double>(maxdelta, delta[ii]);

	}

	for(size_t ii=0; ii<nsamp; ii++) 
		if(std::isinf(delta[ii]))
			delta[ii] = maxdelta; 

	return 0;
}

/**
 * @brief Updates the classifier with new samples, if reinit is true then
 * no prior information will be used. If reinit is false then any existing
 * information will be left intact. In Kmeans that would mean that the
 * means will be left at their previous state.
 *
 * see "Clustering by fast search and find of density peaks"
 * by Rodriguez, a.  Laio, A.
 *
 * This is a brute force solution, order N^2
 *
 * @param samples Samples, S x D matrix with S is the number of samples and
 * D is the dimensionality. This must match the internal dimension count.
 * @param thresh Threshold distance for density calculation
 * @param classes Output classes
 * @param brute whether ther use slower brute force method for density
 * calculation (for testing purposes only)
 *
 * return -1 if maximum number of iterations hit, 0 otherwise (converged)
 */
int fastSearchFindDP(const MatrixXf& samples, double thresh, double outthresh,
		 Eigen::VectorXi& classes, bool brute)
{
	size_t nsamp = samples.rows();
	Eigen::VectorXf delta;
	Eigen::VectorXf rho;
	if(brute)
		findDensityPeaks_brute(samples, thresh, rho, delta, classes);
	else
		findDensityPeaks(samples, thresh, rho, delta, classes);
	
	/************************************************************************
	 * Break Into Clusters 
	 ***********************************************************************/
	vector<int64_t> order(nsamp);
	double mean = 0;
	double stddev = 0;
	for(size_t rr=0; rr<nsamp; rr++) {
		stddev += delta[rr]*delta[rr];
		mean += delta[rr];
	}
	stddev = sqrt(sample_var(nsamp, mean, stddev));
	mean /= nsamp;
	
	std::map<size_t,size_t> classmap;
	size_t nclass = 0;
	for(size_t rr=0; rr<nsamp; rr++) {
		// follow trail of parents until we hit a node with the needed delta
		size_t pp = rr;
		while(delta[pp] < mean + outthresh*stddev && classes[pp] != pp) 
			pp = classes[pp];

		// change the parent to the true parent for later iterations
		classes[rr] = pp;

		auto ret = classmap.insert(make_pair(pp, nclass));
		if(ret.second) 
			nclass++;
	}

	// finally convert parent to classes
	for(size_t rr=0; rr<nsamp; rr++) 
		classes[rr] = classmap[classes[rr]];

	return 0;
}



///**
// * @brief Solves y = Xb (where beta is a vector of parameters, y is a vector
// * of desired results and X is the design/system)
// *
// * @param X Design or system
// * @param y Observations (target)
// *
// * @return Parameters that give the minimum squared error (y - Xb)^2
// */
//VectorXd leastSquares(const MatrixXd& X, const VectorXd& y)
//{
//    assert(X.rows() == y.rows());
//    Eigen::JacobiSVD<MatrixXd> svd;
//    svd.compute(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
//    return svd.solve(y);
//}
//
///**
// * @brief Solves iteratively re-weighted least squares problem. Wy = WXb (where
// * W is a weighting matrix, beta is a vector of parameters, y is a vector of
// * desired results and X is the design/system)
// *
// * @param X Design or system
// * @param y Observations (target)
// * @param w Initial weights (note that zeros will be kept at 0)
// *
// * @return Parameters that give the minimum squared error (y - Xb)^2
// */
//VectorXd IRLS(const MatrixXd& X, const VectorXd& y, VectorXd& w)
//{
//    assert(X.rows() == y.rows());
//    VectorXd beta(X.cols());
//    MatrixXd wX = w.asDiagonal()*X;
//    VectorXd wy = w.asDiagonal()*y;
//    VectorXd err(y.rows());
////    for() {
//        //update beta
//        beta = wX.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(wy);
//        err = (X*beta-y);
//
////        // update weights;
////        for(size_t ii=0; ii<w.size(); ii++) {
////            w[ii] = pow(
////        }
// //   }
//    return beta;
//}


/**
 * @brief Band Lanczos Methof for Hessian Matrices
 *
 * p initial guesses (b_1...b_p) 
 * set v_k = b_k for k = 1,2,...p
 * set p_c = p
 * set I = nullset
 * for j = 1,2, ..., until convergence or p_c = 0; do
 * (3) compute ||v_j||
 *     decide if v_j should be deflated, if yes, then
 *         if j - p_c > 0, set I = I union {j-p_c}
 *         set p_c = p_c-1. If p_c = 0, set j = j-1 and STOP
 *         for k = j, j+1, ..., j+p_c-1, set v_k = v_{k+1}
 *         return to step (3)
 *     set t(j,j-p_c) = ||v_j|| and normalize v_j = v_j/t(j,j-p_c)
 *     for k = j+1, j+2, ..., j+p_c-1, set 
 *         t(j,k-p_c) = v_j^*v_k and v_k = v_k - v_j t(j,k-p_c)
 *     compute v(j+p_c) = Av_j
 *     set k_0 = max{1,j-p_c}. For k = k_0, k_0+1,...,j-1, set 
 *         t(k,j) = conjugate(t(j,k)) and v_{j+p_c} = v_{j+p_c}-v_k t(k,j)
 *     for k in (I union {j}) (in ascending order), set 
 *         t(k,j) = v^*_k v_{j+p_c} and v_{j+p_c} = v_{j+p_c} - v_k t(k,j)
 *     for k in I, set s(j,k) = conjugate(t(k,j))
 *     set T_j^(pr) = T_j + S_j = [t(i,k)] + [s(i,k)] for (i,k=1,2,3...j)
 *     test for convergence
 * end for
 *
 * @param V input/output the initial and final vectors
 */
void bandlanczos(list<VectorXd>& V)
{
	// TODO fill V with random values

	int64_t pc = V.size();
	vector<bool> I(V.size(), false);
	list<VectorXd>::iterator Vk, Vj, Vjpc;

	// Size?
	// sparse MatrixXd T(V.size(), V.size());
	// sparse MatrixXd S(V.size(), V.size());

	Vj = V.begin();
	for(int64_t jj=0; pc >= 0; jj++) {
		// compute norm(vj)
		double vnorm = Vj->norm();
		if(vnorm < DTOL) {
			/* Deflate */
			if(jj - pc >= 0) I[jj-pc] = true;
			if(--pc < 0) {
				jj--;
				break;
			}

			//erase Vj and return to beginning with jj unchanged
			Vj = V.erase(Vj);
			jj--;
		}

		// set t(j,j-pc) = norm(vj) and normalize vj
		if(jj-pc >= 0)
			T(jj, jj-pc) = vnorm;
		(*Vj) /= vnorm;

		Vk = Vj; ++Vk;
		for(int64_t kk=jj+1; kk<jj+pc; kk++, ++Vk) {
			double val = (Vj->transpose()*(*Vk));
			if(kk - pc >= 0) T(jj, kk-pc) = val;
			*Vk -= (*Vj)*val;
		}
		Vjpc = Vk;
		*Vjpc = A*(*Vj);
		k0 = max(0, jj-pc);
		
		Vk = V.begin();
		for(int64_t kk=0; kk<k0; kk++, ++Vk) continue;
		for(int64_t kk=k0; kk<jj; kk++) {
			T(kk, jj) = T(jj,kk);
			*Vjpc -= (*Vk)*T(kk,jj);
		}

		// iterator through k for (I UNION j)
		// temporarily add j to I
		bool jval = I[jj];
		I[jj] = true;
		Vk = V.begin(); 
		for(int64_t kk=0; kk<I.size(); ++Vk, kk++) {
			if(I[kk]) {
				T(kk,jj) = (Vk->transpose())*(*Vjpc);
				*Vjpc -= (*Vk)*T(kk,jj);
			}
		}
		I[jj] = jval;

		Vk = V.begin();
		for(int64_t kk=0; kk<I.size(); ++Vk, kk++) {
			if(I[kk]) 
				S(jj,kk) = T(kk, jj);
		}

		T.col(jj) += S.col(j);
	}
}

} // NPL
