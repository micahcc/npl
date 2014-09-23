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
#include "statistics.h"
#include "basic_functions.h"
#include "macros.h"

#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>

using namespace std;
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
 * @param means Estimated mid-points/means
 */
void approxKMeans(const MatrixXd& samples, size_t nclass, MatrixXd& means)
{
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
 */
void KMeans::update(const MatrixXd& samples, bool reinit)
{
    Eigen::VectorXi classes(samples.rows());

    // initialize with approximate k-means
    if(reinit || !m_valid) 
        approxKMeans(samples, m_k, m_mu);
    m_valid = true;

    // now for the 'real' k-means
    size_t change = SIZE_MAX;
    while(change > 0) {
        change = classify(samples, classes);
        updateMeans(samples, classes);
    }
}

/**
 * @brief Constructor for k-means class
 *
 * @param rank Number of dimensions in input samples.
 * @param k Number of groups to classify samples into
 */
ExpMax::ExpMax(size_t rank, size_t k) : Classifier(rank), m_k(k), m_mu(k, ndim),
    m_cov(k*ndim, ndim)
{ }

void ExpMax::setk(size_t ngroups) 
{
    m_k = ngroups;
    m_mu.resize(m_k, ndim);
    m_cov.resize(ndim*m_k, ndim);
    m_valid = false;
}

/**
 * @brief Sets the mean matrix. Each row of the matrix is a ND-mean, where
 * N is the number of columns.
 *
 * @param newmeans Matrix with new mean
 */
void ExpMax::updateMeanCov(const MatrixXd& newmeans, const MatrixXd& newcov)
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
void ExpMax::updateMeanCov(const MatrixXd samples, const Eigen::VectorXi classes)
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

    VectorXd x(ndim);
    m_mu.setZero();
    vector<size_t> counts(m_k, 0);

    // sum up samples by group
    for(size_t rr = 0; rr < samples.rows(); rr++ ){
        assert(classes[rr] < m_k);
        size_t c = classes[rr]; 
        x = samples.row(rr);
        m_mu.row(c) += x;
        m_cov.block(c*ndim,0, ndim, ndim) += x*x.transpose();
        counts[c]++;
    }

    // normalize
    for(size_t cc=0; cc<m_k; cc++) {
        m_mu.row(cc) /= counts[cc];
        m_cov.block(cc*ndim,0, ndim, ndim) /= counts[cc];
        m_tau[cc] = 1./counts[cc];
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
    classes.resize(samples.size());

    Eigen::FullPivHouseholderQR<MatrixXd> qr(ndim, ndim);
    MatrixXd prob(samples.rows(), m_k);
    VectorXd x;
    MatrixXd Cinv;
    double det = 0;
    size_t change = 0;
	
    //compute Cholesky decomp, then determinant and inverse covariance matrix
	for(int cc = 0; cc < m_k; cc++) {
		if(m_tau[cc] > 0) {
			if(ndim == 1) {
				det = m_cov(0,0);
				m_covinv(cc*ndim,0) = 1./m_cov(cc*ndim,0);
			} else {
                qr.compute(m_cov.block(cc*ndim,0,ndim,ndim));
                Cinv = qr.inverse();
                det = qr.absDeterminant();
			}
		} else {
			//no points in this sample, make inverse covariance matrix infinite
			//dist will be nan or inf, prob will be nan, (dist > max) -> false
            m_cov.fill(INFINITY);
			det = 1;
		}

		//calculate probable location of each point
		for(int pp = 0; pp < samples.rows(); pp++) {
            x = samples.row(pp) - m_mu.row(cc);
			
            //log likelihood = (note that last part is ignored because it is
            // constant for all points)
            //log(tau) - log(sigma)/2 - (x-mu)^Tsigma^-1(x-mu) - dlog(2pi)/2 
            double llike = (x*Cinv*x)(0,0);
            llike += log(m_tau[cc]) - .5*log(det);

			if(std::isinf(llike) || std::isnan(llike))
				llike = -INFINITY;

			prob(pp, cc) = llike;
		}
	}

	//place every point in its most probable group
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
 */
void ExpMax::update(const MatrixXd& samples, bool reinit)
{
    Eigen::VectorXi classes(samples.rows());

    // initialize with approximate k-means
    if(reinit || !m_valid) 
        approxKMeans(samples, m_k, m_mu);
    m_valid = true;

    // now for the 'real' k-means
    size_t change = SIZE_MAX;
    while(change > 0) {
        change = classify(samples, classes);
        updateMeanCov(samples, classes);
    }
}


} // NPL
