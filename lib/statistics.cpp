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
 * @file statistics.cpp Tools for analyzing data, including general linear
 * modeling.
 *
 *****************************************************************************/

#include <Eigen/Eigenvalues>
#include <Eigen/IterativeSolvers>
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

/**********************************************************
 * Basic Statistical Functions
 **********************************************************/

/**
 * @brief Computes mutual information between signal a and signal b which
 * are of length len. Marginal bins used is mbin
 *
 * @param len Length of signal and and b
 * @param a Signal a
 * @param b Signal b
 * @param mbin Bins to use in marginal distribution (mbin*mbin) used in joint
 *
 * @return
 */
double mutualInformation(size_t len, double* a, double* b, size_t mbin)
{
	vector<vector<double>> joint(mbin, vector<double>(mbin, 0));
	vector<double> marg1(mbin, 0);
	vector<double> marg2(mbin, 0);

	double amin = INFINITY;
	double bmin = INFINITY;
	double awidth = -INFINITY;
	double bwidth = -INFINITY;
	for(size_t tt=0; tt<len; tt++) {
		amin = std::min(amin, a[tt]);
		awidth = std::max(awidth, a[tt]);
		bmin = std::min(bmin, b[tt]);
		bwidth = std::max(bwidth, b[tt]);
	}
	awidth = (awidth-amin)/(mbin-1);
	bwidth = (bwidth-bmin)/(mbin-1);

	for(size_t tt=0; tt<len; tt++) {
		int ai = ((a[tt]-amin)/awidth);
		int bi = ((b[tt]-bmin)/bwidth);
		assert(ai >= 0 && ai < mbin);
		assert(bi >= 0 && bi < mbin);

		marg1[ai]++;
		marg2[bi]++;
		joint[ai][bi]++;
	}

	double mi = 0;
	for(size_t ii=0; ii<mbin; ii++) {
		for(size_t jj=0; jj<mbin; jj++) {
			double pj = joint[ii][jj]/len;
			double pa = marg1[ii]/len;
			double pb = marg2[jj]/len;
			if(pj > 0)
				mi += pj*log(pj/(pa*pb));
		}
	}

	return mi;
}

/**
 * @brief Computes correlation between signal a and signal b which
 * are of length len.
 *
 * @param len Length of signal and and b
 * @param a Signal a
 * @param b Signal b
 *
 * @return
 */
double correlation(size_t len, double* a, double* b)
{
	double ab = 0;
	double aa = 0;
	double bb = 0;
	double ma = 0;
	double mb = 0;

	for(size_t ii=0; ii<len; ii++) {
		ab += a[ii]*b[ii];
		aa += a[ii]*a[ii];
		bb += b[ii]*b[ii];
		ma += a[ii];
		mb += b[ii];
	}
	return sample_corr(len, ma, mb, aa, bb, ab);
}

void apply(double* dst, const double* src,  double(*func)(double), size_t sz)
{
	for(unsigned int ii = 0 ; ii < sz; ii++)
		dst[ii] = func(src[ii]);
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
	auto it = std::upper_bound(m_tvals.begin(), m_tvals.end(), t);

	if(it == m_tvals.end()) {
#ifndef NDEBUG
		cerr << "Warning, effectively 0 p-value returned!" << endl;
#endif
		if(negative)
			return 1-m_cdf[m_cdf.size()-1];
		else
			return m_cdf[m_cdf.size()-1];
	}
	assert(it != m_tvals.begin());

	// Linear Interpolate
	int ii = distance(m_tvals.begin(), it);
	double tp = m_tvals[ii-1];
	double tn = m_tvals[ii];
	double prev = m_cdf[ii-1];
	double next = m_cdf[ii];
	out = prev*(tn-t)/(tn-tp) + next*(t-tp)/(tn-tp);

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
	auto it = std::upper_bound(m_tvals.begin(),
			m_tvals.end(), t);
	if(it == m_tvals.end()) {
#ifndef NDEBUG
		cerr << "Warning, effectively 0 p-value returned!" << endl;
#endif
		return 1-m_pdf[m_pdf.size()-1];
	}
	assert(it != m_tvals.begin());

	// Linear Interpolate
	int ii = distance(m_tvals.begin(), it);
	double tp = m_tvals[ii-1];
	double tn = m_tvals[ii];
	double prev = m_pdf[ii-1];
	double next = m_pdf[ii];
	out = prev*(tn-t)/(tn-tp) + next*(t-tp)/(tn-tp);

	if(negative)
		return 1-out;
	else
		return out;
};

double StudentsT::icdf(double p) const
{
	bool negative = false;
	if(p < 0.5) {
		negative = true;
		p = 1-p;
	}

	double out = 0;
	auto it = std::upper_bound(m_cdf.begin(), m_cdf.end(), p);

	if(it == m_cdf.end()) {
#ifndef NDEBUG
		cerr << "Warning, effectively infinite t-value returned!" << endl;
#endif
		return m_tvals[m_tvals.size()-1]*(negative ? -1 : 1);
	}
	assert(it != m_cdf.begin());

	// Linear Interpolate
	int ii = distance(m_cdf.begin(), it);
	double tp = m_tvals[ii-1];
	double tn = m_tvals[ii];
	double cp = m_cdf[ii-1];
	double cn = m_cdf[ii];
	out = tp*(cn-p)/(cn-cp) + tn*(p-cp)/(cn-cp);

	if(negative)
		return -out;
	else
		return out;
};

void StudentsT::init()
{
	m_cdf.resize(m_tmax/m_dt);
	m_pdf.resize(m_tmax/m_dt);
	m_tvals.resize(m_tmax/m_dt);

	double dof = m_dof;
	double logcoeff = lgamma((dof+1)/2)-0.5*(log(dof)+log(M_PI))-lgamma(dof/2);
	double coeff = exp(logcoeff);

	// Evaluate PDF
	for(size_t ii=0; ii < m_pdf.size(); ii++) {
		double t = ii*m_dt;
		m_tvals[ii] = t;
		m_pdf[ii] = coeff*pow(1+t*t/dof, -(dof+1)/2);
	}

	// Perform Integration with Simpsons Rule
	m_cdf.front() = 0.5;
	for(size_t ii = 1; ii<m_cdf.size(); ii++) {
		// Evaluate Integral Up to ii
		// int_a^b f(x) dx = (f(a)+4f((a+b)/2)+f(b))(b-a)/6
		double a = (ii-1)*m_dt;
		double b = ii*m_dt;
		double ab2 = (ii-0.5)*m_dt; // (a+b)/2
		double fab2 = coeff*pow(1+ab2*ab2/dof, -(dof+1)/2); // f((a+b)/2)

		m_cdf[ii] = m_cdf[ii-1] + (m_pdf[ii-1]+ 4*fab2 + m_pdf[ii])*(b-a)/6;
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
 * @param out Struct with Regression Results.
 * @param y response variables
 * @param X independent variables
 * @param covInv Inverse of covariance matrix, to compute us pseudoinverse(X^TX)
 * @param Xinv Pseudo inverse of X. Compute with pseudoInverse(X)
 * @param student_cdf Pre-computed students' T distribution. Example:
 * auto v = students_t_cdf(X.rows()-1, .1, 1000);
 *
 * @return Struct with Regression Results.
 */
void regress(RegrResult* out,
		const Ref<const VectorXd> y,
		const Ref<const MatrixXd> X,
		const Ref<const VectorXd> covInv,
		const Ref<const MatrixXd> Xinv,
		const StudentsT& distrib)
{
	if(!out)
		throw INVALID_ARGUMENT("Regression Result Pointer invalidmismatch");
	if(y.rows() != X.rows())
		throw INVALID_ARGUMENT("y and X matrices row mismatch");
	if(X.rows() != Xinv.cols() || X.cols() != Xinv.rows())
		throw INVALID_ARGUMENT("X and pseudo inverse of X row mismatch");
	if(covInv.rows() != X.cols())
		throw INVALID_ARGUMENT("Cov Invers and X mismatch");

	if(y.rows() != X.rows())
		throw INVALID_ARGUMENT("y and X matrices row mismatch");

	out->bhat = Xinv*y;
	out->yhat = X*out->bhat;
	double ssres = (out->yhat - y).squaredNorm();

	// compute total sum of squares
	double mean = y.mean();
	double sstot = (y-VectorXd::Constant(y.rows(), mean)).squaredNorm();

	// estimate the standard deviation of the error term
	out->sigmahat = ssres/(X.rows()-X.cols());

	// Compute the R-Squared and adjusted R squared
	out->rsqr = 1-ssres/sstot;
	out->adj_rsqr = out->rsqr - (1-out->rsqr)*(X.cols()-1)/(X.rows()-X.cols()-1);

	out->std_err.resize(X.cols());
	out->t.resize(X.cols());
	out->p.resize(X.cols());
	out->dof = X.rows()-1;

	for(size_t ii=0; ii<X.cols(); ii++) {
		out->std_err[ii] = sqrt(out->sigmahat*covInv[ii]);

		double t = out->bhat[ii]/out->std_err[ii];
		out->t[ii] = t;
		double p = distrib.cdf(t);
		if(t > 0) p = 1-p;
		out->p[ii] = 2*p;
	}
}

double gaussian1D(double mean, double sd, double x)
{
	return exp(-(x-mean)*(x-mean)/(2*sd*sd))/(sd*sqrt(2*M_PI))
}


/**
 * @brief PDF for the gamma distribution, if mean is negative then it is
 * assumed that x should be negated as well.
 *
 * mean = k theta
 * var = k theta theta
 *
 * theta = var / mean
 * k = mean / theta
 *
 * prob(mu, sd, x) = x^{k-1}exp(-x/theta)/(gamma(k) theta^k)
 * log prob(mu, sd, x) = (k-1)log(x)-x/theta-log(gamma(k)) - k*log(theta)
 *
 * @param mean
 * @param sd
 * @param x
 *
 * @return
 */
double gamma(double mean, double sd, double x)
{
	if(mean < 0)
		mean = -mean;
		x = -x;
	}
	double theta = sd*sd/mean;
	double k = mean / theta;
	double lp = (k-1)*log(x) - x/theta - lgamma(k) - k*log(theta);
	return exp(lp);
}

void mixtureModel(const Ref<const VectorXd> data,
		vector<std::function<double(double,double,double)>> pdfs,
		Ref<VectorXd> mean, Ref<VectorXd> sd, Ref<VectorXd> prior)
{
	if(mean.rows() != sd.rows() || mean.rows() != pdfs.size())
		throw INVALID_ARGUMENT("Input mean and standard deviation must be "
				"initialized and have the same size as pdfs");

	double THRESH = 0.0001;
	mean = 0;
	sd = 1;
	size_t ndist = loglikelihood.size();
	double total = 0;
	double change = THRESH;
	for(size_t ii=0; ii<ndist; ii++)
		prior[ii] = 1./ndist;

	MatrixXd prob(data.rows(), ndist);
	while(change > THRESH) {
		/*
		 * estimate probabilities
		 */
		prob.setZero();
		for(size_t rr=0; rr<prob.rows(); rr++) {
			double total = 0;
			for(size_t cc=0; cc<prob.cols(); cc++) {
				prob(rr, cc) = prior[cc]*pdfs[cc](mean[cc], sd[cc], data[rr]);
				total += prob(rr,cc);
			}
			for(size_t cc=0; cc<prob.cols(); cc++)
				prob(rr,cc) /= total;
		}

		/*
		 * update means/stddevs
		 */
		change = 0;
		for(size_t tt=0; tt<prior.rows(); tt++) {
			double pmean = mean[tt];
			mean[tt] = 0;
			sd[tt] = 0;
			prior[tt] = 0;
			for(size_t rr=0; rr<data.rows(); rr++) {
				mean[tt] += prob(rr,tt)*data[rr];
				sd[tt] += prob(rr,tt)*data[rr]*data[rr];
				prior[tt] += prob(rr,tt);
			}
			mean[tt] /= prior[tt];
			sd[tt] = (sd[tt] - mean[tt]*mean[tt])/prior[tt];

			if(fabs(pmean - mean[tt]) > change)
				change = fabs(pmean - mean[tt]);
		}

		// Prior was the total probability for each, so then sum the 3 and
		// divide to get the relative proportions
		prior /= prior.sum();
	}

}

/**
 * @brief Computes the Ordinary Least Square predictors, beta for
 *
 * \f$ y = X \hat \beta \f$
 *
 * Returning beta. This is the same as the other regress function, but allows
 * for cacheing of pseudoinverse of X
 *
 * @param out Struct with Regression Results.
 * @param y response variables
 * @param X independent variables
 *
 */
void regress(RegrResult* out,
		const Ref<const VectorXd> y,
		const Ref<const MatrixXd> X)
{
	if(y.rows() != X.rows())
		throw INVALID_ARGUMENT("y and X matrices row mismatch");

	// need to compute the CDF for students_t_cdf
	const double MAX_T = 100;
	const double STEP_T = 0.1;
	StudentsT distrib(out->dof, STEP_T, MAX_T);

	MatrixXd Xinv = pseudoInverse(X);
	VectorXd covInv = pseudoInverse(X.transpose()*X).diagonal();
}

/**
 * @brief Computes the pseudoinverse of the input matrix
 *
 * \f$ P = VE^-1U^* \f$
 *
 * @return Psueodinverse
 */
MatrixXd pseudoInverse(const Ref<const MatrixXd> X)
{
	double THRESH = 0.000001;
	JacobiSVD<MatrixXd> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
	VectorXd singular_values = svd.singularValues();

	size_t rank = 0;
	for(size_t ii=0; ii<svd.singularValues().rows(); ii++) {
		if(singular_values[ii] > THRESH) {
			singular_values[ii] = 1./singular_values[ii];
			rank++;
		} else
			singular_values[ii] = 0;
	}
	return svd.matrixV().leftCols(rank)*
		singular_values.head(rank).asDiagonal()*
		svd.matrixU().leftCols(rank).transpose();
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
void approxKMeans(const Ref<const MatrixXd> samples, size_t nclass,
		Eigen::VectorXi& labels)
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
void approxKMeans(const Ref<const MatrixXd> samples,
		size_t nclass, MatrixXd& means)
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
void KMeans::updateMeans(const Ref<const MatrixXd> newmeans)
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
void KMeans::updateMeans(const Ref<const MatrixXd> samples,
		const Ref<const Eigen::VectorXi> classes)
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
Eigen::VectorXi KMeans::classify(const Ref<const MatrixXd> samples)
{
	Eigen::VectorXi out(samples.rows());
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
size_t KMeans::classify(const Ref<const MatrixXd> samples, Ref<VectorXi> classes)
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
	if(classes.rows() != samples.rows()) {
		throw RUNTIME_ERROR("Number of rows does in samples matrix should "
				"match # of rows in samples, but doesn't");
	}
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
int KMeans::update(const Ref<const MatrixXd> samples, bool reinit)
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
void ExpMax::updateMeanCovTau(const Ref<const MatrixXd> newmeans, const Ref<const MatrixXd> newcov,
		const Ref<const VectorXd> tau)
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
 * @brief Updates the mean/cov/tau coordinates by using the weighted class
 * estimates (rather than hard classification, like the previous)
 *
 * @param samples Matrix of samples, where each row is an ND-sample.
 * @param prob Probability that each sample is in a given distribution
 */
void ExpMax::updateMeanCovTau(const Ref<const MatrixXd> samples, Ref<MatrixXd> prob)
{
	if(prob.rows() != samples.rows()){
		throw RUNTIME_ERROR("Rows in sample and group membership vectors "
				"must match, but do not!");
	}
	if(prob.cols() != m_k){
		throw RUNTIME_ERROR("Cols in group membership prob vectors and m_k "
				"must match, but do not!");
	}
	if(ndim != samples.cols()){
		throw RUNTIME_ERROR("Columns in sample vector must match number of "
				"dimensions, but do not!");
	}

	// Compute Tau
	prob /= prob.sum();
	m_tau = prob.colwise().sum().transpose();

	// compute mean, store counts in tau
	m_mu.setZero();
	for(size_t rr = 0; rr < samples.rows(); rr++ ){
		for(size_t cc=0; cc<m_k; cc++) {
			m_mu.row(cc) += prob(rr, cc)*samples.row(rr);
		}
	}

	for(size_t cc=0; cc<m_k; cc++)
		m_mu.row(cc) /= m_tau[cc];

	// compute covariance
	m_cov.setZero();
	VectorXd x(ndim);
	for(size_t rr = 0; rr < samples.rows(); rr++) {
		for(size_t cc=0; cc<m_k; cc++) {
			x = (samples.row(rr)-m_mu.row(cc));
			m_cov.middleRows(cc*ndim, ndim) += prob(rr,cc)*x*x.transpose();
		}
	}

	for(size_t cc=0; cc<m_k; cc++)
		m_cov.middleRows(cc*ndim, ndim) /= m_tau[cc];

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
Eigen::VectorXi ExpMax::classify(const Ref<const MatrixXd> samples)
{
	Eigen::VectorXi out(samples.rows());
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
size_t ExpMax::classify(const Ref<const MatrixXd> samples, Ref<VectorXi> classes)
{
	if(ndim != samples.cols()){
		throw RUNTIME_ERROR("Columns in sample vector must match number of "
				"dimensions, but do not!");
	}
	if(classes.rows() != samples.rows()) {
		throw RUNTIME_ERROR("Number of rows does in samples matrix should "
				"match # of rows in samples, but doesn't");
	}
	MatrixXd prob(samples.rows(), m_k);
	expectation(samples, prob);

	// Classify Based on Maximum Probability
	size_t change = 0;
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
 * @brief Given a matrix of samples (Samples x Dims, sample on each row),
 * apply the classifier to each sample and return a vector of the classes.
 *
 * @param samples Set of samples, 1 per row
 * @param Class probability of each sample, for each of the potential
 * distributions
 * @param classes input/output samples. Returned value indicates number that
 * changed.
 *
 * @return Number of classes that changed
 */
double ExpMax::expectation(const Ref<const MatrixXd> samples, Ref<MatrixXd> prob)
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
	prob.resize(samples.rows(), m_k);

	static std::default_random_engine rng;

	Eigen::FullPivHouseholderQR<MatrixXd> qr(ndim, ndim);
	VectorXd x(ndim);
	MatrixXd Cinv(ndim, ndim);
	double det = 0;
	double newll = 0;
	vector<int64_t> zero_tau;
	zero_tau.reserve(m_k);

	//compute Cholesky decomp, then determinant and inverse covariance matrix
	for(int cc = 0; cc < m_k; cc++) {
		DBG2(cerr<<"Covariance:\n"<<m_cov.block(cc*ndim, 0, ndim, ndim)<<endl);
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

			//log likelihood =
			//log(tau) - log(sigma)/2 - (x-mu)^Tsigma^-1(x-mu) - dlog(2pi)/2
			double llike = cval - .5*(x.dot(Cinv*x));

			if(std::isinf(llike) || std::isnan(llike))
				llike =  -INFINITY;
			else
				newll += llike;
			prob(pp, cc) = exp(llike);
		}
	}

	for(int pp = 0; pp < samples.rows(); pp++)
		prob.row(pp) /= prob.row(pp).sum();

	double max = -INFINITY;
	int max_class = -1;
	double RANDFACTOR = 10;
	double reassigned = 0;
	//place every point in its most probable group
	std::uniform_real_distribution<double> randf(0, 1);
	if(zero_tau.size() > 0) {
		DBG1(cerr<<"Zero Tau, Randomly Assigning Based on Probabilities"<<endl);
		for(int pp = 0 ; pp < samples.rows(); pp++) {
			max = -INFINITY;
			max_class = -1;
			for(int cc = 0 ; cc < m_k; cc++) {
				if(prob(pp, cc) > max) {
					max = prob(pp,cc);
					max_class = cc;
				}
			}

			double p = pow(1-prob(pp, max_class),RANDFACTOR);
			bool reassign = randf(rng) < p;
			if(reassign) {
				reassigned++;

				// Randomly Set Probabilities
				for(size_t cc=0; cc<m_k; cc++)
					prob(pp, cc) = randf(rng);

				// Reset Class Based on New Probabilities
				max = -INFINITY;
				max_class = -1;
				for(int cc = 0 ; cc < m_k; cc++) {
					if(prob(pp, cc) > max) {
						max = prob(pp,cc);
						max_class = cc;
					}
				}

			}
		}
		DBG1(cerr<<"Reassigned: "<<100*reassigned/samples.rows()<<"%"<< endl);
	}

	swap(newll, m_ll);
	return fabs(newll - m_ll);
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
int ExpMax::update(const Ref<const MatrixXd> samples, bool reinit)
{
	// each row is a sample, each column represents a class prob
	Eigen::MatrixXd classprobs(samples.rows(), m_k);
	Eigen::VectorXi classes(samples.rows());
	m_ll = std::numeric_limits<double>::lowest();

	// initialize with approximate k-means
	if(reinit || !m_valid) {
		approxKMeans(samples, m_k, classes);
		DBG1(cerr<<"Updating Mean/Cov/Tau"<<endl);
		// Set Probabilities from Classes)
		classprobs.setZero();
		for(size_t rr=0; rr<samples.rows(); rr++)
			classprobs(rr, classes[rr]) = 1;

		updateMeanCovTau(samples, classprobs);
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
	double change = SIZE_MAX;
	int ii = 0;
	DBG1(auto c = clock());
	for(ii=0; ii != maxit && change > 1; ++ii, ++maxit) {
		DBG1(c = clock());
		change = expectation(samples, classprobs);
		DBG1(c = clock() - c);
		DBG1(cerr << "Classify Time: " << c << endl);
		DBG1(cerr << "LL="<<m_ll<<" dLL="<<change<<endl);

		DBG1(c = clock());
		updateMeanCovTau(samples, classprobs);
		DBG1(c = clock() -c );
		DBG1(cerr << "Mean/Cov Time: " << c << endl);

#ifndef NDEBUG
		cout << "==========================================" << endl;
		cout << "Changed Prob: " << change << endl;
		cout << "Current Distributions: " << endl;
		for(size_t cc=0; cc<m_k; cc++ ) {
			cout << "Cluster " << cc << ", prob: " << m_tau[cc] << endl;
			cout << "Mean:\n" << m_mu.row(cc) << endl;
			cout << "Covariance:\n" <<
				m_cov.middleRows(cc*ndim, ndim) << endl << endl;
		}
		cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << endl;
#endif
		cerr<<"iter: "<<ii<<", logl: "<<m_ll<<", dlogl: "<<change<<endl;
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
//VectorXd leastSquares(const Ref<const MatrixXd> X, const Ref<const VectorXd> y)
//{
//	assert(X.rows() == y.rows());
//	Eigen::JacobiSVD<MatrixXd> svd;
//	svd.compute(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
//	return svd.solve(y);
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
//VectorXd IRLS(const Ref<const MatrixXd> X, const Ref<const VectorXd> y, VectorXd& w)
//{
//	assert(X.rows() == y.rows());
//	VectorXd beta(X.cols());
//	MatrixXd wX = w.asDiagonal()*X;
//	VectorXd wy = w.asDiagonal()*y;
//	VectorXd err(y.rows());
////	for() {
//		//update beta
//		beta = wX.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(wy);
//		err = (X*beta-y);
//
////		// update weights;
////		for(size_t ii=0; ii<w.size(); ii++) {
////			w[ii] = pow(
////		}
// //	}
//	return beta;
//}

int sign(double v)
{
	return v < 0 ? -1 : (v > 0 ? 1 : 0);
}

/**
 * @brief Performs LASSO regression using the 'activeShooting' algorithm of
 *
 * Peng, J., Wang, P., Zhou, N., & Zhu, J. (2009). Partial Correlation
 * Estimation by Joint Sparse Regression Models. Journal of the American
 * Statistical Association, 104(486), 735–746. doi:10.1198/jasa.2009.0126
 *
 * Essentially solves the equation:
 *
 * y = X * beta
 *
 * where beta is mostly 0's
 *
 * @param X Design matrix
 * @param y Measured value
 * @param gamma Weight of regularization (larger values forces sparser model)
 *
 * @return Beta vector
 */
VectorXd activeShootingRegr(const Ref<const MatrixXd> X, const Ref<const VectorXd> y, double gamma)
{
	double THRESH = 0.1;
//	// Start with least squares
//	JacobiSVD svd(X, ComputeThinV|ComputeThinU);
//	VectorXd beta = svd.solve(y);
	vector<bool> active(X.rows(), false);
	VectorXd beta(X.cols());
	VectorXd Xnorm(X.cols());
	VectorXd sigma(y.rows());
	for(size_t jj=0; jj<X.cols(); jj++)
		Xnorm[jj] = X.col(jj).squaredNorm();

	// Initialize
	for(size_t jj=0; jj<X.cols(); jj++) {
		double ytxj = y.dot(X.col(jj));
		if(ytxj - gamma > 0)
			beta[jj] = sign(ytxj)*(fabs(ytxj)-gamma)/Xnorm[jj];
		else
			beta[jj] = 0;
	}


	double dbeta1 = fabs(THRESH)*1.1;
	while(dbeta1 > THRESH) {
		dbeta1 = 0;

		// Determine Active Set
		for(size_t jj=0; jj<X.cols(); jj++) {
			if(beta[jj] != 0)
				active[jj] = true;
		}

		// Update Active Set until convergence
		double dbeta2 = fabs(THRESH)*1.1;
		while(dbeta2 > THRESH) {
			dbeta2 = 0;

			for(size_t jj=0; jj<X.cols(); jj++) {
				// Update Active Set
				if(active[jj]) {
					double prev = beta[jj];
					double v = (y-X*beta).dot(X.col(jj))/Xnorm[jj] + beta[jj];

					if(fabs(v) > gamma/Xnorm[jj])
						beta[jj] = sign(v)*(fabs(v)-gamma/Xnorm[jj]);
					else
						beta[jj] = 0;

					// to determine convergence
					dbeta2 += fabs(prev-beta[jj]);
				}
			}
			cerr << "dBeta2: " << dbeta2 << endl;
		}

		// Update All
		for(size_t jj=0; jj<X.cols(); jj++) {
			double prev = beta[jj];
			double v = (y-X*beta).dot(X.col(jj))/Xnorm[jj] + beta[jj];
			if(fabs(v) > gamma/Xnorm[jj])
				beta[jj] = sign(v)*(fabs(v)-gamma/Xnorm[jj]);
			else
				beta[jj] = 0;

			// to determine convergence
			dbeta1 += fabs(prev-beta[jj]);
		}
		cerr << "dBeta1: " << dbeta1 << endl;
	}
	return beta;
}

/**
 * @brief Performs LASSO regression using the 'shooting' algorithm of
 *
 * Fu, W. J. (1998). Penalized Regressions: The Bridge versus the Lasso.
 * Journal of Computational and Graphical Statistics, 7(3), 397.
 * doi:10.2307/1390712
 *
 * Essentially solves the equation:
 *
 * y = X * beta
 *
 * where beta is mostly 0's
 *
 * @param X Design matrix
 * @param y Measured value
 * @param gamma Weight of regularization (larger values forces sparser model)
 *
 * @return Beta vector
 */
VectorXd shootingRegr(const Ref<const MatrixXd> X, const Ref<const VectorXd> y, double gamma)
{
	double THRESH = 0.1;
//	// Start with least squares
//	JacobiSVD svd(X, ComputeThinV|ComputeThinU);
//	VectorXd beta = svd.solve(y);
	VectorXd beta(X.cols());
	VectorXd Xnorm(X.cols());
	VectorXd sigma(y.rows());
	for(size_t jj=0; jj<X.cols(); jj++)
		Xnorm[jj] = X.col(jj).squaredNorm();

	// Initialize
	for(size_t jj=0; jj<X.cols(); jj++) {
		double ytxj = y.dot(X.col(jj));
		if(fabs(ytxj)-gamma > 0)
			beta[jj] = sign(ytxj)*(fabs(ytxj)-gamma)/Xnorm[jj];
		else
			beta[jj] = 0;
	}

	double dbeta = fabs(THRESH)*1.1;
	while(dbeta > THRESH) {
		dbeta = 0;
		for(size_t jj=0; jj<X.cols(); jj++) {
			double prev = beta[jj];
			double v = (y-X*beta).dot(X.col(jj))/Xnorm[jj] + beta[jj];
			if(fabs(v) > gamma/Xnorm[jj])
				beta[jj] = sign(v)*(fabs(v)-gamma/Xnorm[jj]);
			else
				beta[jj] = 0;

			// to determine convergence
			dbeta += fabs(prev-beta[jj]);
		}
		cerr << "dBeta: " << dbeta << endl;
	}

	return beta;
}

/**
 * @brief Computes the Principal Components of input matrix X using the
 * covariance matrix.
 *
 * Outputs projections in the columns of a matrix.
 *
 * @param cov 	Covariance matrix of XtX
 * @param varth Variance threshold. Don't include dimensions after this percent
 * of the variance has been explained.
 *
 * @return 		RxP matrix, where P is the number of principal components
 */
MatrixXd pcacov(const Ref<const MatrixXd> cov, double varth)
{
	assert(cov.rows() == cov.cols());

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
			out(rr, jj) = solver.eigenvectors()(rr, ii);
		}
	}

#ifndef NDEBUG
	std::cout  << "  Done" << std::endl;
#endif

	return out;
}

/**
 * @brief
 *
 * @param A M x N
 * @param subsize Columns in projection matrix,
 * @param poweriters Number of power iterations to perform
 * @param U Output Left Singular Vectors
 * @param E OUtput Singular Values
 * @param V Output Right Singular Vectors
 */
void randomizePowerIterationSVD(const Ref<const MatrixXd> A,
		double tol, size_t startrank, size_t maxrank, size_t poweriters,
		MatrixXd& U, VectorXd& E, MatrixXd& V)
{
	// Algorithm 4.4
	MatrixXd Yc;
	MatrixXd Yhc;
	MatrixXd Qtmp;
	MatrixXd Qhat;
	MatrixXd Q;
	MatrixXd Qc;
	MatrixXd Omega;
	VectorXd norms;

	size_t curank = startrank;
	do {
		size_t nextsize = min(curank, A.rows()-curank);
		Omega.resize(A.cols(), nextsize);
		npl::fillGaussian<MatrixXd>(Omega);
		Yc = A*Omega;

		Eigen::HouseholderQR<MatrixXd> qr(Yc);
		Qtmp = qr.householderQ()*MatrixXd::Identity(A.rows(), nextsize);
		Eigen::HouseholderQR<MatrixXd> qrh;
		for(size_t ii=0; ii<poweriters; ii++) {
			Yhc = A.transpose()*Qtmp;
			qrh.compute(Yhc);
			Qhat = qrh.householderQ()*MatrixXd::Identity(A.cols(), nextsize);
			Yc = A*Qhat;
			qr.compute(Yc);
			Qtmp = qr.householderQ()*MatrixXd::Identity(A.rows(), nextsize);
		}

		/*
		 * Orthogonalize new basis with the current basis (Q) and then append
		 */
		if(Q.rows() > 0) {
			// Orthogonalize the additional Q vectors Q with respect to the
			// current Q vectors
			Qc = Qtmp - Q*(Q.transpose()*Qtmp);

			// After orthogonalizing wrt to Q, reorthogonalize wrt each other
			norms.resize(Qc.cols());
			for(size_t cc=0; cc<Qc.cols(); cc++) {
				for(size_t jj=0; jj<cc; jj++)
					Qc.col(cc) -= Qc.col(jj).dot(Qc.col(cc))*Qc.col(jj)/(norms[jj]*norms[jj]);
				norms[cc] = Qc.col(cc).norm();
			}

			// If the matrix is essentially covered by existing space, quit
			size_t keep = 0;
			for(size_t cc=0; cc<Qc.cols(); cc++) {
				if(norms[cc] > tol) {
					Qc.col(cc) /= norms[cc];
					keep++;
				}
			}
			if(keep == 0)
				break;

			// Append Orthogonalized Basis
			Q.conservativeResize(Qc.rows(), Q.cols()+keep);
			keep = 0;
			for(size_t cc=0; cc<Qc.cols(); cc++) {
				if(norms[cc] > tol) {
					Q.col(keep+curank) = Qc.col(cc);
					keep++;
				}
			}
		} else {
			Q = Qtmp;
		}

		curank = Q.cols();
	} while(curank < maxrank && curank < A.cols());

	// Form B = Q* x A
	MatrixXd B = Q.transpose()*A;
	Eigen::JacobiSVD<MatrixXd> smallsvd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
	U = Q*smallsvd.matrixU();
	E = smallsvd.singularValues();

	VectorXd Einv(E.rows());
	for(size_t ii=0; ii<E.rows(); ii++) {
		if(E[ii] > std::numeric_limits<double>::epsilon())
			Einv[ii] = 1./E[ii];
		else
			Einv[ii] = 0;
	}

	V = smallsvd.matrixV();
}

/**
 * @brief
 *
 * @param A M x N
 * @param subsize Columns in projection matrix,
 * @param poweriters Number of power iterations to perform
 * @param U Output Left Singular Vectors
 * @param E OUtput Singular Values
 * @param V Output Right Singular Vectors
 */
void randomizePowerIterationSVD(const Ref<const MatrixXd> A,
		size_t subsize, size_t poweriters, MatrixXd& U, VectorXd& E, MatrixXd& V)
{
	subsize = std::min(subsize, (size_t)A.rows());
	MatrixXd omega(A.cols(), subsize);
	npl::fillGaussian<MatrixXd>(omega);
	MatrixXd Y(A.rows(), subsize);
	MatrixXd Yhat(A.rows(), subsize);
	MatrixXd Q, Qhat;

	Y = A*omega;
	Eigen::HouseholderQR<MatrixXd> qr(Y);
	Q = qr.householderQ()*MatrixXd::Identity(A.rows(), subsize);
	Eigen::HouseholderQR<MatrixXd> qrh;
	for(size_t ii=0; ii<poweriters; ii++) {
		Yhat = A.transpose()*Q;
		qrh.compute(Yhat);
		Qhat = qrh.householderQ()*MatrixXd::Identity(A.cols(), subsize);
		Y = A*Qhat;
		qr.compute(Y);
		Q = qr.householderQ()*MatrixXd::Identity(A.rows(), subsize);
	}

	// Form B = Q* x A
	MatrixXd B = Q.transpose()*A;
	Eigen::JacobiSVD<MatrixXd> smallsvd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
	U = Q*smallsvd.matrixU();
	E = smallsvd.singularValues();

	VectorXd Einv(E.rows());
	for(size_t ii=0; ii<E.rows(); ii++) {
		if(E[ii] > std::numeric_limits<double>::epsilon())
			Einv[ii] = 1./E[ii];
		else
			Einv[ii] = 0;
	}

	// A = U E V*, A* = V E U*, A*U = VE, V = A*UE^-1
	V = smallsvd.matrixV();
}


/**
 * @brief Computes the Principal Components of input matrix X
 *
 * Outputs reduced dimension (fewer cols) in output. Note that prior to this,
 * the columns of X should be 0 mean otherwise the first component will
 * be the mean
 *
 * @param X 	RxC matrix where each column row is a sample, each column a
 * dimension (or feature). The number of columns in the output
 * will be fewer because there will be fewer features
 * @param varth Variance threshold. This is the ratio (0-1) of variance to
 * include in the output. This is used to determine the dimensionality of the
 * output. If this is 1 then all variance will be included. If this < 1 and
 * odim > 0 then whichever gives a larger output dimension will be selected.
 * @param odim Threshold for output dimensions. If this is <= 0 then it is
 * ignored, if it is > 0 then max(dim(varth), odim) is used as the output
 * dimension.
 *
 * @return 		RxP matrix, where P is the number of principal components
 */
MatrixXd pca(const Ref<const MatrixXd> X, double varth, int odim)
{
	double totalv = 0; // total variance
	int outdim = 0;

//#ifndef NDEBUG
	std::cout << "Computing SVD" << std::endl;
//#endif //DEBUG
	Eigen::JacobiSVD<MatrixXd> svd(X, Eigen::ComputeThinU);
//#ifndef NDEBUG
	std::cout << "Done" << std::endl;
//#endif //DEBUG

	const VectorXd& W = svd.singularValues();
	const MatrixXd& U = svd.matrixU();
	//only keep dimensions with variance passing the threshold
	totalv = W.sum();

	double sum = 0;
	for(outdim = 0; outdim < W.rows() && sum < totalv*varth; outdim++)
		sum += W[outdim];
	std::cout << totalv << endl;
	std::cout << varth*totalv << endl;
	std::cout<<W.transpose()<<endl;
//#ifndef NDEBUG
	std::cout << "Output Dimensions: " << outdim
		<< "\nCreating Reduced MatrixXd..." << std::endl;
//#endif //DEBUG

	// Merge the two dimension estimation results
	outdim = std::max(odim, outdim);

	// Return whitened signal
	MatrixXd Xr = U.leftCols(outdim)*W.head(outdim).asDiagonal();
//#ifndef NDEBUG
	std::cout  << "  Done" << std::endl;
//#endif

	return Xr;
}

/*****************************************************************************
 * Metric Functions for ICA
 ****************************************************************************/

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

/*****************************************************************************
 * ICA
 ****************************************************************************/

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
 * @param Xin 	RxC matrix where each row is a sample, each column a
 * dimension (or feature). The number of columns in the output
 * will be fewer because there will be fewer features.
 * Columns should be zero-mean and uncorrelated with one another.
 *
 * @return 		RxP matrix, where P is the number of independent components
 */
MatrixXd ica(const Ref<const MatrixXd> Xin)
{

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

}

