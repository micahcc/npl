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
 * @file statistics.h Tools for analyzing data, including PCA, ICA and
 * general linear modeling.
 *
 *****************************************************************************/

#ifndef STATISTICS_H
#define STATISTICS_H

#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/Dense>
#include "npltypes.h"

namespace npl {

/** \defgroup StatFunctions Statistical Functions
 *
 * @{
 */

/**
 * @brief Computes the Principal Components of input matrix X
 *
 * Outputs reduced dimension (fewer cols) in output. Note that prior to this,
 * the columns of X should be 0 mean otherwise the first component will
 * be the mean
 *
 * @param X 	RxC matrix where each column row is a sample, each column a
 *              dimension (or feature). The number of columns in the output
 *              will be fewer because there will be fewer features
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
MatrixXd pca(const Eigen::MatrixXd& X, double varth = 1, int odim = -1);

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
MatrixXd ica(const Eigen::MatrixXd& X, double varth = 1);

/**
 * @brief Computes the Ordinary Least Square predictors, beta for 
 *
 * \f$ y = \hat \beta X \f$
 *
 * Returning beta
 *
 * @return beta value, the weights of the independent variables to produce the
 * least squares predictor for y.
 */
VectorXd regress(const MatrixXd& y, const MatrixXd& X);

struct RegrResult
{
    /**
     * @brief Predicted y values, based on estimate of Beta
     */
    VectorXd yhat;
    
    /**
     * @brief Estimated Beta
     */
    VectorXd bhat;

    /**
     * @brief Sum of square of the residuals
     */
    double ssres;

    /**
     * @brief Total sum of square of y values
     */
    double sstot;

    /**
     * @brief Coefficient of determination (Rsqr)
     */
    double rsqr;
   
    /**
     * @brief Coefficient of determination, corrected for the number of
     * regressors. 
     */
    double adj_rsqr;

    /**
     * @brief Standard errors for each of the regressors
     */
    VectorXd std_err;

    /**
     * @brief Degrees of freedom in the regression
     */
    double dof;

    /**
     * @brief Students t score of each of the regressors
     */
    VectorXd t;
    
    /**
     * @brief Significance of each of the regressors. 
     */
    VectorXd p;
};

/**
 * @brief Student's T-distribution. A cache of the Probability Density Function and 
 * cumulative density function is created using the analytical PDF.
 */
class StudentsT
{
public:
    /**
     * @brief Defualt constructor takes the degrees of freedom (Nu), step size
     * for numerical CDF computation and Maximum T to sum to for numerical CDF
     *
     * @param dof Degrees of freedom (shape parameter)
     * @param dt step size in x or t to take 
     * @param tmax Maximum t to consider
     */
    StudentsT(int dof = 2, double dt = 0.1, double tmax = 20);

    /**
     * @brief Change the degress of freedom, update cache
     *
     * @param dof Shape parameter, higher values make the distribution more
     * gaussian
     */
    void setDOF(double dof);
    
    /**
     * @brief Step in t to use for computing the CDF, smaller means more
     * precision although in reality the distribution is quite smooth, and
     * linear interpolation should be very good.
     *
     * @param dt Step size for numerical integration
     */
    void setStepT(double dt);

    /**
     * @brief Set the maximum t for numerical integration, and recompute the
     * cdf/pdf caches.
     *
     * @param tmax CDF and PDF are stored as arrays, this is the maximum
     * acceptable t value. Its RARE (like 10^-10 rare) to have a value higher
     * than 20.
     */
    void setMaxT(double tmax);

    /**
     * @brief Get the cumulative probability at some t value.
     *
     * @param t T (or x, distance from center) value to query
     *
     * @return Cumulative probability (probability value of value < t)
     */
    double cumulative(double t) const;

    /**
     * @brief Get the cumulative probability at some t value.
     *
     * @param t T (or x, distance from center) value to query
     *
     * @return Cumulative probability (probability value of value < t)
     */
    double cdf(double t) const { return cumulative(t); };


    /**
     * @brief Get the probability density at some t value.
     *
     * @param t T value to query
     *
     * @return Probability density at t. 
     */
    double density(double t) const;

    /**
     * @brief Get the probability density at some t value.
     *
     * @param t T value to query
     *
     * @return Probability density at t. 
     */
    double pdf(double t) const { return density(t); };

private:
    void init();

    double m_dt;
    double m_tmax;
    int m_dof;
    std::vector<double> m_cdf;
    std::vector<double> m_pdf;
    std::vector<double> m_tvals;
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
 * @param distrib Pre-computed students' T distribution. Example:
 * auto v = students_t_cdf(X.rows()-1, .1, 1000);
 *
 * @return Struct with Regression Results. 
 */
RegrResult regress(const VectorXd& y, const MatrixXd& X, const MatrixXd& covInv,
        const MatrixXd& Xinv, const StudentsT& distrib);

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
RegrResult regress(const VectorXd& y, const MatrixXd& X);

/**
 * @brief Computes the pseudoinverse of the input matrix
 *
 * \f$ P = UE^-1V^* \f$
 *
 * @return Psueodinverse 
 */
MatrixXd pseudoInverse(const MatrixXd& X);

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
VectorXd shootingRegr(const MatrixXd& X, const VectorXd& y, double gamma);

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
VectorXd activeShootingRegr(const MatrixXd& X, const VectorXd& y, double gamma);

/*
 * \defgroup Clustering algorithms
 * @{
 */

/**
 * @brief Approximates k-means using the algorithm of:
 *
 * 'Fast Approximate k-Means via Cluster Closures' by Wang et al
 *
 * This is not really meant to be used by outside users, but as an initializer
 *
 * @param samples Matrix of samples, one sample per row, 
 * @param nclass Number of classes to break samples up into
 * @param means Estimated mid-points/means
 */
void approxKMeans(const MatrixXd& samples, size_t nclass, MatrixXd& means);

/**
 * @brief Base class for all ND classifiers.
 */
class Classifier
{
public:
    /**
     * @brief Initializes the classifier
     *
     * @param rank Number of dimensions of samples
     */
    Classifier(size_t rank) : ndim(rank), maxit(-1), m_valid(false) {};

    /**
     * @brief Given a matrix of samples (Samples x Dims, sample on each row),
     * apply the classifier to each sample and return a vector of the classes.
     *
     * @param samples Set of samples, 1 per row
     *
     * @return Vector of classes, rows match up with input sample rows
     */
    virtual 
    Eigen::VectorXi classify(const MatrixXd& samples) = 0;

    /**
     * @brief Given a matrix of samples (Samples x Dims, sample on each row),
     * apply the classifier to each sample and return a vector of the classes.
     *
     * @param samples Set of samples, 1 per row
     * @param oclass Output classes. This vector will be resized to have the
     * same number of rows as samples matrix.
     *
     * @return the number of changed classifications
     */
    virtual 
    size_t classify(const MatrixXd& samples, Eigen::VectorXi& oclass) = 0;

    /**
     * @brief Updates the classifier with new samples, if reinit is true then
     * no prior information will be used. If reinit is false then any existing
     * information will be left intact. In Kmeans that would mean that the
     * means will be left at their previous state.
     * 
     * @param samples Samples, S x D matrix with S is the number of samples and
     * D is the dimensionality. This must match the internal dimension count.
     * @param reinit whether to reinitialize the  classifier before updating
     *
     * return -1 if maximum number of iterations hit, 0 otherwise (converged)
     */
    virtual
    int update(const MatrixXd& samples, bool reinit = false) = 0;

    /**
     * @brief Alias for updateClasses with reinit = true. This will perform 
     * a classification scheme on all the input samples.
     *
     * @param samples Samples, S x D matrix with S is the number of samples and
     * D is the dimensionality. This must match the internal dimension count.
     */
    void compute(const MatrixXd& samples) { update(samples, true); };

    /**
     * @brief Number of dimensions, must be set at construction. This is the
     * number of columns in input samples.
     */
    const int ndim;
    
    /**
     * @brief Maximum number of iterations. Set below 0 for infinite.
     */
    int maxit;
protected:
    /**
     * @brief Whether the classifier has been initialized yet
     */
    bool m_valid;

};

/**
 * @brief K-means classifier.
 */
class KMeans : public Classifier
{
public:
    /**
     * @brief Constructor for k-means class
     *
     * @param rank Number of dimensions in input samples.
     * @param k Number of groups to classify samples into
     */
    KMeans(size_t rank, size_t k = 2);

    /**
     * @brief Update the number of groups. Note that this invalidates any
     * current information
     *
     * @param ngroups Number of groups to classify
     */
    void setk(size_t ngroups);
    
    /**
     * @brief Sets the mean matrix. Each row of the matrix is a ND-mean, where
     * N is the number of columns.
     *
     * @param newmeans Matrix with new mean
     */
    void updateMeans(const MatrixXd& newmeans);

    /**
     * @brief Updates the mean coordinates by providing a set of labeled samples.
     *
     * @param samples Matrix of samples, where each row is an ND-sample.
     * @param classes Classes, where rows match the rows of the samples matrix.
     * Classes should be integers 0 <= c < K where K is the number of classes
     * in this.
     */
    void updateMeans(const MatrixXd samples, const Eigen::VectorXi classes);

    /**
     * @brief Given a matrix of samples (Samples x Dims, sample on each row),
     * apply the classifier to each sample and return a vector of the classes.
     *
     * @param samples Set of samples, 1 per row
     *
     * @return Vector of classes, rows match up with input sample rows
     */
    Eigen::VectorXi classify(const MatrixXd& samples);

    /**
     * @brief Given a matrix of samples (Samples x Dims, sample on each row),
     * apply the classifier to each sample and return a vector of the classes.
     *
     * @param samples Set of samples, 1 per row
     * @param oclass Output classes. This vector will be resized to have the
     * same number of rows as samples matrix.
     */
    size_t classify(const MatrixXd& samples, Eigen::VectorXi& oclass);

    /**
     * @brief Updates the classifier with new samples, if reinit is true then
     * no prior information will be used. If reinit is false then any existing
     * information will be left intact. In Kmeans that would mean that the
     * means will be left at their previous state.
     * 
     * @param samples Samples, S x D matrix with S is the number of samples and
     * D is the dimensionality. This must match the internal dimension count.
     * @param reinit whether to reinitialize the  classifier before updating
     *
     * @return -1 if maximum number of iterations hit, 0 otherwise
     */
    int update(const MatrixXd& samples, bool reinit = false);

    /**
     * @brief Returns the current mean matrix
     *
     * @return The current mean matrix
     */
    const MatrixXd& getMeans() { return m_mu; };

private:
    /**
     * @brief Number of groups to classify samples into
     */
    size_t m_k;

    /**
     * @brief The means of each group, K x D, where each row is an N-D mean.
     */
    MatrixXd m_mu;
};

/**
 * @brief K-means classifier.
 */
class ExpMax : public Classifier
{
public:
    /**
     * @brief Constructor for k-means class
     *
     * @param rank Number of dimensions in input samples.
     * @param k Number of groups to classify samples into
     */
    ExpMax(size_t rank, size_t k = 2);

    /**
     * @brief Update the number of groups. Note that this invalidates any
     * current information
     *
     * @param ngroups Number of groups to classify
     */
    void setk(size_t ngroups);
    
    /**
     * @brief Sets the mean matrix. Each row of the matrix is a ND-mean, where
     * N is the number of columns.
     *
     * @param newmeans Matrix with new mean, means are stacked so that each row
     * represents a group mean 
     * @param newmeans Matrix with new coviance matrices. Covariance matrices
     * are stacked so that row ndim*k gets the first element of the k'th
     * covance matrix.
     * @param newcovs the new covariance matrices to set in the classifier
     * @param tau the prior probaibilities of each of the mixture gaussians
     */
    void updateMeanCovTau(const MatrixXd& newmeans, const MatrixXd& newcovs, 
            const VectorXd& tau);

    /**
     * @brief Updates the mean and covariance matrices by using a set of
     * classified points.
     *
     * @param samples Matrix of samples, where each row is an ND-sample.
     * @param classes Classes, where rows match the rows of the samples matrix.
     * Classes should be integers 0 <= c < K where K is the number of classes
     * in this.
     */
    void updateMeanCovTau(const MatrixXd samples, const Eigen::VectorXi classes);

    /**
     * @brief Given a matrix of samples (Samples x Dims, sample on each row),
     * apply the classifier to each sample and return a vector of the classes.
     *
     * @param samples Set of samples, 1 per row
     *
     * @return Vector of classes, rows match up with input sample rows
     */
    Eigen::VectorXi classify(const MatrixXd& samples);

    /**
     * @brief Given a matrix of samples (Samples x Dims, sample on each row),
     * apply the classifier to each sample and return a vector of the classes.
     *
     * @param samples Set of samples, 1 per row
     * @param oclass Output classes. This vector will be resized to have the
     * same number of rows as samples matrix.
     */
    size_t classify(const MatrixXd& samples, Eigen::VectorXi& oclass);

    /**
     * @brief Updates the classifier with new samples, if reinit is true then
     * no prior information will be used. If reinit is false then any existing
     * information will be left intact. In Kmeans that would mean that the
     * means will be left at their previous state.
     * 
     * @param samples Samples, S x D matrix with S is the number of samples and
     * D is the dimensionality. This must match the internal dimension count.
     * @param reinit whether to reinitialize the  classifier before updating
     *
     * return 0 if converged, -1 otherwise
     */
    int update(const MatrixXd& samples, bool reinit = false);


    /**
     * @brief Returns the current mean matrix
     *
     * @return The current mean matrix
     */
    const MatrixXd& getMeans() { return m_mu; };
    
    /**
     * @brief Returns the current mean matrix
     *
     * @return The current covariance matrix, with each covariance matrix
     * stacked on top of the next.
     */
    const MatrixXd& getCovs() { return m_cov; };
private:
    /**
     * @brief Number of groups to classify samples into
     */
    size_t m_k;

    /**
     * @brief The means of each group, K x D, where each row is an N-D mean.
     */
    MatrixXd m_mu;

    /**
     * @brief The covariance of each group, Covariances are stacked vertically,
     * with covariance c starting at (ndim*c, 0) and running to 
     * (ndim*(c+1)-1, ndim-1).
     */
    MatrixXd m_cov;

    /**
     * @brief Inverse of covariance matrix
     */
    MatrixXd m_covinv;

    /**
     * @brief Prior probabilities of each distribution based on percent of
     * points that lie within the group
     */
    VectorXd m_tau;
};

/**
 * @brief Algorithm of unsupervised learning (clustering) based on density.
 *
 * see "Clustering by fast search and find of density peaks"
 * by Rodriguez, a.  Laio, A.
 *
 * @param samples Samples, S x D matrix with S is the number of samples and
 * D is the dimensionality. This must match the internal dimension count.
 * @param thresh Threshold distance for density calculation
 * @param outthresh threshold for outlier, ratio of standard devation. Should
 * be > 2 because you want to be well outside the center of the distribution.
 * @param classes Output classes
 * @param brute whether ther use slower brute force method for density
 * calculation (for testing purposes only)
 *
 * return -1 if maximum number of iterations hit, 0 otherwise (converged)
 */
int fastSearchFindDP(const Eigen::MatrixXf& samples, double thresh, 
		double outthresh, Eigen::VectorXi& classes, bool brute = false);

/**
 * @brief Computes Density and Peak computation for Fast Search and Find of
 * Density Peaks algorithm.
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
int findDensityPeaks(const Eigen::MatrixXf& samples, double thresh,
		Eigen::VectorXf& rho, Eigen::VectorXf& delta,
		Eigen::VectorXi& parent);

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
int findDensityPeaks_brute(const Eigen::MatrixXf& samples, double thresh,
		Eigen::VectorXf& rho, Eigen::VectorXf& delta,
		Eigen::VectorXi& parent);

/** @} */

/** @} */

}

#endif // STATISTICS_H
