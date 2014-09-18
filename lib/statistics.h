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
 * Outputs reduced dimension (fewer cols) in output
 *
 * @param X 	RxC matrix where each column row is a sample, each column a
 *              dimension (or feature). The number of columns in the output
 *              will be fewer because there will be fewer features
 * @param varth Variance threshold. Don't include dimensions after this percent
 *              of the variance has been explained.
 *
 * @return 		RxP matrix, where P is the number of principal components
 */
MatrixXd pca(const Eigen::MatrixXd& X, double varth);

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
MatrixXd ica(const Eigen::MatrixXd& X, double varth);

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
        const MatrixXd& Xinv, std::vector<double>& student_cdf);

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
 * @brief Computes cdf at a particular number of degrees of freedom. 
 * Note, this only computes +t values, for negative values invert then use.
 *
 * @param nu
 * @param x
 *
 * @return 
 */
std::vector<double> students_t_cdf(double nu, double dt, double maxt);

/** @} */

}

#endif // STATISTICS_H
