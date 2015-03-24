/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file glm_test1.cpp Tests large scale GLM
 *
 *****************************************************************************/

#include <version.h>
#include <string>
#include <stdexcept>
#include <iostream>
#include <random>

#include <random>
#include <Eigen/Dense>
#include "statistics.h"
#include "utility.h"
#include "basic_plot.h"
#include "macros.h"

using std::string;
using Eigen::MatrixXd;

using namespace std;
using namespace npl;

int main()
{
	VectorXd beta(4);
	beta << 1, -2, 3, -4;

	double noise_sd = 1;
	std::default_random_engine rng;
	std::normal_distribution<double> rdist(0, noise_sd);

	size_t tlen = 1024;
	double se = noise_sd/sqrt(tlen);
	MatrixXd X(tlen, 4);
	for(size_t rr=0; rr<X.rows(); rr++ ) {
		for(size_t cc=0; cc<X.cols(); cc++)
			X(rr,cc) = cos(M_PI*rr/100*(cc+1));
	}

	VectorXd Xsd(X.cols());
	for(size_t ii=0; ii<X.cols(); ii++)
		Xsd[ii] = sqrt(sample_var(X.col(ii)));

	VectorXd y = X*beta;
	for(size_t ii=0; ii<y.rows(); ii++) {
		double e = rdist(rng);
		y[ii] += e;
	}

	// Cache Reused Vectors
	MatrixXd Xinv = pseudoInverse(X);
	VectorXd covInv = pseudoInverse(X.transpose()*X).diagonal();

	const double MAX_T = 30;
	const double STEP_T = 0.1;
	StudentsT stud_dist(X.rows()-1, STEP_T, MAX_T);

	RegrResult ret;
	regress(&ret, y, X, covInv, Xinv, stud_dist);

	Plotter plt;
	plt.addArray(y.rows(), y.data());
	plt.addArray(X.rows(), X.col(0).data());
	plt.addArray(X.rows(), X.col(1).data());
	plt.addArray(X.rows(), X.col(2).data());
	plt.addArray(X.rows(), X.col(3).data());
	plt.write("y_x.svg");

	Plotter plt2;
	plt2.addArray(y.rows(), y.data());
	plt2.addArray(ret.yhat.rows(), ret.yhat.data());
	plt2.write("y_vs_yht.svg");
	cerr << "Beta: Est" << ret.bhat.transpose() << " vs " << beta.transpose()
		<< endl;
	cerr << "Standard Errors: " << ret.std_err.transpose() << endl;
	cerr << "T-Value: " << ret.t.transpose() <<endl;
	cerr << "P-value: " << ret.p.transpose() << endl;
	cerr << "DOF: "<< ret.dof << endl;
	cerr << "Standard Error of Beta: "<<se<< endl;

	for(size_t b=0; b<beta.rows(); b++) {
		if(fabs(beta[b] - ret.bhat[b]) > 2*se) {
			cerr << "Too large a difference in beta ("<<
				beta[b] << " vs " << ret.bhat[b]<<endl;
			return -1;
		}
	}

	return 0;
}



