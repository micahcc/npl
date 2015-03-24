/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file shooting_test.cpp Test shooting algorithm for L1-regularized
 * regression. This is essentially the lasso estimator of regression.
 *
 *****************************************************************************/

#include <string>
#include <stdexcept>
#include <random>
#include <memory>

#include "statistics.h"
#include "basic_plot.h"

using std::string;
using std::shared_ptr;

using namespace Eigen;
using namespace npl;
using namespace std;

void plotMat(std::string filename, const MatrixXd& in)
{
	Plotter plot;
	std::vector<double> tmp(in.rows());
	for(size_t ii=0; ii<in.cols(); ii++) {
		for(size_t rr=0; rr<in.rows(); rr++)
			tmp[rr] = in(rr, ii);
		plot.addArray(tmp.size(), tmp.data());
	}

	plot.write(filename);
}

int main(int argc, char** argv)
{
	double lambda = .1;
	if(argc == 2) {
		lambda = atof(argv[1]);
	}

	std::default_random_engine rng;
	std::uniform_real_distribution<double> unifdist(-1,1);
	std::normal_distribution<> gaussdist(0,1);

	/*
	 * Create the Test Data
	 */

	size_t ntimes = 100;
	size_t ndims = 3;
	MatrixXd X(ntimes, ndims);
	VectorXd y(ntimes);
	VectorXd beta(ndims);

	// create square wave
	bool high = false;
	for(size_t ii=0; ii<ntimes; ii++) {
		if(ii % 20 == 0)
			high = !high;
		X(ii,0) = high;
	}
	// create sin wave
	for(size_t ii=0; ii<ntimes; ii++)
		X(ii,1) = sin(ii/10.);

	// create gaussian
	for(size_t ii=0; ii<ntimes; ii++)
		X(ii,2) = gaussdist(rng);

	plotMat("before_mix.svg", X);

	/*
	 * Mix The Data
	 */

	// create random beta
	for(size_t ii=0; ii<ndims; ii++)
		beta[ii] = unifdist(rng);

	y = X*beta;

	// add noise
	for(size_t ii=0; ii<ntimes; ii++)
		y[ii] += 0.2*gaussdist(rng);

	plotMat("after_mix.svg", X);


	/*
	 * Perform Regression
	 */
	VectorXd ebeta = shootingRegr(X, y, lambda);

	cerr << "True Beta: " << beta.transpose() << endl;
	cerr << "Est. Beta: " << ebeta.transpose() << endl;
	if((ebeta-beta).squaredNorm() > 0.01) {
		cerr << "Problem with shooting algorithm." << endl;
		return -1;
	}

	ebeta = activeShootingRegr(X, y, lambda);
	cerr << "(Active) Est. Beta: " << ebeta.transpose() << endl;
	if((ebeta-beta).squaredNorm() > 0.01) {
		cerr << "Problem with shooting algorithm." << endl;
		return -1;
	}

	return 0;
}


