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
 * @file regression_test.cpp Test of the linear regression function
 *
 *****************************************************************************/

#include "version.h"
#include "statistics.h"
#include "npltypes.h"

#include <iostream>
#include <iomanip>
#include <map>
#include <fstream>
#include <tclap/CmdLine.h>

#include <Eigen/Dense>
#include <Eigen/IterativeSolvers>

using namespace std;
using namespace npl;

TCLAP::SwitchArg a_verbose("v", "verbose", "Write out all the matrices");

std::random_device rd;
std::default_random_engine rng;

int testknown()
{
	MatrixXd X(50, 6);
	X << 1, -0.0398556, 0.0775251, -0.0778678, 1.19892, 0.32024,
		1, -1.107, -0.387552, -0.345707, 1.35331, 1.18566,
		1, -0.0430078, -0.382223, 0.0960757, -1.48062, 1.5028,
		1, -1.01532, 1.84608, -0.2589, 0.222783, 0.475416,
		1, -1.09167, 0.30284, -0.49209, 0.509864, -0.455196,
		1, -0.703596, -0.129298, -2.08837, 0.988401, -0.109849,
		1, -0.789734, 1.43774, -0.454285, -0.674062, -0.21377,
		1, -0.711533, -0.153145, 2.38769, -1.00557, 0.77667,
		1, -1.31482, 1.38907, 1.07463, -0.847792, -2.06152,
		1, -0.0142724, 0.140972, 2.49653, -0.219825, -0.420135,
		1, -1.82509, 1.55991, 0.011301, -0.78255, -0.881528,
		1, 0.4262, 0.413163, 1.66132, 1.66166, 1.89852,
		1, 0.901525, 1.23805, 0.334632, -0.394909, 0.0588878,
		1, -0.0979487, 0.586932, -0.857457, -0.918736, 1.63886,
		1, 0.550112, 0.48066, -0.2645, -0.583528, -0.0432294,
		1, -0.680693, 0.790393, -2.81175, 1.0205, 2.07448,
		1, 0.175807, 0.188655, 0.347521, 0.801807, -0.721153,
		1, 0.680218, 0.37796, -1.2223, -0.746245, -0.101923,
		1, -0.93934, 0.559617, -0.713003, -0.686399, 0.93301,
		1, -1.412, -1.36885, -0.939832, -0.526866, 0.736817,
		1, 0.078045, -0.875778, 0.618061, 1.5617, 0.353743,
		1, -1.8718, 0.261701, 0.391671, -1.54004, -0.478429,
		1, 0.229437, 2.12643, -0.382593, -0.428013, 1.37391,
		1, 0.950427, 1.36793, 0.377052, 0.42131, -1.06224,
		1, 0.597921, 3.24042, 1.12571, -0.416274, 0.934186,
		1, -0.374312, 0.63965, 0.185668, -0.360407, 1.08376,
		1, -2.40382, 0.21589, 0.982043, -0.0509035, -0.411754,
		1, 2.47833, 0.115278, -0.78931, -0.0380982, -0.800705,
		1, -0.23088, -1.08761, -0.0617401, 0.795635, -2.30449,
		1, 0.957238, 0.0402543, 0.25516, -0.587656, 1.10313,
		1, -0.645603, 1.70113, -0.0797718, 1.30369, 1.40809,
		1, -1.56345, -1.27784, -0.910611, -0.394413, 0.533196,
		1, 0.735667, -2.57526, -2.04432, -0.441258, -0.943554,
		1, 0.588167, -0.808995, -0.995099, 1.29167, 0.102834,
		1, -1.19686, 1.29809, -0.88304, -1.74892, 1.14107,
		1, -1.0582, 0.233933, -1.43596, 0.416598, 0.533344,
		1, -0.951429, 1.24673, -0.264601, -0.213223, -0.451737,
		1, 0.121366, -0.207973, 3.62884, -3.36018, 0.756232,
		1, 0.692537, 0.475636, 1.25365, 1.31411, -1.86587,
		1, 0.280593, 1.61257, 0.988025, 0.547012, 0.26319,
		1, -0.794652, -0.86568, 1.04715, -0.178839, -0.525866,
		1, -1.55099, -1.04571, 0.959366, -0.787641, 0.512221,
		1, -0.569629, -1.25275, -0.876406, -1.39311, 2.90733,
		1, 0.770197, 0.618294, -0.0832784, 0.210453, 2.55569,
		1, -0.387763, -0.164914, -0.449936, -0.0212143, 0.460344,
		1, 0.22718, 0.855221, -0.992183, 1.29425, 0.0130175,
		1, -1.16342, 0.535089, -0.473402, -1.03425, -0.673458,
		1, -0.0762733, 0.151017, 1.65391, -0.999451, 0.451698,
		1, 0.719566, -0.35554, -2.16048, -1.06215, -1.76856,
		1, -1.66, 0.755438, 1.37825, -0.0845074, 0.274361;

	VectorXd y(50);
	y << -0.86265, -2.4024, 2.94798, -2.44581, -3.67905, -3.44435, -2.45483,
		1.53931, -4.40622, -1.24535, -4.57873, 1.80432, -0.86215, 3.08747,
		0.70379, -1.54838, -2.55856, -0.908338, -0.312043, 0.140257, -1.20574,
		-1.68833, -1.50211, -2.52589, -3.38935, -0.238696, -2.31781, 0.561894,
		-2.70248, 3.40084, -3.07611, 1.18461, 2.16245, 0.800745, 1.34446,
		-1.51282, -3.92164, 5.31851, -5.71877, -2.68744, -1.4542, 0.775564,
		5.56723, 2.28578, -1.46779, -3.46944, -2.43747, 0.452261, 0.121147,
		-2.21401;

	if(a_verbose.isSet()) {
		cerr << "y = \n" << y.transpose() << endl;
		cerr << "X = \n" << X << endl;
	}

	// true trandard error
	VectorXd Cinv = pseudoInverse(X.transpose()*X).diagonal();
	MatrixXd Xinv = pseudoInverse(X);

	// need to compute the CDF for students_t_cdf
	const double MAX_T = 100;
	const double STEP_T = 0.01;
	StudentsT distrib(X.rows()-X.cols(), STEP_T, MAX_T);

	// Test Regular Regression
	RegrResult result;
	regress(&result, y, X, Cinv, Xinv, distrib);

//	Estimate Std. Error t value Pr(>|t|)
//	(Intercept)       0.1299   -4.66  3.0e-05
//	X1                0.1241    7.64  1.3e-09
//	X2                0.1137  -10.82  5.6e-14
//	X3                0.0983    1.22     0.23
//	X4                0.1203   -9.95  7.7e-13
//	X5                0.1053   12.29  8.1e-16
//	Multiple R-squared:  0.904,	Adjusted R-squared:  0.893

	VectorXd rbeta(6);
	VectorXd rt(6);
	VectorXd rp(6);
	rbeta << -0.6049, 0.9474, -1.2298, 0.1198, -1.1975, 1.2936;
	rt << -4.66, 7.64, -10.82, 1.22, -9.95, 12.29;
	rp << 3.0e-05, 1.3e-09, 5.6e-14, 0.23, 7.7e-13, 8.1e-16;
	double rRsqr = 0.904;
	double rAdjRsqr = 0.893;

	if(fabs(result.rsqr - rRsqr) > 0.001) {
		cerr<<"R-Squared Differs from that calculated from r"<<endl;
		cerr << "R:    " << rRsqr <<endl;
		cerr << "this: " << result.rsqr<<endl;
		return -1;
	} else if(fabs(result.adj_rsqr - rAdjRsqr) > 0.001) {
		cerr<<"Adjusted R-Squared Differs from that calculated from r"<<endl;
		cerr << "R:    " << rAdjRsqr <<endl;
		cerr << "this: " << result.adj_rsqr<<endl;
		return -1;
	} else if((rbeta - result.bhat).cwiseAbs().maxCoeff() > 0.001) {
		cerr<<"Beta Differs from that calculated from r"<<endl;
		cerr << "R:    " << rbeta.transpose()<<endl;
		cerr << "this: " << result.bhat.transpose()<<endl;
		return -1;
	} else if((rt - result.t).cwiseAbs().maxCoeff() > 0.01) {
		cerr<<"T Differs from that calculated from r"<<endl;
		cerr << "R:    " << rt.transpose()<<endl;
		cerr << "this: " << result.t.transpose()<<endl;
		return -1;
	}

	for(size_t cc=0; cc<rp.rows(); cc++) {
		if(rp[cc] > 1e-8) {
			if((rp[cc] - result.p[cc])/max(rp[cc],result.p[cc]) > 0.1) {
				cerr << "Mismatch of p values with R" << endl;
				cerr << "R:    " << rp.transpose()<<endl;
				cerr << "this: " << result.p.transpose()<<endl;
				return -1;
			}
		} else if(rp[cc] < 1e-8) {
			if(result.p[cc] > 1e-8) {
				cerr << "Mismatch of p values with R" << endl;
				cerr << "R:    " << rp.transpose()<<endl;
				cerr << "this: " << result.p.transpose()<<endl;
				return -1;
			}
		}
	}
	return 0;
}

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Performs an SVD. If an input matrix is given then it "
			"performs the SVD on that matrix, otherwise it creates a random "
			"SVD-able matrix and compares the true SVD with the estimated. ",
			' ', __version__ );

	TCLAP::ValueArg<size_t> a_vars("V", "vars", "Number of independent "
			"variables", false, 5, "vars", cmd);
	TCLAP::ValueArg<size_t> a_samples("s", "samples", "Number of samples ",
			false, 1000, "samples", cmd);
	TCLAP::ValueArg<double> a_noise("n", "noise", "Gaussian noise standard "
			"deviation. Note that signals and betas are all gaussian with "
			"standard deviation of 1.", false, 0.1, "sd", cmd);
	TCLAP::ValueArg<size_t> a_seed("S", "seed", "Seed number generation with "
			"the given constant", false, 0, "seed", cmd);

	cmd.add(a_verbose);
	cmd.parse(argc, argv);

	if(testknown() != 0)
		return -1;

	size_t seed = rd();
	if(a_seed.isSet()) {
		seed = a_seed.getValue();
	}
	cerr << "Seed: " << seed << endl;
	rng.seed(seed);

	std::normal_distribution<double> normdist(0, 1);

	VectorXd beta(a_vars.getValue()+1);
	for(size_t rr=0; rr<beta.rows(); rr++)
		beta[rr] = normdist(rng);

	MatrixXd X(a_samples.getValue(), a_vars.getValue()+1);
	for(size_t rr=0; rr<X.rows(); rr++) {
		X(rr, 0) = 1;
		for(size_t cc=1; cc<X.cols(); cc++)
			X(rr,cc) = normdist(rng);
	}

	VectorXd noise(a_samples.getValue());
	VectorXd y = X*beta;
	for(size_t rr=0; rr<y.rows(); rr++)
		noise[rr] = normdist(rng)*a_noise.getValue();
	y += noise;

	if(a_verbose.isSet()) {
		cerr << "y = \n" << y.transpose() << endl;
		cerr << "X = \n" << X << endl;
	}

	// true trandard error
	VectorXd Cinv = pseudoInverse(X.transpose()*X).diagonal();
	MatrixXd Xinv = pseudoInverse(X);

	double nvar = sqrt((noise.array()-noise.mean()).square().sum()/noise.rows());
	VectorXd truestd = (Cinv.cwiseSqrt())*nvar;
	VectorXd trueT = (beta.array()/truestd.array());

	// need to compute the CDF for students_t_cdf
	const double MAX_T = 500;
	const double STEP_T = 0.01;
	StudentsT distrib(X.rows()-X.cols(), STEP_T, MAX_T);

	// Test Regular Regression
	RegrResult result;
	regress(&result, y, X, Cinv, Xinv, distrib);

	if(a_verbose.isSet()) {
		cerr<<"Beta Est:  "<<result.bhat.transpose()<<endl;
		cerr<<"Beta True: "<<beta.transpose()<<endl;
		cerr<<"Rsqr:  "<<result.rsqr<<endl;
		cerr<<"Adj. Rsqr:  "<<result.adj_rsqr<<endl;
		cerr<<"Est. Std. Err: "<<result.std_err.transpose()<<endl;
		cerr<<"True Std. Err:  "<<truestd.transpose()<<endl;
		cerr<<"Est. T: "<<result.t.transpose()<<endl;
		cerr<<"True T: "<<(beta.array()/truestd.array()).transpose()<<endl;
		cerr<<"Est. p: "<<result.p.transpose()<<endl;
	}

	for(size_t cc=0; cc<beta.rows(); cc++) {
		if(beta[cc] > a_noise.getValue()) {
			double bdiff = 2*(beta[cc]-result.bhat[cc])/(beta[cc]+result.bhat[cc]);
			double tdiff = 2*(trueT[cc]-result.t[cc])/(trueT[cc]+result.t[cc]);
			if(bdiff > 0.1) {
				cerr << bdiff << endl;
				cerr<<"Beta Differs from the known value"<<endl;
				cerr << "True: " << beta.transpose()<<endl;
				cerr << "this: " << result.bhat.transpose()<<endl;
				return -1;
			} else if(tdiff > 0.1) {
				cerr << tdiff << endl;
				cerr<<"T Differs from that calculated from r"<<endl;
				cerr << "True: " << trueT.transpose()<<endl;
				cerr << "this: " << result.t.transpose()<<endl;
				return -1;
			}
		}
	}

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr<<"error: "<<e.error()<<" for arg "<<e.argId()<<std::endl;}

	return 0;
}


