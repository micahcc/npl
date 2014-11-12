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
 * @file partial_corr_test_big.cpp Test how long it takes to run large partial
 * correlations.
 *
 *****************************************************************************/

#include <iostream>
#include <fstream>

#include <Eigen/Dense>
#include <Eigen/SVD>

using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;

/**
 * @brief generates gaussian gandom N signals, of length T, output matrix has
 * one signal per row, size N x T
 *
 * @param nsig Number of signals (N)
 * @param ntim Number of timepoints in signals (T)
 *
 * @return Matrix N x T holding generated signals
 */
MatrixXd gensigs(size_t nsig, size_t ntim)
{
	std::random_device rd;
	std::default_random_engine rng(rd());
	std::normal_distribution<double> dist(0, 1);
	MatrixXd out(nsig, ntim);
	for(size_t s=0; s<nsig; s++) {
		for(size_t t=0; t<ntim; t++)
			out(s, t) = dist(rng);
	}
	return out;
};

int main(int argc, char** argv)
{
	if(argc != 2) {
		cerr << "Input should be 1 argument, indicatin number of signals." << endl;
		return -1;
	}

	size_t NSIG = atoi(argv[1]);
	size_t NTIME = 2000;
	cerr << "Generating " << NSIG << " signals with " << NTIME << " timepoints." << endl;
	MatrixXd samples = gensigs(NSIG, NTIME);
	cerr << "Done" << endl;
	
	VectorXd mu = samples.rowwise().sum()/NTIME;

	cerr << "Computing Correlation Matrix" << endl;
	MatrixXd cov = samples*samples.transpose()/NTIME - mu*mu.transpose();
	VectorXd sigmas = cov.diagonal().array().sqrt();
	MatrixXd cor = cov.array()/(sigmas*sigmas.transpose()).array();
	cerr << "Done" << endl;

	// perform decomposition
	Eigen::JacobiSVD<MatrixXd> solver;
	cerr << "Computing SVD..." << endl;
	solver.compute(cor, Eigen::ComputeThinU | Eigen::ComputeThinV);
	cerr << "Done" << endl;
	VectorXd singvals = solver.singularValues();

	cerr << "Computing Pseudo Inverse..." << endl;
	double ALPHA = 1e-15;
	for(size_t ii=0; ii<singvals.rows(); ii++)
		singvals[ii] = 1./(ALPHA+singvals[ii]);
	MatrixXd pinv = solver.matrixV()*singvals.asDiagonal()*solver.matrixU().transpose();
	cerr << "Done" << endl;

	cerr << "Testing Inverse..." << endl;
	double err = (pinv*cor - MatrixXd::Identity(NSIG, NSIG)).array().sum();
	cerr << "Error: " << err << endl;
	if(fabs(err) > 1e-5) {
		cerr << "Deviates too much from identity" << endl;
		return -1;
	}
	cerr << "Done" << endl;

	cerr << "Computing Partial Corr" << endl;
	MatrixXd pcor = -pinv.array()/((pinv.diagonal()*pinv.diagonal().transpose()).array());
	cerr << "Done" << endl;
	return 0;
}


