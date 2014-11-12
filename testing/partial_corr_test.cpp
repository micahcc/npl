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
 * @file partial_corr_test.cpp Compute partial inverse of a matrix, leading 
 * way to partial correlation compute.
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

int main()
{
	size_t NSIG = 10;
	size_t NTIME = 2000;
	MatrixXd samples = gensigs(NSIG, NTIME);
	string sigfile = "pcor_test_sig.txt";
	cerr << "Writing signal to " << sigfile << endl;
	{
		ofstream ofs(sigfile.c_str());
		for(size_t rr=0; rr<NSIG; rr++) {
			for(size_t tt=0; tt<NTIME; tt++) {
				if(tt != 0) ofs << ", ";
				ofs << samples(rr, tt);
			}
			ofs << "\n";
		}
	}

	VectorXd mu = samples.rowwise().sum()/NTIME;
	MatrixXd cov = samples*samples.transpose()/NTIME - mu*mu.transpose();
	cerr << "Covariance: " << endl << cov << endl;
	VectorXd sigmas = cov.diagonal().array().sqrt();
	MatrixXd cor = cov.array()/(sigmas*sigmas.transpose()).array();
	cerr << "Cor:" << endl << cor << endl;

	MatrixXd cor2(NSIG, NSIG);
	for(size_t rr=0; rr<NSIG; rr++) {
		for(size_t cc=0; cc<NSIG; cc++) {
			double m1 = 0, m2 = 0;
			double s1 = 0, s2 = 0;
			double corv = 0;
			for(size_t tt=0; tt<NTIME; tt++) {
				corv += samples(rr, tt)*samples(cc, tt);
				m1 += samples(rr, tt);
				m2 += samples(cc, tt);
				s1 += samples(rr, tt)*samples(rr, tt);
				s2 += samples(cc, tt)*samples(cc, tt);
			}
			m1 /= NTIME;
			m2 /= NTIME;
			s1 = sqrt(s1/NTIME - m1*m1);
			s2 = sqrt(s2/NTIME - m2*m2);
			corv = (corv/NTIME - m1*m2)/(s1*s2);
			cor2(rr, cc) = corv;
		}
	}
	
	cerr << "Slow Cor: " << endl << cor2 << endl;
	for(size_t rr=0; rr<NSIG; rr++) {
		for(size_t cc=0; cc<NSIG; cc++) {
			if(fabs(cor(rr,cc) - cor2(rr,cc)) > 0.0000001) {
				cerr << "CORRELATION DISAGREES!" << endl;
				return -1;
			}
		}
	}
	
	// perform decomposition
	Eigen::JacobiSVD<MatrixXd> solver;
	cerr << "Done\nComputing..." << endl;
	solver.compute(cor, Eigen::ComputeThinU | Eigen::ComputeThinV);
	cerr << endl << solver.matrixU() << endl;
	VectorXd singvals = solver.singularValues();

	double ALPHA = 1e-15;
	for(size_t ii=0; ii<singvals.rows(); ii++)
		singvals[ii] = 1./(ALPHA+singvals[ii]);
	MatrixXd pinv = solver.matrixV()*singvals.asDiagonal()*solver.matrixU().transpose();

	cerr << "Inverse Corr: " << endl << pinv << endl << endl;
	cerr << "Should be ident: " << endl << (pinv*cor) << endl << endl;

	double err = (pinv*cor - MatrixXd::Identity(NSIG, NSIG)).array().sum();
	cerr << "Error: " << err << endl;
	if(fabs(err) > 1e-5) {
		cerr << "Deviates too much from identity" << endl;
		return -1;
	}

	MatrixXd pcor = -pinv.array()/((pinv.diagonal()*pinv.diagonal().transpose()).array());
	cerr << "Partial Corr: " << endl << pcor << endl;
	return 0;
}

