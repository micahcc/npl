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
 * @file generalLinearModel.cpp Tests large scale GLM
 *
 *****************************************************************************/

#include <version.h>
#include <string>
#include <stdexcept>
#include <iostream>
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

	size_t tlen = 1024;
	MatrixXd X(tlen, 4);
	for(size_t rr=0; rr<X.rows(); rr++ ) {
		for(size_t cc=0; cc<X.cols(); cc++) {
			X(rr,cc) = cos(M_PI*rr/100*cc);
		}
	}

	VectorXd y = X*beta + VectorXd::Random(tlen);

	// Cache Reused Vectors
	auto Xinv = pseudoInverse(X);
	auto covInv = pseudoInverse(X.transpose()*X);

	const double MAX_T = 30;
	const double STEP_T = 0.1;
	StudentsT stud_dist(X.rows()-1, STEP_T, MAX_T);

	RegrResult ret;
	regress(ret, y, X, covInv, Xinv, stud_dist);

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
	cerr << "T-Value: " << ret.t.transpose() << endl;
	cerr << "P-value: " << ret.p.transpose() << endl;
	cerr << "DOF: "<< ret.dof << endl;

	return 0;
}



