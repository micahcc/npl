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
 * @file array_test1.cpp
 *
 *****************************************************************************/

#include <Eigen/Dense>
#include "statistics.h"
#include <iostream>
#include <ctime>

using namespace std;
using namespace npl;

int main()
{
	// Generate data according to a mixture of gaussian, gamma and -gamma
	std::random_device rd;
	std::default_random_engine rng(rd());

	std::gamma_distribution<double> gsdist1(8,2);
	std::gamma_distribution<double> gsdist2(8,2);
	std::normal_distribution<double> normdist(0.5, 4);
	std::uniform_int_distribution<int> undist(0,6);
	VectorXd samples(1000);
	VectorXd truemean(3);
	VectorXd truesd(3);
	VectorXd trueprior(3);
	truemean.setZero();
	truesd.setZero();
	trueprior.setZero();

	for(size_t ii=0; ii<samples.rows(); ii++) {
		int g = undist(rng);
		double v = 0;
		switch(g) {
			case 0:
				v = -gsdist1(rng);
				trueprior[0]++;
				truemean[0] += v;
				truesd[0] += v*v;
			break;
			case 1:
				v = gsdist2(rng);
				trueprior[2]++;
				truemean[2] += v;
				truesd[2] += v*v;
			break;
			default:
				v = normdist(rng);
				trueprior[1]++;
				truemean[1] += v;
				truesd[1] += v*v;
			break;
			break;
		}

		samples[ii] = v;
	}

	for(size_t cc=0; cc<3; cc++) {
		truemean[cc] /= trueprior[cc];
		truesd[cc] = sqrt(truesd[cc]/trueprior[cc]-truemean[cc]*truemean[cc]);
		trueprior[cc] /= samples.size();
	}
	cerr<<"Negative Gauss: "<<truemean[0]<<", "<<truesd[0]<<endl;
	cerr<<"Positive Gauss: "<<truemean[2]<<", "<<truesd[2]<<endl;
	cerr<<"Normal Dist : "<<truemean[1]<<", "<<truesd[1]<<endl;

	cerr<<"Fitting T-maps"<<endl;
	VectorXd mu(3);
	VectorXd sd(3);
	VectorXd prior(3);
	vector<std::function<double(double,double,double)>> pdfs;
	pdfs.push_back(gammaPDF_MS);
	pdfs.push_back(gaussianPDF); // middle is the gaussian that we care about
	pdfs.push_back(gammaPDF_MS);
	mu << -1,0,1;
	sd << 1,1,1;
	prior << .3,.3,.3;
	expMax1D(samples, pdfs, mu, sd, prior, "expmax1d_test.svg");

	cerr<<"Negative Gauss: "<<mu[0]<<", "<<sd[0]<<endl;
	cerr<<"Positive Gauss: "<<mu[2]<<", "<<sd[2]<<endl;
	cerr<<"Normal Dist : "<<mu[1]<<", "<<sd[1]<<endl;

	if((mu[0] - truemean[0])/sd[0] > 0.1)
		return -1;
	if((mu[1] - truemean[1])/sd[1] > 0.1)
		return -1;
	if((mu[2] - truemean[2])/sd[2] > 0.1)
		return -1;
	return 0;
}

