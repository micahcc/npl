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
 * @file FSFDP_test.cpp Test of the Fast Search and Find of Density Peaks 
 * algorith from Rodriguez et al 2014.
 *
 *****************************************************************************/

#include "statistics.h"
#include "npltypes.h"
#include <iostream>
#include <iomanip>
#include <map>
#include <fstream>
#include <Eigen/Dense>
#include <ctime>

using namespace std;
using namespace npl;

int main(int argc, char** argv)
{
	/****************************
	 * create points randomly clustered around a few means
	 ***************************/
	const size_t NCLUSTER = 4;
	const size_t NDIM = 2;
	const size_t NSAMPLES = 1000;

	std::random_device rd;
	//    std::default_random_engine rng(rd());
	std::default_random_engine rng(13);
	std::uniform_int_distribution<int> randUI(0, NCLUSTER-1);
	std::normal_distribution<double> randGD(0, 1);

	// add mean to each sample
	Eigen::VectorXi trueclass(NSAMPLES);
	Eigen::MatrixXd samples(NSAMPLES, NDIM);
	for(size_t ii=0; ii<NSAMPLES; ii++) {
		// choose random group
		int radius = 1+randUI(rng);
		double angle = randGD(rng)/10;
		samples(ii, 0) = radius*cos(angle);
		samples(ii, 1) = radius*sin(angle);
		trueclass[ii] = radius-1;
	}
	
	/*************************************************************************
	 * Test That Preliminary Parts of algorithms return the same thing
	 *************************************************************************/
	Eigen::VectorXi rho1, rho2, parent1, parent2;
	Eigen::VectorXd delta1, delta2;
	findDensityPeaks(samples, .5, rho1, delta1, parent1);
	findDensityPeaks_brute(samples, .5, rho2, delta2, parent2);
	for(size_t ii=0; ii<samples.rows(); ii++) {
		if(rho1[ii] != rho2[ii]) {
			cerr << "Mismatched Rho at "<< ii << " with brute: " << rho2[ii] <<
				" vs " << rho1[ii] << endl;
			return -1;
		}
		if(delta1[ii] != delta1[ii]) {
			cerr << "Mismatched delta"<< endl;
			return -1;
		}
		if(parent1[ii] != parent2[ii]) {
			cerr << "Mismatched parent"<< endl;
			return -1;
		}
	}

	/****************************
	 * Perform Clustering
	 ***************************/
	Eigen::VectorXi classes;
	if(fastSearchFindDP_brute(samples, classes, .5) != 0) {
		cerr << "Clustering Failed" << endl;
		return -1;
	}

	/****************************
	 * Test output
	 ***************************/
	// align based on maximum match
	Eigen::VectorXi cmap(NCLUSTER);
	for(size_t ii=0; ii<NCLUSTER; ii++)
		cmap[ii] = ii;


	// TODO better comparison metric
	// for each true class, find the best matching estimated class
	size_t total = 0;
	for(int64_t tc=0; tc<NCLUSTER; tc++) {
		map<int64_t, int64_t> matchcounts;
		for(size_t rr=0; rr<classes.rows(); rr++) {
			auto ret = matchcounts.insert(make_pair(classes[rr], 0));
			ret.first->second++;
		}

		cerr << "Class: " << tc << endl;
		int64_t bestmatch = 0;
		for(auto& p: matchcounts) {
			bestmatch = max(bestmatch, p.second);
			cerr << p.first << " : " << p.second << endl;
		}
		total += bestmatch;
	}
	double ratio = total/(double)samples.rows();
	cerr << "Match Percent: " << ratio << endl;

	if(argc > 1) {
		cerr << "Writing classification to " << argv[1] << endl;
		ofstream ofs(argv[1]);
		ofs << setw(10) << "Index ";
		for(size_t cc=0; cc<samples.cols(); cc++) 
			ofs << setw(10) << cc;
		ofs << setw(10) << "TrueClass" << setw(10) << "EstClass" << endl;
		for(size_t rr=0; rr<samples.rows(); rr++) {
			ofs << rr;
			for(size_t cc=0; cc<samples.cols(); cc++) 
				ofs << setw(15) << samples(rr, cc);
			ofs << setw(15) << trueclass[rr] << setw(15) << classes[rr] << endl;
		}
		ofs << endl;
	}

	if((1-ratio) > 0.1) {
		cerr << "Fail" << endl;
		return -1;
	}

	cerr << "OK!" << endl;
	return 0;
}


