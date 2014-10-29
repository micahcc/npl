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
 * @file comparison_test.cpp Tests various metrics of image comparison
 *
 *****************************************************************************/

#include "mrimage.h"
#include "iterators.h"
#include "accessors.h"
#include "ndarray_utils.h"
#include "mrimage_utils.h"
#include "registration.h"
#include "byteswap.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <memory>
#include <cstring>

using namespace std;
using namespace npl;

shared_ptr<MRImage> squareImage()
{
    // create test image
	int64_t index[3];
	size_t sz[] = {32, 32, 32};
	auto in = createMRImage(sizeof(sz)/sizeof(size_t), sz, FLOAT64);

	// fill with square
	OrderIter<double> sit(in);
	while(!sit.eof()) {
		sit.index(3, index);
		if(index[0] > sz[0]/4 && index[0] < 2*sz[0]/3 && 
				index[1] > sz[1]/5 && index[1] < sz[1]/2 && 
				index[2] > sz[2]/3 && index[2] < 2*sz[2]/3) {
			sit.set(1);
		} else {
			sit.set(0);
		}
		++sit;
	}

    return in;
};

int main()
{
    auto origimg = squareImage();
	auto noisyimg = origimg->copy();
	for(NDIter<double> it(noisyimg); !it.eof(); ++it) 
		it.set(*it + 0.1*rand()/(double)RAND_MAX);
	double v;
	
	cerr << "Noisy versus Original" << endl;
	v = mse(origimg, noisyimg);
	cerr << "MSE: " << v << endl;
	if(v > 0.1) {
		cerr << "MSE is too large" << endl;
		return -1;
	}

	v = corr(origimg, noisyimg);
	cerr << "Correlation: " << v << endl;
	if(v < 0.9) {
		cerr << "Corr is too small" << endl;
		return -1;
	}

	v = information(origimg, noisyimg, 64, 4, METRIC_MI);
	cerr << "Mutual Info: " << v << endl;
	if(v < 0.14) {
		cerr << "Mutual information is too small" << endl;
		return -1;
	}

	v = information(origimg, noisyimg, 64, 4, METRIC_NMI);
	cerr << "Normalized Mutual Info: " << v << endl;
	if(v < 1) {
		cerr << "Normalized Mutual information is too small" << endl;
		return -1;
	}
	
	// Negate and compare
	cerr << "Inverted versus Original" << endl;
	for(NDIter<double> it(noisyimg); !it.eof(); ++it) 
		it.set(-*it);
	
	v = mse(origimg, noisyimg);
	cerr << "MSE: " << v << endl;
	if(v < 0.1) {
		cerr << "MSE is too small" << endl;
		return -1;
	}

	v = corr(origimg, noisyimg);
	cerr << "Correlation: " << v << endl;
	if(v > -0.9) {
		cerr << "Corr is too large" << endl;
		return -1;
	}

	v = information(origimg, noisyimg, 100, 4, METRIC_MI);
	cerr << "Mutual Info: " << v << endl;
	if(v < 0.14) {
		cerr << "Mutual information is too small" << endl;
		return -1;
	}

	v = information(origimg, noisyimg, 100, 4, METRIC_NMI);
	cerr << "Mutual Info: " << v << endl;
	if(v < 1) {
		cerr << "Normalized Mutual information is too small" << endl;
		return -1;
	}

    return 0;
}




