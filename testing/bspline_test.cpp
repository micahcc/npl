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
 * @file bspline_test.cpp Test the b-spline interpolation class
 *
 *****************************************************************************/

#include <iostream>
#include <memory>
#include <ctime>

#include "bspline.h"
#include "iterators.h"
#include "iterators.h"

using namespace std;
using namespace npl;

shared_ptr<MRImage> squareImage()
{
    // create test image
	int64_t index[3];
	size_t sz[] = {64, 64, 64};
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
}

int main()
{
	// create a square image
	auto img = squareImage();

	vector<double> pt(img->ndim());
	CubicBSpline spline;
	spline.ras = true;
	spline.createOverlay(img, 10);
	
	// Randomly set parameters
	for(FlatIter<double> it(spline.params); !it.eof(); ++it) 
		it.set(rand()/(double)RAND_MAX);

	// Multiply Image By Sampled Spline
	clock_t c = clock();
	for(NDIter<double> it(img); !it.eof(); ++it) 
		it.set((*it)*spline.sample(pt));
	c = clock() - c;
	cerr << "Sampling Time: " << c << endl;

	auto sbspline = dPtrCast<MRImage>(img->createAnother()); 
	for(FlatIter<double> it(sbspline); !it.eof(); ++it)
		it.set(1./(*it));

	c = clock();
	spline.sample(sbspline);
	for(FlatIter<double> it1(img), it2(sbspline); !it1.eof(); ++it1, ++it2) 
		it1.set((*it1)*(*it2));
	c = clock() - c;

	auto newsq = squareImage();
	for(FlatIter<double> it1(newsq), it2(img); !it1.eof(); ++it1, ++it2) {
		if(fabs(*it1-*it2) > 0.0000000000) {
			cerr << "Mismatch!" << endl;
			return -1;
		}
	}
	return 0;
}

