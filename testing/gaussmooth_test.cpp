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
 * @file gaussmooth_test.cpp Tests gaussian smoothing of images.
 *
 *****************************************************************************/

#include "mrimage.h"
#include "ndarray.h"
#include "ndarray_utils.h"
#include "macros.h"
#include "iterators.h"
#include "accessors.h"

#include <iostream>

using namespace npl;
using namespace std;

double gaussGen(double x, double y, double z, double xsz, double ysz, double zsz,
		double sx, double sy, double sz)
{
    double v = 1./(sx*sy*sz*pow(2*M_PI,1.5))*
		exp(-pow(xsz/2-x,2)/pow(sx,2)/2)*exp(-pow(ysz/2-y,2)/pow(sy,2)/2)*
		exp(-pow(zsz/2-z,2)/pow(sz,2)/2);
	return v;
}

shared_ptr<MRImage> impulseImage()
{
    // create an image
    size_t sz[] = {64,64,64};
    vector<int64_t> index(3);
    auto in = createMRImage(sizeof(sz)/sizeof(size_t), sz, FLOAT32);
	for(FlatIter<double> fit(in); !fit.eof(); ++fit)
		fit.set(0);

	NDView<double> view(in);
	index[0] = 32;
	index[1] = 32;
	index[2] = 32;
	view.set(index, 1);
    return in;
}

shared_ptr<MRImage> gaussianImage(double sx, double sy, double sz)
{
    // create an image
    size_t size[] = {64,64,64};
    int64_t index[3];
    auto in = createMRImage(sizeof(size)/sizeof(size_t), size, FLOAT32);

    // fill with a shape that is somewhat unique when rotated. 
    OrderIter<double> sit(in);
    double sum = 0;
    while(!sit.eof()) {
        sit.index(3, index);
		double v= gaussGen(index[0], index[1], index[2], size[0], size[1],
				size[2], sx, sy, sz);
        sit.set(v);
        sum += v;
        ++sit;
    }

    for(sit.goBegin(); !sit.eof(); ++sit) 
        sit.set(sit.get()/sum);

    return in;
}

int main()
{
	double sx = 3;
	double sy = 4; 
	double sz = 5;

    // create image with gaussian kernel in it
    auto gimg = gaussianImage(sx, sy, sz);
	gimg->write("gaussmooth_test_gimg.nii.gz");

	auto iimg = impulseImage();
	iimg->write("gaussmooth_test.nii.gz");

	// smooth
	gaussianSmooth1D(iimg, 0, sx);
	gaussianSmooth1D(iimg, 1, sy);
	gaussianSmooth1D(iimg, 2, sz);
	iimg->write("gaussmooth_test_smoothed.nii.gz");

	for(FlatConstIter<double> git(gimg), iit(iimg); !git.eof() && !iit.eof();
				++iit, ++git) {
		if(fabs(*git - *iit) > 0.0001) {
			cerr << "Mismatch!" << *git << " vs " << *iit << endl;
			return -1;
		}
	}
    return 0;
}

