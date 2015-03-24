/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file dc_reg_test1.cpp Tests mutual information derivative for deformation
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

double gaussGen(double x, double y, double z, double xsz, double ysz, double zsz,
		double sx, double sy, double sz)
{
    double v = 1./(sx*sy*sz*pow(2*M_PI,1.5))*
		exp(-pow(xsz/2-x,2)/pow(sx,2)/2)*exp(-pow(ysz/2-y,2)/pow(sy,2)/2)*
		exp(-pow(zsz/2-z,2)/pow(sz,2)/2);
	return v;
}

shared_ptr<MRImage> gaussianImage(double sx, double sy, double sz)
{
    // create an image
    size_t size[] = {12,16,17};
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
    auto distimg = gaussianImage(sx, sy, sz);
	distimg->write("gaussmooth_test_distimg.nii.gz");

	auto origimg = gaussianImage(4, 4, 4);
	origimg->write("gaussmooth_test_origimg.nii.gz");

    if(distcorDerivTest(0.001, 0.157, distimg, origimg, 0, 100) != 0)
        return -1;
    if(distcorDerivTest(0.001, 0.155, distimg, origimg, 100, 0) != 0)
        return -1;
    if(distcorDerivTest(0.001, 0.14, distimg, origimg, 100, 100) != 0)
        return -1;
    return 0;
}



