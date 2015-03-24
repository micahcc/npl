/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file bspline_test.cpp Test the b-spline interpolation class
 *
 *****************************************************************************/

#include <iostream>
#include <memory>
#include <ctime>

#include "iterators.h"
#include "accessors.h"

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
	img->write("bspline_square.nii.gz");

	vector<double> pt(img->ndim());
	BSplineView<double> spline;
	spline.m_ras = true;
	spline.createOverlay(img, 10);

	// Randomly set parameters
	for(FlatIter<double> it(spline.getParams()); !it.eof(); ++it)
		it.set(rand()/(double)RAND_MAX);
	spline.getParams()->write("bspline_params.nii.gz");

	// Multiply Image By Sampled Spline
	clock_t c = clock();
	for(NDIter<double> it(img); !it.eof(); ++it) {
		it.index(pt.size(), pt.data());
		img->indexToPoint(pt.size(), pt.data(), pt.data());
		it.set((*it)-spline.get(pt.size(), pt.data()));
	}
	c = clock() - c;
	cerr << "Sampling Time: " << c << endl;
	img->write("bspline_sub_square.nii.gz");

	c = clock();
	auto sbspline = spline.reconstruct(img);
	sbspline->write("bspline_sampled.nii.gz");
	for(FlatIter<double> it1(img), it2(sbspline); !it1.eof(); ++it1, ++it2)
		it1.set((*it1)+(*it2));
	c = clock() - c;
	cerr << "Recon Time:    " << c << endl;
	img->write("bspline_add_square.nii.gz");

	auto newsq = squareImage();
	for(FlatIter<double> it1(newsq), it2(img); !it1.eof(); ++it1, ++it2) {
		if(fabs(*it1-*it2) > 0.000000001) {
			cerr << "Mismatch!" << (*it1-*it2) << endl;
			return -1;
		}
	}
	return 0;
}

