/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file shear_decompose_test.cpp A test of the shift decomposition of a
 * rotation matrix
 *
 *****************************************************************************/

#include <string>
#include <stdexcept>

#include <Eigen/Geometry>

#include "mrimage.h"
#include "mrimage_utils.h"
#include "utility.h"
#include "ndarray_utils.h"
#include "iterators.h"
#include "accessors.h"

using namespace npl;
using namespace std;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::AngleAxisd;

int closeCompare(shared_ptr<const MRImage> a, shared_ptr<const MRImage> b,
        double thresh = .01)
{
	if(a->ndim() != b->ndim()) {
		cerr << "Error image dimensionality differs" << endl;
		return -1;
	}

	for(size_t dd=0; dd<a->ndim(); dd++) {
		if(a->dim(dd) != b->dim(dd)) {
			cerr << "Image size in the " << dd << " direction differs" << endl;
			return -1;
		}
	}

	OrderConstIter<double> ita(a);
	OrderConstIter<double> itb(b);
	itb.setOrder(ita.getOrder());
	for(ita.goBegin(), itb.goBegin(); !ita.eof() && !itb.eof(); ++ita, ++itb) {
		double diff = fabs(*ita - *itb);
		if(diff > thresh) {
			cerr << "Images differ by " << diff << endl;
			return -1;
		}
	}

	return 0;
}

int main()
{
	// create an image
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

	cerr << "Testing Shear Decompositions" << endl;
	std::list<Matrix3d> terms;
	const double PI = acos(-1);
	int64_t iters = 10;
	for(int64_t ii=-iters/2; ii<iters/2; ii++) {
		for(int64_t jj=-iters/2; jj<iters/2; jj++) {
			for(int64_t kk=-iters/2; kk<iters/2; kk++) {
				double rx = (PI/2.)*ii/(double)iters;
				double ry = (PI/2.)*jj/(double)iters;
				double rz = (PI/2.)*kk/(double)iters;
				cerr << rx << "," << ry << "," << rz << endl;
				if(shearTest(rx,ry,rz) != 0) {
					cerr << "Failed Shear Test for " <<
						rx << ", " << ry << ", " << rz << endl;
					return -1;
				}
				if(shearDecompose(terms, rx, ry, rz) != 0) {
					cerr << "Failure!" << endl;
					return -1;
				}
			}
		}
	}
	cerr << "Success" << endl;
	return 0;
}



