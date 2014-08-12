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
 * @file fft_rotate_test.cpp A test of automatica computation of the rotation 
 * axis, based on the pseudo-polar fourier transform
 *
 *****************************************************************************/

#include <version.h>
#include <string>
#include <stdexcept>

#include <Eigen/Geometry> 

#define DEBUG 1

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

/**
 * @brief Performs a rotation of the image first by rotating around z, then
 * around y, then around x.
 *
 * @param rx Rotation around x, radians
 * @param ry Rotation around y, radians
 * @param rz Rotation around z, radians
 * @param in Input image
 *
 * @return 
 */
shared_ptr<MRImage> bruteForceRotate(Vector3d axis, double theta,
		shared_ptr<const MRImage> in)
{
	Matrix3d m;
	// negate because we are starting from the destination and mapping from
	// the source
	m = AngleAxisd(-theta, axis);
	LinInterp3DView<double> lin(in);
	auto out = dynamic_pointer_cast<MRImage>(in->copy());
	Vector3d ind;
	Vector3d cind;
	Vector3d center;
	for(size_t ii=0; ii<3 && ii<in->ndim(); ii++) {
		center[ii] = (in->dim(ii)-1)/2.;
	}

	for(Vector3DIter<double> it(out); !it.isEnd(); ++it) {
		it.index(3, ind.array().data());
		cind = m*(ind-center)+center;

		// set for each t
		for(size_t tt = 0; tt<in->tlen(); tt++) 
			it.set(tt, lin(cind[0], cind[1], cind[2], tt));
	}

	return out;
}

int closeCompare(shared_ptr<const MRImage> a, shared_ptr<const MRImage> b)
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
		if(diff > 1E-10) {
			cerr << "Images differ!" << endl;
			return -1;
		}
	}

	return 0;
}

Vector3d getAxis(shared_ptr<const MRImage> a, shared_ptr<const MRImage> b)
{
	Vector3d axis;
	
//	ppiX = copyCast
//


	return axis;
}

int main()
{
	// create an image
	int64_t index[3];
	size_t sz[] = {128, 128, 128};
	auto in = createMRImage(sizeof(sz)/sizeof(size_t), sz, FLOAT64);

	// fill with square
	OrderIter<double> sit(in);
	while(!sit.eof()) {
		sit.index(3, index);
		if(index[0] > sz[0]/4 && index[0] < sz[0]/3 && 
				index[1] > sz[1]/5 && index[1] < sz[1]/2 && 
				index[2] > sz[2]/3 && index[2] < 2*sz[2]/3) {
			sit.set(1);
		} else {
			sit.set(0);
		}
		++sit;
	}

	in->write("original.nii.gz");
	cerr << "Rotating manually" << endl;
	Vector3d ax(.1, .2, .3);
	ax.normalize();
	auto out = bruteForceRotate(ax, .2, in);
	out->write("brute_rotated.nii.gz");
	cerr << "Done" << endl;

	return 0;
}



