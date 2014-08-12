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
 * @file fft_rotate_test.cpp A test of the shear/fourier shift based rotation
 * function
 *
 *****************************************************************************/

#include <version.h>
#include <string>
#include <stdexcept>

#include <Eigen/Geometry> 

#define DEBUG 1

#include "mrimage.h"
#include "mrimage_utils.h"
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
shared_ptr<MRImage> bruteForceRotate(double rx, double ry, double rz, 
		shared_ptr<const MRImage> in)
{
	Matrix3d m;
	m = AngleAxisd(rx, Vector3d::UnitX())*AngleAxisd(ry,  Vector3d::UnitY())*
				AngleAxisd(rz, Vector3d::UnitZ());
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
		cind = m*(ind-center);

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
				index[1] > sz[1]/4 && index[1] < sz[1]/3 && 
				index[2] > sz[2]/4 && index[2] < sz[2]/3) {
			sit.set(1);
		} else {
			sit.set(0);
		}
		++sit;
	}

	in->write("original.nii.gz");
	auto out = bruteForceRotate(.3, .1, .01, in);
	out->write("rotated.nii.gz");

	double shearterms[4][2]; 

	std::list<Matrix3d> terms;
	const double PI = acos(-1);
	size_t iters = 10;
	size_t failure = 0;
	size_t count = 0;
	for(size_t ii=0; ii<iters; ii++) {
		for(size_t jj=0; jj<iters; jj++) {
			for(size_t kk=0; kk<iters; kk++) {
				double rx = ii/(2*PI);
				double ry = jj/(2*PI);
				double rz = kk/(2*PI);
				count++;
				if(shearTest(rx,ry,rz) != 0) {
					cerr << "Failed Shear Test for " <<
						rx << ", " << ry << ", " << rz << endl;
					return -1;
				}
				if(shearDecompose(terms, rx, ry, rz) != 0)
					failure++;
			}
		}
	}

	cerr << "Failure Rate:" << failure/(double)count << endl;;
//	// perform fourier rotation, +a
//	// strictly the frequency for component k (where k = k-N/2,N/2]
//	// double T = fft->dim(d)*in->spacing()[d];
//	// double f = k/T; // where T is the total sampling period
//	double rotate[3] = {1, 5, 10};
//
//	// algorithm (registration) will produce axis, we will rotate that axis to
//	// make it align with +Z. 
//
//	// need euler angles for shearing, use mathematica to solve shearing for
//	// angle-axis version?
//	
//	// manual shift
//	auto mrotate = dynamic_pointer_cast<MRImage>(in->copy());
//	NDAccess<double> acc(in);
//	for(OrderIter<double> it(mrotate); !it.eof(); ++it) {
//		it.index(3, index);
//		
//
//		// rotate point
//		for(size_t dd = 0 ; dd < in->ndim(); dd++)
//			index[dd] = clamp<int64_t>(0, in->dim(dd)-1, index[dd]-shift[dd]);
//
//		it.set(acc.get(3, index));
//
//	}
//	mrotate->write("manual_rotate.nii.gz");
////	
////	
////	for(size_t ii=0; ii<sizeof(shift)/sizeof(double); ii++)
////		shiftImage(in, ii, shift[ii]);
////	
////	in->write("fourier_shift.nii.gz");
////
//	if(closeCompare(in, mrotate) != 0)
//		return -1;
//	

	return 0;
}


