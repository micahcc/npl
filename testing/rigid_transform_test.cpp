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
 * @file rigid_transform_test.cpp Tests rigid transform struct.
 *
 *****************************************************************************/

#include <string>
#include <stdexcept>

#include <Eigen/Geometry> 

#include "mrimage.h"
#include "mrimage_utils.h"
#include "utility.h"
#include "ndarray_utils.h"
#include "registration.h"
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

	in->write("original.nii.gz");

	Rigid3DTrans rigid;
	rigid.rotation[0] = -3.14159/7;
	rigid.rotation[1] = 3.14159/5;
	rigid.rotation[2] = 3.14159/10;
	rigid.shift[0] = 3;
	rigid.shift[1] = 7;
	rigid.shift[2] = -3;

	cerr << "Original Transform:\n" << rigid << endl;
	auto transformed = dPtrCast<MRImage>(in->copy());
	rotateImageShearKern(transformed, rigid.rotation[0], rigid.rotation[1],
			rigid.rotation[2]);
	for(size_t dd=0; dd<(sizeof(sz)/sizeof(size_t)); dd++)
		shiftImageKern(transformed, dd, rigid.shift[dd]);
	transformed->write("transformed.nii.gz");

	rigid.invert();
	cerr << "Inverted Transform:\n" << rigid << endl;
	auto itransformed = dPtrCast<MRImage>(transformed->copy());
	rotateImageShearKern(itransformed, rigid.rotation[0], rigid.rotation[1],
			rigid.rotation[2]);
	for(size_t dd=0; dd<(sizeof(sz)/sizeof(size_t)); dd++)
		shiftImageKern(itransformed, dd, rigid.shift[dd]);
	itransformed->write("itransformed.nii.gz");

	rigid.invert();
	cerr << "\nDouble Inverted Transform:\n" << rigid << endl;

	cerr << "Original, Post Transformed" << endl;
	if(closeCompare(in, itransformed, 0.3) != 0)
		return -1;

	return 0;
}


