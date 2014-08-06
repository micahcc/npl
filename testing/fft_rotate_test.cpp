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

/******************************************************************************
 * @file fft_test.cpp
 * @brief This file is specifically to test forward, reverse of fft image
 * procesing functions.
 ******************************************************************************/

#include <version.h>
#include <string>
#include <stdexcept>

#include "mrimage.h"
#include "mrimage_utils.h"
#include "ndarray_utils.h"
#include "iterators.h"
#include "accessors.h"

using namespace npl;
using namespace std;

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
	cerr << sizeof(sz) << endl;
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
	
	// perform fourier rotation, +a
	// strictly the frequency for component k (where k = k-N/2,N/2]
	// double T = fft->dim(d)*in->spacing()[d];
	// double f = k/T; // where T is the total sampling period
	double rotate[3] = {1, 5, 10};

	// manual shift
	auto mrotate = dynamic_pointer_cast<MRImage>(in->copy());
	NDAccess<double> acc(in);
//	for(OrderIter<double> it(mshift); !it.eof(); ++it) {
//		it.index(3, index);
//		
//		// rotate point
//		for(size_t dd = 0 ; dd < in->ndim(); dd++)
//			index[dd] = clamp<int64_t>(0, in->dim(dd)-1, index[dd]-shift[dd]);
//
//		it.set(acc.get(3, index));
//
//	}
//	mshift->write("manual_shift.nii.gz");
//	
//	
//	for(size_t ii=0; ii<sizeof(shift)/sizeof(double); ii++)
//		shiftImage(in, ii, shift[ii]);
//	
//	in->write("fourier_shift.nii.gz");
//
	if(closeCompare(in, mrotate) != 0)
		return -1;
	

	return 0;
}


