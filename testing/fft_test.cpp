/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file fft_test.cpp
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
	auto in = createMRImage(sizeof(sz)/sizeof(size_t), sz, FLOAT64);

	// fill with sphere
	OrderIter<double> it(in);
	while(!it.eof()) {
		it.index(3, index);
		double dist = 0;
		for(size_t ii=0; ii<3 ; ii++) {
			dist += (index[ii]-sz[ii]/2.)*(index[ii]-sz[ii]/2.);
		}
		if(sqrt(dist) < 5)
			it.set(dist);
		else it.set(0);

		++it;
	}

	// fourier transform image in xyz direction
	in->write("pre-fft.nii.gz");
    std::vector<size_t> iosize(sz, sz+3);
	auto fft = dynamic_pointer_cast<MRImage>(fft_forward(in, iosize));
	auto ifft = dynamic_pointer_cast<MRImage>(fft_backward(fft, iosize));
    ifft = dynamic_pointer_cast<MRImage>(ifft->copyCast(FLOAT64));
	ifft->write("post-ifft.nii.gz");
	if(closeCompare(ifft, in) != 0)
		return -1;


	return 0;
}



