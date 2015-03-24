/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file fft_shift_test.cpp
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
		if(diff > .5) {
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

	// fill with sphere
	OrderIter<double> sit(in);
	while(!sit.eof()) {
		sit.index(3, index);
		double dist = 0;
		for(size_t ii=0; ii<3 ; ii++) {
			dist += (index[ii]-sz[ii]/2.)*(index[ii]-sz[ii]/2.);
		}
		if(sqrt(dist) < 5)
			sit.set(1);
		else
			sit.set(0);

		++sit;
	}

	// perform fourier shift, +a
	// strictly the frequency for component k (where k = k-N/2,N/2]
	// double T = fft->dim(d)*in->spacing()[d];
	// double f = k/T; // where T is the total sampling period
	double shift[3] = {1, 5, 10};

	// manual shift
	auto kshift = dynamic_pointer_cast<MRImage>(in->copy());
	for(size_t ii=0; ii<sizeof(shift)/sizeof(double); ii++)
		shiftImageKern(kshift, ii, shift[ii]);
	kshift->write("kern_shift.nii.gz");


	auto fshift = dynamic_pointer_cast<MRImage>(in->copy());
	for(size_t ii=0; ii<sizeof(shift)/sizeof(double); ii++)
		shiftImageFFT(fshift, ii, shift[ii]);
	fshift->write("fourier_shift.nii.gz");

	if(closeCompare(kshift, fshift) != 0)
		return -1;


	return 0;
}

