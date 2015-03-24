/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file imgiter_test1.cpp
 *
 *****************************************************************************/

#include <iostream>

#include "mrimage.h"
#include "nplio.h"
#include "iterators.h"

using namespace std;
using namespace npl;

int main()
{
	auto img = readMRImage("../../data/grad_imag.nii.gz");
	OrderIter<int> it(img);
	it.setOrder({0,1,2});
	it.goBegin();
	for(size_t ii=1; ii<=125; ii++, ++it) {
		if(*it != ii) {
			cerr << "Difference between read image and theoretical image" << endl;
			return -1;
		}
	}

	it.setOrder({}, true);
	it.goBegin();
	for(size_t ii=1; ii<=125; ii++, ++it) {
		if(*it != ii) {
			cerr << "Difference between read image and theoretical image"
				" when using reversed default order" << endl;
			return -1;
		}
	}

	return 0;
}

