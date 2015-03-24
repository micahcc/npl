/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file txt_write1.cpp Creates a new image then reads it back
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
	// create an image
	size_t sz[] = {3, 1, 1, 10};

	{
		auto in = createMRImage(sizeof(sz)/sizeof(size_t), sz, FLOAT64);

		NDIter<double> sit(in);
		for(int ii=0; !sit.eof(); ++sit, ++ii) {
			sit.set(ii);
		}
		in->write("test_txt1.csv");
	}

	{
		auto reread = dPtrCast<MRImage>(readMRImage("test_txt1.csv"));

		NDIter<double> sit(reread);
		for(int ii=0; !sit.eof(); ++sit, ++ii) {
			if(*sit != ii) {
				cerr << "Value Mismatch! " << ii << " vs " << *sit << endl;
				return -1;
			}
		}

	}
	return 0;
}



