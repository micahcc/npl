/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file imgiter_test.cpp
 *
 *****************************************************************************/

#include <iostream>

#include "mrimage.h"

using namespace std;
using namespace npl;

int main()
{
	/* Create an image with: x+y*100+z*10000*/
	size_t sz[3] = {10, 23, 39};
	MRImageStore<3, double> testimage(sz);
	MRImage* oimage = &testimage;

	for(size_t zz=0; zz < oimage->dim(2); zz++) {
		for(size_t yy=0; yy < oimage->dim(1); yy++) {
			for(size_t xx=0; xx < oimage->dim(0); xx++) {
				double pix = xx+yy*100+zz*10000;
				oimage->dbl({xx,yy,zz}, pix);
			}
		}
	}

	/* Write the Image */
	writeMRImage(oimage, "test4.nii.gz");

	/* Read the Image */
	MRImage* iimage = readMRImage("test4.nii.gz", true);

	/* Check the Image */
	for(size_t zz=0; zz < oimage->dim(2); zz++) {
		for(size_t yy=0; yy < oimage->dim(1); yy++) {
			for(size_t xx=0; xx < oimage->dim(0); xx++) {
				double pix = xx+yy*100+zz*10000;
				if(pix != oimage->dbl({xx,yy,zz})) {
					cerr << "Mismatch!" << endl;
					return -1;
				}
			}
		}
	}

	return 0;
}


