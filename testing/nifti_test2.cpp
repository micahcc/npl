/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file nifti_test2.cpp
 *
 *****************************************************************************/

#include <iostream>

#include "mrimage.h"
#include "nplio.h"
#include "accessors.h"

using namespace std;
using namespace npl;

int main()
{
	/* Create an image with: x+y*100+z*10000*/
	auto oimage = createMRImage({10, 23, 39}, FLOAT64);
	NDView<double> acc(oimage);

	for(int64_t zz=0; zz < oimage->dim(2); zz++) {
		for(int64_t yy=0; yy < oimage->dim(1); yy++) {
			for(int64_t xx=0; xx < oimage->dim(0); xx++) {
				double pix = xx+yy*100+zz*10000;
				acc.set({xx,yy,zz}, pix);
			}
		}
	}

	/* Write the Image */
	oimage->write("test4.nii.gz", false);

	/* Read the Image */
	auto iimage = readMRImage("test4.nii.gz", true);
	NDView<double> in(iimage);

	/* Check the Image */
	for(int64_t zz=0; zz < oimage->dim(2); zz++) {
		for(int64_t yy=0; yy < oimage->dim(1); yy++) {
			for(int64_t xx=0; xx < oimage->dim(0); xx++) {
				double pix = xx+yy*100+zz*10000;
				if(pix != in[{xx,yy,zz}]) {
					cerr << "Mismatch!" << endl;
					return -1;
				}
			}
		}
	}

	return 0;
}


