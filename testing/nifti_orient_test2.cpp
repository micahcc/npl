/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file nifti_orient_test2.cpp
 *
 *****************************************************************************/

#include <iostream>

#include "mrimage.h"
#include "nplio.h"

using namespace std;
using namespace npl;

int main()
{
	/* Read the Image */
	shared_ptr<MRImage> img = readMRImage("../../data/test_nifti2.nii.gz", true);
	if(!img) {
		std::cerr << "Failed to open image!" << std::endl;
		return -1;
	}

    Eigen::Vector4d corrorigin;
    corrorigin << 1.3, 75, 9, 0;

    Eigen::Vector4d corrspacing;
    corrspacing << 3, 47, 12, 3;

    Eigen::Matrix4d corrdir;
    corrdir << -0.16000,   -0.70293,     -0.69303, 0,
               -0.01707,   -0.70000,      0.71394, 0,
                0.98697,   -0.12606,     -0.10000, 0,
                      0,          0,      0.00000, 1;

	cerr << "Correct Direction:\n" << corrdir << endl;
	cerr << "Image Direction:\n" << img->getDirection() << endl;

    cerr << "Correct Spacing:\n" << corrspacing << endl;
	cerr << "Image Spacing:\n" << img->getSpacing() << endl;

    cerr << "Correct Origin:\n" << corrorigin << endl;
	cerr << "Image Origin:\n" << img->getOrigin() << endl;

	for(size_t ii=0; ii < img->ndim(); ii++) {
        if(fabs(img->origin(ii) - corrorigin[ii]) > 0.000001)
            cerr << "Error, origin vector mismatches" << endl;
        if(fabs(img->spacing(ii) - corrspacing[ii]) > 0.000001)
            cerr << "Error, origin vector mismatches" << endl;

		for(size_t jj=0; jj < img->ndim(); jj++) {
			if(fabs(img->direction(ii,jj) - corrdir(ii,jj)) > 0.0001) {
				cerr << "Error, direction matrix mismatches" << endl;
				return -1;
			}
		}
	}

	return 0;
}

