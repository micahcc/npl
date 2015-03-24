/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file nifti_orient_test1.cpp
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
	ptr<MRImage> img = readMRImage("../../data/test_nifti1.nii.gz", true);
	if(!img) {
		std::cerr << "Failed to open image!" << std::endl;
		return -1;
	}

    Eigen::Vector4d corrorigin;
    corrorigin << 1.3, 75, 9, 0;

    Eigen::Vector4d corrspacing;
    corrspacing << 4.3, 4.7, 1.2, .3;

    Eigen::Matrix4d corrdir;
    corrdir <<
            -0.16000, -0.98424,  0.07533, 0,
            0.62424, -0.16000, -0.76467, 0,
            0.76467, -0.07533,  0.64000, 0,
                  0,        0,        0, 1;

	cerr << "Correct Direction:\n" << corrdir << endl;
	cerr << "Image Direction:\n" << img->getDirection() << endl;

    cerr << "Correct Spacing:\n" << corrspacing << endl;
	cerr << "Image Spacing:\n" << img->getSpacing() << endl;

    cerr << "Correct Origin:\n" << corrorigin << endl;
	cerr << "Image Origin:\n" << img->getOrigin() << endl;

	for(size_t ii=0; ii < img->ndim(); ii++) {
        if(fabs(img->origin(ii) - corrorigin[ii]) > 0.001) {
            cerr << "Error, origin vector mismatches" << endl;
            cerr << "Correct: " << corrorigin.transpose() << endl;
            cerr << "Image's: " << img->getOrigin().transpose() << endl;
            return -1;
        }
        if(fabs(img->spacing(ii) - corrspacing[ii]) > 0.001)  {
            cerr << "Error, Spacing vector mismatches" << endl;
            cerr << "Correct: " << corrspacing.transpose() << endl;
            cerr << "Image's: " << img->getSpacing().transpose() << endl;
            return -1;
        }

		for(size_t jj=0; jj < img->ndim(); jj++) {
			if(fabs(img->direction(ii,jj) - corrdir(ii,jj)) > 0.001) {
				cerr << "Error, direction matrix mismatches" << endl;
				cerr << "\n" << img->getDirection() << "\nvs.";
				cerr << "\n" << img->getDirection() << "\n";
				return -1;
			}
		}
	}

	return 0;
}

