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
	shared_ptr<MRImage> img = readMRImage("../../data/quatern_test.nii.gz", true);
	if(!img) {
		std::cerr << "Failed to open image!" << std::endl;
		return -1;
	}

    Eigen::Vector3d corrorigin;
	corrorigin << 122.711868, -93.597809, -42.948792;
    
    Eigen::Vector3d corrspacing;
    corrspacing << .9375, .9375, 3.5;

    Eigen::Matrix3d corrdir;
	corrdir << -1,   0,  -0, 0,   1,0.0262, -0,-0.0262,   1; 

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
			if(fabs(img->direction(ii,jj) - corrdir(ii,jj)) > 0.01) {
				cerr << "Error, direction matrix mismatches" << endl;
				return -1;
			}
		}
	}

	return 0;
}


