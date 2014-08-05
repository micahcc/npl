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
 * @file nifti_orient_test1.cpp
 *
 *****************************************************************************/

#include <iostream>

#include "mrimage.h"
#include "mrimage_utils.h"

using namespace std;
using namespace npl;

int main()
{
	/* Read the Image */
	shared_ptr<MRImage> img = readMRImage("../../testing/test_nifti1.nii.gz", true);
	if(!img) {
		std::cerr << "Failed to open image!" << std::endl;
		return -1;
	}

	std::vector<double> aff({
			-0.688,	-4.625922,	0.090391,	0,	1.3,
			2.684227,	-0.752000,	-0.917609,	0,	75.000000,	
			3.288097,	-0.354033,	0.768000,	0,	9.000000,
					0,			0,			0,	0.3,	0,	
					0,			0,			0,	0,	1});
	Matrix<5,5> corraff(aff);

	cerr << "Correct Affine: " << endl << corraff << endl;
	cerr << "Image Affine: " << img->affine() << endl;

	for(size_t ii=0; ii < img->ndim()+1; ii++) {
		for(size_t jj=0; jj < img->ndim()+1; jj++) {
			if(fabs(img->affine()(ii,jj) - corraff(ii,jj)) > 0.000001) {
				cerr << "Error, affine matrix mismatches" << endl;
				return -1;
			}
		}
	}

	return 0;
}



