/*******************************************************************************
This file is part of Neural Program Library (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neural Program Library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The Neural Programs and Libraries are distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the Neural Programs Library.  If not, see <http://www.gnu.org/licenses/>.
*******************************************************************************/

#include <iostream>

#include "mrimage.h"

using namespace std;
using namespace npl;

int main()
{
	/* Read the Image */
	MRImage* img = readMRImage("../../testing/test_nifti2.nii.gz", true);
	if(!img) {
		std::cerr << "Failed to open image!" << std::endl;
		return -1;
	}

	std::vector<double> aff({
				-0.480000,  -33.037640,  	-8.316368,  	0,	1.3,
				-0.051215,  -32.900002,  	8.567265,  		0,	75,
				2.960908, 	-5.924880,  	-1.199999, 		0,	9,
				0.000000, 	0,				0,				3,	0,
				0.000000, 	0,				0,				0,	1});
	Matrix<5,5> corraff(aff);

	cerr << "Correct Affine: " << endl << corraff << endl;
	cerr << "Image Affine: " << img->affine() << endl;

	for(size_t ii=0; ii < img->ndim()+1; ii++) {
		for(size_t jj=0; jj < img->ndim()+1; jj++) {
			if(fabs(img->affine()(ii,jj) - corraff(ii,jj)) > 0.00001) {
				cerr << "Error, affine matrix mismatches" << endl;
				return -1;
			}
		}
	}

	return 0;
}



