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



