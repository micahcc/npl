#include <iostream>

#include "ndimage.h"

using namespace std;
using namespace npl;

int main()
{
	/* Create an image with: x+y*100+z*10000*/
	size_t sz[3] = {10, 23, 39};
	NDImageStore<3, double> testimage(sz);
	NDImage* oimage = &testimage;

	for(size_t zz=0; zz < oimage->dim(2); zz++) {
		for(size_t yy=0; yy < oimage->dim(1); yy++) {
			for(size_t xx=0; xx < oimage->dim(0); xx++) {
				double pix = xx+yy*100+zz*10000;
				oimage->dbl({xx,yy,zz}, pix);
			}
		}
	}

	/* Write the Image */
	writeNDImage(oimage, "test4.nii", false);

//	/* Read the Image */
//	NDImage* iimage = readNDImage("test4.nii.gz", true);
//
//	/* Check the Image */
//	for(size_t zz=0; zz < iimage->dim(2); zz++) {
//		for(size_t yy=0; yy < iimage->dim(1); yy++) {
//			for(size_t xx=0; xx < iimage->dim(0); xx++) {
//				double pix = xx+yy*100+zz*10000;
//				if(pix != iimage->dbl({xx,yy,zz})) {
//					cerr << "Mismatch!" << endl;
//					return -1;
//				}
//			}
//		}
//	}

	return 0;
}



