#include <iostream>

#include "ndimage.h"

using namespace std;

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
				total += oimage->setD(pix, {xx,yy,zz});
			}
		}
	}

	/* Write the Image */
	writeImage(oimage, "test4.nii.gz");

	/* Read the Image */
	NDImage* iimage = readImage("test4.nii.gz");
	
	/* Check the Image */
	for(size_t zz=0; zz < oimage->dim(2); zz++) {
		for(size_t yy=0; yy < oimage->dim(1); yy++) {
			for(size_t xx=0; xx < oimage->dim(0); xx++) {
				double pix = xx+yy*100+zz*10000;
				if(pix != oimage->getD({xx,yy,zz})) {
					cerr << "Mismatch!" << endl;
					return -1;
				}
			}
		}
	}

	return 0;
}

