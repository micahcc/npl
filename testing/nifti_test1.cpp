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
	/* Create an image with: x+y*100+z*10000*/
	MRImageStore<3, double> testimage({10, 23, 39});
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
	writeMRImage(oimage, "test4.nii", false);

//	/* Read the Image */
//	MRImage* iimage = readMRImage("test4.nii.gz", true);
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



