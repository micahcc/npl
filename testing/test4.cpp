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
 * @file test4.cpp
 *
 *****************************************************************************/

#include <iostream>

#include "mrimage.h"

using namespace std;
using namespace npl;

int main()
{
	/* Create an image with: x+y*100+z*10000*/
	size_t sz[3] = {10, 23, 39};
	MRImageStore<3, double> testimage(sz);
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
	writeMRImage(oimage, "test4.nii.gz");

	/* Read the Image */
	MRImage* iimage = readMRImage("test4.nii.gz", true);
	
	/* Check the Image */
	for(size_t zz=0; zz < iimage->dim(2); zz++) {
		for(size_t yy=0; yy < iimage->dim(1); yy++) {
			for(size_t xx=0; xx < iimage->dim(0); xx++) {
				double pix = xx+yy*100+zz*10000;
				if(pix != iimage->dbl({xx,yy,zz})) {
					cerr << "Mismatch!" << endl;
					return -1;
				}
			}
		}
	}

	cerr << "PASS!" << endl;
	return 0;
}

