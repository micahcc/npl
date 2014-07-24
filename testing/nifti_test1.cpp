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
#include <memory>

#include "mrimage.h"
#include "accessors.h"
#include "mrimage_utils.h"

using namespace std;
using namespace npl;

int main()
{
	{
	/* Create an image with: 0.5+x+y*100+z*10000*/
	auto dblversion = createMRImage({10, 23, 39}, FLOAT64);
	NDAccess<double> acc(dblversion);

	for(int64_t zz=0; zz < dblversion->dim(2); zz++) {
		for(int64_t yy=0; yy < dblversion->dim(1); yy++) {
			for(int64_t xx=0; xx < dblversion->dim(0); xx++) {
				double pix = xx+yy*100+zz*10000+.5;
				acc.set(pix, {xx,yy,zz});
			}
		}
	}

	/* Write the Image */
	dblversion->write("test1a.nii.gz", false);

	/* Create a Cast Copy and write */
	auto intversion = dblversion->copyCast(dblversion->ndim(), 
			dblversion->dim(), INT64);
	dynamic_pointer_cast<MRImage>(intversion)->write("test1b.nii.gz");
	}

	/* Read the Image */
	auto dblversion = readMRImage("test1a.nii.gz", true);
	auto intversion = readMRImage("test1b.nii.gz", true);
	
	NDAccess<int> v1(dblversion);
	NDAccess<int> v2(intversion);
	if(intversion->ndim() != dblversion->ndim())
		return -1;
	if(intversion->dim(0) != dblversion->dim(0) || intversion->dim(1) != 
			dblversion->dim(1) || intversion->dim(2) != dblversion->dim(2)) {
		std::cerr << "Mismatch between dimension of written images!" << std::endl;
		return -1;
	}

	/* Check the Image */
	for(int64_t zz=0; zz < dblversion->dim(2); zz++) {
		for(int64_t yy=0; yy < dblversion->dim(1); yy++) {
			for(int64_t xx=0; xx < dblversion->dim(0); xx++) {
				if(v1[{xx,yy,zz}] != v2[{xx,yy,zz}]) {
					cerr << "Mismatch!" << endl;
					return -1;
				}
			}
		}
	}

	return 0;
}


