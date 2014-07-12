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

int main(int argc, char** argv)
{
	if(argc != 2) {
		cerr << "Need a filename!" << endl;
		return -1;
	}

	/* Create an image with: x+y*100+z*10000*/
	std::vector<size_t> sz({10, 23, 39, 8});
	MRImage* testimg = createMRImage(sz, FLOAT64);

	int64_t ii = 0;
	for(auto iter = testimg->begin(); !iter.isEnd(); ++iter) {
		iter.int64(ii++);
	}

	/* Write the Image */
	writeMRImage(testimg, argv[1]);

	/* Read the Image */
	MRImage* iimage = readMRImage(argv[1], true);
	
	/* Check the Image */
	ii = 0;
	for(auto iter = iimage->begin(); !iter.isEnd(); ++iter) {
		if(iter.int64() != ii++) {
			cerr << "Error, mismatch in read image" << endl;
			return -1;
		}
	}

	cerr << "PASS!" << endl;
	return 0;
}

