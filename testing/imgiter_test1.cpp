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
#include "mrimage_utils.h"
#include "iterators.h"

using namespace std;
using namespace npl;

int main()
{
	auto img = readMRImage("../../data/grad_imag.nii.gz");
	OrderIter<int> it(img);
	it.setOrder({0,1,2});
	it.goBegin();
	for(size_t ii=1; ii<=125; ii++, ++it) {
		if(*it != ii) {
			cerr << "Difference between read image and theoretical image" << endl;
			return -1;
		}
	}
	
	it.setOrder({}, true);
	it.goBegin();
	for(size_t ii=1; ii<=125; ii++, ++it) {
		if(*it != ii) {
			cerr << "Difference between read image and theoretical image" 
				" when using reversed default order" << endl;
			return -1;
		}
	}
	
	return 0;
}

