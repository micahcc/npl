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

	MRImageStore<3, double> testimage({10, 23, 39});

	MRImage* testbase = &testimage;

	testimage.dbl({0,0,0}, 0);
	cerr << testimage.dbl({0,0,0}) << endl;
	cerr << testbase->dbl({0,0,0}) << endl;
	
	testimage.dbl({0,0,0}, 10);
	cerr << testimage.dbl({0,0,0}) << endl;
	cerr << testbase->dbl({0,0,0}) << endl;

}
