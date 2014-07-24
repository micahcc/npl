/*******************************************************************************
This file is part of Neuro Programs and Libraries (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neuro Programs and Libraries are free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

The Neural Programs and Libraries are distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the Neural Programs Library.  If not, see <http://www.gnu.org/licenses/>.
*******************************************************************************/

#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include "nd_algos.h"
#include "ndarray.h"
#include "npltypes.h"
#include "matrix.h"
#include "fftw3.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <memory>

namespace npl {

using std::vector;
using std::shared_ptr;

int hob (int num)
{
	if (!num)
		return 0;

	int ret = 1;

	while (num >>= 1)
		ret <<= 1;

	return ret;
}

int64_t round2(int64_t in)
{
	return (hob(in) << 1);
}

//shared_ptr<NDArray> fft(shared_ptr<const NDArray> in)
//{
//	// pad 
//	std::vector<int> newsize(in->ndim());
//	size_t npixel = 1;
//	for(size_t ii=0; ii<in->ndim(); ii++) {
//		newsize[ii] = round2(in->dim(ii));
//		npixel *= newsize[ii];
//	}
//	
//	auto data = new fftw_complex[npixel];
//
//	// fill data
//	OrderConstIter<cdouble_t> it(in);
//	
//
//	for(size_t ii=0; ii<;;) {
//
//		// TODO
//	}
//
//	fftw_plan plan = fftw_plan_dft(newsize.size(), newsize.data(), data, data, 
//			1, FFTW_MEASURE);
//	
//	// copy data
//	
//}

} // npl
#endif  //IMAGE_PROCESSING_H


