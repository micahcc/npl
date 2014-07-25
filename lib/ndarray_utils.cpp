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

#include "ndarray_utils.h"
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

shared_ptr<NDArray> ifft(shared_ptr<const NDArray> in)
{
	// pad 
	std::vector<int> newsizeI(in->ndim());
	std::vector<size_t> newsizeZ(in->ndim());
	size_t npixel = 1;
	for(size_t ii=0; ii<in->ndim(); ii++) {
		newsizeI[ii] = round2(in->dim(ii));
		newsizeZ[ii] = round2(in->dim(ii));
		npixel *= newsizeZ[ii];
	}
	
	// create output image
	auto out = in->copyCast(in->ndim(), newsizeZ.data(), COMPLEX128);

	// plan and execute DFT in place
	fftw_complex* idata = (fftw_complex*)in->data();
	fftw_complex* odata = (fftw_complex*)out->data();
	fftw_plan plan = fftw_plan_dft(in->ndim(), newsizeI.data(), idata, odata, 
				FFTW_BACKWARD, FFTW_MEASURE);
	fftw_execute(plan);

	return out;
}

shared_ptr<NDArray> fft(shared_ptr<const NDArray> in)
{
	// pad 
	std::vector<int> newsize(in->ndim());
	std::vector<size_t> newsize2(in->ndim());
	size_t npixel = 1;
	for(size_t ii=0; ii<in->ndim(); ii++) {
		newsize[ii] = round2(in->dim(ii));
		newsize2[ii] = round2(in->dim(ii));
		npixel *= newsize[ii];
	}
	
	// create output image
	auto out = in->copyCast(in->ndim(), newsize2.data(), COMPLEX128);

	// plan and execute DFT in place
	fftw_complex* data = (fftw_complex*)out->data();
	fftw_plan plan = fftw_plan_dft(newsize.size(), newsize.data(), data, data, 
				FFTW_FORWARD, FFTW_MEASURE);
	fftw_execute(plan);

	return out;
}


/**
 * @brief Returns whether two NDArrays have the same dimensions, and therefore
 * can be element-by-element compared/operated on. elL is set to true if left
 * is elevatable to right (ie all dimensions match or are missing or are unary).
 * elR is the same but for the right. 
 *
 * Strictly R is elevatable if all dimensions that don't match are missing or 1
 * Strictly L is elevatable if all dimensions that don't match are missing or 1
 *
 * Examples of *elR = true (return false):
 *
 * left = [10, 20, 1]
 * right = [10, 20, 39]
 *
 * left = [10]
 * right = [10, 20, 39]
 *
 * Examples where neither elR or elL (returns true):
 *
 * left = [10, 20, 39]
 * right = [10, 20, 39]
 *
 * Examples where neither elR or elL (returns false):
 *
 * left = [10, 20, 9]
 * right = [10, 20, 39]
 *
 * left = [10, 1, 9]
 * right = [10, 20, 1]
 *
 * @param left	NDArray input
 * @param right NDArray input
 * @param elL Whether left is elevatable to right (see description of function)
 * @param elR Whether right is elevatable to left (see description of function)
 *
 * @return 
 */
bool comparable(const NDArray* left, const NDArray* right, bool* elL, bool* elR)
{
	bool ret = true;

	bool rightEL = true;
	bool leftEL = true;

	for(size_t ii = 0; ii < left->ndim(); ii++) {
		if(ii < right->ndim()) {
			if(right->dim(ii) != left->dim(ii)) {
				ret = false;
				// if not 1, then R is not elevateable
				if(right->dim(ii) != 1)
					rightEL = false;
			}
		}
	}
	
	for(size_t ii = 0; ii < right->ndim(); ii++) {
		if(ii < left->ndim()) {
			if(right->dim(ii) != left->dim(ii)) {
				ret = false;
				// if not 1, then R is not elevateable
				if(left->dim(ii) != 1)
					leftEL = false;
			}
		}
	}
	
	if(ret) {
		leftEL = false;
		rightEL = false;
	}

	if(elL) *elL = leftEL;
	if(elR) *elR = rightEL;

	return ret;
}

} // npl
#endif  //IMAGE_PROCESSING_H


