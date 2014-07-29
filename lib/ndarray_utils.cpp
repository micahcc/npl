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
#include "iterators.h"
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

/**
 * @brief Performs in-place fft. Note that the input should already be padded
 * and a complex type
 *
 * @param in
 * @param dim
 */
void fft1d(shared_ptr<NDArray> in, size_t dd, bool inverse)
{
	// plan and execute 
	double* indata = fftw_alloc_real(in->dim(dd));
	fftw_complex *data = fftw_alloc_complex(in->dim(dd));
	fftw_plan plan;
	if(inverse) {
		plan = fftw_plan_dft_1d(in->dim(dd), data, data, FFTW_BACKWARD,
				FFTW_MEASURE);
	} else {
		plan = fftw_plan_dft_1d(in->dim(dd), data, data, FFTW_FORWARD,
				FFTW_MEASURE);
	}

	OrderIter<cdouble_t> oit(in);
	oit.setOrder({dd}); 
	OrderConstIter<cdouble_t> iit(in);
	iit.setOrder(oit.getOrder()); // make dd the fastest
	while(!iit.eof() && !oit.eof()) {

		// fill array
		for(size_t ii=0; ii<in->dim(dd); ++iit, ii++) {
			data[ii][0] = (*iit).real();
			data[ii][1] = (*iit).imag();
		}
		
		// execute
		fftw_execute(plan);
		
		// write array
		cdouble_t tmp;
		for(size_t ii=0; ii<in->dim(dd); ++oit, ii++) {
			tmp.real(data[ii][0]);
			tmp.real(data[ii][1]);
			data[ii][0] = (*iit).real();
			data[ii][1] = (*iit).imag();
			oit.set(tmp);
		}
	}

	fftw_fr
}

/**
 * @brief Perform fourier transform on the dimensions specified. Those
 * dimensions will be padded out. The output of this will be a complex double
 *
 * @param in Input image to fourier trnasform
 * @param len
 * @param dim
 * @param inverse perform backward fourier transform
 *
 * @return 
 */
shared_ptr fft(shared_ptr<NDArray> in, size_t len, size_t* dim)
{
	/*
	 * pad out the given dimensions
	 */

	// start with original size and then round up specified dimensions
	cerr << "Input Dimensions: [" << endl;
	for(size_t dd=0; dd<in->ndim(); dd++)
		cerr << in->dim(dd) << ",";
	cerr << "]";
	std::vector<size_t> newsize(in->dim(), in->dim()+in->ndim());
	for(size_t ii=0; ii<len; ii++) {
		if(dim[ii] >= in->ndim()) {
			throw std::out_of_bounds("Error, invalid dimension specified "
					"for fouirer transform");
		}
		newsize[dim[ii]] = round2(newsize[dim[ii]]);
	}

	// create padded image with complex pixel type
	auto out = in->copyCast(newsize.size(), newsize.data(), COMPLEX128);
	cerr << "Padded Dimensions: [" << endl;
	for(size_t dd=0; dd<out->ndim(); dd++)
		cerr << out->dim(dd) << ",";
	cerr << "]";

	for(size_t ii=0; ii<len; ii++) {
		assert(dim[ii] < in->ndim());
		// perform 1D fourier transform

		// plan and execute 
		size_t orig_insz = in->dim(dim[ii]);
		size_t pad_insz = out->dim(dim[ii]);
		size_t osize = pad_insz/2+1;
		
		double* idata = fftw_alloc_real(pad_insz);
		fftw_complex *odata = fftw_alloc_complex(osize);
		
		fftw_plan plan = fftw_plan_dft_r2c_1d(pad_insz, idata, odata, 
					FFTW_FORWARD, FFTW_MEASURE);

		OrderIter<cdouble_t> oit(in);
		oit.setOrder({dd}); 
		OrderConstIter<cdouble_t> iit(in);
		iit.setOrder(oit.getOrder()); // make dd the fastest
		while(!iit.eof() && !oit.eof()) {

			// fill array
			for(size_t ii=0; ii<in->dim(dd); ++iit, ii++) {
				data[ii][0] = (*iit).real();
				data[ii][1] = (*iit).imag();
			}

			// execute
			fftw_execute(plan);

			// write array
			cdouble_t tmp;
			for(size_t ii=0; ii<in->dim(dd); ++oit, ii++) {
				tmp.real(data[ii][0]);
				tmp.real(data[ii][1]);
				data[ii][0] = (*iit).real();
				data[ii][1] = (*iit).imag();
				oit.set(tmp);
			}
		}







// done with 1D fourier transform
		fft1d(out, dim[ii], inverse);
	}

	return out;
}

shared_ptr<NDArray> ppfft(shared_ptr<NDArray> in, size_t len, size_t* dims)
{
	// compute the 3D foureir transform for the entire image (or at least the
	// dimensions specified)
	auto fourier = fft(in, len, dims);

	// compute for each fan
	for(size_t ii=0; ii<len; ii++) {

	}
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

/**
 * @brief Erode an binary array repeatedly
 *
 * @param in Input to erode
 * @param reps Number of radius-1 kernel erosions to perform
 *
 * @return Eroded Image
 */
shared_ptr<NDArray> erode(shared_ptr<NDArray> in, size_t reps)
{
	std::vector<int64_t> index1(in->ndim(), 0);
	std::vector<int64_t> index2(in->ndim(), 0);
	auto prev = in->copy();
	auto out = in->copy();
	for(size_t rr=0; rr<reps; ++rr) {
		std::swap(prev, out);
		
		KernelIter<int> it(prev);
		it.setRadius(1);
		OrderIter<int> oit(out);
		oit.setOrder(it.getOrder());
		// for each pixels neighborhood, smooth neightbors
		for(oit.goBegin(), it.goBegin(); !it.eof(); ++it, ++oit) {
			oit.index(index1.size(), index1.data());
			it.center_index(index2.size(), index2.data());

			// if any of the neighbors are 0, then set to 0
			bool erodeme = false;
			for(size_t ii=0; ii<it.ksize(); ++ii) {
				if(it.offset(ii) == 0) {
					erodeme = true;
				}
			}

			if(erodeme) 
				oit.set(0);
		}
	}

	return out;
}

/**
 * @brief Dilate an binary array repeatedly
 *
 * @param in Input to dilate
 * @param reps Number of radius-1 kernel dilations to perform
 *
 * @return Dilated Image
 */
shared_ptr<NDArray> dilate(shared_ptr<NDArray> in, size_t reps)
{
	std::vector<int64_t> index1(in->ndim(), 0);
	std::vector<int64_t> index2(in->ndim(), 0);
	auto prev = in->copy();
	auto out = in->copy();
	for(size_t rr=0; rr<reps; ++rr) {
		std::swap(prev, out);
		
		KernelIter<int> it(prev);
		it.setRadius(1);
		OrderIter<int> oit(out);
		oit.setOrder(it.getOrder());
		// for each pixels neighborhood, smooth neightbors
		for(oit.goBegin(), it.goBegin(); !it.eof(); ++it, ++oit) {
			oit.index(index1.size(), index1.data());
			it.center_index(index2.size(), index2.data());
			for(size_t ii=0; ii<in->ndim(); ++ii) {
				if(index1[ii] != index2[ii]) {
					throw std::logic_error("Error differece in iteration!");
				}
			}

			// if any of the neighbors are 0, then set to 0
			bool dilateme = false;
			int dilval = 0;
			for(size_t ii=0; ii<it.ksize(); ++ii) {
				if(it.offset(ii) != 0) {
					dilval = it.offset(ii);
					dilateme = true;
				}
			}

			if(dilateme) 
				oit.set(dilval);
		}
	}

	return out;
}


} // npl
#endif  //IMAGE_PROCESSING_H


