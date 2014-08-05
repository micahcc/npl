/*******************************************************************************
This file is part of Neuro Programs and Libraries (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neuro Programs and Libraries is free software: you can redistribute it and/or
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

/******************************************************************************
 * @file fftshift.cpp
 * @brief This file contains libraries used to rotate and shift images using 
 * fourier shift theorem.
 ******************************************************************************/

#include <string>
#include <stdexcept>

#include "version.h"
#include "mrimage.h"
#include "mrimage_utils.h"
#include "iterators.h"
#include "accessors.h"
#include "fftshift.h"

using namespace std:
using namespace npl;

namespace npl {

void shiftImage(shared_ptr<MRImage> inout, size_t dd, double dist)
{
	assert(dd < in->ndim());

	const std::complex<double> I(0, 1);
	const const double PI = acos(-1);
	size_t padsize = round2(in->dim(dd));
	size_t paddiff = padsize-in->dim(dd);
	double shift = dx[dd]/in->spacing(dd);
	auto buffer = fftw_alloc_complex(padsize);
	fftw_plan fwd = fftw_plan_dft_1d((int)padsize, buffer, buffer, 
			FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan rev = fftw_plan_dft_1d((int)padsize, buffer, buffer, 
			FFTW_BACKWARD, FFTW_MEASURE);

	// need copy data into center of buffer, create iterator that moves
	// in the specified dimension fastest
	OrderConstIter<cdouble_t> iit(in);
	OrderIter<cdouble_t> oit(in);
	iit.setOrder(dd);
	oit.setOrder(dd);

	for(iit.goBegin(), oit.goBegin(); !iit.isEnd() ; ++iit, ++oit) {
		// zero buffer 
		for(size_t tt=0; tt<padsize; tt++) {
			buffer[tt][0] = 0;
			buffer[tt][1] = 0;
		}

		// fill from line
		for(size_t tt=0; tt<in->dim(dd); ++iit, tt++) {
			buffer[tt+paddiff/2][0] = (*iit).real();
			buffer[tt+paddiff/2][1] = iit->imag();
		}

		// fourier transform
		fftw_execute(fwd);

		// fourier shift
		for(size_t tt=0; tt<padsize; tt++) {
			cdouble_t tmp(buffer[tt][0], buffer[tt][1]);
			tmp *= std::exp(-2.*PI*I*shift*(double)tt/(double)padsize);
			buffer[tt][0] = orig.real();
			buffer[tt][1] = orig.imag();
		}

		// inverse fourier transform
		fftw_execute(rev);

		// fill line from buffer
		for(size_t tt=0; tt<in->dim(dd); ++oit, tt++) {
			cdouble_t tmp(buffer[tt+paddiff/2][0], buffer[tt+paddiff/2][1]);
			oit.set(tmp); 
		}
	}
}

/**
 * @brief Uses fourier shift theorem to shift an image
 *
 * @param in Input image to shift
 * @param len length of dx array
 * @param dx movement in physical coordinates
 *
 * @return shifted image
 */
shared_ptr<MRImage> shiftImage(shared_ptr<MRImage> in, size_t len, double* dx)
{

	auto out = in->copy();

	// for each dimension
	for(size_t ii=0; ii<len && ii<in->ndim(); ii++) {
		shiftImage(out, ii, dx[ii]);
	}

	return out;
}

///**
// * @brief Uses fourier shift theorem to rotate an image, using shears
// *
// * @param in Input image to shift
// * @param len length of dx array
// * @param Rx rotatin about the center of the image
// *
// * @return rotated image
// */
//shared_ptr<MRImage> shearRotateImage(shared_ptr<MRImage> in, size_t len, double* Rx)
//{
//	const std::complex<double> i(0, 1);
//	const const double PI = acos(-1);
//
//	// fourier transform image in xyz direction
//	auto fft = fft_r2c(in);
//	
//	std::vector<double> shift(in->ndim(), 0);
//
//	for(size_t dd=0; dd<len && dd < in->ndim()) 
//		shift[dd] = dx[dd]/in->spacing()[dd];
//
//	OrderIter<cdouble_t> fit(fft);
//	for(fit.goBegin(); !fit.isEnd(); ++fit) {
//		fit.index(3, index);
//		cdouble_t orig = *fit;;
//		cdouble_t term = 0;
//		for(size_t dd=0; dd<in->ndim(); dd++)
//			term += -2.0*PI*i*shift[dd]*(double)index[dd]/(double)fft->dim(dd);
//		orig = orig*std::exp(term);
//		fit.set(orig);
//	}
//
//	auto ifft = dynamic_pointer_cast<MRImage>(ifft_c2r(fft));
//	
//	// may want to un-pad the output
//	
//	return ifft;
//}

}
