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
#include <cmath>
#include <complex>
#include <stdexcept>

#include "version.h"
#include "mrimage.h"
#include "mrimage_utils.h"
#include "iterators.h"
#include "accessors.h"
#include "fftshift.h"

using namespace std;
using namespace npl;

namespace npl {

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
