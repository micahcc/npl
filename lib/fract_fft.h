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
 * @file fract_fft.h
 *
 *****************************************************************************/

/******************************************************************************
 * @file fract_fft.h
 * @brief Fractional fourier transform based on FFT
 ******************************************************************************/

#ifndef FRACT_FFT
#define FRACT_FFT

#include <string>
#include "fftw3.h"

namespace npl {

/**
 * @brief Comptues the Fractional Fourier transform using FFTW for nlogn
 * performance.
 *
 * @param isize size of input/output
 * @param in Input array, may be the same as output, length sz
 * @param out Output array, may be the same as input, length sz
 * @param a Fraction, 1 = fourier transform, 2 = reverse, 
 * 3 = inverse fourier transform, 4 = identity
 * @param Buffer size
 * @param buffer Buffer to do computations in, may be null, in which case new
 * memory will be allocated and deallocated during processing. Note that if
 * the provided buffer is not sufficient size a new buffer will be allocated
 * and deallocated, and a warning will be produced. 4x the padded value is
 * needed, which means this value should be around 16x sz
 * @param nonfft
 */
void fractional_ft(size_t sz, fftw_complex* in, fftw_complex* out, double a,
		size_t bsz = 0, fftw_complex* buffer = NULL, bool nonfft = false);

/**
 * @brief Comptues the power fractional fourier transform using FFTW for n log
 * n performance. 
 *
 * Definition:
 *
 * \hat{I}(\omega) = \Sum^{N/2}_{k=-N/2} I(k) \exp(-2 \pi i \alpha \omega k)
 *
 * @param isize 	Size of input/output
 * @param in 		Input array, may be the same as out, length sz
 * @param out 		Output array, may be the same as input, length sz
 * @param alpha 	Fraction of full space to compute
 * @param bsz 		Buffer size, if NULL or the value is less than the needed buffer,
 * 					a realocation will occur. If this is non-null then the
 * 					length of the new array will be placed in the variable.
 * @param buffer 	Buffer to do computations in, may be null. If not null,
 * 					then it is assumed that bsz contains the length of this
 * 					array. If the buffer is sufficient size (around 16x input),
 * 					then this will be used, otherwise new memory is allocated.
 * 					If new memory is allocated and this and bsz are non-null,
 * 					the bsz and this will be updated with the size and address
 * 					of the new memory allocated. If this points to non-null
 * 					then fftw_free() will be called on the given address. 
 * @param nonfft
 */
void powerFFT(size_t sz, fftw_complex* in, fftw_complex* out, double a,
		size_t* bsz = 0, fftw_complex** buffer = NULL);

void writePlotReIm(std::string reFile, std::string imFile, size_t insz,
		fftw_complex* in);

void writePlotAbsAng(std::string absFile, std::string angFile, size_t insz,
		fftw_complex* in);

}

#endif
