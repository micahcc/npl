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
 * @brief Interpolate the input array, filling the output array
 *
 * @param isize 	Size of in
 * @param in 		Values to interpolate
 * @param osize 	Size of out
 * @param out 		Output array, filled with interpolated values of in
 */
void interp(int64_t isize, fftw_complex* in, int64_t osize, fftw_complex* out);

/**
 * @brief Fills the input array (chirp) with a chirp of the specified type
 *
 * @param sz 		Size of output array
 * @param chirp 	Output array
 * @param origsz 	Original size, decides maximum frequency reached
 * @param upratio 	Ratio of upsampling performed. This may be different than 
 * 					sz/origsz
 * @param alpha 	Positive term in exp
 * @param beta 		Negative term in exp
 * @param fft 		Whether to fft the output (put it in frequency domain)
 */
void createChirp(int64_t sz, fftw_complex* chirp, int64_t origsz,
		double upratio, double alpha, double beta, bool fft);

/**
 * @brief Comptues the Fractional Fourier transform using FFTW for nlogn
 * performance.
 *
 * The definition of the fractional fourier transform is:
 * \f[
 * F(u) = SUM f(j) exp(-2 PI i a u j / (N+1)
 * \f]
 * where 
 * \f$j = [-N/2,N/2], u = [-N/2, N/2]\f$
 *
 * @param isize size of input/output
 * @param in Input array, may be the same as output, length sz
 * @param out Output array, may be the same as input, length sz
 * @param bsz Buffer size
 * @param a Fraction, 1 = fourier transform, 2 = reverse, 
 * 3 = inverse fourier transform, 4 = identity
 * @param buffer Buffer to do computations in, may be null, in which case new
 * memory will be allocated and deallocated during processing. Note that if
 * the provided buffer is not sufficient size a new buffer will be allocated
 * and deallocated, and a warning will be produced
 * @param nonfft Whether to use brute force method (non-fft)
 *
 */
void fractional_ft(size_t isize, fftw_complex* in, fftw_complex* out, double a,
		size_t bsz = 0, fftw_complex* buffer = NULL, bool nonfft = false);

/**
 * @brief Writes a 2D plot of the complex array where both real and imaginary
 * values are plotted together.
 *
 * @param file Filename base
 * @param insz Size of in array
 * @param in Array of complex values
 */
void writePlotReIm(std::string file, size_t insz, fftw_complex* in);

/**
 * @brief Writes a 2D plot of the complex array where both absolute value and 
 * angle * values are plotted together.
 *
 * @param file Filename base
 * @param insz Size of in array
 * @param in Array of complex values
 */
void writePlotAbsAng(std::string file, size_t insz, fftw_complex* in);

}

#endif
