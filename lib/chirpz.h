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
 * @file chirpz.cpp Functions for performing the chirpz transform
 *
 *****************************************************************************/

#ifndef CHIRPZ_H
#define CHIRPZ_H

#include <cmath>

namespace npl {

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
		double upratio, double alpha, bool fft);

/**
 * @brief Comptues the chirpzFFT transform using FFTW for n log n performance.
 *
 * This version needs for chirps to already been calculated. This is useful
 * if you are running a large number of inputs with the same alpha
 *
 * @param isize 	Size of input/output
 * @param usize 	Size that we should upsample input to 
 * @param inout		Input array, length sz
 * @param uppadsize	Padded+upsampled array size
 * @param buffer	Complex buffer used for upsampling, size = uppadsize
 * @param negchirp	Chirp defined as exp(-i PI alpha x^2), centered with 
 * 					frequencies ranging: [-(isize-1.)/2.,-(isize-1.)/2.]
 * 					and with size uppadsize. Computed using:
 *
 * 					createChirp(uppadsize, nega_chirp, isize, 
 * 							(double)usize/(double)isize, -alpha, false);
 *
 * @param poschirpF	Chirp defined as F{ exp(-i PI alpha x^2) }, centered with 
 * 					frequencies ranging: [-(isize-1.)/2.,-(isize-1.)/2.]
 * 					and with size uppadsize. Computed using:
 *
 * 					createChirp(uppadsize, posa_chirp, isize, 
 * 							(double)usize/(double)isize, alpha, true);
 */
void chirpzFFT(size_t isize, size_t usize, fftw_complex* inout, 
		size_t uppadsize, fftw_complex* buffer, fftw_complex* negchirp, 
		fftw_complex* poschirpF);

/**
 * @brief Comptues the chirpzFFT transform using FFTW for n log n performance.
 *
 * @param isize Size of input/output
 * @param in Input array, may be the same as out, length sz
 * @param out Output array, may be the same as input, length sz
 * @param alpha Fraction of full space to compute
 */
void chirpzFFT(size_t isize, fftw_complex* in, fftw_complex* out, double a);

/**
 * @brief Performs chirpz transform with a as fractional parameter by N^2 
 * algorithm.
 *
 * @param len	Length of input array
 * @param in	Input Array (length = len)
 * @param out	Output Array (length = len)
 * @param a		Fraction/Alpha To raise exp() term to
 */
void chirpzFT_brute(size_t len, fftw_complex* in, fftw_complex* out, double a);

}

#endif //CHIRPZ_H

