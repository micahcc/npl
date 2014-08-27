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

#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>
#include <complex>

#include "fftw3.h"

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
 * @param center	Whether to center, or start at 0 
 * @param fft 		Whether to fft the output (put it in frequency domain)
 */
void createChirp(int64_t sz, fftw_complex* chirp, int64_t origsz,
		double upratio, double alpha, bool center, bool fft);

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
 * @param prechirp 	Pre-multiply chirp.
 * @param convchirp	Chirp that we need to convolve with
 * @param postchirp	Post-multiply chirp.
 * @param debug	whether to write out diagnostic plots
 */
void chirpzFFT(size_t isize, size_t usize, fftw_complex* inout, 
		size_t uppadsize, fftw_complex* buffer, fftw_complex* prechirp,
		fftw_complex* convchirp, fftw_complex* postchirp, bool debug = false);
		

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
 * @param prechirp 	Pre-multiply chirp.
 * @param convchirp	Chirp that we need to convolve with
 * @param postchirp	Post-multiply chirp.
 * @param debug	whether to write out diagnostic plots
 */
void chirpzFFT(size_t isize, size_t usize, fftw_complex* inout, 
		fftw_complex* buffer, bool debug = false);

/**
 * @brief Comptues the chirpzFFT transform using FFTW for n log n performance.
 *
 * @param isize Size of input/output
 * @param in Input array, may be the same as out, length sz
 * @param out Output array, may be the same as input, length sz
 * @param alpha Fraction of full space to compute
 * @param debug	whether to write out diagnostic plots
 */
void chirpzFFT(size_t isize, fftw_complex* in, fftw_complex* out, double a, 
		bool debug = false);

/**
 * @brief Performs chirpz transform with a as fractional parameter by N^2 
 * algorithm.
 *
 * @param len	Length of input array
 * @param in	Input Array (length = len)
 * @param out	Output Array (length = len)
 * @param a		Fraction/Alpha To raise exp() term to
 */
void chirpzFT_brute2(size_t len, fftw_complex* in, fftw_complex* out, double a, 
		bool debug = false);

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


/**
 * @brief Performs the chirpz-transform using linear intpolation and the FFT.
 * For greater speed pre-allocate the buffer and use the other chirpzFT_zoom
 *
 * @param isize Input size
 * @param in	Input line
 * @param out	Output line
 * @param a		Zoom factor (alpha)
 */
void chirpzFT_zoom(size_t isize, fftw_complex* in, fftw_complex* out, 
		double a);

/**
 * @brief Performs the chirpz-transform using linear intpolation and teh FFT.
 *
 * @param isize 	Input size
 * @param in		Input line
 * @param out		Output line
 * @param buffer	Buffer, should be size isize
 * @param a			Zoom factor (alpha)
 */
void chirpzFT_zoom(size_t isize, fftw_complex* in, fftw_complex* out, 
		fftw_complex* buffer, double a);

/**
 * @brief Plots an array of complex points with the Real and Imaginary Parts
 *
 * @param file	Filename
 * @param insz	Size of in
 * @param in	Array input
 */
void writePlotAbsAng(std::string file, size_t insz, fftw_complex* in);

void writePlotReIm(std::string file, const std::vector<std::complex<double>>& in);

/**
 * @brief Plots an array of complex points with the Real and Imaginary Parts
 *
 * @param file	Filename
 * @param insz	Size of in
 * @param in	Array input
 */
void writePlotReIm(std::string file, size_t insz, fftw_complex* in);

}

#endif //CHIRPZ_H

