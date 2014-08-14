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

}
#endif
