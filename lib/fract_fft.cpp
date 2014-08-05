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
 * @file fract_fft.cpp
 *
 *****************************************************************************/

/******************************************************************************
 * @file fract_fft.cpp
 * @brief Fractional fourier transform based on FFT
 ******************************************************************************/

#include "fract_fft.h"
#include "utility.h"

#include <cstdlib>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <list>
#include <complex>

//#define DEBUG

using std::complex;

namespace npl {

/**
 * @brief Provides a list of the prime-fractors of the input number
 *
 * @param f input number
 *
 * @return list of factors
 */
std::list<int64_t> factor(int64_t f)
{
	std::list<int64_t> factors;
	for(int64_t ii = 2; ii<=f; ii++) {
		while(f % ii == 0) {
			f = f/ii;
			factors.push_back(ii);
		}
	}

	return factors;
}

/**
 * @brief Rounds a number up to the nearest number that can be broken down into
 * 3,5,7
 *
 * @param in Input number
 *
 * @return Next number up that matches the requirement
 */
int64_t round357(int64_t in)
{
	// make it odd
	if(in %2 == 0)
		in++;

	bool acceptable = false;
	while(!acceptable) {
		acceptable = true;
		in += 2;

		// check the factors
		auto factors = factor(in);
		for(auto f : factors) {
			if(f != 3 && f != 5 && f != 7) {
				acceptable = false;
				break;
			}
		}
	}

	return in;
}


/**
 * @brief Fills the input array (chirp) with a chirp of the specified type
 *
 * @param sz Size of output array
 * @param chirp Output array
 * @param origsz Original size, decides maximum frequency reached
 * @param upratio Ratio of upsampling performed. This may be different than
 * sz/origsz
 * @param alpha Positive term in exp
 * @param beta Negative term in exp
 * @param fft Whether to fft the output (put it in frequency domain)
 */
void createChirp(int64_t sz, fftw_complex* chirp, int64_t origsz,
		double upratio, double alpha, double beta, bool fft)
{
	assert(sz%2==1);
	const double PI = acos(-1);
	const complex<double> I(0,1);
	
	auto fwd_plan = fftw_plan_dft_1d((int)sz, chirp, chirp, FFTW_FORWARD,
				FFTW_MEASURE | FFTW_PRESERVE_INPUT);

	for(int64_t ii=-sz/2; ii<=sz/2; ii++) {
		double ff = ((double)ii)/upratio;
		auto tmp = std::exp(I*PI*(alpha-beta)*ff*ff/(double)origsz);
		chirp[ii+sz/2][0] = tmp.real();
		chirp[ii+sz/2][1] = tmp.imag();
	}
	
	if(fft) {
		fftw_execute(fwd_plan);
		double norm = sqrt(1./sz);
		for(size_t ii=0; ii<sz; ii++) {
			chirp[ii][0] *= norm;
			chirp[ii][1] *= norm;
		}
	}

	fftw_destroy_plan(fwd_plan);
}

/**
 * @brief Lanczos window, used for sampling
 *
 * @param v locaiton
 * @param a radius
 *
 * @return weight
 */
double lanczos(double v, double a)
{
	const double PI = acos(-1);
	if(v == 0)
		return 1;
	else if(abs(v) < a) {
		return a*sin(PI*v)*sin(PI*v/a)/(PI*PI*v*v);
	} else {
		return 0;
	}
}

/**
 * @brief Interpolate the input array, filling the output array
 *
 * @param isize Size of in
 * @param in Values to interpolate
 * @param osize Size of out
 * @param out Output array, filled with interpolated values of in
 */
void interp(int64_t isize, fftw_complex* in, int64_t osize, fftw_complex* out)
{
	// fill/average pad
	int64_t radius = 3;
	double ratio = (double)(isize)/(double)osize;
	
	// copy/center
	for(size_t oo=0; oo<osize; oo++) {
		double cii = ratio*oo;
		int64_t center = round(cii);

		complex<double> sum = 0;
		for(int64_t ii=center-radius; ii<=center+radius; ii++) {
			if(ii>=0 && ii<isize) {
				complex<double> tmp(in[ii][0], in[ii][1]);
				sum += lanczos(ii-cii, radius)*tmp;
			}
		}
		out[oo][0] = sum.real();
		out[oo][1] = sum.imag();
	}
}

/**
 * @brief Does the same thing as fractional_fft (also limited to 0.5 <= a <=
 * 1.5) but does NOT use the fourier transform.
 *
 * @param isize Size of input array
 * @param usize Size of upsampled input
 * @param uppadsize Size of padded and upsampled input
 * @param inout Array used for input and output
 * @param buffer Buffer which may be preallocated
 * @param a Fraction of fourier transform to perform
 */
void fractional_brute_ft(int64_t isize, int64_t usize, int64_t uppadsize,
		fftw_complex* inout, fftw_complex* buffer, double a)
{
	assert(a <= 1.5 && a>= 0.5);
	assert(uppadsize%2 != 0);
	assert(usize%2 != 0);
	
	const double PI = acos(-1);
	const complex<double> I(0,1);
	double phi = a*PI/2;
	complex<double> A_phi = std::exp(-I*PI/4.+I*phi/2.) / (usize*sqrt(sin(phi)));
	double alpha = 1./tan(phi);
	double beta = 1./sin(phi);
	if(a == 1) {
		alpha = 0;
		beta = 1;
	}

	// zero
	for(size_t ii=0; ii<uppadsize*3; ii++) {
		buffer[ii][0] = 0;
		buffer[ii][1] = 0;
	}

	fftw_complex* upsampled = &buffer[0];
	fftw_complex* sigbuff = &buffer[usize];
	fftw_complex* ab_chirp = &buffer[uppadsize];
	fftw_complex* b_chirp = &buffer[uppadsize*2];

	// pre-compute chirps
	createChirp(uppadsize, ab_chirp, isize, (double)usize/(double)isize,
			alpha, beta, false);
	createChirp(uppadsize, b_chirp, isize, (double)usize/(double)isize,
			beta, 0, false);
	
	interp(isize, inout, usize, upsampled);
	
	// pre-multiply
	for(int64_t nn = -usize/2; nn<=usize/2; nn++) {
		complex<double> tmp1(ab_chirp[nn+uppadsize/2][0],
				ab_chirp[nn+uppadsize/2][1]);
		complex<double> tmp2(upsampled[nn+usize/2][0],
				upsampled[nn+usize/2][1]);
		tmp1 *= tmp2;
		upsampled[nn+usize/2][0] = tmp1.real();
		upsampled[nn+usize/2][1] = tmp1.imag();
	}
#ifdef DEBUG
	{
		std::vector<double> tmp(usize);
		for(size_t ii=0; ii<usize; ii++)
			tmp[ii] = upsampled[ii][0];
		writePlot("brute_premult.tga", tmp);
	}
#endif //DEBUG

#ifdef DEBUG
	{
		std::vector<double> tmp(uppadsize);
		for(size_t ii=0; ii<uppadsize; ii++)
			tmp[ii] = b_chirp[ii][0];
		writePlot("brute_b_chirp.tga", tmp);
	}
#endif //DEBUG
//
	
	// multiply
	/*
	 * convolve
	 */
	for(int64_t mm = -usize/2; mm<=usize/2; mm++) {
		sigbuff[mm+usize/2][0] = 0;
		sigbuff[mm+usize/2][1] = 0;

		for(int64_t nn = -usize/2; nn<= usize/2; nn++) {
			complex<double> tmp1(b_chirp[mm-nn+uppadsize/2][0],
					b_chirp[mm-nn+uppadsize/2][1]);
			complex<double> tmp2(upsampled[nn+usize/2][0],
					upsampled[nn+usize/2][1]);
			tmp1 = tmp1*tmp2;

			sigbuff[mm+usize/2][0] += tmp1.real();
			sigbuff[mm+usize/2][1] += tmp1.imag();
		}
	}

#ifdef DEBUG
	{
		std::vector<double> tmp(usize);
		for(size_t ii=0; ii<usize; ii++)
			tmp[ii] = sqrt(sigbuff[ii][0]*sigbuff[ii][0]+
					sigbuff[ii][1]*sigbuff[ii][1]);
		writePlot("brute_convolve.tga", tmp);
	}
#endif //DEBUG
	
	// post-multiply
	for(int64_t ii=-usize/2; ii<=usize/2; ii++) {
		complex<double> tmp1(ab_chirp[ii+uppadsize/2][0],
				ab_chirp[ii+uppadsize/2][1]);
		complex<double> tmp2(sigbuff[ii+usize/2][0],
				sigbuff[ii+usize/2][1]);
		tmp1 = tmp1*tmp2*A_phi;
		upsampled[ii+usize/2][0] = tmp1.real();
		upsampled[ii+usize/2][1] = tmp1.imag();
	}
	
	interp(usize, upsampled, isize, inout);
}

void fractional_fft(int64_t isize, int64_t usize, int64_t uppadsize,
		fftw_complex* inout, fftw_complex* buffer, double a)
{
	assert(usize%2 == 1);
	assert(uppadsize%2 == 1);

	const double PI = acos(-1);
	const complex<double> I(0,1);
	double phi = a*PI/2;
	complex<double> A_phi = std::exp(-I*PI/4.+I*phi/2.) / (usize*sqrt(sin(phi)));
	double alpha = 1./tan(phi);
	double beta = 1./sin(phi);
	if(a == 1) {
		alpha = 0;
		beta = 1;
	}

	// zero
	for(size_t ii=0; ii<uppadsize*3; ii++) {
		buffer[ii][0] = 0;
		buffer[ii][1] = 0;
	}

	fftw_complex* sigbuff = &buffer[0]; // note the overlap with upsampled
	fftw_complex* upsampled = &buffer[uppadsize/2-usize/2];
	fftw_complex* ab_chirp = &buffer[uppadsize];
	fftw_complex* b_chirp = &buffer[uppadsize*2];

	// create buffers and plans
	createChirp(uppadsize, ab_chirp, isize, (double)usize/(double)isize, alpha,
			beta, false);
	createChirp(uppadsize, b_chirp, isize, (double)usize/(double)isize,
			beta, 0, true);

#ifdef DEBUG
	{
		std::vector<double> tmp(uppadsize);
		for(size_t ii=0; ii<uppadsize; ii++)
			tmp[ii] = ab_chirp[ii][0];
		writePlot("fft_abchirp.tga", tmp);
	}
#endif //DEBUG

#ifdef DEBUG
	{
		std::vector<double> tmp(uppadsize);
		for(size_t ii=0; ii<uppadsize; ii++)
			tmp[ii] = b_chirp[ii][0];
		writePlot("fft_bchirp.tga", tmp);
	}
#endif //DEBUG

	fftw_plan sigbuff_plan_fwd = fftw_plan_dft_1d(uppadsize, sigbuff, sigbuff,
			FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan sigbuff_plan_rev = fftw_plan_dft_1d(uppadsize, sigbuff, sigbuff,
			FFTW_BACKWARD, FFTW_MEASURE);

	// upsample input
	interp(isize, inout, usize, upsampled);
#ifdef DEBUG
	{
		std::vector<double> tmp(usize);
		for(size_t ii=0; ii<usize; ii++)
			tmp[ii] = upsampled[ii][0];
		writePlot("upin.tga", tmp);
	}
#endif //DEBUG
	
	// pre-multiply
	for(int64_t nn = -usize/2; nn<=usize/2; nn++) {
		complex<double> tmp1(ab_chirp[nn+uppadsize/2][0],
				ab_chirp[nn+uppadsize/2][1]);
		complex<double> tmp2(upsampled[nn+usize/2][0],
				upsampled[nn+usize/2][1]);
		tmp1 *= tmp2;
		upsampled[nn+usize/2][0] = tmp1.real();
		upsampled[nn+usize/2][1] = tmp1.imag();
	}
#ifdef DEBUG
	{
		std::vector<double> tmp(usize);
		for(size_t ii=0; ii<usize; ii++)
			tmp[ii] = upsampled[ii][0];
		writePlot("fft_premult.tga", tmp);
	}
#endif //DEBUG

	/*
	 * convolve
	 */
	fftw_execute(sigbuff_plan_fwd);

	// not 100% clear on why sqrt works here, might be that the sqrt should be
	// b_chirp fft
	double normfactor = sqrt(1./(uppadsize));
	for(size_t ii=0; ii<uppadsize; ii++) {
		complex<double> tmp1(sigbuff[ii][0], sigbuff[ii][1]);
		complex<double> tmp2(b_chirp[ii][0], b_chirp[ii][1]);
		tmp1 *= tmp2*normfactor;
		sigbuff[ii][0] = tmp1.real();
		sigbuff[ii][1] = tmp1.imag();
	}
	fftw_execute(sigbuff_plan_rev);

#ifdef DEBUG
	{
		std::vector<double> tmp(uppadsize);
		for(size_t ii=0; ii<uppadsize; ii++)
			tmp[ii] = sqrt(sigbuff[ii][0]*sigbuff[ii][0] +
					sigbuff[ii][1]*sigbuff[ii][1]);
		writePlot("fft_convolve.tga", tmp);
	}
#endif //DEBUG

	// circular shift
	std::rotate(&sigbuff[0][0], &sigbuff[(uppadsize-1)/2][0],
			&sigbuff[uppadsize][0]);
#ifdef DEBUG
	{
		std::vector<double> tmp(uppadsize);
		for(size_t ii=0; ii<uppadsize; ii++)
			tmp[ii] = sqrt(sigbuff[ii][0]*sigbuff[ii][0] +
					sigbuff[ii][1]*sigbuff[ii][1]);
		writePlot("rotated.tga", tmp);
	}
#endif //DEBUG
	
	// post-multiply
	for(int64_t ii=-usize/2; ii<=usize/2; ii++) {
		complex<double> tmp1(ab_chirp[ii+uppadsize/2][0],
				ab_chirp[ii+uppadsize/2][1]);
		complex<double> tmp2(upsampled[ii+usize/2][0],
				upsampled[ii+usize/2][1]);
		tmp1 = tmp1*tmp2*A_phi;
		upsampled[ii+usize/2][0] = tmp1.real();
		upsampled[ii+usize/2][1] = tmp1.imag();
	}

#ifdef DEBUG
	{
		std::vector<double> tmp(uppadsize);
		for(size_t ii=0; ii<uppadsize; ii++)
			tmp[ii] = sigbuff[ii][0];
		writePlot("mult.tga", tmp);
	}
#endif //DEBUG
	
	interp(usize, upsampled, isize, inout);

	fftw_destroy_plan(sigbuff_plan_rev);
	fftw_destroy_plan(sigbuff_plan_fwd);
}

/**
 * @brief Comptues the Fractional Fourier transform using FFTW for nlogn
 * performance.
 *
 * @param isize size of input/output
 * @param in Input array, may be the same as output, length sz
 * @param out Output array, may be the same as input, length sz
 * @param Buffer size
 * @param a Fraction, 1 = fourier transform, 3 = inverse fourier transform,
 * 4 = identity
 * @param buffer Buffer to do computations in, may be null, in which case new
 * memory will be allocated and deallocated during processing. Note that if
 * the provided buffer is not sufficient size a new buffer will be allocated
 * and deallocated, and a warning will be produced
 * @param nonfft
 */
void fractional_ft(size_t isize, fftw_complex* in, fftw_complex* out, double a,
		size_t bsz, fftw_complex* buffer, bool nonfft)
{
	double MINTHRESH = 0.000000000001;

	// bring a into range
	while(a < 0)
		a += 4;
	a = fmod(a, 4);
	
	// there are 3 sizes: isize: the original size of the input array, usize :
	// the size of the upsampled array, and uppadsize the padded+upsampled
	// size, we want both uppadsize and usize to be odd, and we want uppadsize
	// to be the product of small primes (3,5,7)
	double approxratio = 4;
	int64_t uppadsize = round357(isize*approxratio);
	int64_t usize;
	while( (usize = (uppadsize-1)/2) % 2 == 0) {
		uppadsize = round357(uppadsize+2);
	}

	// check/allocate buffer
	bool freemem = false;
	if(bsz < isize+3*uppadsize || !buffer) {
		bsz = 4*uppadsize;
		buffer = fftw_alloc_complex(bsz);
		freemem = true;
	}

	fftw_complex* current = &buffer[0];
	fftw_plan curr_to_out_fwd = fftw_plan_dft_1d(isize, current, out,
			FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan curr_to_out_rev = fftw_plan_dft_1d(isize, current, out,
			FFTW_BACKWARD, FFTW_MEASURE);
	fftw_plan curr_to_curr_fwd = fftw_plan_dft_1d(isize, current, current,
			FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan curr_to_curr_rev = fftw_plan_dft_1d(isize, current, current,
			FFTW_BACKWARD, FFTW_MEASURE);

	// copy input to buffer
	for(size_t ii=0; ii<isize; ii++) {
		current[ii][0] = in[ii][0];
		current[ii][1] = in[ii][1];
	}

	while(a != 0) {
		if(fabs(a-1) < MINTHRESH) {
			// fourier transform
			fftw_execute(curr_to_out_fwd);
			a = 0;
		} else if(fabs(a-2) < MINTHRESH) {
			// reverse
			for(size_t ii=0; ii<isize; ii++) {
				out[ii][0] = current[isize-1-ii][0];
				out[ii][1] = current[isize-1-ii][1];
			}
			a = 0;
		} else if(fabs(a-3) < MINTHRESH) {
			// inverse fourier transform
			fftw_execute(curr_to_out_rev);
			a = 0;
		} else if(a < 0.5) {
			// below range, do a FFT to bring into range of fractional_fft
			fftw_execute(curr_to_curr_fwd);
			a += 1;
		} else if(a <= 1.5) {
			// can do the general purpose fractional_fft
			if(nonfft)
				fractional_brute_ft(isize, usize, uppadsize, current,
						&buffer[isize], a);
			else
				fractional_fft(isize, usize, uppadsize, current,
						&buffer[isize], a);
			a = 0;
		} else if(a < 2.5) {
			// above range, do an rev FFT to bring into range of fractional_fft
			fftw_execute(curr_to_curr_rev);
			a -= 1;
		} else if(a < 3.5) {
			// way above range, do an signal reversal to bring into range of
			// fractional_fft
			// reverse
			for(size_t ii=0; ii<isize; ii++) {
				current[ii][0] = current[isize-1-ii][0];
				current[ii][1] = current[isize-1-ii][1];
			}
			a -= 2;
		} else {
			// 3.5-4, if we add 1 (subtract 3) we get to 0.5-1.5
			fftw_execute(curr_to_curr_rev);
			a -= 3;
		}
	}

	// copy current to output
	for(size_t ii=0; ii<isize; ii++) {
		out[ii][0] = current[ii][0];
		out[ii][1] = current[ii][1];
	}

	if(freemem)
		fftw_free(buffer);

	fftw_destroy_plan(curr_to_curr_fwd);
	fftw_destroy_plan(curr_to_curr_rev);
	fftw_destroy_plan(curr_to_out_fwd);
	fftw_destroy_plan(curr_to_out_rev);
}

}

