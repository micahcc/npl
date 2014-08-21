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

#include "chirpz.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <algorithm>

#include "basic_functions.h"
#include "basic_plot.h"

using namespace std;
//#define DEBUG

namespace npl {

/**
 * @brief Interpolate the input array, filling the output array
 *
 * @param isize 	Size of in
 * @param in 		Values to interpolate
 * @param osize 	Size of out
 * @param out 		Output array, filled with interpolated values of in
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
				sum += lanczosKernel(ii-cii, radius)*tmp;
			}
		}
		out[oo][0] = sum.real();
		out[oo][1] = sum.imag();
	}
}

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
		double upratio, double alpha, bool fft)
{
//	assert(sz%2==1);
	const double PI = acos(-1);
	const complex<double> I(0,1);
	
	auto fwd_plan = fftw_plan_dft_1d((int)sz, chirp, chirp, FFTW_FORWARD,
				FFTW_MEASURE | FFTW_PRESERVE_INPUT);

	cerr << "Upsample: " << upratio << endl;
	for(int64_t ii=0; ii<sz; ii++) {
		double xx = (ii-(sz-1.)/2.)/upratio;
		auto tmp = std::exp(I*PI*alpha*xx*xx/(double)origsz);
		chirp[ii][0] = tmp.real();
		chirp[ii][1] = tmp.imag();
	}
	
	if(fft) {
		fftw_execute(fwd_plan);
		double norm = 1./sz;
		for(size_t ii=0; ii<sz; ii++) {
			chirp[ii][0] *= norm;
			chirp[ii][1] *= norm;
		}
	}

	fftw_destroy_plan(fwd_plan);
}

/**
 * @brief Comptues the chirpzFFT transform using FFTW for n log n performance.
 *
 * @param isize Size of input/output
 * @param in Input array, may be the same as out, length sz
 * @param out Output array, may be the same as input, length sz
 * @param alpha Fraction of full space to compute
 */
void chirpzFFT(size_t isize, fftw_complex* in, fftw_complex* out, double a)
{
	// there are 3 sizes: isize: the original size of the input array, usize :
	// the size of the upsampled array, and uppadsize the padded+upsampled
	// size, we want both uppadsize and usize to be odd, and we want uppadsize
	// to be the product of small primes (3,5,7)
	double approxratio = 2;
	int64_t usize = round2(isize*approxratio);
	int64_t uppadsize = usize*2;
	double upratio = (double)usize/(double)isize;

	size_t bsz = isize+3*uppadsize;
	fftw_complex* buffer = fftw_alloc_complex(bsz);
	fftw_complex* current = &buffer[0];
	fftw_complex* nchirp = &buffer[isize+uppadsize];
	fftw_complex* pchirp = &buffer[isize+2*uppadsize];
	
	createChirp(uppadsize, nchirp, isize, upratio, -a, false);
	createChirp(uppadsize, pchirp, isize, upratio, a, true);

	// copy input to buffer
	for(size_t ii=0; ii<isize; ii++) {
		current[ii][0] = in[ii][0];
		current[ii][1] = in[ii][1];
	}

	chirpzFFT(isize, usize, current, uppadsize, &buffer[isize], nchirp, pchirp);

	// copy current to output
	for(size_t ii=0; ii<isize; ii++) {
		out[ii][0] = current[ii][0];
		out[ii][1] = current[ii][1];
	}

	fftw_free(buffer);
}

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
		fftw_complex* poschirpF)
{
	const complex<double> I(0,1);

	// zero
	for(size_t ii=0; ii<uppadsize; ii++) {
		buffer[ii][0] = 0;
		buffer[ii][1] = 0;
	}

	fftw_complex* sigbuff = &buffer[0]; // note the overlap with upsampled
	fftw_complex* upsampled = &buffer[uppadsize/2-usize/2];

	fftw_plan sigbuff_plan_fwd = fftw_plan_dft_1d(uppadsize, sigbuff, sigbuff,
			FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan sigbuff_plan_rev = fftw_plan_dft_1d(uppadsize, sigbuff, sigbuff,
			FFTW_BACKWARD, FFTW_MEASURE);

#ifdef DEBUG
	writePlotReIm("fft_poschirpf.svg", uppadsize, poschirpF);
	writePlotReIm("fft_negchirp.svg", uppadsize, negchirp);
	writePlotReIm("fft_in.svg", isize, inout);
#endif //DEBUG
	// upsample input
	interp(isize, inout, usize, upsampled);

#ifdef DEBUG
	writePlotReIm("fft_upin.svg", usize, upsampled);
#endif //DEBUG
	
	// pre-multiply
	for(int64_t nn = 0; nn<usize; nn++) {
		size_t cc = nn+(uppadsize-usize)/2;
		complex<double> tmp1(negchirp[cc][0], negchirp[cc][1]);
		complex<double> tmp2(upsampled[nn][0], upsampled[nn][1]);
		tmp1 *= tmp2;
		upsampled[nn][0] = tmp1.real();
		upsampled[nn][1] = tmp1.imag();
	}

#ifdef DEBUG
	writePlotReIm("fft_premult.svg", usize, upsampled);
#endif //DEBUG

	/*
	 * convolve
	 */
	fftw_execute(sigbuff_plan_fwd);
	double normfactor = 1./uppadsize;
	for(size_t ii=0; ii<uppadsize; ii++) {
		sigbuff[ii][0] *= normfactor;
		sigbuff[ii][1] *= normfactor;
	}

	for(size_t ii=0; ii<uppadsize; ii++) {
		complex<double> tmp1(sigbuff[ii][0], sigbuff[ii][1]);
		complex<double> tmp2(poschirpF[ii][0], poschirpF[ii][1]);
		tmp1 *= tmp2;
		sigbuff[ii][0] = tmp1.real();
		sigbuff[ii][1] = tmp1.imag();
	}
	fftw_execute(sigbuff_plan_rev);
	
	// not sure why this works...
	normfactor = 2*isize; 
	for(size_t ii=0; ii<uppadsize; ii++) {
		sigbuff[ii][0] *= normfactor;
		sigbuff[ii][1] *= normfactor;
	}
	
#ifdef DEBUG
	writePlotReIm("fft_convolve.svg", uppadsize, sigbuff);
#endif //DEBUG

	// circular shift
	std::rotate(&sigbuff[0][0], &sigbuff[(uppadsize-1)/2][0],
			&sigbuff[uppadsize][0]);
#ifdef DEBUG
	writePlotReIm("fft_rotated.svg", uppadsize, sigbuff);
#endif //DEBUG
	
	// post-multiply
	for(int64_t ii=0; ii<usize; ii++) {
		size_t cc = ii+(uppadsize-usize)/2;
		complex<double> tmp1(negchirp[cc][0], negchirp[cc][1]);
		complex<double> tmp2(upsampled[ii][0], upsampled[ii][1]);
		tmp1 = tmp1*tmp2;
		upsampled[ii][0] = tmp1.real();
		upsampled[ii][1] = tmp1.imag();
	}

#ifdef DEBUG
	writePlotReIm("fft_postmult.svg", uppadsize, sigbuff);
#endif //DEBUG
	
	interp(usize, upsampled, isize, inout);
	
#ifdef DEBUG
	writePlotReIm("fft_out.svg", isize, inout);
#endif //DEBUG

	fftw_destroy_plan(sigbuff_plan_rev);
	fftw_destroy_plan(sigbuff_plan_fwd);
}

/**
 * @brief Performs chirpz transform with a as fractional parameter by N^2 
 * algorithm.
 *
 * @param len	Length of input array
 * @param in	Input Array (length = len)
 * @param out	Output Array (length = len)
 * @param a		Fraction/Alpha To raise exp() term to
 */
void chirpzFT_brute(size_t len, fftw_complex* in, fftw_complex* out, double a)
{
	const complex<double> I(0,1);
	const double PI = acos(-1);
	int64_t ilen = len;

	for(int64_t ii=0; ii<ilen; ii++) {
		double ff=(ii-(ilen)/2.);
		out[ii][0]=0;
		out[ii][1]=0;

		for(int64_t jj=0; jj<ilen; jj++) {
			double xx=(jj-(ilen-1)/2.);
			complex<double> tmp1(in[jj][0], in[jj][1]);
			complex<double> tmp2 = tmp1*std::exp(-2.*PI*I*a*xx*ff/(double)ilen);
			
			out[ii][0] += tmp2.real();
			out[ii][1] += tmp2.imag();
		}
	}
}


/**
 * @brief Plots an array of complex points with the Real and Imaginary Parts
 *
 * @param file	Filename
 * @param insz	Size of in
 * @param in	Array input
 */
void writePlotReIm(std::string file, size_t insz, fftw_complex* in)
{
	std::vector<double> realv(insz);
	std::vector<double> imv(insz);
	for(size_t ii=0; ii<insz; ii++) {
		realv[ii] = in[ii][0];
		imv[ii] = in[ii][1];
	}

	Plotter plt;
	plt.addArray(insz, realv.data());
	plt.addArray(insz, imv.data());
	plt.write(file);
}

/**
 * @brief Plots an array of complex points with the Real and Imaginary Parts
 *
 * @param file	Filename
 * @param insz	Size of in
 * @param in	Array input
 */
void writePlotAbsAng(std::string file, size_t insz, fftw_complex* in)
{
	double phasemax = -INFINITY;
	double phasemin = INFINITY;
	double absmax = -INFINITY;
	double absmin = INFINITY;
	std::vector<double> absv(insz);
	std::vector<double> angv(insz);
	for(size_t ii=0; ii<insz; ii++) {
		angv[ii] = atan2(in[ii][0], in[ii][1]);
		absv[ii] = sqrt(pow(in[ii][0],2)+pow(in[ii][1],2));
		phasemax = std::max(phasemax, angv[ii]);
		absmax = std::max(absmax, absv[ii]);
		phasemin = std::min(phasemin, angv[ii]);
		absmin = std::min(absmin, absv[ii]);
	}

	Plotter plt;
	plt.addArray(insz, absv.data());
	plt.addArray(insz, angv.data());
	plt.write(file);
}

}

