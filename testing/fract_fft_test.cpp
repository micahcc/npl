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
 * @file fft_test.cpp
 * @brief This file is specifically to test forward, reverse of fft image
 * procesing functions.
 ******************************************************************************/

#include <string>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <complex>
#include <cassert>

#include "utility.h"

#include "fftw3.h"

const std::complex<double> I(0,1);
const double PI = acos(-1);

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
	int64_t just_hob = hob(in);
	if(just_hob == in)
		return in;
	else
		return (in<<1);
}

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

int64_t round357(int64_t in)
{
	// make it odd
	if(in %2 == 0)
		in++;

	// factor number
	auto factors = factor(in);
	
	// increase the largest factors first
	factors.sort();
	for(auto rit = factors.rbegin(); rit != factors.rend(); rit++) {

		// once we get to the primes we like, quit
		if(*rit == 3 || *rit == 5 || *rit == 7) 
			break;

		// round up to the product of the givne factors
		(*rit)++;
		*rit = round357(*rit);
	}

	int64_t out = 1;
	for(auto f : factors) {
		out *= f;
	}

	return out;
}

using namespace std;
using namespace npl;

void writeComplex(string filename, size_t len, const fftw_complex* input)
{
	ofstream of(filename.c_str());
	for(size_t ii=0; ii<len; ii++) {
		of << input[ii][0] << ", " << input[ii][1] << endl;
	}
	of.close();
}

void writeComplex(string filename, const std::vector<complex<double>>& input)
{
	ofstream of(filename.c_str());
	for(auto it: input) {
		of << it.real() << ", " << it.imag() << endl;
	}
	of.close();
}

void IFFT(const std::vector<complex<double>>& input, 
		vector<complex<double>>& out)
{
	size_t sz = round2(input.size());
	auto buffer = fftw_alloc_complex(sz); 
	fftw_plan buffer_plan = fftw_plan_dft_1d((int)sz, buffer, buffer,
				FFTW_BACKWARD, FFTW_MEASURE);

	// pad the (middle) upper frequencies
	for(size_t ii=0; ii < sz; ii++) {
		buffer[ii][0] = 0;
		buffer[ii][1] = 0;
	}

	// positive frequencies
	for(size_t ii=0; ii<(1+input.size())/2; ii++) {
		 buffer[ii][0] = input[ii].real();
		 buffer[ii][1] = input[ii].imag();
	}
	//negative
	for(int64_t ii=0; ii<(int64_t)input.size()/2; ii--) {
		 buffer[sz-1-ii][0] = input[input.size()-1-ii].real();
		 buffer[sz-1-ii][1] = input[input.size()-1-ii].imag();
	}

	fftw_execute(buffer_plan);

	// just take the end
	out.resize(input.size());
	for(size_t ii=0; ii<input.size(); ii++) {
		out[ii].real(buffer[ii][0]);
		out[ii].imag(buffer[ii][1]);
	}
}

void FFT(const std::vector<complex<double>>& input, 
		vector<complex<double>>& out)
{
	size_t sz = round2(input.size());
	auto buffer = fftw_alloc_complex(sz); 
	fftw_plan buffer_plan = fftw_plan_dft_1d((int)sz, buffer, buffer,
				FFTW_FORWARD, FFTW_MEASURE);
	for(size_t ii=0; ii<sz; ii++) {
		if(ii<input.size()) {
			buffer[ii][0] = input[ii].real();
			buffer[ii][1] = input[ii].imag();
		} else {
			buffer[ii][0] = 0;
			buffer[ii][1] = 0;
		}
	}

	fftw_execute(buffer_plan);

	// positive frequencies
	out.resize(input.size());
	for(size_t ii=0; ii<(1+input.size())/2; ii++) {
		 out[ii].real(buffer[ii][0]);
		 out[ii].imag(buffer[ii][1]);
	}
	//negative
	for(int64_t ii=0; ii<(int64_t)input.size()/2; ii--) {
		 out[input.size()-1-ii].real(buffer[sz-1-ii][0]);
		 out[input.size()-1-ii].imag(buffer[sz-1-ii][1]);
	}
}

//void integralFrFFT(const std::vector<complex<double>>& input, int a_frac,
//		vector<complex<double>>& out)
//{
//	while(a_frac < 0)
//		a_frac+= 4;
//	a_frac = a_frac%4;
//
//	switch(a_frac) {
//		case 0:
//			// identity
//			out.assign(input.begin(), input.end());
//			break;
//		case 1:
//			// FFT
//			FFT(input, out);
//			break;
//		case 2:
//			// reverse
//			out.resize(input.size());
//			std::reverse_copy(input.begin(), input.end(), out.begin());
//			break;
//		case 3:
//			// inverse fft
//			IFFT(input, out);
//			break;
//	}
//}

template <typename T> int sgn(T val) {
	return (T(0) < val) - (val < T(0));
}

/**
 * @brief Downsamples input, note input will be destroyed, and out should
 * already have the desired size.
 *
 * @param insz Input size
 * @param in in array
 * @param out output array, should already be allocated
 */
void downsample(size_t insz, fftw_complex* in, std::vector<complex<double>>& out)
{
	assert(out.size() > 0);
	assert(out.size() < insz);

	auto inbuff = fftw_alloc_complex((int)insz);
	fftw_plan in_plan = fftw_plan_dft_1d((int)insz, inbuff, inbuff,
				FFTW_FORWARD, FFTW_MEASURE);
	size_t roundout = round357(out.size());
	auto outbuff = fftw_alloc_complex((int)roundout);
	fftw_plan out_plan = fftw_plan_dft_1d((int)roundout, outbuff, outbuff,
				FFTW_BACKWARD, FFTW_MEASURE);
	
	//fill
	for(size_t ii=0; ii<insz; ii++) {
		inbuff[ii][0] = in[ii][0];
		inbuff[ii][1] = in[ii][1];
	}

	fftw_execute(in_plan);

	// fill outbuff with the matching frequencies
	for(size_t ii=0; ii<roundout; ii++) {
		outbuff[ii][0] = 0;
		outbuff[ii][1] = 0;
	}
	for(int64_t ii=0; ii<(1+out.size())/2; ii++) {
		outbuff[ii][0] = inbuff[ii][0];
		outbuff[ii][1] = inbuff[ii][1];
	}
	for(int64_t ii=0; ii<out.size()/2; ii++) {
		outbuff[roundout-ii-1][0] = inbuff[insz-1-ii][0];
		outbuff[roundout-ii-1][1] = inbuff[insz-1-ii][1];
	}

	fftw_execute(out_plan);

	for(int64_t ii=0; ii<out.size(); ii++) {
		out[ii].real(outbuff[ii][0]);
		out[ii].imag(outbuff[ii][1]);
	}

	fftw_free(inbuff);
	fftw_free(outbuff);
	fftw_destroy_plan(in_plan);
	fftw_destroy_plan(out_plan);
}

void upsampleShift(const std::vector<complex<double>>& in, 
		size_t outsize, fftw_complex* out)
{
	assert(out != NULL);

	size_t roundin = round357(in.size());
	auto inbuff = fftw_alloc_complex((int)roundin);
	fftw_plan in_plan = fftw_plan_dft_1d((int)roundin, inbuff, inbuff,
				FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan out_plan = fftw_plan_dft_1d((int)outsize, out, out,
				FFTW_BACKWARD, FFTW_MEASURE);
	
	// fill/average pad
	complex<double> avg = 0;
	int64_t shift = (roundin-1)/2 - (in.size()-1)/2;
	for(size_t ii=0; ii<in.size(); ii++) {
		avg += in[ii];
	}
	avg /= (double)in.size();
	// fill with average
	for(size_t ii=0; ii<roundin; ii++) {
		inbuff[ii][0] = avg.real();
		inbuff[ii][1] = avg.imag();
	}
	// copy/center
	for(size_t ii=0; ii<in.size(); ii++) {
		inbuff[ii+shift][0] = in[ii].real();
		inbuff[ii+shift][1] = in[ii].imag();
	}
	
	fftw_execute(in_plan);

	// fill outbuff with the matching frequencies
	for(size_t ii=0; ii<outsize; ii++) {
		out[ii][0] = 0;
		out[ii][1] = 0;
	}
	for(int64_t ii=0; ii<(1+roundin)/2; ii++) {
		out[ii][0] = inbuff[ii][0];
		out[ii][1] = inbuff[ii][1];
	}
	for(int64_t ii=0; ii<roundin/2; ii++) {
		out[outsize-1-ii][0] = inbuff[roundin-1-ii][0];
		out[outsize-1-ii][1] = inbuff[roundin-1-ii][1];
	}

	fftw_execute(out_plan);

	fftw_destroy_plan(in_plan);
	fftw_destroy_plan(out_plan);
	fftw_free(inbuff);
}

fftw_complex* createChirp(int64_t sz, double alpha, double beta, double Dx, bool fft)
{
	fftw_complex* out = fftw_alloc_complex(sz);
	auto fwd_plan = fftw_plan_dft_1d((int)sz, out, out, FFTW_FORWARD,
				FFTW_MEASURE);

	complex<double> imag(0, 1);
	complex<double> eterm = imag*PI*(alpha-beta)/(4*Dx*Dx);
	for(int64_t ii=-sz/2; ii<= sz/2; ii++) {
		auto tmp = std::exp(eterm*(double)(ii*ii));
		out[ii+sz/2][0] = tmp.real();
		out[ii+sz/2][1] = tmp.imag();
	}

	if(fft)
		fftw_execute(fwd_plan);

	fftw_destroy_plan(fwd_plan);
	return out;
}

void floatFrFFT(const std::vector<complex<double>>& input, float a_frac,
		vector<complex<double>>& out)
{
	assert(a_frac <= 1.5 && a_frac >= 0.5);
	assert(a_frac != 1);
	writeComplex("input.txt", input);

	const double PI = acos(-1);
	double phi = a_frac*PI/2;
	complex<double> imag(0,1);
	double alpha = 1./tan(phi);
	double beta = 1./sin(phi);
	std::complex<double> avg(0);
//	complex<double> A_phi = std::exp(-imag*PI*sgn(sin(phi))/4+imag*phi/2.)/
//			sqrt(fabs(sin(phi)));
	// since phi [.78,2.35], sin(phi) is positive, sgn(sin(phi)) = 1:
	complex<double> A_phi = std::exp(-imag*PI/4.+imag*phi/2.) / sqrt(sin(phi));
	double Dx = sqrt((double)input.size());
	complex<double> tmp1, tmp2;

	// upsample input, and maintain center location
	size_t upsize = round357(input.size()*2);
	auto upsampled = fftw_alloc_complex(upsize);
	auto outchirp = createChirp(upsize, alpha, beta, Dx, false);
	auto convchirp = createChirp(upsize, beta, 0, Dx, true);
	
	writeComplex("chirp.txt", upsize, outchirp);
	writeComplex("freq_chirp.txt", upsize, convchirp);
	
	// .. but first create plans which would overwrite upsampled...
	auto fwd_plan = fftw_plan_dft_1d((int)upsize, upsampled, upsampled,
			FFTW_FORWARD, FFTW_MEASURE);
	auto back_plan = fftw_plan_dft_1d((int)upsize, upsampled, upsampled,
			FFTW_BACKWARD, FFTW_MEASURE);

	// now upsample
	upsampleShift(input, upsize, upsampled);

	writeComplex("upsampled.txt", upsize, upsampled);
	
	// pre-multiply with chirp
	size_t count = 0;
	for(int64_t ii=0; ii<upsize; ii++) {
		tmp1.real(upsampled[ii][0]);
		tmp1.imag(upsampled[ii][1]);
		tmp2.real(outchirp[ii][0]);
		tmp2.imag(outchirp[ii][1]);

		tmp1 = tmp1*tmp2;

		upsampled[ii][0] = tmp1.real();
		upsampled[ii][1] = tmp1.imag();
		++count;
	}
	assert(count == upsize);

	fftw_execute(fwd_plan);

	// perform convolution with chirp
	for(int64_t ii=0; ii<upsize; ii++) {
		tmp1.real(upsampled[ii][0]);
		tmp1.imag(upsampled[ii][1]);
		tmp2.real(convchirp[ii][0]);
		tmp2.imag(convchirp[ii][1]);

		tmp1 = tmp1*tmp2;

		upsampled[ii][0] = tmp1.real();
		upsampled[ii][1] = tmp1.imag();
	}
	
	fftw_execute(back_plan);
	
	// multiply with final chirp
	out.resize(upsize);
	for(int64_t ii=0; ii<upsize; ii++) {
		tmp1.real(upsampled[ii][0]);
		tmp1.imag(upsampled[ii][1]);
		tmp2.real(outchirp[ii][0]);
		tmp2.imag(outchirp[ii][1]);

		tmp1 = tmp1*tmp2*A_phi/(2*Dx);

		out[ii] = tmp1;
	}

	writeComplex("full_fract.txt", out);

	fftw_free(upsampled);
	fftw_free(convchirp);
	fftw_free(outchirp);
	fftw_destroy_plan(fwd_plan);
	fftw_destroy_plan(back_plan);
}

//// based on Ozaktas 1996
//void fractionalFFT(const std::vector<complex<double>>& input, double a_frac,
//		vector<complex<double>>& out)
//{
//	
//	// modifications due to integral a_fac values, at the end this will be the
//	// input to the smaller value a_fac transform. The actual FrFFT needs a
//	// value in the range [0.5, 1.5]
//	auto buffer = fftw_alloc_complex(sz); 
//	fftw_plan buffer_fwd = fftw_plan_dft_1d((int)sz, buffer, buffer,
//				FFTW_FORWARD, FFTW_MEASURE);
//	fftw_plan buffer_rev = fftw_plan_dft_1d((int)sz, buffer, buffer,
//				FFTW_BACKWARD, FFTW_MEASURE);
//
//	out.assign(input.begin(), input.end());
//	
//	/*
//	 * first perform fourier transfor to the needed degree to get within 0.5 of
//	 * the desired fract. Since 
//	 */
//	
//	// make positive
//	if(a_frac < 0) {
//		while(a_frac < 0)
//			a_frac+= 4;
//	}
//
//	// remove excessive values, since it repeats every 4
//	a_frac = fmod(a_frac, 4);
//
//	// [0, 4)
//	// 2 is equal to reversing the signal
//	if(a_frac >= 2) {
//		a_frac -= 2;
//
//		for(size_t ii=0; ii<out.size()/2; ii++) 
//			std::swap(out[ii], out[out.size()-1-ii]);
//	}
//	// [0, 2)
//	if(fabs(a_frac) < 0.00000000001 ) {
//		// nothing more to do, just return the current output
//		return;
//	}
//	// (0, 2)
//	
//	// all the remaining operations require the buffer, so fill it
//	for(size_t ii=0; ii<sz; ii++) {
//		// time to start using the buffer...
//		if(ii < out.size()) {
//			buffer[ii][0] += out[ii].real();
//			buffer[ii][1] += out[ii].imag();
//		} else {
//			buffer[ii][0] += 0;
//			buffer[ii][1] += 0;
//		}
//	}
//
//	// (0, 2)
//	if(a_frac > 1.5) {
//		// fourier transform adds 1 to the effective a_frac, meaning we need to
//		// do 1 less
//
//		a_frac -= 1;
//	}
//	// (0, 1.5)
//	else if(a_frac < 0.5) {
//		// inverse fourier transform adds 3 (-1) to the effective a_frac, meaning we
//		// need to do 1 more 
//
//		a_frac += 1;
//	}
//	// (0.5, 1.5)
//	else if(fabs(a_frac - 1) < 0.0000001 ) {
//		// 1 is equal to a fourier transform, so just do that if we are really
//		// close to 1
//		for(size_t ii=0; ii<sz; ii++) {
//
//		fftw_execute(buffer_fwd);
//		
//		// just take the low frequencies, ignoring high end
//		// positive frequencies
//		for(int64_t ii=0; ii<input.size()/2; ii++) {
//			out[ii].real(buffer[ii][0]);
//			out[ii].imag(buffer[ii][1]);
//		}
//		
//		// negative frequencies
//		int64_t jj = sz-1;
//		for(int64_t ii=input.size()-1, jj=sz-1; ii>input.size()/2; ii--, jj--) {
//			out[ii].real(buffer[jj][0]);
//			out[ii].imag(buffer[jj][1]);
//		}
//
//		return ;
//	}
//
//	// 
//	if(% 2
//		// TODO
//	}
//	a_frac = a_frac - a_frac_near;
//
//	
//	// we round the size to the product of 3,5,7 so the output will be odd,
//	// this is to satisfy the requirement of summing from -N to N
//	auto yvec = fftw_alloc_complex(sz);
//	auto zvec = fftw_alloc_complex(sz);
//	auto buffer = fftw_alloc_complex(sz);
//
//	fftw_plan yvec_plan = fftw_plan_dft_1d((int)sz, yvec, yvec, FFTW_FORWARD, FFTW_MEASURE);
//	fftw_plan zvec_plan = fftw_plan_dft_1d((int)sz, zvec, zvec, FFTW_FORWARD, FFTW_MEASURE);
//	fftw_plan buffer_plan = fftw_plan_dft_1d((int)sz, buffer, buffer, FFTW_BACKWARD, FFTW_MEASURE);
//
//	// multiply input by chirps 
//	const std::complex<double> alpha = PI*I*alpha/(double)sz;
//	const std::complex<double> beta = PI*I*alpha/(double)sz;
//
//	// create chirp
//	for(size_t ii=0; ii<sz; ii++){
//		if(ii < input.size()) {
//			auto tmp = input[ii]*std::exp(-term*(double)(ii*ii));
//			yvec[ii][0] = tmp.real();
//			yvec[ii][1] = tmp.imag();
//
//			tmp = std::exp(term*(double)(ii*ii));
//			zvec[ii][0] = tmp.real();
//			zvec[ii][1] = tmp.imag();
//		} else if(ii < sz-input.size()) {
//			yvec[ii][0] = 0;
//			yvec[ii][1] = 0;
//			
//			zvec[ii][0] = 0;
//			zvec[ii][1] = 0;
//		} else {
//			yvec[ii][0] = 0;
//			yvec[ii][1] = 0;
//			
//			auto tmp = std::exp(term*(double)((sz-ii)*(sz-ii)));
//			zvec[ii][0] = tmp.real();
//			zvec[ii][1] = tmp.imag();
//		}
//	}
//
//	writeComplex("yvec.txt", sz, yvec);
//	writeComplex("zvec.txt", sz, zvec);
//	fftw_execute(yvec_plan);
//	fftw_execute(zvec_plan);
//	writeComplex("fyvec.txt", sz, yvec);
//	writeComplex("fzvec.txt", sz, zvec);
//
//	// multiply (convolve)
//	for(size_t ii=0; ii<sz; ii++){
//		std::complex<double> tmp1(yvec[ii][0], yvec[ii][1]);
//		std::complex<double> tmp2(zvec[ii][0], zvec[ii][1]);
//		tmp1 = tmp1*tmp2/((double)sz*sz); //normalize convolution
//		buffer[ii][0] = tmp1.real();
//		buffer[ii][1] = tmp1.imag();
//	}
//	
//	// invert
//	writeComplex("prebuffer.txt", sz, buffer);
//	fftw_execute(buffer_plan);
//	writeComplex("postbuffer.txt", sz, buffer);
//
//	// chirp again, we only need the first elements, the rest were padding
//	out.resize(input.size());
//	for(size_t ii=0; ii<sz; ii++){
//		std::complex<double> tmp1(buffer[ii][0], buffer[ii][1]);
//		std::complex<double> tmp2 = std::exp(-term*((double)(ii*ii)));
//		tmp1 *= tmp2;
//		buffer[ii][0] = tmp1.real();
//		buffer[ii][1] = tmp1.imag();
//	}
//
//	writeComplex("postmult.txt", sz, buffer);
//
//	fftw_free(yvec);
//	fftw_free(zvec);
//	fftw_free(buffer);
//
//	fftw_destroy_plan(yvec_plan);
//	fftw_destroy_plan(zvec_plan);
//	fftw_destroy_plan(buffer_plan);
//}

//// Averbuch 2006
//void fractionalFFT3(const std::vector<complex<double>>& input, double alpha,
//		vector<complex<double>>& out)
//{
//	size_t sz = round2(input.size())*2;
//	auto EcF = fftw_alloc_complex(sz);
//	auto EdF = fftw_alloc_complex(sz);
//	auto g_k = fftw_alloc_complex(sz);
//
//	fftw_plan EcF_plan = fftw_plan_dft_1d((int)sz, EcF, EcF, FFTW_FORWARD, FFTW_MEASURE);
//	fftw_plan EdF_plan = fftw_plan_dft_1d((int)sz, EdF, EdF, FFTW_FORWARD, FFTW_MEASURE);
//	fftw_plan inv_plan = fftw_plan_dft_1d((int)sz, g_k, g_k, FFTW_BACKWARD, FFTW_MEASURE);
//
//	// multiply input by chirps 
//	const double PI = acos(-1);
//	const std::complex<double> imag(0,1);
//	std::complex<double> yj;
//	std::complex<double> zj;
//	std::complex<double> term = PI*imag*alpha/(double)sz;
//	for(size_t ii=0; ii<sz; ii++){
//		if(ii<input.size()) {
//			yj = 0;
//			zj = 0;
//		} else if(ii < input.size()*2) {
//			yj = input[ii-input.size()]*std::exp(-term*((double)ii*ii));
//			zj = std::exp(term*((double)ii*ii));
//		} else {
//			yj = 0;
//			zj = 0;
//		}
//
//		
//		EcF[ii][0] = yj.real();
//		EcF[ii][1] = yj.imag();
//		
//		EdF[ii][0] = zj.real();
//		EdF[ii][1] = zj.imag();
//	}
//	
//	// fourier transform both chirp-multiplied
//	fftw_execute(EcF_plan);
//	fftw_execute(EdF_plan);
//
//	// multiply (really convolve)
//	for(size_t ii=0; ii<sz; ii++) {
//		yj.real(EcF[ii][0]);
//		yj.imag(EcF[ii][1]);
//		zj.real(EdF[ii][0]);
//		zj.imag(EdF[ii][1]);
//
//		// normalize during convolution
//		yj = yj*zj/((double)sz*sz); ///(double)(sz*sz);
//
//		g_k[ii][0] = yj.real();
//		g_k[ii][1] = yj.imag();
//	}
//
//	// invert
//	fftw_execute(inv_plan);
//
//	// chirp again, we only need the first elements, the rest were padding
//	out.resize(input.size());
//	for(size_t ii=input.size(); ii<2*input.size(); ii++){
//		yj.real(g_k[ii][0]);
//		yj.imag(g_k[ii][1]);
//		yj *= std::exp(-term*(double)(ii*ii));
//		out[ii-input.size()] = yj;
//	}
//
//	fftw_free(EdF);
//	fftw_free(EcF);
//	fftw_free(g_k);
//}
//
//// based on Bultheel
//void fractionalFFT4(const std::vector<complex<double>>& input, double alpha,
//		vector<complex<double>>& out)
//{
//	size_t sz = round2(input.size())*4;
//	auto EcF = fftw_alloc_complex(sz);
//	auto EdF = fftw_alloc_complex(sz);
//	auto g_k = fftw_alloc_complex(sz);
//
//	fftw_plan EcF_plan = fftw_plan_dft_1d((int)sz, EcF, EcF, FFTW_FORWARD, FFTW_MEASURE);
//	fftw_plan EdF_plan = fftw_plan_dft_1d((int)sz, EdF, EdF, FFTW_FORWARD, FFTW_MEASURE);
//	fftw_plan inv_plan = fftw_plan_dft_1d((int)sz, g_k, g_k, FFTW_BACKWARD, FFTW_MEASURE);
//
//	// multiply input by chirps 
//	const double PI = acos(-1);
//	const std::complex<double> imag(0,1);
//	std::complex<double> edf;
//	std::complex<double> ecf;
//	std::complex<double> term = PI*imag*alpha;
//	size_t nonzero = 0;
//	for(size_t ii=0; ii<sz; ii++){
//		if(ii<input.size()) {
//			edf = input[ii]*std::exp(-imag*tan(alpha/2.)*((double)ii*ii));
//			ecf = input[ii]*std::exp(imag*(1./sin(alpha))*((double)ii*ii));
//			EcF[ii][0] = ecf.real();
//			EcF[ii][1] = ecf.imag();
//			EdF[ii][0] = edf.real();
//			EdF[ii][1] = edf.imag();
//		} else if(ii < sz) {
//			EcF[ii][0] = 0;
//			EcF[ii][1] = 0;
//			EdF[ii][0] = 0;
//			EdF[ii][1] = 0;
//		}
//	}
//	
//	// fourier transform both chirp-multiplied
//	fftw_execute(EcF_plan);
//	fftw_execute(EdF_plan);
//
//	// multiply (really convolve)
//	for(size_t ii=0; ii<sz; ii++) {
//		ecf.real(EcF[ii][0]);
//		ecf.imag(EcF[ii][1]);
//		edf.real(EdF[ii][0]);
//		edf.imag(EdF[ii][1]);
//
//		// normalize during convolution
//		auto tmp = ecf*edf/((double)sz*sz); ///(double)(sz*sz);
//
//		g_k[ii][0] = tmp.real();
//		g_k[ii][1] = tmp.imag();
//	}
//
//	// invert
//	fftw_execute(inv_plan);
//
//	// chirp again, we only need the first elements, the rest were padding
//	out.resize(input.size());
//	for(size_t ii=0; ii<input.size(); ii++){
//		std::complex<double> tmp(g_k[ii][0], g_k[ii][1]);
//		tmp *= (sqrt(1.-imag/tan(alpha))/(2*sqrt(input.size())))*
//				std::exp(-imag*tan(alpha/2.)*((double)ii*ii));
//		out[ii] = tmp;
//	}
//
//	fftw_free(EdF);
//	fftw_free(EcF);
//	fftw_free(g_k);
//	
//	fftw_destroy_plan(EcF_plan);
//	fftw_destroy_plan(EdF_plan);
//	fftw_destroy_plan(inv_plan);
//}
//
//void fractionalFFT(const std::vector<complex<double>>& input, double alpha,
//		vector<complex<double>>& out)
//{
//	size_t sz = round2(input.size())*2;
//	auto EcF = fftw_alloc_complex(sz);
//	auto EdF = fftw_alloc_complex(sz);
//	auto g_k = fftw_alloc_complex(sz);
//
//	fftw_plan EcF_plan = fftw_plan_dft_1d((int)sz, EcF, EcF, FFTW_FORWARD, FFTW_MEASURE);
//	fftw_plan EdF_plan = fftw_plan_dft_1d((int)sz, EdF, EdF, FFTW_FORWARD, FFTW_MEASURE);
//	fftw_plan inv_plan = fftw_plan_dft_1d((int)sz, g_k, g_k, FFTW_BACKWARD, FFTW_MEASURE);
//
//	// multiply input by chirps 
//	const double PI = acos(-1);
//	const std::complex<double> imag(0,1);
//	std::complex<double> yj;
//	std::complex<double> zj;
//	std::complex<double> term = PI*imag*alpha;
//	size_t nonzero = 0;
//	for(size_t ii=0; ii<sz; ii++){
//		if(ii<input.size()) {
//			nonzero++;
//			yj = input[ii]*std::exp(-PI*imag*(alpha*ii*ii));
//			zj = std::exp(PI*imag*(alpha*ii*ii));
//		} else if(ii < sz-input.size()) {
//			yj = 0;
//			zj = 0;
//		} else if(ii < sz) {
//			yj = 0;
//			zj = std::exp(PI*imag*(alpha*(sz-ii)*(sz-ii)));
//			nonzero++;
//		}
//
//		
//		EcF[ii][0] = yj.real();
//		EcF[ii][1] = yj.imag();
//		
//		EdF[ii][0] = zj.real();
//		EdF[ii][1] = zj.imag();
//	}
//	
//	// fourier transform both chirp-multiplied
//	fftw_execute(EcF_plan);
//	fftw_execute(EdF_plan);
//
//	// multiply (really convolve)
//	for(size_t ii=0; ii<sz; ii++) {
//		yj.real(EcF[ii][0]);
//		yj.imag(EcF[ii][1]);
//		zj.real(EdF[ii][0]);
//		zj.imag(EdF[ii][1]);
//
//		// normalize during convolution
//		yj = yj*zj/((double)sz*sz); ///(double)(sz*sz);
//
//		g_k[ii][0] = yj.real();
//		g_k[ii][1] = yj.imag();
//	}
//
//	// invert
//	fftw_execute(inv_plan);
//
//	// chirp again, we only need the first elements, the rest were padding
//	out.resize(input.size());
//	for(size_t ii=0; ii<input.size(); ii++){
//		yj.real(g_k[ii][0]);
//		yj.imag(g_k[ii][1]);
//		yj = std::exp(-PI*imag*(alpha*ii*ii))*yj;
//		out[ii] = yj;
//	}
//
//	fftw_free(EdF);
//	fftw_free(EcF);
//	fftw_free(g_k);
//}

int main(int argc, char** argv)
{
	if(argc != 3) {
		return -1;
	}
	double alpha = atof(argv[1]);
	size_t sz = atoi(argv[2]);
	vector<std::complex<double>> in(sz);
	vector<std::complex<double>> out(sz);
	for(size_t ii=0; ii<in.size(); ii++) {
		if(ii < in.size()*2/3 && ii > in.size()/3)
			in[ii] = 1.;
		else
			in[ii] = 0.;
	}
	// make nonreal signal sin(x) signal
//	for(size_t ii=0; ii<in.size(); ii++) {
//		in[ii] = std::exp(-2.*PI*I*(double)ii/100.);
//	}
	std::vector<double> realv(sz);
	std::vector<double> imagv(sz);

	for(size_t ii=0; ii<sz; ii++) {
		realv[ii] = in[ii].real();
		imagv[ii] = in[ii].imag();
	}
	writePlot("orig_real.tga", realv);
	writePlot("orig_imag.tga", imagv);

	floatFrFFT(in, alpha, out);

	realv.resize(out.size());
	imagv.resize(out.size());
	for(size_t ii=0; ii<out.size(); ii++) {
		realv[ii] = abs(out[ii]);
		imagv[ii] = arg(out[ii]);
	}
	writePlot("frft_real.tga", realv);
	writePlot("frft_imag.tga", imagv);

//	for(size_t ii=0; ii<in.size(); ii++) {
//		if(norm(in[ii]-out[ii]) > 0.01) {
//			cerr << "Different: " << ii << endl;
//			cerr << in[ii] << " vs " << out[ii] << endl;
//			return -1;
//		}
//	}


	fftw_cleanup();
	return 0;
}




