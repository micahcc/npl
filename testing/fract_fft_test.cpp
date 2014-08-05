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

#define DEBUG

#include "utility.h"

#include "fftw3.h"

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

using namespace std;
using namespace npl;

void writeComplex(string filename, size_t len, const fftw_complex* input)
{
	ofstream of(filename.c_str());
	for(size_t ii=0; ii<len; ii++) {
		of << input[ii][0] << ", " << input[ii][1] << std::endl;
	}
	of.close();
}

void writeComplex(string filename, const std::vector<complex<double>>& input)
{
	ofstream of(filename.c_str());
	for(auto it: input) {
		of << it.real() << ", " << it.imag() << std::endl;
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

void interp(const std::vector<complex<double>>& in, 
		std::vector<complex<double>>& out)
{
	// fill/average pad
	int64_t radius = 3;
	double ratio = (double)(in.size())/(double)out.size();
	
	// copy/center
	for(size_t oo=0; oo<out.size(); oo++) {
		double cii = ratio*oo;
		int64_t center = round(cii);

		complex<double> sum = 0;
		for(int64_t ii=center-radius; ii<=center+radius; ii++) {
			if(ii>=0 && ii<in.size()) 
				sum += lanczos(ii-cii, radius)*in[ii];
		}
		out[oo] = sum;
	}
}

fftw_complex* createChirp(int64_t sz, int64_t origsz, double upratio, 
		double alpha, double beta, bool fft)
{
	assert(sz%2 == 1);
	const double PI = acos(-1);
	const complex<double> I(0,1);
	
	auto chirp = fftw_alloc_complex(sz);
	auto fwd_plan = fftw_plan_dft_1d((int)sz, chirp, chirp, FFTW_FORWARD,
				FFTW_MEASURE);

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
	return chirp;
}

/**
 * @brief Brute force version of fractional fourier transform. For testing
 * purposes only. 
 *
 * @param input Input vector,
 * @param a_frac fractional level (repeats every 4).
 * @param out output vector
 */
void floatFrFFTBrute(const std::vector<complex<double>>& input, float a_frac,
		vector<complex<double>>& out)
{
	assert(a_frac <= 1.5 && a_frac >= 0.5);

	const std::complex<double> I(0,1);
	const double PI = acos(-1);
	double phi = a_frac*PI/2;

	double alpha = 1./tan(phi);
	double beta = 1./sin(phi);
	if(a_frac == 1) {
		alpha = 0;
		beta = 1;
	}

	// there are 3 sizes: isize: the origina size of the input array, usize :
	// the size of the upsampled array, and uppadsize the padded+upsampled
	// size, we want both uppadsize and usize to be odd, and we want uppadsize
	// to be the product of small primes (3,5,7)
	double approxratio = 4;
	int64_t isize = input.size();
	int64_t uppadsize = round357(isize*approxratio); 
	int64_t usize;
	while( (usize = (uppadsize-1)/2) % 2 == 0) {
		uppadsize = round357(uppadsize+2);
	}
	assert(uppadsize%2 != 0);
	assert(usize%2 != 0);

//	complex<double> A_phi = std::exp(-imag*PI*sgn(sin(phi))/4+imag*phi/2.)/
//			sqrt(fabs(sin(phi)));
	// since phi [.78,2.35], sin(phi) is positive, sgn(sin(phi)) = 1:
	complex<double> A_phi = std::exp(-I*PI/4.+I*phi/2.) / (usize*sqrt(sin(phi)));
	
	// upsampled version of input
	std::vector<complex<double>> upsampled(usize);
	
	// pre-compute chirps
	auto sigbuff = fftw_alloc_complex(usize);
	auto ab_chirp = createChirp(uppadsize, isize, (double)usize/(double)isize,
			alpha, beta, false);
	auto b_chirp = createChirp(uppadsize, isize, (double)usize/(double)isize,
			beta, 0, false);
	

	interp(input, upsampled);
	
	// pre-multiply 
	for(int64_t nn = -usize/2; nn<=usize/2; nn++) {
		complex<double> tmp1(ab_chirp[nn+uppadsize/2][0], 
				ab_chirp[nn+uppadsize/2][1]);
		upsampled[nn+usize/2] *= tmp1;;
	}
	
	// multiply
	for(int64_t mm = -usize/2; mm<=usize/2; mm++) {
		sigbuff[mm+usize/2][0] = 0;
		sigbuff[mm+usize/2][1] = 0;

		for(int64_t nn = -usize/2; nn<= usize/2; nn++) {
			complex<double> tmp1(b_chirp[mm-nn+uppadsize/2][0], 
					b_chirp[mm-nn+uppadsize/2][1]);
			tmp1 = tmp1*upsampled[nn+usize/2];

			sigbuff[mm+usize/2][0] += tmp1.real();
			sigbuff[mm+usize/2][1] += tmp1.imag();
		}
	}

#ifdef DEBUG
	std::vector<double> mag;
	mag.resize(usize);
	for(size_t ii=0; ii<usize; ii++) {
		mag[ii] = sqrt(sigbuff[ii][0]*sigbuff[ii][0] + 
					sigbuff[ii][1]*sigbuff[ii][1]);
	}
	writePlot("brute_premult.tga", mag);
#endif //DEBUG

	for(int64_t ii=-usize/2; ii<usize/2; ii++) {
		complex<double> tmp1(ab_chirp[ii+uppadsize/2][0], 
				ab_chirp[ii+uppadsize/2][1]);
		complex<double> tmp2(sigbuff[ii+usize/2][0], sigbuff[ii+usize/2][1]);

		upsampled[ii+usize/2] = tmp1*tmp2*A_phi;
	}
	

	out.resize(input.size());
	interp(upsampled, out);
	
	fftw_free(sigbuff);
	fftw_free(b_chirp);
	fftw_free(ab_chirp);
}
	
/**
 * @brief Computes the fractional fourier transform of the input vector and 
 * writes out an equal length array in out. This function is ONLY valid for 
 * 0.5 <= a <= 1.5, use the more general FrFFT for other values.
 *
 * @param input Array to perform frft on
 * @param a_frac Fractional level of fft. 
 * @param out output vector, will be the same size as the input
 */
void floatFrFFT(const std::vector<complex<double>>& input, float a_frac,
		vector<complex<double>>& out)
{
	assert(a_frac <= 1.5 && a_frac >= 0.5);

	const std::complex<double> I(0,1);
	const double PI = acos(-1);
	double phi = a_frac*PI/2;

	double alpha = 1./tan(phi);
	double beta = 1./sin(phi);
	if(a_frac == 1) {
		alpha = 0;
		beta = 1;
	}

	// there are 3 sizes: isize: the origina size of the input array, usize :
	// the size of the upsampled array, and uppadsize the padded+upsampled
	// size, we want both uppadsize and usize to be odd, and we want uppadsize
	// to be the product of small primes (3,5,7)
	double approxratio = 4;
	int64_t isize = input.size();
	int64_t uppadsize = round357(isize*approxratio); 
	int64_t usize;
	while( (usize = (uppadsize-1)/2) % 2 == 0) {
		uppadsize = round357(uppadsize+2);
	}
	assert(uppadsize%2 != 0);
	assert(usize%2 != 0);

//	complex<double> A_phi = std::exp(-imag*PI*sgn(sin(phi))/4+imag*phi/2.)/
//			sqrt(fabs(sin(phi)));
	// since phi [.78,2.35], sin(phi) is positive, sgn(sin(phi)) = 1:
	complex<double> A_phi = std::exp(-I*PI/4.+I*phi/2.) / (usize*sqrt(sin(phi)));

	// upsampled version of input
	std::vector<complex<double>> upsampled(usize); // CACHE

	// create buffers and plans
	auto sigbuff = fftw_alloc_complex(uppadsize); // CACHE
	auto ab_chirp = createChirp(uppadsize, isize, (double)usize/(double)isize,
			alpha, beta, false); // CACHE
	auto b_chirp = createChirp(uppadsize, isize, (double)usize/(double)isize, 
			beta, 0, true); // CACHE

#ifdef DEBUG
	{
		std::vector<double> tmp(uppadsize);
		for(size_t ii=0; ii<uppadsize; ii++)
			tmp[ii] = ab_chirp[ii][0];
		writePlot("abchirp.tga", tmp);
	}
#endif //DEBUG

	fftw_plan sigbuff_plan_fwd = fftw_plan_dft_1d(uppadsize, sigbuff, sigbuff, 
			FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan sigbuff_plan_rev = fftw_plan_dft_1d(uppadsize, sigbuff, sigbuff, 
			FFTW_BACKWARD, FFTW_MEASURE);

	// upsample input
	interp(input, upsampled);
	
#ifdef DEBUG
	{
		std::vector<double> tmp(usize);
		for(size_t ii=0; ii<usize; ii++)
			tmp[ii] = upsampled[ii].real();
		writePlot("upin.tga", tmp);
	}
#endif //DEBUG

	// pre-multiply 
	for(int64_t nn = -usize/2; nn<=usize/2; nn++) {
		complex<double> tmp1(ab_chirp[nn+uppadsize/2][0], 
				ab_chirp[nn+uppadsize/2][1]);
		upsampled[nn+usize/2] *= tmp1;;
	}

	// copy to padded buffer
	for(int64_t nn = -uppadsize/2; nn<=uppadsize/2; nn++) {
		if(nn <= usize/2 && nn >= -usize/2) {
			sigbuff[nn+uppadsize/2][0] = upsampled[nn+usize/2].real();
			sigbuff[nn+uppadsize/2][1] = upsampled[nn+usize/2].imag();
		} else {
			sigbuff[nn+uppadsize/2][0] = 0;
			sigbuff[nn+uppadsize/2][1] = 0;
		}
	}

	// convolve
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
			tmp[ii] = sigbuff[ii][0];
		writePlot("postconvolve.tga", tmp);
	}
#endif //DEBUG

	// I am actually still not 100% on why these should be shifted up (-1) in 
	// sigbuff indexing
	// copy out, negatives
	for(int64_t ii=-usize/2; ii<=0; ii++) {
		upsampled[ii+usize/2].real(sigbuff[uppadsize-1+ii][0]);
		upsampled[ii+usize/2].imag(sigbuff[uppadsize-1+ii][1]);
	}
	// positives
	for(int64_t ii=1; ii<=usize/2; ii++) {
		upsampled[ii+usize/2].real(sigbuff[ii-1][0]);
		upsampled[ii+usize/2].imag(sigbuff[ii-1][1]);
	}

#ifdef DEBUG
	std::vector<double> mag;
	mag.resize(usize);
	for(size_t ii=0; ii<usize; ii++) 
		mag[ii] = abs(upsampled[ii]);
	writePlot("fft_premult.tga", mag);
#endif //DEBUG

	// post-multiply
	for(int64_t ii=-usize/2; ii<=usize/2; ii++) {
		complex<double> tmp1(ab_chirp[ii+uppadsize/2][0], 
				ab_chirp[ii+uppadsize/2][1]);

		upsampled[ii+usize/2] *= tmp1*A_phi;
	}
	

	out.resize(input.size());
	interp(upsampled, out);

	fftw_free(sigbuff);
	fftw_free(b_chirp);
	fftw_free(ab_chirp);
	fftw_destroy_plan(sigbuff_plan_rev);
	fftw_destroy_plan(sigbuff_plan_fwd);
}

int main(int argc, char** argv)
{
	double alpha;
	size_t sz;
	if(argc == 3) {
		alpha = atof(argv[1]);
		sz = atoi(argv[2]);
	} else {
		alpha = .5;
		sz = 127;
	}

	vector<std::complex<double>> in(sz);
	vector<std::complex<double>> fftout(sz);
	vector<std::complex<double>> bruteout(sz);
	for(size_t ii=0; ii<in.size(); ii++) {
		if(ii < in.size()*3/5 && ii > in.size()*2/5)
			in[ii] = 1.;
		else
			in[ii] = 0.;
	}
	
	std::vector<double> phasev(sz);
	std::vector<double> absv(sz);
	for(size_t ii=0; ii<sz; ii++) {
		phasev[ii] = arg(in[ii]);
		absv[ii] = abs(in[ii]);
	}
	writePlot("orig_phase.tga", phasev);
	writePlot("orig_abs.tga", absv);

	floatFrFFTBrute(in, alpha, bruteout);
	floatFrFFT(in, alpha, fftout);

	for(size_t ii=0; ii<fftout.size(); ii++) {
		phasev[ii] = arg(fftout[ii]);
		absv[ii] = abs(fftout[ii]);
	}
	writePlot("fft_phase.tga", phasev);
	writePlot("fft_abs.tga", absv);
	
	for(size_t ii=0; ii<bruteout.size(); ii++) {
		phasev[ii] = arg(bruteout[ii]);
		absv[ii] = abs(bruteout[ii]);
	}
	writePlot("brute_phase.tga", phasev);
	writePlot("brute_abs.tga", absv);

	for(size_t ii=0; ii<sz; ii++) {
		if(fabs(bruteout[ii].real() - fftout[ii].real()) > 0.01) {
			cerr << "Brute force and fft versions differ in real!" << endl;
			cerr << bruteout[ii].real() << " vs " << fftout[ii].real() << endl;
			cerr << bruteout[ii].real()/fftout[ii].real() << endl;
			return -1;
		}
		if(fabs(bruteout[ii].imag() - fftout[ii].imag()) > 0.01) {
			cerr << "Brute force and fft versions differ in imag!" << endl;
			cerr << bruteout[ii].imag() << " vs " << fftout[ii].imag() << endl;
			cerr << bruteout[ii].imag()/fftout[ii].imag() << endl;
			return -1;
		}
	}

	fftw_cleanup();
	return 0;
}




