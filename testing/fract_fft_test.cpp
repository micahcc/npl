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

fftw_complex* createChirp(int64_t sz, double alpha, double beta, double rate, bool fft)
{
	const double PI = acos(-1);
	
	fftw_complex* out = fftw_alloc_complex(sz);
	auto fwd_plan = fftw_plan_dft_1d((int)sz, out, out, FFTW_FORWARD,
				FFTW_MEASURE);

	complex<double> imag(0, 1);
	complex<double> eterm = imag*PI*(alpha-beta)/((double)rate*sz);
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

void floatFrFFTBrute(const std::vector<complex<double>>& input, float a_frac,
		vector<complex<double>>& out)
{
	assert(a_frac <= 1.5 && a_frac >= 0.5);
	writeComplex("input.txt", input);

	const std::complex<double> I(0,1);
	const double PI = acos(-1);
	double phi = a_frac*PI/2;

	double alpha = 1./tan(phi);
	double beta = 1./sin(phi);
	if(a_frac == 1) {
		alpha = 0;
		beta = 1;
	}
	std::complex<double> avg(0);
//	complex<double> A_phi = std::exp(-imag*PI*sgn(sin(phi))/4+imag*phi/2.)/
//			sqrt(fabs(sin(phi)));
	// since phi [.78,2.35], sin(phi) is positive, sgn(sin(phi)) = 1:
	complex<double> A_phi = std::exp(-I*PI/4.+I*phi/2.) / sqrt(sin(phi));
	complex<double> tmp1, tmp2;

	double approxratio = 2;
	int64_t isize = input.size();
	int64_t uppadsize = round357(isize*approxratio); 
	int64_t usize;
	while( (usize = (uppadsize-1)/2) % 2 == 0) {
		uppadsize += 2;
	}
	assert(uppadsize%2 != 0);
	assert(usize%2 != 0);
	std::vector<complex<double>> upsampled(usize);
	double upratio = (double)usize/(double)isize;
	double space_u = 1./usize;
	
	assert(uppadsize%2 == 1);
	assert(usize%2 == 1);

	// upsample input
	interp(input, upsampled);
	
	// for debug purposes, write the magnitiude
	std::vector<double> mag(usize);
	for(size_t ii=0; ii<usize; ii++) {
		mag[ii] = real(upsampled[ii]);
	}
	writePlot("up_abs.tga", mag);

	// copy to padded buffer
	auto sigbuff = fftw_alloc_complex(uppadsize);
	for(int64_t ii=-uppadsize/2; ii<=uppadsize/2; ii++) {
		if(ii <= usize/2 && ii >= -usize/2) {
			sigbuff[ii+uppadsize/2][0] = upsampled[ii+usize/2].real();
			sigbuff[ii+uppadsize/2][1] = upsampled[ii+usize/2].imag();
			assert(!std::isnan(sigbuff[ii+uppadsize/2][0]));
			assert(!std::isnan(sigbuff[ii+uppadsize/2][1]));
		} else {
			sigbuff[ii+uppadsize/2][0] = 0;
			sigbuff[ii+uppadsize/2][1] = 0;
		}
	}
	
	// debug
	mag.resize(uppadsize);
	for(size_t ii=0; ii<uppadsize; ii++) {
		mag[ii] = sqrt((sigbuff[ii][0]*sigbuff[ii][0])+(sigbuff[ii][1]*sigbuff[ii][1]));
		assert(!std::isnan(mag[ii]));
	}
	writePlot("sigbuff.tga", mag);

	// pre-compute alpha-beta chirp
	auto ab_chirp = fftw_alloc_complex(uppadsize);
	for(int64_t ii=-uppadsize/2; ii<=uppadsize/2; ii++) {
		double ff = ((double)ii)/upratio;
		auto tmp = std::exp(I*PI*(alpha-beta)*ff*ff/(double)isize);
		ab_chirp[ii+uppadsize/2][0] = tmp.real();
		ab_chirp[ii+uppadsize/2][1] = tmp.imag();
	}
	for(size_t ii=0; ii<uppadsize; ii++) {
		mag[ii] = ab_chirp[ii][0];
		assert(!std::isnan(mag[ii]));
	}
	writePlot("ab_chirp.tga", mag);
	
	// pre-compute beta chirp
	auto b_chirp = fftw_alloc_complex(uppadsize);
	for(int64_t ii=-uppadsize/2; ii<=uppadsize/2; ii++) {
		double ff = ((double)ii)/upratio;
		auto tmp = std::exp(I*PI*beta*ff*ff/(double)isize);
		b_chirp[ii+uppadsize/2][0] = tmp.real();
		b_chirp[ii+uppadsize/2][1] = tmp.imag();
	}
	for(size_t ii=0; ii<uppadsize; ii++) {
		mag[ii] = b_chirp[ii][0];
		assert(!std::isnan(mag[ii]));
	}
	writePlot("b_chirp.tga", mag);
	
	// multiply
	auto buff = fftw_alloc_complex(usize);
	for(int64_t mm = -usize/2; mm<=usize/2; mm++) {
		buff[mm+usize/2][0] = 0;
		buff[mm+usize/2][1] = 0;

		for(int64_t nn = -usize/2; nn<= usize/2; nn++) {
			complex<double> tmp1(b_chirp[mm-nn+uppadsize/2][0], 
					b_chirp[mm-nn+uppadsize/2][1]);
			complex<double> tmp2(ab_chirp[nn+uppadsize/2][0], 
					ab_chirp[nn+uppadsize/2][1]);
			tmp1 = tmp1*tmp2*upsampled[nn+usize/2];

			buff[mm+usize/2][0] += tmp1.real();
			buff[mm+usize/2][1] += tmp1.imag();
		}
	}

	mag.resize(usize);
	for(size_t ii=0; ii<usize; ii++) 
		mag[ii] = sqrt(buff[ii][0]*buff[ii][0]+buff[ii][1]*buff[ii][1]);
	writePlot("buff.tga", mag);

	for(int64_t ii=-usize/2; ii<usize/2; ii++) {
		complex<double> tmp1(ab_chirp[ii+uppadsize/2][0], 
				ab_chirp[ii+uppadsize/2][1]);
		complex<double> tmp2(buff[ii+usize/2][0], buff[ii+usize/2][1]);

		upsampled[ii+usize/2] = tmp1*tmp2*A_phi*space_u;
	}
	
	mag.resize(usize);
	for(size_t ii=0; ii<usize; ii++) 
		mag[ii] = abs(upsampled[ii]);
	writePlot("up_outmag.tga", mag);

	out.resize(input.size());
	interp(upsampled, out);
}
	
void floatFrFFT(const std::vector<complex<double>>& input, float a_frac,
		vector<complex<double>>& out)
{
	assert(a_frac <= 1.5 && a_frac >= 0.5);
	writeComplex("input.txt", input);

	const std::complex<double> I(0,1);
	const double PI = acos(-1);
	double phi = a_frac*PI/2;

	double alpha = 1./tan(phi);
	double beta = 1./sin(phi);
	if(a_frac == 1) {
		alpha = 0;
		beta = 1;
	}
	std::complex<double> avg(0);
//	complex<double> A_phi = std::exp(-imag*PI*sgn(sin(phi))/4+imag*phi/2.)/
//			sqrt(fabs(sin(phi)));
	// since phi [.78,2.35], sin(phi) is positive, sgn(sin(phi)) = 1:
	complex<double> A_phi = std::exp(-I*PI/4.+I*phi/2.) / sqrt(sin(phi));
	complex<double> tmp1, tmp2;

	double approxratio = 2;
	int64_t isize = input.size();
	int64_t uppadsize = round357(isize*approxratio); 
	int64_t usize;
	while( (usize = (uppadsize-1)/2) % 2 == 0) {
		uppadsize += 2;
	}
	assert(uppadsize%2 != 0);
	assert(usize%2 != 0);
	std::vector<complex<double>> upsampled(usize);
	double upratio = (double)usize/(double)isize;
	double space_u = 1./usize;

	// create buffers and plans
	auto sigbuff = fftw_alloc_complex(uppadsize);
	auto ab_chirp = fftw_alloc_complex(uppadsize);
	auto b_chirp = fftw_alloc_complex(uppadsize);

	fftw_plan sigbuff_plan_fwd = fftw_plan_dft_1d(uppadsize, sigbuff, sigbuff, 
			FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan sigbuff_plan_rev = fftw_plan_dft_1d(uppadsize, sigbuff, sigbuff, 
			FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan b_chirp_plan_fwd = fftw_plan_dft_1d(uppadsize, b_chirp, b_chirp, 
			FFTW_FORWARD, FFTW_MEASURE);

	assert(uppadsize%2 == 1);
	assert(usize%2 == 1);

	// upsample input
	interp(input, upsampled);

#ifdef NDEBUG
	// for debug purposes, write the magnitiude
	std::vector<double> mag(usize);
	for(size_t ii=0; ii<usize; ii++) {
		mag[ii] = real(upsampled[ii]);
	}
	writePlot("up_abs.tga", mag);
#endif //NDEBUG

	
#ifdef NDEBUG
	// debug
	mag.resize(uppadsize);
	for(size_t ii=0; ii<uppadsize; ii++) {
		mag[ii] = sqrt((sigbuff[ii][0]*sigbuff[ii][0])+(sigbuff[ii][1]*sigbuff[ii][1]));
		assert(!std::isnan(mag[ii]));
	}
	writePlot("sigbuff.tga", mag);
#endif //NDEBUG

	// pre-compute alpha-beta chirp
	for(int64_t ii=-uppadsize/2; ii<=uppadsize/2; ii++) {
		double ff = ((double)ii)/upratio;
		auto tmp = std::exp(I*PI*(alpha-beta)*ff*ff/(double)isize);
		ab_chirp[ii+uppadsize/2][0] = tmp.real();
		ab_chirp[ii+uppadsize/2][1] = tmp.imag();
	}

#ifdef NDEBUG
	for(size_t ii=0; ii<uppadsize; ii++) {
		mag[ii] = ab_chirp[ii][0];
		assert(!std::isnan(mag[ii]));
	}
	writePlot("ab_chirp.tga", mag);
#endif //NDEBUG
	
	// pre-compute beta chirp (only frequency domain is needed)
	for(int64_t ii=-uppadsize/2; ii<=uppadsize/2; ii++) {
		double ff = ((double)ii)/upratio;
		auto tmp = std::exp(I*PI*beta*ff*ff/(double)isize);
		b_chirp[ii+uppadsize/2][0] = tmp.real();
		b_chirp[ii+uppadsize/2][1] = tmp.imag();
	}
	fftw_execute(b_chirp_plan_fwd);

#ifdef NDEBUG
	for(size_t ii=0; ii<uppadsize; ii++) {
		mag[ii] = b_chirp[ii][0];
		assert(!std::isnan(mag[ii]));
	}
	writePlot("b_chirp.tga", mag);
#endif //NDEBUG
	
	// pre-multiply // copy to padded buffer
	for(int64_t nn = -uppadsize/2; nn<=uppadsize/2; nn++) {
		if(nn <= usize/2 && nn >= -usize/2) {
			complex<double> tmp1(ab_chirp[nn+uppadsize/2][0], 
					ab_chirp[nn+uppadsize/2][1]);
			tmp1 *= upsampled[nn+usize/2];
			sigbuff[nn+uppadsize/2][0] = tmp1.real();
			sigbuff[nn+uppadsize/2][1] = tmp1.imag();
		} else {
			sigbuff[nn+uppadsize/2][0] = 0;
			sigbuff[nn+uppadsize/2][1] = 0;
		}
	}

	// convolve
	fftw_execute(sigbuff_plan_fwd);
	for(size_t ii=0; ii<uppadsize; ii++) {
		complex<double> tmp1(sigbuff[ii][0], sigbuff[ii][1]);
		complex<double> tmp2(b_chirp[ii][0], b_chirp[ii][1]);
		tmp1 *= tmp2;
		sigbuff[ii][0] = tmp1.real();
		sigbuff[ii][1] = tmp1.imag();
	}
	fftw_execute(sigbuff_plan_rev);

#ifdef NDEBUG
	mag.resize(usize);
	for(size_t ii=0; ii<usize; ii++) 
		mag[ii] = sqrt(buff[ii][0]*buff[ii][0]+buff[ii][1]*buff[ii][1]);
	writePlot("buff.tga", mag);
#endif //NDEBUG

	// post-multiply, and copy out valid region
	for(int64_t ii=-usize/2; ii<usize/2; ii++) {
		complex<double> tmp1(ab_chirp[ii+uppadsize/2][0], 
				ab_chirp[ii+uppadsize/2][1]);
		complex<double> tmp2(sigbuff[ii+usize/2][0], sigbuff[ii+usize/2][1]);

		upsampled[ii+usize/2] = tmp1*tmp2*A_phi*space_u;
	}
	
#ifdef NDEBUG
	mag.resize(usize);
	for(size_t ii=0; ii<usize; ii++) 
		mag[ii] = abs(upsampled[ii]);
	writePlot("up_outmag.tga", mag);
#endif //NDEBUG

	out.resize(input.size());
	interp(upsampled, out);
}

int main(int argc, char** argv)
{
//	for(int64_t ii=1; ii<200; ii++) {
//		std::cerr << ii << " -> " << round357(ii) << std::endl;
//	}
//
	if(argc != 3) {
		return -1;
	}
	double alpha = atof(argv[1]);
	size_t sz = atoi(argv[2]);
	vector<std::complex<double>> in(sz);
	vector<std::complex<double>> out(sz);
	for(size_t ii=0; ii<in.size(); ii++) {
		if(ii < in.size()*3/5 && ii > in.size()*2/5)
			in[ii] = 1.;
		else
			in[ii] = 0.;
	}
	// make nonreal signal sin(x) signal
//	for(size_t ii=0; ii<in.size(); ii++) {
//		in[ii] = std::exp(-2.*PI*I*(double)ii/100.);
//	}
	std::vector<double> phasev(sz);
	std::vector<double> absv(sz);

	for(size_t ii=0; ii<sz; ii++) {
		phasev[ii] = arg(in[ii]);
		absv[ii] = abs(in[ii]);
	}
	writePlot("orig_phase.tga", phasev);
	writePlot("orig_abs.tga", absv);

	floatFrFFTBrute(in, alpha, out);

	phasev.resize(out.size());
	absv.resize(out.size());
	for(size_t ii=0; ii<out.size(); ii++) {
		phasev[ii] = arg(out[ii]);
		absv[ii] = abs(out[ii]);
	}
	writePlot("frft_phase.tga", phasev);
	writePlot("frft_abs.tga", absv);

//	for(size_t ii=0; ii<in.size(); ii++) {
//		if(norm(in[ii]-out[ii]) > 0.01) {
//			std::cerr << "Different: " << ii << std::endl;
//			std::cerr << in[ii] << " vs " << out[ii] << std::endl;
//			return -1;
//		}
//	}


	fftw_cleanup();
	return 0;
}




