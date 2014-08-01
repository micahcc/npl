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
#include <complex>

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

void fractionalFFT_brute(const std::vector<complex<double>>& input, double alpha,
		vector<complex<double>>& out)
{
	size_t sz = round2(input.size()*2);
	auto yvec = fftw_alloc_complex(sz);
	auto zvec = fftw_alloc_complex(sz);
	auto buffer = fftw_alloc_complex(sz);

	// multiply input by chirps 
	const double PI = acos(-1);
	const std::complex<double> term = PI*I*alpha/(double)sz;

	writeComplex("input.txt", input);

	// create chirp
	for(size_t ii=0; ii<sz; ii++){
		if(ii < input.size()) {
			auto tmp = input[ii]*std::exp(-term*(double)(ii*ii));
			yvec[ii][0] = tmp.real();
			yvec[ii][1] = tmp.imag();

			tmp = std::exp(term*(double)(ii*ii));
			zvec[ii][0] = tmp.real();
			zvec[ii][1] = tmp.imag();
		} else {
			yvec[ii][0] = 0;
			yvec[ii][1] = 0;
			
			zvec[ii][0] = 0;
			zvec[ii][1] = 0;
		}
	}

	writeComplex("bf_yvec.txt", sz, yvec);
	writeComplex("bf_zvec.txt", sz, zvec);
	for(int64_t kk=0; kk<sz; kk++) {
		std::complex<double> sum(0,0);
		for(int64_t jj=0; jj<sz; jj++) {
			std::complex<double> tmp1(yvec[jj][0], yvec[jj][1]);
			if(kk-jj < 0) {
				std::complex<double> tmp2(zvec[jj-kk][0], zvec[jj-kk][1]);
				tmp1 *= tmp2;
			} else {
				std::complex<double> tmp2(zvec[kk-jj][0], zvec[kk-jj][1]);
				tmp1 *= tmp2;
			}
			sum += tmp1;
		}
	}
	
	writeComplex("bf_postconvolv.txt", sz, buffer);

	// chirp again, we only need the first elements, the rest were padding
	out.resize(input.size());
	for(size_t ii=0; ii<sz; ii++){
		std::complex<double> tmp1(buffer[ii][0], buffer[ii][1]);
		std::complex<double> tmp2 = std::exp(-term*((double)(ii*ii)));
		tmp1 *= tmp2;
		buffer[ii][0] = tmp1.real();
		buffer[ii][1] = tmp1.imag();
	}

	writeComplex("bf_postmult.txt", sz, buffer);

	fftw_free(yvec);
	fftw_free(zvec);
	fftw_free(buffer);
}

// based on Baily
void fractionalFFT(const std::vector<complex<double>>& input, double alpha,
		vector<complex<double>>& out)
{
	size_t sz = round2(input.size()*2);
	auto yvec = fftw_alloc_complex(sz);
	auto zvec = fftw_alloc_complex(sz);
	auto buffer = fftw_alloc_complex(sz);

	fftw_plan yvec_plan = fftw_plan_dft_1d((int)sz, yvec, yvec, FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan zvec_plan = fftw_plan_dft_1d((int)sz, zvec, zvec, FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan buffer_plan = fftw_plan_dft_1d((int)sz, buffer, buffer, FFTW_BACKWARD, FFTW_MEASURE);

	// multiply input by chirps 
	const std::complex<double> term = PI*I*alpha/(double)sz;

	writeComplex("input.txt", input);

	// create chirp
	for(size_t ii=0; ii<sz; ii++){
		if(ii < input.size()) {
			auto tmp = input[ii]*std::exp(-term*(double)(ii*ii));
			yvec[ii][0] = tmp.real();
			yvec[ii][1] = tmp.imag();

			tmp = std::exp(term*(double)(ii*ii));
			zvec[ii][0] = tmp.real();
			zvec[ii][1] = tmp.imag();
		} else if(ii < sz-input.size()) {
			yvec[ii][0] = 0;
			yvec[ii][1] = 0;
			
			zvec[ii][0] = 0;
			zvec[ii][1] = 0;
		} else {
			yvec[ii][0] = 0;
			yvec[ii][1] = 0;
			
			auto tmp = std::exp(term*(double)((sz-ii)*(sz-ii)));
			zvec[ii][0] = tmp.real();
			zvec[ii][1] = tmp.imag();
		}
	}

	writeComplex("yvec.txt", sz, yvec);
	writeComplex("zvec.txt", sz, zvec);
	fftw_execute(yvec_plan);
	fftw_execute(zvec_plan);
	writeComplex("fyvec.txt", sz, yvec);
	writeComplex("fzvec.txt", sz, zvec);

	// multiply (convolve)
	for(size_t ii=0; ii<sz; ii++){
		std::complex<double> tmp1(yvec[ii][0], yvec[ii][1]);
		std::complex<double> tmp2(zvec[ii][0], zvec[ii][1]);
		tmp1 = tmp1*tmp2/((double)sz*sz); //normalize convolution
		buffer[ii][0] = tmp1.real();
		buffer[ii][1] = tmp1.imag();
	}
	
	// invert
	writeComplex("prebuffer.txt", sz, buffer);
	fftw_execute(buffer_plan);
	writeComplex("postbuffer.txt", sz, buffer);

	// chirp again, we only need the first elements, the rest were padding
	out.resize(input.size());
	for(size_t ii=0; ii<sz; ii++){
		std::complex<double> tmp1(buffer[ii][0], buffer[ii][1]);
		std::complex<double> tmp2 = std::exp(-term*((double)(ii*ii)));
		tmp1 *= tmp2;
		buffer[ii][0] = tmp1.real();
		buffer[ii][1] = tmp1.imag();
	}

	writeComplex("postmult.txt", sz, buffer);

	fftw_free(yvec);
	fftw_free(zvec);
	fftw_free(buffer);

	fftw_destroy_plan(yvec_plan);
	fftw_destroy_plan(zvec_plan);
	fftw_destroy_plan(buffer_plan);
}

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
	// make nonreal signal sin(x) signal
	for(size_t ii=0; ii<in.size(); ii++) {
		in[ii] = std::exp(-2.*PI*I*(double)ii/100.);
	}
	std::vector<double> realv(sz);
	std::vector<double> imagv(sz);

	for(size_t ii=0; ii<sz; ii++) {
		realv[ii] = in[ii].real();
		imagv[ii] = in[ii].imag();
	}
	writePlot("real_orig.tga", realv);
	writePlot("imag_orig.tga", imagv);

	double PI = acos(-1);
	
	fractionalFFT(in, alpha, out);
	fractionalFFT_brute(in, alpha, out);

//	for(size_t ii=0; ii<in.size(); ii++) {
//		if(norm(in[ii]-out[ii]) > 0.01) {
//			cerr << "Different: " << ii << endl;
//			cerr << in[ii] << " vs " << out[ii] << endl;
//			return -1;
//		}
//	}


	return 0;
}




