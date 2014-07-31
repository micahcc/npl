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
#include <vector>
#include <complex>

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

using namespace std;
using namespace npl;

void fractionalFFT(const std::vector<complex<double>>& input, double alpha,
		vector<complex<double>>& out)
{
	size_t sz = round2(input.size())*2;
	auto EcF = fftw_alloc_complex(sz);
	auto EdF = fftw_alloc_complex(sz);
	auto g_k = fftw_alloc_complex(sz);

	fftw_plan EcF_plan = fftw_plan_dft_1d((int)sz, EcF, EcF, FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan EdF_plan = fftw_plan_dft_1d((int)sz, EdF, EdF, FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan inv_plan = fftw_plan_dft_1d((int)sz, g_k, g_k, FFTW_BACKWARD, FFTW_MEASURE);

	// multiply input by chirps 
	const double PI = acos(-1);
	const std::complex<double> imag(0,1);
	std::complex<double> yj;
	std::complex<double> zj;
	std::complex<double> term = PI*imag*alpha;
	size_t nonzero = 0;
	for(size_t ii=0; ii<sz; ii++){
		if(ii<input.size()) {
			nonzero++;
			yj = input[ii]*std::exp(-PI*imag*(alpha*ii*ii));
			zj = std::exp(PI*imag*(alpha*ii*ii));
		} else if(ii < sz-input.size()) {
			yj = 0;
			zj = 0;
		} else if(ii < sz) {
			yj = 0;
			zj = std::exp(PI*imag*(alpha*(sz-ii)*(sz-ii)));
			nonzero++;
		}

		
		EcF[ii][0] = yj.real();
		EcF[ii][1] = yj.imag();
		
		EdF[ii][0] = zj.real();
		EdF[ii][1] = zj.imag();
	}
	
	// fourier transform both chirp-multiplied
	fftw_execute(EcF_plan);
	fftw_execute(EdF_plan);

	// multiply (really convolve)
	for(size_t ii=0; ii<sz; ii++) {
		yj.real(EcF[ii][0]);
		yj.imag(EcF[ii][1]);
		zj.real(EdF[ii][0]);
		zj.imag(EdF[ii][1]);

		// normalize during convolution
		yj = yj*zj/((double)nonzero*nonzero); ///(double)(sz*sz);

		g_k[ii][0] = yj.real();
		g_k[ii][1] = yj.imag();
	}

	// invert
	fftw_execute(inv_plan);

	// chirp again, we only need the first elements, the rest were padding
	out.resize(input.size());
	for(size_t ii=0; ii<input.size(); ii++){
		yj.real(g_k[ii][0]);
		yj.imag(g_k[ii][1]);
		yj = std::exp(-PI*imag*(alpha*ii*ii))*yj;
		out[ii] = yj;
	}

	fftw_free(EdF);
	fftw_free(EcF);
	fftw_free(g_k);
}

int main()
{
	vector<std::complex<double>> in(200);
	vector<std::complex<double>> out(200);
	// make square wave signal
	for(size_t ii=0; ii<in.size(); ii++) {
		if(fabs((double)in.size()/2.0 - ii) < 0.1*in.size()) {
			in[ii].real(1);
		} else {
			in[ii].real(0);
		}
		in[ii].imag(0);
	}

	std::vector<std::complex<double>> real, imag;
	fractionalFFT(in, 1./in.size(), out);
	fractionalFFT(out, -1./in.size(), out);

	for(size_t ii=0; ii<in.size(); ii++) {
		if(norm(in[ii]-out[ii]) > 0.0001) {
			cerr << "Different:" << endl;
			cerr << in[ii] << " vs " << out[ii] << endl;
			return -1;
		}
	}
//	writePlot("square.tga", in);
//	writePlot("real.tga", real);
//	writePlot("imag.tga", imag);

	return 0;
}




