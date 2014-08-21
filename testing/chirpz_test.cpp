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
 * @file fft_rotate_test.cpp A test of automatica computation of the rotation 
 * axis, based on the pseudo-polar fourier transform
 *
 *****************************************************************************/

#include <version.h>
#include <string>
#include <stdexcept>

#include <Eigen/Geometry> 

#define DEBUG 1

#include "basic_plot.h"
#include "chirpz.h"

#include "fftw3.h"


using namespace npl;
using namespace std;


int testPowerFFT(size_t length, double alpha)
{
	auto line = fftw_alloc_complex(length);
	auto line_brute = fftw_alloc_complex(length);
	auto line_fft = fftw_alloc_complex(length);
	
	// fill with a noisy square
	double sum = 0;
	for(size_t ii=0; ii<length; ii++){ 
		if(ii > 2.*(length-1)/5 && ii < (length-1)*3./5) {
			line[ii][0] = 1;
			line[ii][1] = 0;
			sum += 1;
		} else {
			line[ii][0] = 0;
			line[ii][1] = 0;
		}
	}
	for(size_t ii=0; ii<length; ii++) 
		line[ii][0] /= sum;
	
	//writePlotReIm("input.svg", length, line);

	writePlotReIm("input.svg", length, line);
	chirpzFT_brute(length, line, line_brute, alpha);
	chirpzFFT(length, line, line_fft, alpha);

	writePlotReIm("powerBruteFT.svg", length, line_brute);
	writePlotReIm("chirpzFFT.svg", length, line_fft);

	for(size_t ii=0; ii<length; ii++) {
		complex<double> a(line_brute[ii][0], line_brute[ii][1]);
		complex<double> b(line[ii][0], line[ii][1]);
		
		if(abs(abs(a) - abs(b)) > 0.001) {
			cerr << "Error, absolute difference in chirpzFFT" << endl;
			return -1;
		}
		if(abs(arg(a) - arg(b)) > 0.1) {
			cerr << "Error, angle difference in chirpzFFT" << endl;
			return -1;
		}
	}

	return 0;
}

int main()
{
	// test the 'Power' Fourier Transform
	if(testPowerFFT(128, 1) != 0)
		return -1;

	return 0;
}





