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

#define DEBUG 1

#include <complex>
#include "basic_plot.h"
#include "chirpz.h"

#include "fftw3.h"


using namespace npl;
using namespace std;

clock_t brute1_time = 0;
clock_t brute2_time = 0;
clock_t fft_time = 0;

int testChirpz(size_t length, double alpha, bool debug = false)
{
	cerr << "Testing length = " << length << " with alpha = " << alpha << endl;
	ostringstream oss;
	oss << "test_" << length << "_" << alpha;

	auto line = fftw_alloc_complex(length);
	auto line_brute = fftw_alloc_complex(length);
	auto line_brute2 = fftw_alloc_complex(length);
	auto line_fft = fftw_alloc_complex(length);
	
	// fill with a noisy square
	double sum = 0;
	for(size_t ii=0; ii<length; ii++){ 
		double v = 0;
		if(ii > (length)/2. - 10 && ii < length/2. + 10) {
			v = 1;
//			v = std::exp(-pow(ii-length/2.,2)/16);
			line[ii][0] = v;
			line[ii][1] = 0;
			sum += v;
		} else {
			line[ii][0] = 0;
			line[ii][1] = 0;
		}
	}
	if(debug) cerr << "Test Signal:\n";
	for(size_t ii=0; ii<length; ii++) {
		line[ii][0] /= sum;
		if(debug) cerr << line[ii][0] << endl;
	}
	
	if(debug) {
		writePlotReIm(oss.str()+"_input.svg", length, line);
	}

	clock_t n = clock();
	chirpzFT_brute(length, line, line_brute, alpha);
	brute1_time += clock()-n;

	n = clock();
	chirpzFT_brute2(length, line, line_brute2, alpha, debug);
	brute2_time += clock()-n;
	
	n = clock();
	chirpzFFT(length, line, line_fft, alpha, debug);
	fft_time += clock()-n;

	if(debug) {
		writePlotReIm(oss.str()+"_chirpzBruteFT.svg", length, line_brute);
		writePlotReIm(oss.str()+"_chirpzBruteFT2.svg", length, line_brute2);
		writePlotReIm(oss.str()+"_chirpzFFT.svg", length, line_fft);
	}

	for(size_t ii=0; ii<length; ii++) {
		complex<double> a(line_brute[ii][0], line_brute[ii][1]);
		complex<double> b(line_fft[ii][0], line_fft[ii][1]);
		
		if(abs(a.real() - b.real()) > 0.3) {
			cerr << "Error, absolute difference in chirpzFFT" << endl;
			cerr << a.real() << " vs " << b.real() << endl;
			return -1;
		}
		if(abs(a.imag() - b.imag()) > 0.3) {
			cerr << "Error, absolute difference in chirpzFFT" << endl;
			cerr << a.imag() << " vs " << b.imag() << endl;
			return -1;
		}
		
	}

	return 0;
}

int main(int argc, char** argv)
{
	if(argc == 3) {
		int length = atoi(argv[1]);
		double alpha = atof(argv[2]);
		cerr << length << "," << alpha << endl;
		if(testChirpz(length, alpha, true) != 0)
			return -1;
	} else {
		if(testChirpz(256, 1) != 0)
			return -1;
		if(testChirpz(1024, .95) != 0)
			return -1;
		if(testChirpz(321, .95) != 0)
			return -1;
		if(testChirpz(727, .15) != 0)
			return -1;
		if(testChirpz(727, .55) != 0)
			return -1;
		if(testChirpz(727, -1) != 0)
			return -1;
		if(testChirpz(128, -.95) != 0)
			return -1;
		if(testChirpz(1024, -.95) != 0)
			return -1;
		if(testChirpz(321, -.95) != 0)
			return -1;
		if(testChirpz(727, -.15) != 0)
			return -1;
		if(testChirpz(727, -.55) != 0)
			return -1;
		if(testChirpz(727, -1) != 0)
			return -1;
	}
	cerr << "Brute1 Time:" << brute1_time << endl;
	cerr << "Brute2 Time:" << brute2_time << endl;
	cerr << "Fast Chirp Time:" << fft_time << endl;
	return 0;
}





