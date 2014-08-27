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
clock_t zoom_time = 0;

double alpha = -1;

fftw_complex line[] = {{0.03125,0},
{-0.000216305,0.000747931},
{-0.000380475,0.000946459},
{2.50957e-05,-0.000388651},
{0.000989912,-0.000162022},
{0.000808298,-0.000945903},
{-5.06103e-05,0.000847835},
{-0.00220405,-0.00140134},
{-0.000976563,0.000976562},
{-0.000391243,-0.00211353},
{0.0101042,0.0148596},
{0.00362542,0.00364142},
{-0.00219561,-0.00196647},
{0.00151088,0.00510937},
{-0.00162308,-0.000469246},
{0.00149291,0.000545992},
{0,-0.00195312},
{0.00138813,0.000517917},
{-0.00142822,-0.000510394},
{-0.000233746,0.000641898},
{-0.00113858,-0.000585406},
{0.00130033,0.000708506},
{2.44012e-05,0.000225943},
{0.000877585,0.000567459},
{-0.000976563,-0.000976562},
{0.000520198,-0.000834397},
{-0.000647485,-0.000449016},
{0.000286254,0.00122494},
{-0.00156197,0.00121905},
{0.00048997,0.000247633},
{0.00181375,-0.00138292},
{0.00634527,-0.00139419},
{-0.00390625,0},
{0.00634527,0.00139419},
{0.00181375,0.00138292},
{0.00048997,-0.000247633},
{-0.00156197,-0.00121905},
{0.000286254,-0.00122494},
{-0.000647485,0.000449016},
{0.000520198,0.000834397},
{-0.000976563,0.000976562},
{0.000877585,-0.000567459},
{2.44012e-05,-0.000225943},
{0.00130033,-0.000708506},
{-0.00113858,0.000585406},
{-0.000233746,-0.000641898},
{-0.00142822,0.000510394},
{0.00138813,-0.000517917},
{0,0.00195312},
{0.00149291,-0.000545992},
{-0.00162308,0.000469246},
{0.00151088,-0.00510937},
{-0.00219561,0.00196647},
{0.00362542,-0.00364142},
{0.0101042,-0.0148596},
{-0.000391243,0.00211353},
{-0.000976563,-0.000976562},
{-0.00220405,0.00140134},
{-5.06103e-05,-0.000847835},
{0.000808298,0.000945903},
{0.000989912,0.000162022},
{2.50957e-05,0.000388651},
{-0.000380475,-0.000946459},
{-0.000216305,-0.000747931}};

// this won't work because the frequencies will cause aliasing!
//fftw_complex line[] = {
//	{0.015625,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0},
//	{0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0},
//	{0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0},
//	{0,0}, {0,0}, {0,0}, {0.015625,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0},
//	{0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0},
//	{0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0},
//	{0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}
//};

int testChirpz(double alpha)
{
	bool debug = true;
	size_t length = sizeof(line)/sizeof(fftw_complex);
	cerr << "Testing length = " << length << " with alpha = " << alpha << endl;
	ostringstream oss;
	oss << "test_" << length << "_" << alpha;

	auto line_brute = fftw_alloc_complex(length);
	auto line_brute2 = fftw_alloc_complex(length);
	auto line_fft = fftw_alloc_complex(length);
	auto line_zoom = fftw_alloc_complex(length);
	
	if(debug) {
		cerr << "Test Signal:\n";
		for(size_t ii=0; ii<length; ii++) {
			cerr << line[ii][0] << endl;
		}
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
	
	n = clock();
	chirpzFT_zoom(length, line, line_zoom, alpha);
	zoom_time += clock()-n;

	if(debug) {
		writePlotReIm(oss.str()+"_chirpzBruteFT.svg", length, line_brute);
		writePlotReIm(oss.str()+"_chirpzBruteFT2.svg", length, line_brute2);
		writePlotReIm(oss.str()+"_chirpzFFT.svg", length, line_fft);
		writePlotReIm(oss.str()+"_chirpzZoom.svg", length, line_zoom);
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
	double alpha = 1;
	if(argc == 2)
		alpha = atof(argv[1]);

	if(testChirpz(alpha) != 0)
		return -1;
	return 0;
}






