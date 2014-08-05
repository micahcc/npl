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
 * @file fract_fft_test2.cpp
 *
 *****************************************************************************/

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

//#define DEBUG

#include "utility.h"
#include "fract_fft.h"

#include "fftw3.h"

using namespace npl;
using namespace std;

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

	
	fftw_complex* in = fftw_alloc_complex(sz);
	fftw_complex* fftout = fftw_alloc_complex(sz);
	fftw_complex* bruteout = fftw_alloc_complex(sz);
	
	// fill with rectangle
	for(size_t ii=0; ii<sz; ii++) {
		if(ii < sz*3/5 && ii > sz*2/5) {
			in[ii][0] = 1.;
			in[ii][1] = 0.;
		} else {
			in[ii][0] = 0.;
			in[ii][1] = 0.;
		}
	}

	std::vector<double> phasev(sz);
	std::vector<double> absv(sz);
	for(size_t ii=0; ii<sz; ii++) {
		phasev[ii] = atan2(in[ii][0], in[ii][1]);
		absv[ii] = sqrt(pow(in[ii][0],2)+pow(in[ii][1],2));
	}
	writePlot("orig_phase.tga", phasev);
	writePlot("orig_abs.tga", absv);

	fractional_ft(sz, in, bruteout, alpha, 0, NULL, true);
	fractional_ft(sz, in, fftout, alpha);

	for(size_t ii=0; ii<sz; ii++) {
		phasev[ii] = atan2(fftout[ii][0], fftout[ii][1]);
		absv[ii] = sqrt(pow(fftout[ii][0],2)+pow(fftout[ii][1],2));
	}
	writePlot("fft_phase.tga", phasev);
	writePlot("fft_abs.tga", absv);
	
	for(size_t ii=0; ii<sz; ii++) {
		phasev[ii] = atan2(bruteout[ii][0], bruteout[ii][1]);
		absv[ii] = sqrt(pow(bruteout[ii][0],2)+pow(bruteout[ii][1],2));
	}
	writePlot("brute_phase.tga", phasev);
	writePlot("brute_abs.tga", absv);

	for(size_t ii=0; ii<sz; ii++) {
		if(fabs(bruteout[ii][0] - fftout[ii][0]) > 0.01) {
			cerr << "Brute force and fft versions differ in real!" << endl;
			cerr << bruteout[ii][0] << " vs " << fftout[ii][0] << endl;
			cerr << bruteout[ii][0]/fftout[ii][0] << endl;
			return -1;
		}
		if(fabs(bruteout[ii][1] - fftout[ii][1]) > 0.01) {
			cerr << "Brute force and fft versions differ in imag!" << endl;
			cerr << bruteout[ii][1] << " vs " << fftout[ii][1] << endl;
			cerr << bruteout[ii][1]/fftout[ii][1] << endl;
			return -1;
		}
	}

	fftw_free(in);
	fftw_free(fftout);
	fftw_free(bruteout);
	fftw_cleanup();
	return 0;
}





