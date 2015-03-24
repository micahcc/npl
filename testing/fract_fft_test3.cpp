/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
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

int main()
{
	size_t sz = 127;
	std::vector<double> phasev(sz);
	std::vector<double> absv(sz);
	fftw_complex* in = fftw_alloc_complex(sz);
	fftw_complex* fftout = fftw_alloc_complex(sz);
	writePlotAbsAng("start-abs.tga", "start-ang.tga", sz, in);

	for(double alpha = 1.5; alpha<4; alpha += .4) {
		// fill in with rectangle
		for(size_t ii=0; ii<sz; ii++) {
			if(ii < sz*3/5 && ii > sz*2/5) {
				in[ii][0] = 1.;
				in[ii][1] = 0.;
			} else {
				in[ii][0] = 0.;
				in[ii][1] = 0.;
			}
		}
	}

	fractional_ft(sz, in, fftout, 0.5, 0, NULL, true);
	fractional_ft(sz, fftout, fftout, 0.5, 0, NULL, true);
	fractional_ft(sz, fftout, fftout, -1, 0, NULL, true);

	writePlotAbsAng("final-abs.tga", "final-ang.tga", sz, fftout);
	for(size_t ii=0; ii<sz; ii++) {
		std::complex<double> a(fftout[ii][0], fftout[ii][1]);
		std::complex<double> b(in[ii][0], in[ii][1]);
		cerr << a << " vs. " << b << endl;
		double err = std::abs(a-b);
		cerr << err << endl;
		if(err > .01) {
			cerr << "Error greater than threshold!" << endl;
			return -1;
		}
	}

//
//		for(size_t ii=0; ii<sz; ii++) {
//			phasev[ii] = atan2(in[ii][0], in[ii][1]);
//			absv[ii] = sqrt(pow(in[ii][0],2)+pow(in[ii][1],2));
//		}
//		writePlot("orig_phase.tga", phasev);
//		writePlot("orig_abs.tga", absv);
//		fractional_ft(sz, in, fftout, alpha);
//		fractional_ft(sz, fftout, fftout, -alpha);
//
//		for(size_t ii=0; ii<sz; ii++) {
//			phasev[ii] = atan2(fftout[ii][0], fftout[ii][1]);
//			absv[ii] = sqrt(pow(fftout[ii][0],2)+pow(fftout[ii][1],2));
//		}
//		writePlot("restored_phase.tga", phasev);
//		writePlot("restored_abs.tga", absv);
//
//	}

	fftw_free(in);
	fftw_free(fftout);
	fftw_cleanup();
	return 0;
}


