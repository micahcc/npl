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
 * @file ols.cpp Perform ordinary least Tool to apply a deformation field to another image.
 * Not yet functional
 *
 *****************************************************************************/

#include <iostream>
#include <cstdio>
#include <iomanip>
#include <string>
#include <fstream>

#include <set>
#include <vector>
#include <list>
#include <cmath>
#include <memory>

#include "fftw3.h"
#include <tclap/CmdLine.h>

#include "basic_plot.h"
#include "statistics.h"
#include "npltypes.h"
#include "nplio.h"
#include "utility.h"
#include "mrimage.h"
#include "mrimage_utils.h"
#include "iterators.h"
#include "version.h"

using std::list;
using std::set;
using std::vector;
using std::string;
using std::endl;
using std::cerr;
using std::cout;
using namespace npl;

const double PI = acos(-1);

int main(int argc, char* argv[])
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */
	TCLAP::CmdLine cmd("This is a program to perform basic OLS regression of "
			"an fMRI data image. It takes a single fMRI image and a single "
			"csv file. ", ' ', __version__ );

	// arguments
	TCLAP::ValueArg<std::string> a_input("i","in","Input fMRI Image",true,"",
			"4D Image", cmd);

	// for regressing-out
	TCLAP::ValueArg<std::string> a_tscore("t","tscore","Output 4D Image, "
			"containing t-statistics. One volume per input regressor", false,
			"", "4D Image", cmd);

	TCLAP::ValueArg<std::string> a_beta("b","bimg","Output 4D Image, "
			"containing beta (slope). One volume per input regressor", false,
			"", "4D Image", cmd);

	TCLAP::ValueArg<std::string> a_regressors("r","regressor",
			"Input Regressor.", false,"", "*.csv", cmd);

	// parse arguments
	cmd.parse(argc, argv);

	auto fmri = readMRImage(a_input.getValue());
	size_t tlen = fmri->tlen();
	if(fmri->ndim() < 4 || fmri->tlen() <= 1) {
		cerr << "Input image should be 4D!" << endl;
		return -1;
	}

	// Load Regressors into X
	auto regr = readNumericCSV(a_regressors.getValue());
	if(regr.size() != tlen) {
		cerr<<"Regressor matrix rows does not match fMRI timepoints!"<<endl;
		return -1;
	}

	size_t nregr = regr[0].size();
	MatrixXd X(regr.size(), nregr+1);
	for(size_t rr=0; rr<X.rows(); rr++) {
		if(regr[rr].size() != nregr) {
			cerr<<"Regressor rows have different sizes!"<<endl;
			return -2;
		}

		for(size_t cc=0; cc<nregr; cc++) {
			X(rr, cc) = regr[rr][cc];
		}
	}

	// Add Intercept
	for(size_t rr=0; rr<X.rows(); rr++)
		X(rr, nregr) = 1;

	// Create Statistic Images
	size_t statsz[4];
	for(size_t dd=0; dd<3; dd++)
		statsz[dd] = fmri->dim(dd);
	statsz[3] = X.cols();
	auto timg = dPtrCast<MRImage>(fmri->createAnother(4, statsz, FLOAT32));
	auto betaimg = dPtrCast<MRImage>(fmri->createAnother(4, statsz, FLOAT32));

	// Load Each Timeseries and perform regression
	auto Xinv = pseudoInverse(X);
	auto covInv = pseudoInverse(X.transpose()*X);
	VectorXd y(tlen);

	const double MAX_T = 100;
	const double STEP_T = 0.1;
	StudentsT distrib(X.rows(), STEP_T, MAX_T);

	RegrResult result;
	Vector3DIter<double> tit(timg);
	Vector3DIter<double> bit(betaimg);
	Vector3DIter<double> iit(fmri);
	for(; !iit.eof(); ++iit, ++bit, ++tit) {
		for(size_t tt=0; tt<tlen; tt++)
			y[tt] = iit[tt];

		regress(&result, y, X, covInv, Xinv, distrib);
		for(size_t cc=0; cc<nregr; cc++) {
			tit.set(cc, result.t[cc]);
			bit.set(cc, result.bhat[cc]);
		}
	}

	if(a_beta.isSet())
		betaimg->write(a_beta.getValue());

	if(a_tscore.isSet())
		timg->write(a_tscore.getValue());

	// done, catch all argument errors
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}

