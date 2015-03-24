/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file generalLinearModel.cpp
 *
 *****************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include <Eigen/Dense>
#include "statistics.h"
#include "mrimage.h"
#include "mrimage_utils.h"
#include "kdtree.h"
#include "iterators.h"
#include "accessors.h"
#include "utility.h"
#include "nplio.h"
#include "basic_plot.h"

using std::string;
using namespace npl;
using std::shared_ptr;
using Eigen::MatrixXd;

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Performs a General Linear Model statistical test on "
			"an fMRI image. ",
			' ', __version__ );

	TCLAP::ValueArg<string> a_fmri("i", "input", "fMRI image.",
			true, "", "*.nii.gz", cmd);
	TCLAP::MultiArg<string> a_events("e", "event-reg", "Event-related regression "
			"variable. Three columns, ONSET DURATION VALUE. If these overlap, "
			"an error will be thrown. ", true, "*.txt", cmd);

    TCLAP::ValueArg<string> a_odir("o", "outdir", "Output directory.", false,
            ".", "dir", cmd);

	cmd.parse(argc, argv);

	// read fMRI image
	shared_ptr<MRImage> fmri = readMRImage(a_fmri.getValue());
	if(fmri->ndim() != 4) {
		cerr << "Input should be 4D!" << endl;
		return -1;
	}

	// read the event-related designs, will have rows to match time, and cols
	// to match number of regressors
	int tlen = fmri->tlen();
	double TR = fmri->spacing(3);
	MatrixXd X(tlen, a_events.getValue().size());
	size_t regnum = 0;
	for(auto it=a_events.begin(); it != a_events.end(); it++, regnum++) {
		auto events = readNumericCSV(*it);
		auto v = getRegressor(events, TR, tlen, 0);

		// draw
		writePlot(a_odir.getValue()+"/ev1.svg", v);

		// copy to output
		for(size_t ii=0; ii<tlen; ii++)
			X(ii, regnum) = v[ii];
	}

	ptr<MRImage> bimg, Timg, pimg;
	bimg=dPtrCast<MRImage>(fmri->copyCast(4, osize.data(), FLOAT32));
	Timg=dPtrCast<MRImage>(fmri->copyCast(4, osize.data(), FLOAT32));
	pimg=dPtrCast<MRImage>(fmri->copyCast(4, osize.data(), FLOAT32));

	fmriGLM(fmri, X, bimg, Timg, pimg);

	if() {

	}

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}


