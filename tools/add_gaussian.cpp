/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file applyDeform.cpp Tool to apply a deformation field to another image.
 * Not yet functional
 *
 *****************************************************************************/

#include "version.h"
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include "nplio.h"
#include "mrimage.h"
#include "mrimage_utils.h"
#include "iterators.h"
#include "statistics.h"
#include "accessors.h"

using std::string;
using namespace npl;
int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Adds a gaussian random variable to every pixel.",
			' ', __version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input image.",
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_out("o", "output", "Output image.",
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<double> a_mean("m", "mean", "Mean of gaussian.",
			false, 0, "mu", cmd);
	TCLAP::ValueArg<double> a_sd("s", "sd", "Standard deviation.",
			false, 1, "sd", cmd);
	TCLAP::ValueArg<double> a_snr("n", "snr", "Choose 0 mean and sd equal to "
			"this ratio of the input standard deviation.",
			false, 0.1, "snr", cmd);

	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	ptr<MRImage> inimg(readMRImage(a_in.getValue()));

	double sd = a_sd.getValue();
	double mu = a_mean.getValue();
	if(a_snr.isSet()) {
		double mu = 0;
		double var = 0;
		for(FlatIter<double> it(inimg); !it.eof(); ++it) {
			mu += *it;
			var += (*it)*(*it);
		}
		var = sample_var(inimg->elements(), mu, var);
		sd = sqrt(var);
		mu = 0;
	}

	cerr << "Mean: " << mu << ", sd: " << sd << endl;
	std::normal_distribution<> randn(mu, sd);
	std::default_random_engine rng;

	for(FlatIter<double> it(inimg); !it.eof(); ++it)
		it.set(*it + randn(rng));
	inimg->write(a_out.getValue());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}

