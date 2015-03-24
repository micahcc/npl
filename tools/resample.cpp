/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file resample.cpp Resample images (change resolution)
 *
 *****************************************************************************/

#include <iostream>
#include <tclap/CmdLine.h>

#include "nplio.h"
#include "mrimage.h"
#include "version.h"
#include "mrimage_utils.h"
#include "macros.h"
#include "npltypes.h"

using namespace std;
using namespace npl;

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Changes the resolution of input images. If isotropic "
			"spacing is chosen (-I) then the output image will have isotropic "
			"spacing in the spatial domain. Otherwise individual dimsions "
			"can set with -x,-y,-z-t. Warning upsampling is currently broken, "
			"for lanczos resampling but not for nearest neighbor. "
			"Also the windowing might need to be improved.", ' ', __version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input image.",
		true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_out("o", "output", "Output image.",
		true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<double> a_isospacing("I", "iso-spacing",
			"Make isotropic spacing.", false, 1, "mm", cmd);
	TCLAP::ValueArg<double> a_xspacing("x", "x-space",
			"New x-spacing. " , false, 1, "mm", cmd);
	TCLAP::ValueArg<double> a_yspacing("y", "y-space",
			"New y-spacing. " , false, 1, "mm", cmd);
	TCLAP::ValueArg<double> a_zspacing("z", "z-space",
			"New z-spacing. " , false, 1, "mm", cmd);
	TCLAP::ValueArg<double> a_tspacing("t", "t-space",
			"New t-spacing. " , false, 1, "mm", cmd);
	TCLAP::SwitchArg a_nn("n", "nearest-neighbor",
			"Perform Nearest Neighbor Interpolation. ", cmd);

	vector<string> knownWin({"rect", "hann", "hamming", "sinc", "lanczos",
			"welch"});
	TCLAP::ValuesConstraint<string> consWin(knownWin);

	TCLAP::ValueArg<string> a_window("w", "window", "Window function during "
			"fourier resampling. ", false, "sinc", &consWin, cmd);

	cmd.parse(argc, argv);

	/****************************************************
	 * Read a Single 3D Volume from Input Image
	 ****************************************************/
	ptr<MRImage> fullres = readMRImage(a_in.getValue());

	vector<double> newspace(fullres->ndim());
	for(size_t dd=0; dd<fullres->ndim(); dd++)
		newspace[dd] = fullres->spacing(dd);

	if(a_isospacing.isSet()) {
		for(size_t dd=0; dd<newspace.size(); dd++)
			newspace[dd] = a_isospacing.getValue();
	}

	if(a_xspacing.isSet() && newspace.size() >= 1)
		newspace[0] = a_xspacing.getValue();
	if(a_yspacing.isSet() && newspace.size() >= 2)
		newspace[1] = a_yspacing.getValue();
	if(a_zspacing.isSet() && newspace.size() >= 3)
		newspace[2] = a_zspacing.getValue();
	if(a_tspacing.isSet() && newspace.size() >= 4)
		newspace[3] = a_tspacing.getValue();

	for(size_t dd=0; dd<newspace.size(); dd++)
		cerr << "New Spacing: " << newspace[dd] << endl;

	ptr<MRImage> out;
	if(a_nn.isSet())
		out = resampleNN(fullres, newspace.data());
	else if(a_window.getValue() == "rect")
		out = resample(fullres, newspace.data(), rectWindow);
	else if(a_window.getValue() == "hann")
		out = resample(fullres, newspace.data(), hannWindow);
	else if(a_window.getValue() == "hamming")
		out = resample(fullres, newspace.data(), hammingWindow);
	else if(a_window.getValue() == "sinc" || a_window.getValue() == "lanczos")
		out = resample(fullres, newspace.data(), sincWindow);
	else if(a_window.getValue() == "welch")
		out = resample(fullres, newspace.data(), welchWindow);

	if(a_out.isSet())
		out->write(a_out.getValue());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

	return 0;
}

