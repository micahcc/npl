/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file smooth.cpp Smooth images one volume at a time.
 *
 *****************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <iostream>

#include "mrimage.h"
#include "nplio.h"
#include "mrimage_utils.h"
#include "macros.h"

using namespace npl;
using namespace std;

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Smooth image one volume at a time. Image will be cast "
			"to double then converted back to input type (or whatever "
			"out-type is", ' ', __version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input image (may be >3D).",
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<double> a_stddev("s", "stddev", "Standard deviation of "
			"gaussian kernel (in mm)." , false, 2, "mm", cmd);
	TCLAP::ValueArg<string> a_out("o", "output", "Output image identical "
			"to input image. .", true, "", "*.nii.gz", cmd);

	vector<string> allowed;
	allowed.push_back("int");
	allowed.push_back("short");
	allowed.push_back("float");
	allowed.push_back("double");
	allowed.push_back("input");
	TCLAP::ValuesConstraint<string> allowedVals( allowed );
	TCLAP::ValueArg<string> a_wtype("t","type","Work type, input causes the "
			"image to be kept the same as input type. (Default float)",false, "float",
			&allowedVals);
	TCLAP::ValueArg<string> a_type("T","outtype","Output type, input causes "
			"the output to match the input. (Default input)",false, "input",&allowedVals);
	cmd.add(a_wtype);
	cmd.add(a_type);

	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	auto inimg = readMRImage(a_in.getValue());
	if(a_wtype.getValue() == "int") {
		inimg = dPtrCast<MRImage>(inimg->copyCast(INT32));
	} else if(a_wtype.getValue() == "short") {
		inimg = dPtrCast<MRImage>(inimg->copyCast(INT16));
	} else if(a_wtype.getValue() == "float") {
		inimg = dPtrCast<MRImage>(inimg->copyCast(FLOAT32));
	} else if(a_wtype.getValue() == "double") {
		inimg = dPtrCast<MRImage>(inimg->copyCast(FLOAT64));
	}

	for(size_t ii=0; ii<3 && ii<inimg->ndim(); ii++)
		gaussianSmooth1D(inimg, ii, a_stddev.getValue());

	if(a_type.getValue() == "int") {
		inimg = dPtrCast<MRImage>(inimg->copyCast(INT32));
	} else if(a_type.getValue() == "short") {
		inimg = dPtrCast<MRImage>(inimg->copyCast(INT16));
	} else if(a_type.getValue() == "float") {
		inimg = dPtrCast<MRImage>(inimg->copyCast(FLOAT32));
	} else if(a_type.getValue() == "double") {
		inimg = dPtrCast<MRImage>(inimg->copyCast(FLOAT64));
	}

	inimg->write(a_out.getValue());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

    return 0;
}



