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
#include "kdtree.h"
#include "iterators.h"
#include "accessors.h"

using std::string;
using namespace npl;

/**
 * @brief Computes the overlap of the two images' in 3-space.
 *
 * @param a Image
 * @param b Image
 *
 * @return Ratio of b that overlaps with a's grid
 */
double overlapRatio(ptr<MRImage> a, ptr<MRImage> b)
{
	int64_t index[3];
	double point[3];
	size_t incount = 0;
	size_t maskcount = 0;
	for(OrderIter<int64_t> it(a); !it.eof(); ++it) {
		it.index(3, index);
		a->indexToPoint(3, index, point);
		maskcount++;
		incount += (b->pointInsideFOV(3, point));
	}
	return (double)(incount)/(double)(maskcount);
}

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Applies 3D deformation to a volume."
			" If you have a *.svreg.map.* file use "
			"' nplApplyDeform -1 -l -d *.map.nii.gz -i *.atlas.space.nii.gz'"
			"Note that there may be issue if your input has different "
			"orientation from the atlas. It is usually better to use physical "
			"space offset maps (which is why that is the default).",
			' ', __version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input image.",
			true, "", "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_deform("d", "deform", "Deformation field.",
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_interp("I", "interp", "Interpolation method. "
			"One of: lanczos, linear, nn. Default is lanczos", false,
			"lanczos", "type", cmd);
	TCLAP::SwitchArg a_indexmap("l", "lookup", "Index lookup map "
			"(rather offsets in physical space). Indexes refer to indexes in "
			"input image. ", cmd);
	TCLAP::SwitchArg a_onebased("1", "one-based", "One based indexes (rather "
			"than 0 based). Just subtracts 1 from values in the deformation "
			"IF -l /--lookup is set.", cmd);

	TCLAP::ValueArg<string> a_out("o", "out", "Output image.",
			true, "", "*.nii.gz", cmd);


	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	ptr<MRImage> inimg(readMRImage(a_in.getValue()));
	if(inimg->ndim() > 4 || inimg->ndim() < 3) {
		cerr << "Expected input to be 3D/4D Image!" << endl;
		return -1;
	}

	ptr<MRImage> defimg(readMRImage(a_deform.getValue()));
	if(defimg->ndim() > 5 || defimg->ndim() < 4 || defimg->tlen() != 3) {
		cerr << "Expected dform to be 4D/5D Image, with 3 volumes!" << endl;
		return -1;
	}

	ptr<Vector3DConstView<double>> interp;
	if(a_interp.getValue() == "lanczos")
		interp.reset(new LanczosInterp3DView<double>(inimg));
	else if(a_interp.getValue() == "nn")
		interp.reset(new NNInterp3DView<double>(inimg));
	else if(a_interp.getValue() == "linear")
		interp.reset(new LinInterp3DView<double>(inimg));
	else
		interp.reset(new LanczosInterp3DView<double>(inimg));

	ptr<MRImage> outimg;
	vector<size_t> outsize(inimg->ndim());
	for(size_t ii=0; ii<inimg->ndim(); ii++)
		outsize[ii] = defimg->dim(ii);
	if(inimg->ndim() == 4)
		outsize[3] = inimg->dim(3);

	if(a_interp.getValue() == "nn")
		outimg = dPtrCast<MRImage>(defimg->createAnother(inimg->ndim(),
					outsize.data(), inimg->type()));
	else
		outimg = dPtrCast<MRImage>(defimg->createAnother(inimg->ndim(),
					outsize.data(), FLOAT32));

	double pt[3];
	for(Vector3DIter<double> dit(defimg), oit(outimg); !oit.eof(); ++oit, ++dit) {

		if(a_indexmap.isSet()) {
			// Just look at point in input
			for(size_t dd=0; dd < 3; dd++)
				pt[dd] = dit[dd] - a_onebased.isSet();
		} else {
			// convert index to point add offset, then convert point to in
			// input image index
			oit.index(3, pt);
			outimg->indexToPoint(3, pt, pt);
			for(size_t dd=0; dd < 3; dd++)
				pt[dd] += dit[dd];
			inimg->pointToIndex(3, pt, pt);
		}

		for(size_t tt=0; tt<inimg->tlen(); tt++)
			oit.set(tt, interp->get(pt[0], pt[1], pt[2], tt));
	}

	outimg->write(a_out.getValue());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}

