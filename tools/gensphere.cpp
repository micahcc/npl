/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file gensphere.cpp Generate a sphere image
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

int main(int argc, char* argv[])
{
	try {

	TCLAP::CmdLine cmd("This program takes a coordinate, a label number and "
			"creates a sphere centered at the coordinate of the given radius",
			' ', __version__ );

	TCLAP::ValueArg<string> a_example("e", "example", "Example image", true, "",
			"image", cmd);

	TCLAP::ValueArg<string> a_out("o", "out", "Output image", true, "", "image",
			cmd);

	TCLAP::UnlabeledMultiArg<double> a_coord("coordinate", "Coordinate in image (LPS!).",
			true, "coord");
	cmd.add(a_coord);

	TCLAP::ValueArg<double> a_radius("R", "radius", "Radius of sphere in mm.",
			false, 5, "radius", cmd);
	TCLAP::ValueArg<int> a_label("L", "label", "Label to use in sphere.",
			false, 1, "label", cmd);
	TCLAP::ValueArg<int> a_default("D", "default", "Outside sphere label to use. "
			"If not set then no other points will be touched.",
			false, 1, "label", cmd);

	TCLAP::SwitchArg a_index("I", "index", "Use index coordinates rather than RAS "
			"coordinates.", cmd);

	cmd.parse(argc, argv);

	if(a_coord.getValue().size() != 3) {
		cerr << "Error, must provide x,y,z coordinates" << endl;
		return -1;
	}

	auto output = readMRImage(a_example.getValue());

	double pt[3];
	for(NDIter<int> it(output); !it.eof(); ++it) {
		it.index(3, pt);
		double dist = 0;
		if(a_index.isSet()) {
			for(size_t ii=0; ii<3; ii++)
				dist += pow(pt[ii]-a_coord.getValue()[ii],2)*output->spacing(ii);
		} else {
			output->indexToPoint(3, pt, pt);
			for(size_t ii=0; ii<3; ii++)
				dist += pow(pt[ii]-a_coord.getValue()[ii],2);
		}

		dist = sqrt(dist);
		if(dist < a_radius.getValue()) {
			it.set(a_label.getValue());
		} else if(a_default.isSet()) {
			it.set(a_default.getValue());
		}
	}

	output->write(a_out.getValue());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

	return 0;
}

