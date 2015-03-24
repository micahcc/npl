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

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include "mrimage.h"
#include "nplio.h"
#include "iterators.h"

using namespace npl;
using namespace std;

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Reads an image (supported types are .json[.gz] and "
            ".nii[.gz]) then rewrites it, if no input is provided then "
            "an example image is generated and written. Useful for conversion "
            "between json and nii, or for seeing what the json format looks like",
            ' ', __version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input image.",
			false, "", "input", cmd);
	TCLAP::ValueArg<string> a_out("o", "output", "Output image.",
			true, "", "out", cmd);

    cmd.parse(argc, argv);

    if(!a_in.isSet()) {
        // create an image
        size_t sz[] = {2, 1, 4, 3};
        Eigen::VectorXd neworigin(4);
        neworigin << 1.3, 75, 9, 0;

        Eigen::VectorXd newspacing(4);
        newspacing << 4.3, 4.7, 1.2, .3;

        Eigen::MatrixXd newdir(4,4);
        newdir <<
            -0.16000, -0.98424,  0.07533, 0,
            0.62424, -0.16000, -0.76467, 0,
            0.76467, -0.07533,  0.64000, 0,
            0,        0,        0, 1;

        auto in = createMRImage(sizeof(sz)/sizeof(size_t), sz, FLOAT64);

        NDIter<double> sit(in);
        for(int ii=0; !sit.eof(); ++sit, ++ii) {
            sit.set(ii);
        }
        in->setOrient(neworigin, newspacing, newdir);
        in->write(a_out.getValue());
        return 0;
    } else {
        auto img = dPtrCast<MRImage>(readMRImage(a_in.getValue()));
        img->write(a_out.getValue());
    }

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}

