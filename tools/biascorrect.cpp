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
 * @file applyDeform.cpp Tool to apply a deformation field to another image. 
 * Not yet functional
 *
 *****************************************************************************/

#include <unordered_map>
#include <version.h>
#include <string>
#include <stdexcept>

#include "mrimage.h"
#include "nplio.h"
#include "mrimage_utils.h"
#include "kdtree.h"
#include "iterators.h"
#include "accessors.h"
#include "mathexpression.h"

using namespace npl;
using namespace std;

void usage(int status)
{
	cerr << "Usage: nplMath --out <image> [options] [-a <image>] [-b <image>] ... \"<equation>\"" << endl;
	cerr << "\tMath is performed in the space of the first image on the "
		"command line which is not necessarily -a. All other images are "
		"lanczos resampled to that space (unless --nn/--lin are provided). "
		"Pixel type is by default the same as this image, but it can be set "
		"with --short/--double/--float. Any single character can follow a - to "
		"create a variable for use in the equation. ";
	cerr << "Options:\n"<<endl;
	cerr << '\t' << setw(10) << left << "--nn"     << "Nearest neighbor resampling" << endl;
	cerr << '\t' << setw(10) << left << "--lin"    << "Linear resampling" << endl;
	cerr << '\t' << setw(10) << left << "--short"  << "Use short int out for type" << endl;
	cerr << '\t' << setw(10) << left << "--int"    << "Use int for out type" << endl;
	cerr << '\t' << setw(10) << left << "--float"  << "Use float for out type" << endl;
	cerr << '\t' << setw(10) << left << "--double" << "Use double for out type" << endl;
	cerr << "\nAcceptable operations in the equation are:\n";
	listops();
	exit(status);
}

int main(int argc, char** argv)
{
	try {
		/*
		 * Command Line
		 */

		TCLAP::CmdLine cmd("Computes a bias-field from an image and a mask. ",
						' ', __version__ );

		TCLAP::ValueArg<string> a_in("i", "input", "Input image.",
				true, "", "*.nii.gz", cmd);
		TCLAP::ValueArg<string> a_mask("m", "mask", "Mask image. If this image is "
						"a scalar then values are taken as weights.", true, "", "*.nii.gz",
						cmd);
		TCLAP::ValueArg<double> a_spacing("s", "spacing", "Space between knots "
						"for bias-field estimation.", false, 10, "[mm]", cmd);

		TCLAP::ValueArg<string> a_biasfield("o", "out", "Bias Field Image.",
						false, "", "*.nii.gz", cmd);
		TCLAP::ValueArg<string> a_corimage("c", "corr", "Bias Field Corrected "
						"version of input image.", false, "", "*.nii.gz", cmd);

		cmd.parse(argc, argv);

		// read input
		auto in = readMRImage(a_in->getValue());
		auto mask = readMRImage(a_mask->getValue());
		size_t ndim = in->ndim();

		VectorXd spacing;
		VectorXd origin;
		vector<size_t> osize(ndim, 0);
		for(size_t dd=0; dd<osize.size(); ++dd) {
				osize[dd] = 4+in->dim(dd)*in->spacing(dd)/a_spacing->getValue();
				spacing[dd] = a_spacing->getValue();
		}

		auto biasfield = createMRImage(osize.size(), osize.data(), FLOAT64);
		biasfield->setDirection(in->getDirection());
		biasfield->setSpacing(spacing);
		
		// compute center of input
		VectorXd indc(ndim); // center index
		for(size_t dd=0; dd<ndim; dd++) 
				indc[dd] = (in->dim(dd)-1.)/2.;
		VectorXd ptc(ndim); // point center
		in->indexToPoint(ndim, indc.array().data(), ptc.array().data());

		// compute origin from center index (x_c) and center of input (c): 
		// o = c-R(sx_c)
		for(size_t dd=0; dd<ndim; dd++) 
				indc[dd] = (osize[dd]-1.)/2.;
		VectorXd origin = ptc - in->getDirection()*(spacing.asDiagonal()*indc);
		biasfield->setOrigin(origin);

		biasfied->write("biasfield_empty.nii.gz");

		size_t nparams = 1;
		size_t npixels = 1;
		for(size_t ii=0; ii<ndim; ii++) {
				nparams *= osize[ii];
				npixels *= in->dim(ii);
		}
		VectorXd params(nparams);
		MatrixXd pixweights(npixels, nparams);

		// for each parameter
		for(NDIter<double> pit(biasfield); !pit.eof(); ++pit) {
				// construct ROI
		}
		// for each parameter
		// for each pixel within 2*spacing 
		// compute weight
		
		
		// compute weights of parameters at each pixel location
		auto pweightimg = in->createAnother(FLOAT32);
		for(auto pweightimg
		MatrixXd pweights(npixels, nparams);

	} catch (TCLAP::ArgException &e)  // catch any exceptions
}

