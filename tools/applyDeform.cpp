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
			" If you have a *.svreg.map.* file use '$ convertDeform --in-index -1 "
			"-i *.svreg.map.nii.gz -a atlas.bfc.nii.gz "
			"-o offset.nii.gz' to generate an "
			"appropriate input (offset.nii.gz).", ' ', __version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input image.",
			true, "", "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_deform("d", "deform", "Deformation field.",
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_interp("I", "interp", "Interpolation method. "
			"One of: lanczos, linear, nn. Default is lanczos", false,
			"lanczos", "type", cmd);
	TCLAP::SwitchArg a_ignoreorient("O", "orient-ignore", "Ignore orientation. "
			"Warning this is risky. However it is necessary if you know that "
			"the inputs are in the same pixel space but someone left "
			"out the orienation. ", cmd);

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

	if(a_ignoreorient.isSet() || !defimg->isOriented()) {
		cerr << "Assuming input and deform have matching pixels" << endl;
		for(size_t dd=0; dd<3; dd++) {
			if(defimg->dim(dd) != inimg->dim(dd)) {
				cerr << "Input pixel sizes differ (deform versus input!)"
					<< endl;
				return -1;
			}
		}

		defimg->setOrient(inimg->getOrigin(), inimg->getSpacing(),
				inimg->getDirection(), false);
	} else if(defimg->getDirection() != inimg->getDirection() ||
			defimg->getOrigin() != inimg->getOrigin() ||
			defimg->getSpacing() != inimg->getSpacing()) {
		if(overlapRatio(inimg, defimg) < 0.5) {
			cerr << "Deformation and Input do not overlap!" << endl;
			return -1;
		}

		cerr << "Linearly Resampling Deform into space of input." << endl;
		size_t newsize[4] = {inimg->dim(0), inimg->dim(1), inimg->dim(2), 3};
		auto odef = dPtrCast<MRImage>(inimg->createAnother(4, newsize, FLOAT32));
		LinInterp3DView<double> definterp(defimg);
		definterp.m_ras = true;

		double pt[3];
		for(Vector3DIter<double> dit(odef); !dit.eof(); ++dit) {
			dit.index(3, pt);
			odef->indexToPoint(3, pt, pt);

			for(size_t dd=0; dd<3; dd++)
				dit.set(dd, definterp.get(pt[0], pt[1], pt[2], dd));
		}

		odef->write("definterp.nii.gz");
		defimg = odef;
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

	ptr<MRImage> out;
	if(a_interp.getValue() == "nn")
		dPtrCast<MRImage>(inimg->createAnother());
	else
		dPtrCast<MRImage>(inimg->createAnother(FLOAT32));

	double pt[3];
	for(Vector3DIter<double> dit(defimg), oit(out); !oit.eof(); ++oit, ++dit) {
		oit.index(3, pt);
		out->indexToPoint(3, pt, pt);
		for(size_t dd=0; dd < 3; dd++)
			pt[dd] += dit[dd];
		out->pointToIndex(3, pt, pt);

		for(size_t tt=0; tt<inimg->tlen(); tt++)
			oit.set(tt, interp->get(pt[0], pt[1], pt[2], tt));

	}

	out->write(a_out.getValue());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}

