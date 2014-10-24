/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file distortioncorr.cpp Distortion correction (1D b-spline registration)
 *
 *****************************************************************************/

#include <tclap/CmdLine.h>
#include <version.h>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <fstream>

#include "mrimage.h"
#include "nplio.h"
#include "mrimage_utils.h"
#include "ndarray_utils.h"
#include "iterators.h"
#include "accessors.h"
#include "registration.h"

using namespace npl;
using namespace std;

#define VERYDEBUG
#include "macros.h"


/**
 * @brief Information based registration between two 3D volumes. note
 * that the two volumes should have identical sampling and identical
 * orientation. 
 *
 * @param fixed Image which will be the target of registration. 
 * @param moving Image which will be rotated then shifted to match fixed.
 * @param sigmas Standard deviation of smoothing at each level
 * @param bins Number of bins in marginal PDF
 * @param parzrad radius of parzen window, to smooth pdf
 * @param metric Type of information based metric to use
 *
 * @return parameters of bspline
 */
ptr<MRImage> distortionCorr(ptr<const MRImage> fixed, ptr<const MRImage> moving,
		const std::vector<double>& sigmas, int bins, int parzrad, string metric);

int main(int argc, char** argv)
{
try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Computes a 1D B-spline deformation to morph the moving "
			"image to match a fixed image. For a 4D input, the 0'th volume "
			"will be used. TODO: correlation", ' ', __version__ );

	TCLAP::ValueArg<string> a_fixed("f", "fixed", "Fixed image.", true, "",
			"*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_moving("m", "moving", "Moving image. ", true, 
			"", "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_metric("M", "metric", "Metric to use. "
			"(NMI, MI, VI, COR-unimplemented)", false, "", "metric", cmd);
	TCLAP::ValueArg<string> a_out("o", "out", "Registered version of "
			"moving image. ", false, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_transform("t", "transform", "File to write "
			"transform parameters to. This will be a nifti image the B-spline "
			"parameters and the the phase encoding direction set to the "
			"direction of deformation." , false, "", "*.nii", cmd);

	TCLAP::MultiArg<double> a_sigmas("s", "sigmas", "Smoothing standard "
			"deviations at each step of the registration.", false, 
			"sd", cmd);

	TCLAP::ValueArg<char> a_dir("d", "direction", "Distortion direction "
			"x or y or z etc.... By default the phase encode direction of "
			"the moving image will be used (if set).", false, 200,
			"n", cmd);
	TCLAP::ValueArg<int> a_bins("b", "bins", "Bins to use in information "
			"metric to estimate the joint distribution. This is the "
			"the number of bins in the marginal distribution.", false, 200,
			"n", cmd);
	TCLAP::ValueArg<int> a_parzen("r", "radius", "Radius in parzen window "
			"for bins", false, 5, "n", cmd);

	cmd.parse(argc, argv);

	// set up sigmas
	vector<double> sigmas({3,2,1,0});
	if(a_sigmas.isSet()) 
		sigmas.assign(a_sigmas.begin(), a_sigmas.end());

	/*************************************************************************
	 * Read Inputs
	 *************************************************************************/

	// fixed image
	cout << "Reading Inputs...";
	ptr<MRImage> fixed, moving, in_moving;
	{
	ptr<MRImage> fixed = readMRImage(a_fixed.getValue());
	ptr<MRImage> in_moving = readMRImage(a_moving.getValue());

	size_t ndim = min(fixed->ndim(), in_moving->ndim());
	auto moving = dPtrCast<MRImage>(in_moving->copyCast(ndim, in_moving->dim(), 
				FLOAT32));

	cout << "Done\nPutting Fixed Image into Moving Space...";

	// Create Copy of Moving image, then sample input fixed image on its grid
	fixed = dPtrCast<MRImage>(moving->createAnother());
	vector<int64_t> ind(ndim);
	vector<double> point(ndim);
	
	// Create Interpolator, ensuring that radius is at least 1 pixel in the
	// output space 
	LanczosInterpNDView<double> interp(in_fixed);
	interp.setRadius(ceil(in_moving->spacing(0)/in_fixed->spacing(0)));
	interp.m_ras = true;
	for(NDIter<double> it(fixed); !it.eof(); ++it) {
		// get point 
		it.index(ind.size(), ind.data());
		fixed->indexToPoint(ind.size(), ind.data(), point.data());

		// sample 
		it.set(interp.get(point));
	}
	fixed->write("resampled.nii.gz");
	}

	// Get direction
	int dir = moving->m_phasedim;
	if(a_dir.isSet()) {
		if(a_dir.getValue() < 'x' || a_dir.getValue() > 'z') {
			cerr << "Invalid direction (use x,y or z): " << a_dir.getValue()
				<< endl;
			return -1;
		}
		dir = a_dir.getValue() - (int)'x';
	else if(dir < 0) {
		cerr << "Error, no direction set, and no phase-encode direction set "
			"in moving image!" << endl;
		return -1;
	}
	
	/*************************************************************************
	 * Registration
	 *************************************************************************/
	ptr<MRImage> transform;
	if(a_metric.getValue() == "COR") {
		cerr << "COR not yet implemented!" << endl;
	} else {
		cout << "Done\nRigidly Registering with " << a_metric.getValue() 
			<< "..." << endl;
		Metric metric;
		
		if(a_metric.getValue() == "MI")
			metric = METRIC_MI;
		else if(a_metric.getValue() == "NMI")
			metric = METRIC_NMI;
		else if(a_metric.getValue() == "VI")
			metric = METRIC_VI;

		transform = distortionCorr(fixed, moving, dir, sigmas,
				a_bins.getValue(), a_parzen.getValue(), metric); 
	}

	fixed.reset();
	moving.reset();

	cout << "Finished\nWriting output.";
	if(a_transform.isSet()) 
		transform->write(a_transform.getValue());

	if(a_out.isSet()) {
		// Apply Rigid Transform.
		// Copy input moving then sample
		auto out = dPtrCast<MRImage>(in_moving->copyCast(FLOAT32));
		
		// Create Sampler for Transform and moving view
		BSplineView<double> bvw(transform);
		bvw.m_ras = true;
		LanczosInterpNDView<double> mvw(in_moving);

		// Temps
		vector<double> cind(in_moving->ndim());
		vector<double> pt(in_moving->ndim());
		double def;
		double ddef;

		// Iterate over output, and pply transform
		for(NDIter<double> it(out); !it.eof(); ++it) {
			it.index(cind);
			out->indexToPoint(cind, pt);
			bvw.get(pt.size(), pt.data(), transform->m_phasedim, def, ddef);
			cind[dir] += def;
			it.set(mvw.get(cind)*(1+ddef));
		}
		out->write(a_out.getValue());
	}
	cout << "Done" << endl;

} catch (TCLAP::ArgException &e)  // catch any exceptions
{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}



