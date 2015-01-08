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
 * @file fmri_gica2.cpp New version of group ICA that performs two SVD's to
 * reduce the memory footprint, prior to performing ICA.
 *
 *****************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/IterativeSolvers>

#include "ica_helpers.h"
#include "mrimage.h"
#include "nplio.h"
#include "iterators.h"
#include "statistics.h"
#include "macros.h"

using std::string;
using std::shared_ptr;
using std::to_string;

using namespace Eigen;
using namespace npl;

TCLAP::MultiSwitchArg a_verbose("v", "verbose", "Write lots of output. "
		"Be wary for large datasets!", 1);

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Perform ICA analysis on an image, or group of images. "
			"By default this will be a temporal ICA. Group analysis is "
			"done by concatinating in time. By default a Temporal ICA will "
			"be performed, to do a spatial ICA select -S.", ' ', __version__ );

	TCLAP::MultiArg<string> a_in("i", "input", "Input fMRI images. Ordered in "
			"time-major (adjacent inputs will be next to each other during "
			"concatination). Every s images will correspond to a mask (where "
			"s is the value provided by -s).",
			true, "*.nii.gz", cmd);
	TCLAP::MultiArg<string> a_masks("m", "mask", "Mask images.",
			true, "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_prefix("p", "prefix", "Output "
			"prefix for ICA significance maps and intermediate data files. "
			"The number of maps will "
			"depend on the number of components, and will be in the same "
			"space as the input fMRI image. names will be the "
			"$prefix_$input_$num.nii.gz where $prefix is this, $input "
			"is the basename from -i and $num is the component number",
			true, "./", "/", cmd);

	TCLAP::ValueArg<int> a_time_append("t", "time-concat", "Number of images "
			"in a rows of time-concatination. ", false, -1,
			"#vecs", cmd);
	TCLAP::ValueArg<int> a_space_append("s", "space-concat", "Number of images "
			"in a rows of spatial-concatination. ", false, -1,
			"#vecs", cmd);

	TCLAP::ValueArg<double> a_evthresh("", "ev-thresh", "Threshold on "
			"ratio of total variance to account for (default 0.99)",
			false, 0.1, "ratio", cmd);
	TCLAP::ValueArg<double> a_varthresh("", "ev-thresh", "Threshold on "
			"ratio of total variance to account for (default 0.99)",
			false, 0.99, "ratio", cmd);
	TCLAP::ValueArg<int> a_simultaneous("V", "simul-vectors", "Simultaneous "
			"vectors to estimate eigenvectors for in lambda. Bump this up "
			"if you have convergence issues. This is part of the "
			"Band Lanczos Eigenvalue Decomposition of the covariance matrix. "
			"By default the covariance size is used, which is conservative. "
			"Values less than 1 will default back to that", false, -1,
			"#vecs", cmd);
	TCLAP::ValueArg<int> a_iters("", "max-iters", "Maximum iterations "
			"in PCA. This sets are hard limit on the number of estimated "
			"eigenvectors in the Band Lanczos Solver (and thus the maximum "
			"number of singular values in the SVD. Be wary of making it "
			"too small). -1 removes limit", false, -1, "#vecs", cmd);
	TCLAP::ValueArg<double> a_gbram("M", "memory-max", "Maximum number of GB "
			"of RAM to use for chunks. This is needed to decide how to divide "
			"up data into spatial chunks. ", false, -1, "#vecs", cmd);

	TCLAP::SwitchArg a_spatial_ica("S", "spatial-ica", "Perform a spatial ICA"
			", reducing unmixing timepoints to produce spatially independent "
			"maps.", cmd);
	TCLAP::ValueArg<string> a_tmap("T", "t-maps", "Significance of acivation "
			"throughout the brain. This is computed with Regression for "
			"temporal ICA, and a Mixture Model for spacial ICA",
			false, "", "#vecs", cmd);

	cmd.add(a_verbose);
	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	if(!a_in.isSet()) {
		cerr << "Need to provide at least 1 input image!" << endl;
		return -1;
	}

//	MatrixXd ics = fmri_gica(a_time_append.getValue(),
//			a_space_append.getValue(), a_prefix.getValue(), a_masks.getValue(),
//			a_in.getValue(), a_evthresh.getValue(), a_varthresh.getValue(),
//			a_simultaneous.getValue(), a_iters.getValue(), a_gbram.getValue(),
//			a_spatial_ica.isSet());
	GICAfmri gica(a_prefix.getValue());
	gica.evthresh = a_evthresh.getValue();
	gica.varthresh = a_varthresh.getValue();
	gica.initbasis = a_simultaneous.getValue();
	gica.maxiters = a_iters.getValue();
	gica.maxmem = a_gbram.getValue();
	
	gica.compute(a_time_append.getValue(), a_space_append.getValue(),
			a_masks.getValue(), a_in.getValue(), a_spatial_ica.isSet());

//	gica.writeMaps(a_tmap.getValue());
//	gica.writeSignals(a_tmap.getValue());
//	gica.maps()->write(a_tmap.getValue());
//	gica.signals()->write(a_tmap.getValue());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

	return 0;
}


