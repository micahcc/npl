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

#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/IterativeSolvers>

#include "version.h"
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
			false, "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_prefix("p", "prefix", "Output "
			"prefix for ICA significance maps and intermediate data files. "
			"The number of maps will "
			"depend on the number of components, and will be in the same "
			"space as the input fMRI image. names will be the "
			"$prefix_$input_$num.nii.gz where $prefix is this, $input "
			"is the basename from -i and $num is the component number",
			true, "./", "/", cmd);

	TCLAP::ValueArg<int> a_time_append("t", "time-concat", "Number of images "
			"in a rows of time-concatination. ", false, 1,
			"time-columns", cmd);
	TCLAP::ValueArg<int> a_space_append("s", "space-concat", "Number of images "
			"in a rows of spatial-concatination. ", false, 1,
			"spatial-rows", cmd);

	TCLAP::ValueArg<double> a_varthresh("", "var-thresh", "Cut off after this "
			"ratio of the variance has been accounted for.", false, 0.99, "ratio", cmd);
	TCLAP::ValueArg<double> a_tol("", "tol", "Tolerance for low rank "
			"approximation to true matrix (all timeseries). ", false,
			1e-5, "tol", cmd);
	TCLAP::ValueArg<size_t> a_poweriters("", "power-iters", "Power iteration "
			"can increase accuracy of eigenvalues when they are clustered. "
			"Setting this to 2 or 3 could improve the results, but will cost "
			"2*i more computation time", false, 0, "iters", cmd);
	TCLAP::ValueArg<int> a_initrank("", "initrank", "Initial rank estimate. "
			"estimate will go up by powers of 2 until maxrank is reached. "
			"Default is to decide automatically",
			false, -1, "rank", cmd);
	TCLAP::ValueArg<int> a_maxrank("", "maxrank", "Maximum rank estimate. "
			"estimate will go up by powers of 2 until maxrank is reached. "
			"Default is to decide automatically",
			false, -1, "rank", cmd);

	TCLAP::ValueArg<double> a_gbram("M", "memory-max", "Maximum number of GB "
			"of RAM to use for chunks. This is needed to decide how to divide "
			"up data into spatial chunks. ", false, -1, "#GB", cmd);

	TCLAP::SwitchArg a_spatial_ica("S", "spatial-ica", "Perform a spatial ICA"
			", reducing unmixing timepoints to produce spatially independent "
			"maps.", cmd);
	TCLAP::SwitchArg a_no_norm_ts("N", "no-norm-ts", "Do not normalize each "
			"timeseries prior to data-reduction.", cmd);
//	TCLAP::ValueArg<string> a_tmap("T", "t-maps", "Significance of acivation "
//			"throughout the brain. This is computed with Regression for "
//			"temporal ICA, and a Mixture Model for spacial ICA",
//			false, "", "#vecs", cmd);

	cmd.add(a_verbose);
	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	if(!a_in.isSet()) {
		cerr << "Need to provide at least 1 input image!" << endl;
		return -1;
	}

	GICAfmri gica(a_prefix.getValue());
	gica.varthresh = a_varthresh.getValue();
	gica.maxmem = a_gbram.getValue();
	gica.spatial = a_spatial_ica.isSet();
	gica.normts = !a_no_norm_ts.isSet();
	gica.verbose = a_verbose.isSet();
	gica.tolerance = a_tol.getValue();
	gica.initrank = a_initrank.getValue();
	gica.maxrank = a_maxrank.getValue();
	gica.poweriters = a_poweriters.getValue();

	gica.compute(a_time_append.getValue(), a_space_append.getValue(),
			a_masks.getValue(), a_in.getValue());
	gica.computeSpatialMaps();

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

	return 0;
}


