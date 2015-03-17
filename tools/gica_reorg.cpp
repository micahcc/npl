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
 * @file gica_reorg.cpp New version of group ICA that performs two SVD's to
 * reduce the memory footprint, prior to performing ICA.
 *
 *****************************************************************************/

#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/IterativeSolvers>

#include "version.h"
#include "fmri_inference.h"
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

	TCLAP::CmdLine cmd("Perform reorganization of multiple fMRI. This is the "
			"first phase of ICA analysis.", ' ', __version__ );

	TCLAP::MultiArg<string> a_in("i", "input", "Input fMRI images. Ordered in "
			"time-major (adjacent inputs will be next to each other during "
			"concatination). Every s images will correspond to a mask (where "
			"s is the value provided by -s).",
			true, "*.nii.gz", cmd);
	TCLAP::MultiArg<string> a_masks("m", "mask", "Mask images.",
			false, "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_prefix("", "reorg-prefix", "Output "
			"prefix for ICA tall matrices and masks. "
			"The number of tall matrices will depend on the memory size -M GB "
			"and the number of masks will depend on the number of columns "
			"(-s #)", true, "./", "/", cmd);

	TCLAP::ValueArg<int> a_time_append("t", "time-concat", "Number of images "
			"in a rows of time-concatination. ", false, 1,
			"time-columns", cmd);
	TCLAP::ValueArg<int> a_space_append("s", "space-concat", "Number of images "
			"in a rows of spatial-concatination. ", false, 1,
			"spatial-rows", cmd);

	TCLAP::ValueArg<double> a_gbram("M", "memory-max", "Maximum number of GB "
			"of RAM to use for chunks. This is needed to decide how to divide "
			"up data into spatial chunks. ", false, -1, "#GB", cmd);

	TCLAP::SwitchArg a_no_norm_ts("N", "no-norm-ts", "Do not normalize each "
			"timeseries.", cmd);

	cmd.add(a_verbose);
	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	if(!a_in.isSet()) {
		cerr << "Need to provide at least 1 input image!" << endl;
		return -1;
	}

	size_t tb = 0;
	size_t sb = 0;
	if(!a_time_append.isSet() && !a_space_append.isSet()) {
		tb = a_in.getValue().size();
		sb = 1;
	} else {
		tb = a_time_append.getValue();
		sb = a_space_append.isSet();
	}

	cerr << "Reorganizing data into matrices..."<<endl;
	gicaCreateMatrices(tb, sb, a_masks.getValue(), a_in.getValue(),
			a_prefix.getValue(), a_gbram.getValue(), !a_no_norm_ts.isSet(),
			a_verbose.isSet());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

	return 0;
}



