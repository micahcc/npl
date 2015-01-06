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
 * @file fmri_psd_ica.cpp Tool for performing ICA on multiple fMRI images by
 * first concatinating in space. This requires much less memory for the SVD
 * than concatinating in time. For more general usage you can optionally 
 * perform ICA on Power Spectral Density, which makes the specific time-courses
 * irrelevent, and insteady groups data by frequency.
 *****************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/IterativeSolvers>

#include "fftw3.h"

#include "ica_helpers.h"
#include "mrimage.h"
#include "nplio.h"
#include "iterators.h"
#include "statistics.h"
#include "utility.h"
#include "ndarray_utils.h"
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

	TCLAP::CmdLine cmd("Perform ICA analysis on an image, or group of "
			"images by concating in space. This only works if 1) the images "
			"have identical timecourses or 2) you select -p to transform the "
			"time-series into the power-spectral-density prior to performing "
			"ICA analysis. The SVD memory usage only depends on the squre of "
			"the longest single subjects time-course rather than the "
			"concatined time-course (which is what the normal memory "
			"complexity of PCA is.", ' ', __version__ );

	TCLAP::SwitchArg a_spatial("S", "spatial-ica", "Compute the spatial ICA. "
			"Note that LxS values must fit into memory to do this, where L is "
			"the PCA-reduced dataset and S is the totally number of masked "
			"spatial locations across all subjects", cmd);
	
	TCLAP::SwitchArg a_psd("p", "power-spectral-density", "Compute the power "
			"spectral density of timer-series. The advantage of this is that "
			"concating in space will work even if the timeseries of the "
			"individual images are not identical. The only assumption is that "
			"similar tasks have similar domonant frequencies. ", cmd);

	TCLAP::MultiArg<string> a_in("i", "input", "Input fMRI images.",
			true, "*.nii.gz", cmd);
	TCLAP::MultiArg<string> a_masks("m", "mask", "Input masks for fMRI. Note "
			"that it is not necessary to mask if all the time-series outsize "
			"the brain are 0, since those regions will be removed anyway",
			false, "*.nii.gz", cmd);

//    TCLAP::ValueArg<string> a_components("o", "out-components", "Output "
//            "Independent Components as a 1x1xCxT image.",
//			true, "", "*.nii.gz", cmd);
    TCLAP::ValueArg<string> a_workdir("d", "dir", "Output "
            "directory. Large intermediate matrices will be written here as "
			"well as the spatial maps. ", true, "./", "path", cmd);

	TCLAP::ValueArg<double> a_evthresh("T", "ev-thresh", "Threshold on "
			"ratio of total variance to account for (default 0.99)", false,
			INFINITY, "ratio", cmd);
	TCLAP::ValueArg<int> a_simultaneous("V", "simul-vectors", "Simultaneous "
			"vectors to estimate eigenvectors for in lambda. Bump this up "
			"if you have convergence issues. This is part of the "
			"Band Lanczos Eigenvalue Decomposition of the covariance matrix. "
			"By default the covariance size is used, which is conservative. "
			"Values less than 1 will default back to that", false, -1, 
			"#vecs", cmd);
	TCLAP::ValueArg<int> a_maxrank("R", "max-rank", "Maximum rank of output "
			"in PCA. This sets are hard limit on the number of estimated "
			"eigenvectors in the Band Lanczos Solver (and thus the maximum "
			"number of singular values in the SVD)", false, -1, "#vecs", cmd);

	cmd.add(a_verbose);
	cmd.parse(argc, argv);

	spcat_ica(a_psd.isSet(), a_in.getValue(), a_masks.getValue(), 
			a_workdir.getValue(), a_evthresh.getValue(),
			a_simultaneous.getValue(), a_maxrank.getValue(), 
			a_spatial.isSet());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

    return 0;
}
