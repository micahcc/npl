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
 * @file gica_reduce.cpp Reduce reorganized matrices into U,E,V matrices
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

	TCLAP::CmdLine cmd("Perform reduction of reorganized fMRI matrices. This "
			"is the second phase of ICA analysis.", ' ', __version__ );

	TCLAP::ValueArg<double> a_cvarthresh("", "cvar-thresh", "Cut off after this "
			"ratio of the cumulative explained variance has been found.", false,
			0.99, "ratio", cmd);
	TCLAP::ValueArg<double> a_varthresh("", "var-thresh", "Cut off after this "
			"ratio of the maximum variance component has been reached .",
			false, 0.1, "ratio", cmd);
	TCLAP::ValueArg<size_t> a_poweriters("", "power-iters", "Power iteration "
			"can increase accuracy of eigenvalues when they are clustered. "
			"Setting this to 2 or 3 could improve the results, but will cost "
			"2*i more computation time. Not applicable for full SVD",
			false, 0, "iters", cmd);
	TCLAP::ValueArg<int> a_rank("", "rank", "Initial rank estimate. "
			"You usually want to high ball this a bit so that var thresh "
			"can be used to reduce the rank automatically. Not applicable "
			"to full SVD", false, 100, "rank", cmd);

	TCLAP::ValueArg<string> a_in("i", "input-prefix", "Prefix for input tall "
			"matrices and masks", true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_out("o", "output-prefix", "Prefix for output "
			"UEV matrices.", true, "", "*.nii.gz", cmd);
	TCLAP::SwitchArg a_full("F", "full-svd", "Perform full SVD rather than "
			"probabilistic one", cmd);

	cmd.add(a_verbose);
	cmd.parse(argc, argv);
	if(a_full.isSet()) {
		gicaReduceFull(a_in.getValue(), a_out.getValue(),
				a_varthresh.getValue(), a_cvarthresh.getValue(),
				a_verbose.isSet());
	} else {
		gicaReduceProb(a_in.getValue(), a_out.getValue(),
				a_varthresh.getValue(), a_cvarthresh.getValue(),
				a_rank.getValue(), a_poweriters.getValue(),
				a_verbose.isSet());
	}

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

	return 0;
}




