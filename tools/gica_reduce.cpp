/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
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

	TCLAP::CmdLine cmd("Perform reduction of reorganized fMRI matrices. This "
			"is the second phase of ICA analysis.", ' ', __version__ );

	double cvarthresh = 1;
//	TCLAP::ValueArg<double> a_cvarthresh("", "cvar-thresh", "Cut off after this "
//			"ratio of the cumulative explained variance has been found.", false,
//			0.99, "ratio", cmd);
	TCLAP::ValueArg<double> a_varthresh("", "var-thresh", "Cut off after this "
			"ratio of the maximum variance component has been reached .",
			false, 0.1, "ratio", cmd);
	TCLAP::ValueArg<size_t> a_poweriters("", "power-iters", "Power iteration "
			"can increase accuracy of eigenvalues when they are clustered. "
			"Setting this to 2 or 3 could improve the results, but will cost "
			"2*i more computation time. Not applicable for full SVD",
			false, 0, "iters", cmd);
	TCLAP::ValueArg<int> a_rank("", "rank", "Maximum output rank. If "
			"randomized SVD is applied, then this is the initial reduction. "
			"If full SVD is being performed then this is the maximum number of "
			"rows to save. ", false, 100, "rank", cmd);

	TCLAP::ValueArg<string> a_reorgprefix("", "reorg-prefix", "Prefix for input tall "
			"matrices and masks", true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_reduceprefix("", "reduce-prefix", "Prefix for output "
			"UEV matrices.", true, "", "*.nii.gz", cmd);
	TCLAP::SwitchArg a_full("F", "full-svd", "Perform full SVD rather than "
			"probabilistic one", cmd);

	cmd.add(a_verbose);
	cmd.parse(argc, argv);
	cerr<<"Outputs will be:"<<endl;
	cerr<<a_reduceprefix.getValue()<<"_Umat"<<endl;
	cerr<<a_reduceprefix.getValue()<<"_Vmat"<<endl;
	cerr<<a_reduceprefix.getValue()<<"_Evec"<<endl;
	if(a_full.isSet()) {
		gicaReduceFull(a_reorgprefix.getValue(), a_reduceprefix.getValue(),
				a_varthresh.getValue(), cvarthresh, a_rank.getValue(),
				a_verbose.isSet());
	} else {
		gicaReduceProb(a_reorgprefix.getValue(), a_reduceprefix.getValue(),
				a_varthresh.getValue(), cvarthresh,
				a_rank.getValue(), a_poweriters.getValue(),
				a_verbose.isSet());
	}

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

	return 0;
}




