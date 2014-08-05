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
 * @file generalLinearModel.cpp
 *
 *****************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include <Eigen/Dense>
#include "mrimage.h"
#include "mrimage_utils.h"
#include "kernel_slicer.h"
#include "kdtree.h"
#include "iterators.h"
#include "accessors.h"
#include "utility.h"

using std::string;
using namespace npl;
using std::shared_ptr;
using Eigen::MatrixXd;

int main(int argc, char** argv)
{
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Performs a General Linear Model statistical test on "
			"an fMRI image. ",
			' ', __version__ );

	TCLAP::MultiArg<string> a_fmri("i", "input", "fMRI image.",
			true, "*.nii.gz", cmd);
	TCLAP::MultiArg<string> a_events("e", "event-reg", "Event-related regression "
			"variable. Three columns, ONSET DURATION VALUE. If these overlap, "
			"an error will be thrown. ", false, "*.txt", cmd);
	TCLAP::MultiArg<string> a_tscore("t", "t-score", "Output statistics, there "
			"may be one of these for each variable passed through '-e'",
			false, "*.nii.gz", cmd);
	TCLAP::MultiArg<string> a_pval("p", "p-val", "Output statistics, there "
			"may be one of these for each variable passed through '-e'",
			false, "*.nii.gz", cmd);
	TCLAP::MultiArg<string> a_response("b", "response", "Output response, there "
			"may be one of these for each variable passed through '-e'",
			false, "*.nii.gz", cmd);

	cmd.parse(argc, argv);
	
	// read fMRI images
	std::list<shared_ptr<MRImage>> fmri;
	int tlen = -1;
	double TR = -1;
	for(auto it=a_fmri.begin(); it != a_fmri.end(); it++) {
		fmri.push_back(readMRImage(*it));

		if(fmri.back()->ndim() != 4) {
			cerr << "Inputs should be 4D!" << endl;
			return -1;
		}

		// TODO resample volumes as necessary

		// check number of volumes
		if(tlen == -1) {
			tlen = fmri.back()->tlen();
		} else if(tlen != fmri.back()->tlen()) {
			cerr << "Inputs should have same time-length!" << endl;
			return -1;
		}
		
		// check TR
		if(TR < 0) {
			TR = fmri.back()->spacing()[3];
		} else if(fabs(TR-fmri.back()->spacing()[3] > 0.00000001)) {
			cerr << "Inputs should have same timing!" << endl;
			return -1;
		}
	}

	// read the event-related designs, will have rows to match time, and cols
	// to match number of regressors
	MatrixXd X(tlen, a_events.getValue().size());
	vector<double> upsampled(tlen*10);
	for(auto it=a_events.begin(); it != a_events.end(); it++) {
		auto events = readNumericCSV(*it);
		


		if(tlen == -1) {
			tlen = fmri.back()->tlen();
		} else if(tlen != fmri.back()->tlen()) {
			std::length_error("Input images have # of timepoints");
		}
	}

	//
	
	Eigen::VectorXd y(tlen*fmri.size());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}


