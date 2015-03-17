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

#include <iostream>
#include <cstdio>
#include <iomanip>
#include <string>
#include <fstream>

#include <set>
#include <vector>
#include <cmath>
#include <memory>

#include "fftw3.h"
#include <tclap/CmdLine.h>

#include "basic_plot.h"
#include "statistics.h"
#include "npltypes.h"
#include "nplio.h"
#include "utility.h"
#include "mrimage.h"
#include "mrimage_utils.h"
#include "ndarray_utils.h"
#include "ica_helpers.h"
#include "iterators.h"
#include "version.h"

TCLAP::SwitchArg a_verbose("v", "verbose", "Be verbose (More for debugging).");

const double DELTA = 1e-30;

using std::set;
using std::vector;
using std::string;
using std::endl;
using std::cerr;
using std::cout;
using namespace npl;

int main(int argc, char* argv[])
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */
	TCLAP::CmdLine cmd("This is a program to perform cleaning of fMRI data "
			"prior to other analysis. This includes regressing out known "
			"confounding signals, regressing out signal from non-GM reagions "
			"and removing frequency bands outside the band of interest."
			"Regressors may be passed through CSV files (-r). This is the sort "
			"of regressor that motion parameters would be passed through, and "
			"these are not convolved with the HRF. Signal from specified "
			"labels can also be added as confounds. This includes the average, "
			"(-A) or principal components (-P) or independent components (-I)"
			"Event-related regressors can also be passed in (-e), these ARE "
			"convolved with the cannonical HRF (from SPM). Time-filtering is "
			"also performed using a 2nd order butterworth filter. (-f/-F) ",
			' ', __version__ );

	// arguments
	TCLAP::ValueArg<std::string> a_input("i","in","Input fMRI Image",true,"",
			"4D Image", cmd);

	TCLAP::ValueArg<std::string> a_labelmap("L","labelmap",
			"Input labelmap. Can be used to generate label-based regressors.",
			false,"","3D Image", cmd);

	TCLAP::MultiArg<std::string> a_labels("l","label", "Regress average of given"
			" label[s] then subtract the fit from each timeseries. Multiple "
			"options will result in multiple simultaneous linear regressions. "
			"Note that multiple labels can put together separated by commas "
			"(NOT SPACE) to average over multiple labels. Labels in labelmap "
			"(-L) to use as regressors. Timeseries from these regions will "
			"be extracted and will be used as regressors.", false,"int", cmd);

	TCLAP::ValueArg<int> a_components("c", "components", "Number of PCA or "
			"ICA components to extract from a large set of voxels when "
			"generating regressors", false, 7, "int", cmd);

	vector<TCLAP::Arg*> xorlist;
	TCLAP::SwitchArg a_ica("I", "ica", "Compute independent components from "
			"labeled time-series. For example -l label.nii.gz -L 1 "
			"-I -c 7 would therefore extract all the fMRI timepoints in label "
			"1 of  label.nii.gz and extract 7 independent time-series from that "
			"data, then use those as regressors.", cmd);
	TCLAP::SwitchArg a_pca("P", "pca", "Compute principal components from "
			"labeled time-series. For example -l label.nii.gz -L 1 "
			"-P -c 7 would therefore extract all the fMRI timepoints in label "
			"1 of  label.nii.gz and extract 7 uncorrelated time-series from that "
			"data, then use those as regressors.", cmd);
	TCLAP::SwitchArg a_avg("A", "average", "Compute average of "
			"labeled time-series within each label. For example -l label.nii.gz "
			"-L 1 -L 2 -A would therefore extract all the fMRI timepoints in label "
			"1 of label.nii.gz and compute their average, then do the same for "
			"label 2, thus creating 2 regressors.", cmd);
	TCLAP::SwitchArg a_noreduce("N", "no-reduce", "Do not reduce timeseries "
			"at all. Thus ALL voxels in labeled regions specified by -l would be "
			"used as regressors. Probably not a good idea.", cmd);

	vector<string> allowed({"none","cannonical"});
	TCLAP::ValuesConstraint<string> allowedVals( allowed );

//	TCLAP::ValueArg<std::string> a_hrf("H", "hrf", "Hemodynamic response function"
//			" to convolve regressors (-r) with. Note that any regressors "
//			"extracted from fMRI are not convolved. Presummably if its from the "
//			"brain it is already convolved by nature", false, "none",
//			&allowedVals, cmd);

	TCLAP::ValueArg<std::string> a_rplot("", "plot-regressors",
			"Image filename to save a plot of the regressors.", false, "",
			"*.tga", cmd);

//	TCLAP::ValueArg<std::string> a_hplot("", "plot-hrf",
//			"Image filename to save a plot of the hemodynamic response function.",
//			false, "", "*.tga", cmd);

	TCLAP::SwitchArg a_deriv("d", "derivative", "Add temporal derivatives "
			"of loaded regressors only, as regressors.", cmd);

	// for regressing-out
	TCLAP::ValueArg<std::string> a_output("o","out","Output fMRI Image",false,"",
			"4D Image", cmd);

	// for creating maps
	TCLAP::MultiArg<std::string> a_maps("m","map","Output map. Each output map "
			"corresponds to a column in the output regressors. (-R)",
			false,"3D Image", cmd);

	TCLAP::MultiArg<std::string> a_regressors("r","regressor",
			"Input Regressor.", false,"*.csv", cmd);
	TCLAP::MultiArg<std::string> a_events("e","events",
			"Input Regressor in the form of an event (1 column: even times,"
			" 3 column: onset duration value). These will be convolved with "
			"the cannonical hemodynamic response function from SPM",
			false,"*.csv", cmd);

	TCLAP::ValueArg<std::string> a_oregressors("R","outregress",
			"Output Regressors. Single file with all the regression timeseries "
			"used", false,"","*.csv", cmd);

	TCLAP::ValueArg<double> a_minfreq("f", "minfreq",
				"Remove signal above below the given frequency in hz. "
				"Note that if not provided, then 0 will be used ", false, 0,
				"hz", cmd);
	TCLAP::ValueArg<double> a_maxfreq("F", "maxfreq",
				"Remove signal above the given frequency in hz. If not set "
				"then all high frequency will be kept", false, INFINITY, "hz",
				cmd);

	cmd.add(a_verbose);

	// parse arguments
	cmd.parse(argc, argv);

	/*
	 * Main Processing
	 */

	/* Initialize Data */
	ptr<MRImage> fmri = readMRImage(a_input.getValue());
	ptr<MRImage> labelmap;

	if(fmri->tlen() <= 1) {
		std::cerr << "Error Image Input Not 4D" << endl;
		return -1;
	}

	int tlen = fmri->tlen();
	double tr = fmri->spacing(3);
	std::cout << "Time Points " << tlen << endl;

	/*
	 * Setting up Design Matrix X from:
	 * 	input files
	 * 	average of particular labels
	 * 	derivative of those
	 */
	//load design matrices from as many files are provided
	MatrixXd X;
	// Read Each CSV File then push the regressors
	for(auto it = a_regressors.begin(); it != a_regressors.end(); it++) {
		auto tmp = readNumericCSV(*it);
		if(tmp.size() != tlen)
			throw INVALID_ARGUMENT("Input regressor "+*it+" does not have the"
					"right number of rows (expected: "+to_string(tlen)+" got "
					+to_string(tmp.size()));
		size_t cols = X.cols();
		X.conservativeResize(tlen, cols+tmp[0].size());
		for(size_t rr=0; rr<tmp.size(); rr++) {
			if(tmp[rr].size() != X.cols()-cols)
				throw INVALID_ARGUMENT("Input regressor "+*it+" has "
						"mismatching number of columns in different rows ");
			for(size_t cc=0; cc<tmp[rr].size(); cc++)
				X(rr, cc+cols) = tmp[rr][cc];
		}
	}

	for(auto it = a_events.begin(); it != a_events.end(); it++) {
		auto tmp = readNumericCSV(*it);
		auto reg = getRegressor(tmp, tr, tlen, 0);
		size_t cols = X.cols();
		X.conservativeResize(tlen, cols+1);
		for(size_t rr=0; rr<tlen; rr++) {
			X(rr, cols) = reg[rr];
		}
	}

	// Derivatives
	if(a_deriv.isSet()) {
		// Compute Numeric Derivative
		MatrixXd Xd(X.rows(), X.cols());
		Xd.bottomRows(X.rows()-1) = X.topRows(X.rows()-1)-
			X.bottomRows(X.rows()-1);
		Xd.row(0).setZero();

		// Append Derivatives
		X.conservativeResize(tlen, Xd.cols());
		X.rightCols(Xd.cols()) = Xd;
	}

	if(a_rplot.isSet()) {
		Plotter plotter;
		vector<double> tmp(X.rows());
		for(size_t cc=0; cc<X.cols(); cc++) {
			for(size_t rr=0; rr<X.rows(); rr++)
				tmp[rr]= X(rr,cc);
			plotter.addArray(tmp.size(), tmp.data());
		}
		plotter.write(a_rplot.getValue());
	}

	/* fMRI Extract Regressors */
	if(a_pca.isSet() || a_ica.isSet() || a_avg.isSet()) {
		if(!(a_labelmap.isSet()&&a_labels.isSet())) {
			cerr << "Error, must provide labels and labelmap to regress "
				"ICA/PCA/MEAN fmri components" << endl;
			return 0;
		}

		//load labelmap and put it in fmri space
		if(!a_labelmap.getValue().empty()) {
			try{
				labelmap = readMRImage(a_labelmap.getValue());
				cerr << "Labelmap voxels: " << labelmap->elements() << endl;
				auto tmp = dPtrCast<MRImage>(fmri->extractCast(3, fmri->dim()));
				labelmap = resampleNN(labelmap, tmp, INT32);
				cerr << "Labelmap voxels: " << labelmap->elements() << endl;
			} catch(...) {
				std::cerr << "Failed to load image" << a_labelmap.getValue() << endl;
				return -1;
			}
		}

		// convert each group of labels into its own unique label. In the
		// following functions it is assumed that each non-zero label group
		// forms a set which should be merged then reduced
		auto rlabelmap = labelmap->cloneImage();
		fillZero(rlabelmap);

		int64_t newlabel = 1;
		NDIter<int64_t> lit(labelmap), rit(rlabelmap);
		for(auto it=a_labels.begin(); it!=a_labels.end(); it++){
			auto tmp = parseLine(*it, ",");
			std::set<int64_t> labels;
			cout << "Creating regressor from labels: " << endl;
			for(size_t ll = 0 ; ll < tmp.size() ; ll++)  {
				int label = atoi(tmp[ll].c_str());
				labels.insert(label);
				cout << label << " ";
			}

			for(lit.goBegin(), rit.goBegin(); !lit.eof(); ++lit, ++rit) {
				if(labels.count(*lit) > 0)
					rit.set(newlabel);
			}
			newlabel++;
		}
		cerr<<"Writing re-mapped lablemap, later on this code should be removed"<<endl;
		rlabelmap->write("relabeled.nii.gz");
		MatrixXd Xnew;

		// Create Regression Term, and add to X
		if(a_pca.isSet()) {
			Xnew = extractLabelPCA(fmri, rlabelmap, a_components.getValue());
		} else if(a_ica.isSet()) {
			Xnew = extractLabelICA(fmri, labelmap, a_components.getValue());
		} else if(a_avg.isSet()) {
			Xnew = extractLabelAVG(fmri, labelmap);
		}
	}

	if(a_oregressors.isSet()) {
		std::cout << "Saving design file to " << a_oregressors.getValue() << endl;
		std::ofstream ofs(a_oregressors.getValue().c_str());
		for(size_t rr=0; rr<X.rows(); rr++) {
			if(rr != 0)
				ofs << "\n";
			for(size_t cc=0; cc<X.cols(); cc++) {
				if(cc != 0)
					ofs << ",";
				ofs << X(rr,cc);
			}
		}
	}

	cerr << "Performing Regression...";
	if(X.cols() > 0) {
		try {
			fmri = regressOut(fmri, X);
		} catch(...) {
			std::cerr << "Error problems regressing out a parameter" << endl;
			return -1;
		}
		if(!fmri) {
			std::cerr << "Error problems regressing out a parameter" << endl;
			return -1;
		}
	}
	cerr << "Done!" << endl;

	/* Time Filter */
	cerr << "Performing Bandpass fileter...";
	if(a_minfreq.isSet() || a_maxfreq.isSet())
		fmriBandPass(fmri, a_minfreq.getValue(), a_maxfreq.getValue());
	cerr << "Done" << endl;

	if(a_output.isSet()) {
		fmri->write(a_output.getValue());
	}

	// done, catch all argument errors
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}
