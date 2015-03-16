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
#include <list>
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
#include "iterators.h"
#include "version.h"

TCLAP::SwitchArg a_verbose("v", "verbose", "Be verbose (More for debugging).");

const double DELTA = 1e-30;

using std::list;
using std::set;
using std::vector;
using std::string;
using std::endl;
using std::cerr;
using std::cout;
using namespace npl;

const double PI = acos(-1);

/***************************
 * Regression
 ***************************
*/

ptr<MRImage> regressOut(ptr<const MRImage> inimg, list<vector<double>> designs);

/**
 * @brief Saves the average value at each time point in the given label into ret
 *
 * Note that labelmap and fmri should be in the same pixel space (except for
 * dimension 3)
 *
 * @param fmri		fmri with timeseries to extract, then average
 * @param labelmap 	labelmap to use when looking for voxels within the given label
 * @param labels 	label to search for when computing average
 * @param design 	output design matrix, an additional vector will be added
 *
 * @return number of relevent voxels found
 */
int extractLabelAvgTS(ptr<const MRImage> fmri,
		ptr<const MRImage> labelmap,
		const set<int>& labels, list<vector<double>>& design);

/**
 * @brief Creates a matrix of timeseries, then perfrorms principal components
 * analysis on it to reduce the number of timeseries to outsz. It appends
 * the resulting reduced timeseries onto design directly
 *
 * Note that labelmap and fmri should be in the same pixel space (except for
 * dimension 3)
 *
 * @param fmri 		FMRI image with timeseres to extract
 * @param labelmap	Labelmap used to identify relevent input timeseries
 * @param labels	Set of labels to identify relelvent input timeseries
 * @param outsz		Number of output timeseres ti append to design
 * @param design	Output Design matrix with the main principal componets added
 *
 * @return Number of relevent voxels found
 */
int extractLabelPcaTS(ptr<const MRImage> fmri,
		ptr<const MRImage> labelmap,
		const set<int>& labels, size_t outsz, list<vector<double>>& design);

/**
 * @brief Creates a matrix of timeseries, then perfrorms principal components
 * analysis on it to reduce the number of timeseries to outsz. It appends
 * the resulting reduced timeseries onto design directly
 *
 * Note that labelmap and fmri should be in the same pixel space (except for
 * dimension 3)
 *
 * @param fmri 		FMRI image with timeseres to extract
 * @param labelmap	Labelmap used to identify relevent input timeseries
 * @param labels	Set of labels to identify relelvent input timeseries
 * @param outsz		Number of output timeseres ti append to design
 * @param design	Output Design matrix with the main principal componets added
 *
 * @return Number of relevent voxels found
 */
int extractLabelIcaTS(ptr<const MRImage> fmri, ptr<const MRImage> labelmap,
		const set<int>& labels, size_t outsz, list<vector<double>>& design);

void computeAppendDerivs(list<vector<double>>& design,
			int start, int number);

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

	int timepoints = fmri->tlen();
	double tr = fmri->spacing(3);
	std::cout << "Time Points " << timepoints << endl;

	/*
	 * Setting up Design Matrix X from:
	 * 	input files
	 * 	average of particular labels
	 * 	derivative of those
	 */
	//load design matrices from as many files are provided
	list<vector<double>> X;
	// Read Each CSV File then push the regressors
	for(auto it = a_regressors.begin(); it != a_regressors.end(); it++) {
		auto tmp = readNumericCSV(*it);
		for(size_t ii=0; ii<tmp.size(); ii++) {
			X.push_back(std::move(tmp[ii]));
		}
	}

	for(auto it = a_events.begin(); it != a_events.end(); it++) {
		auto tmp = readNumericCSV(*it);
		X.push_back(getRegressor(tmp, tr, timepoints, 0));
	}

	// Derivatives
	if(a_deriv.isSet()) {
		//Don't take the derivative of the input have input X
		computeAppendDerivs(X, 0, X.size());
	}

	if(a_rplot.isSet()) {
		Plotter plotter;
		for(auto& v : X)
			plotter.addArray(v.size(), v.data());
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

		// For Each group of labels
		std::vector<string> tmp;
		for(auto it=a_labels.begin(); it!=a_labels.end(); it++){
			tmp = parseLine(*it, ",");
			std::set<int> labels;
			cout << "Creating regressor from labels: " << endl;
			for(size_t ll = 0 ; ll < tmp.size() ; ll++)  {
				int label = atoi(tmp[ll].c_str());
				labels.insert(label);
				cout << label << " ";
			}

			// Create Regression Term, and add to X
			if(a_pca.isSet()) {
				int count = extractLabelPcaTS(fmri, labelmap, labels,
						a_components.getValue(), X);
				std::cout << "Matching Voxels: " << count << endl;
			} else if(a_ica.isSet()) {
				int count = extractLabelIcaTS(fmri, labelmap, labels,
						a_components.getValue(), X);
				std::cout << "Matching Voxels: " << count << endl;
			} else if(a_avg.isSet()) {
				int count = extractLabelAvgTS(fmri, labelmap, labels, X);
				std::cout << "Matching Voxels: " << count << endl;
			}
		}


	}

	if(a_oregressors.isSet()) {
		std::cout << "Saving design file to " << a_oregressors.getValue() << endl;
		std::ofstream ofs(a_oregressors.getValue().c_str());
		for(auto it=X.begin(); it != X.end(); ++it) {
			if(it != X.begin())
				ofs << "\n";
			for(size_t cc=0; cc<it->size(); cc++) {
				if(cc != 0)
					ofs << ",";
				ofs << (*it)[cc];
			}
		}
	}

	cerr << "Performing Regression...";
	if(X.size() > 0) {
		try{
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

/***************************
 * Regression
 ***************************
*/

/**
 * @brief Removes the effects of X from signal (y). Note that this takes both X
 * and the pseudoinverse of X because the bulk of the computation will be on
 * the pseudoinverse.
 *
 * Each column of X should be a regressor, Xinv should be the pseudoinverse of X
 *
 * Beta in OLS may be computed with the pseudoinverse (P):
 * B = Py
 * where P is the pseudoinverse:
 * P = VE^{-1}U^*
 *
 * @param signal response term (y), will be modified to remove the effects of X
 * @param X Design matrix, or independent values in colums
 * @param Xinv the pseudoinverse of X
 */
inline
void regressOutLS(VectorXd& signal, const MatrixXd& X, const MatrixXd& Xinv)
{
	signal = signal - (Xinv*signal)*X;
}

ptr<MRImage> regressOut(ptr<const MRImage> inimg, list<vector<double>> designs)
{
	MatrixXd X(inimg->tlen(), designs.size()+1);

	size_t cc = 0, rr = 0;
	for(auto it=designs.begin(); it!=designs.end(); ++it, ++cc) {
		if(it->size() != inimg->tlen()) {
			cerr << "Error input image time-length does not match one of "
				"the regressors! (" << cc << ")" << endl;
		}

		for(rr=0; rr<it->size(); rr++)
			X(rr, cc) = it->at(rr);
	}

	// Add Intercept
	for(rr=0; rr<X.rows(); rr++)
		X(rr, X.cols()-1) = 1;

	size_t tlen = inimg->tlen();
	MatrixXd Xinv = pseudoInverse(X);
	VectorXd y(X.rows());
	VectorXd beta(X.cols());

	//allocate output
	auto out = dPtrCast<MRImage>(inimg->copyCast(FLOAT32));

	// Create Iterators
	Vector3DIter<double> oit(out);
	Vector3DConstIter<double> iit(inimg);

	// Iterate through and regress each line
	for(iit.goBegin(), oit.goBegin(); !iit.eof(); ++iit, ++oit) {

		// statistics
		double minv = std::numeric_limits<double>::max();
		double maxv = std::numeric_limits<double>::min();

		for(size_t tt=0; tt<tlen; ++tt) {
			minv = std::min(minv, iit[tt]);
			maxv = std::max(maxv, iit[tt]);
			y[tt] = iit[tt];
		}

		if(fabs(maxv - minv) > DELTA)
			regressOutLS(y, X, Xinv);
		else
			y.setZero();

		for(size_t tt=0; tt<tlen; ++tt) {
			oit.set(tt, y[tt]);
		}
	}

	return out;
}

/**
 * @brief Saves the average value at each time point in the given label into ret
 *
 * Note that labelmap and fmri should be in the same pixel space
 *
 * @param fmri		fmri with timeseries to extract, then average
 * @param labelmap 	labelmap to use when looking for voxels within the given label
 * @param labels 	label to search for when computing average
 * @param design 	output design matrix, an additional vector will be added
 *
 * @return number of relevent voxels found
 */
int extractLabelAvgTS(ptr<const MRImage> fmri, ptr<const MRImage> labelmap,
		const set<int>& labels, list<vector<double>>& design)
{
	std::cout << "Average of: " ;
	for(auto it = labels.begin(); it != labels.end() ; it++)
		std::cout << *it << ", " ;
	std::cout << endl;

	// Check Inputs
	if(!fmri->matchingOrient(labelmap, false, true)) {
		cerr << "Input image orientations do not match!" << endl;
		return -1;
	}

	int tlen = fmri->tlen();
	if(tlen <= 1) {
		cerr << "Input image is not 4D!" << endl;
		return -1;
	}

	// Append to Design
	design.push_back(vector<double>(tlen, 0));

	// Create Local Variables
	Vector3DConstIter<double> it(fmri);
	NDConstIter<int> lit(labelmap);
	lit.setOrder(it.getOrder());

	size_t matchcount = 0;
	double mean, stddev;
	for(it.goBegin(), lit.goBegin(); !lit.eof(); ++lit, ++it) {
		if(labels.count(*lit) > 0) {
			mean = 0;
			stddev = 0;
			for(int tt = 0 ; tt<tlen; tt++) {
				mean += it[tt];
				stddev += it[tt]*it[tt];
			}

			//skip invalid timeseries
			if(std::isnan(mean) || std::isinf(mean))
				continue;
			stddev = sqrt(sample_var(tlen, mean, stddev));
			mean /= tlen;
			matchcount++;

			//since levels might vary across or over labels, subtract
			//mean of voxels, then add to group mean
			for(int tt = 0 ; tt<tlen; ++tt)
				design.back()[tt] += (it[tt] - mean)/stddev;
		}
	}

	if(matchcount == 0) {
		//if there were no matching voxels, remove the added element
		design.pop_back();
	} else {
		for(int tt = 0 ; tt < tlen; tt++)
			design.back()[tt] /= matchcount;
	}

	return matchcount;
}

/**
 * @brief Creates a matrix of timeseries, then perfrorms principal components
 * analysis on it to reduce the number of timeseries to outsz. It appends
 * the resulting reduced timeseries onto design directly
 *
 * Note that labelmap and fmri should be in the same pixel space (except for
 * dimension 3)
 *
 * @param fmri 		FMRI image with timeseres to extract
 * @param labelmap	Labelmap used to identify relevent input timeseries
 * @param labels	Set of labels to identify relelvent input timeseries
 * @param outsz		Number of output timeseres ti append to design
 * @param design	Output Design matrix with the main principal componets added
 *
 * @return Number of relevent voxels found
 */
int extractLabelPcaTS(ptr<const MRImage> fmri, ptr<const MRImage> labelmap,
		const set<int>& labels, size_t outsz, list<vector<double>>& design)
{
	std::cout << "PCA of: " ;
	for(auto it = labels.begin(); it != labels.end() ; it++)
		std::cout << *it << ", " << endl;
	std::cout << endl;

	// Check inputs
	if(!fmri->matchingOrient(labelmap, false, true)) {
		cerr << "Input image orientations do not match!" << endl;
		return -1;
	}

	int tlen = fmri->tlen();
	if(tlen <= 1) {
		cerr << "Input image is not 4D!" << endl;
		return -1;
	}

	// Append to Design
	design.push_back(vector<double>(tlen, 0));

	// Create Local Variables
	Vector3DConstIter<double> it(fmri);
	NDConstIter<int> lit(labelmap);
	lit.setOrder(it.getOrder());

	unsigned int ndims = 0;

	for(lit.goBegin(); !lit.eof(); ++lit){
		if(labels.count(*lit) > 0)
			ndims++;
	}

	// Create Output
	std::cout << "Timeseries Matching Labels: " << ndims << endl;
	MatrixXd X(tlen, ndims);

	// Fill Matrix with X
	double mean, stddev;
	size_t dd=0;
	for(it.goBegin(), lit.goBegin(); !lit.eof(); ++lit, ++it) {
		if(labels.count(*lit) > 0) {
			mean = 0;
			stddev = 0;

			// Compute Mean and Fill Column of X
			for(int tt = 0 ; tt<tlen; tt++) {
				X(tt, dd) = it[tt];
				mean += it[tt];
				stddev += it[tt]*it[tt];
			}
			stddev = sqrt(sample_var(tlen, mean, stddev));
			mean /= tlen;

			//skip invalid timeseries (left as zeros)
			if(std::isnan(mean) || std::isinf(mean)) {
				for(int tt = 0 ; tt<tlen; ++tt)
					X(tt, dd) = 0;
			} else {
				for(int tt = 0 ; tt<tlen; ++tt)
					X(tt, dd) = (X(tt,dd) - mean)/stddev;
			}

			//only iterate through output if we find match
			dd++;
		}
	}

	std::cout << "PCA" << endl;
	X = pca(X, 1, outsz);
	std::cout << "Done" << endl;

	std::cout << "Copying Reduced dimensions to output" << endl;
	//copy relevent timeseries:
	for(unsigned int dd = 0 ; dd < outsz; dd++) {
		design.push_back(vector<double>(tlen));
		for(unsigned int tt = 0 ; tt < tlen; tt++)  {
			design.back()[tt] = X(tt, dd);
		}
	}
	std::cout << "Done" << endl;

	return ndims;
}

/**
 * @brief Creates a matrix of timeseries, then perfrorms principal components
 * analysis on it to reduce the number of timeseries to outsz. It appends
 * the resulting reduced timeseries onto design directly
 *
 * Note that labelmap and fmri should be in the same pixel space (except for
 * dimension 3)
 *
 * @param fmri 		FMRI image with timeseres to extract
 * @param labelmap	Labelmap used to identify relevent input timeseries
 * @param labels	Set of labels to identify relelvent input timeseries
 * @param outsz		Number of output timeseres ti append to design
 * @param design	Output Design matrix with the main principal componets added
 *
 * @return Number of relevent voxels found
 */
int extractLabelIcaTS(ptr<const MRImage> fmri, ptr<const MRImage> labelmap,
		const set<int>& labels, size_t outsz, list<vector<double>>& design)
{
	std::cout << "ICA of: " ;
	for(auto it = labels.begin(); it != labels.end() ; it++)
		std::cout << *it << ", " << endl;
	std::cout << endl;

	// Check inputs
	if(!fmri->matchingOrient(labelmap, false, true)) {
		cerr << "Input image orientations do not match!" << endl;
		return -1;
	}

	int tlen = fmri->tlen();
	if(tlen <= 1) {
		cerr << "Input image is not 4D!" << endl;
		return -1;
	}

	// Append to Design
	design.push_back(vector<double>(tlen, 0));

	// Create Local Variables
	Vector3DConstIter<double> it(fmri);
	NDConstIter<int> lit(labelmap);
	lit.setOrder(it.getOrder());

	unsigned int ndims = 0;

	for(lit.goBegin(); !lit.eof(); ++lit){
		if(labels.count(*lit) > 0)
			ndims++;
	}

	// Create Output
	std::cout << "Timeseries Matching Labels: " << ndims << endl;
	MatrixXd X(tlen, ndims);

	// Fill Matrix with X
	double mean, stddev;
	size_t dd=0;
	for(it.goBegin(), lit.goBegin(); !lit.eof(); ++lit, ++it) {
		if(labels.count(*lit) > 0) {
			mean = 0;
			stddev = 0;

			// Compute Mean and Fill Column of X
			for(int tt = 0 ; tt<tlen; tt++) {
				X(tt, dd) = it[tt];
				mean += it[tt];
				stddev += it[tt]*it[tt];
			}

			stddev = sqrt(sample_var(tlen, mean, stddev));
			mean /= tlen;

			//skip invalid timeseries (left as zeros)
			if(std::isnan(mean) || std::isinf(mean)) {
				for(int tt = 0 ; tt<tlen; ++tt)
					X(tt, dd) = 0;
			} else {
				for(int tt = 0 ; tt<tlen; ++tt)
					X(tt, dd) = (X(tt,dd) - mean)/stddev;
			}

			//only iterate through output if we find match
			dd++;
		}
	}

	std::cout << "PCA" << endl;
	X = pca(X, 1, outsz);
	std::cout << "Done" << endl;

	std::cout << "ICA...";
	X = ica(X);
	std::cout << "Done" << endl;

	std::cout << "Copying Reduced dimensions to output" << endl;
	//copy relevent timeseries:
	for(unsigned int dd = 0 ; dd < outsz; dd++) {
		design.push_back(vector<double>(tlen));
		for(unsigned int tt = 0 ; tt < tlen; tt++)  {
			design.back()[tt] = X(tt, dd);
		}
	}
	std::cout << "Done" << endl;

	return ndims;
}

void computeAppendDerivs(list<vector<double>>& design,
			int start, int number)
{
	if(design.size() == 0)
		return;

	int sz = design.size();
	int tlen = 0;
	int maxlength = design.front().size();
	std::list< std::vector<double> >::iterator it = design.begin();

	double next = 0;
	double prev = 0;

	for(int ii = 0 ; ii < sz ; ii++) {
		//skip ones we arent taking the derivatives of
		if(ii < start || ii >= start + number) {
			std::cerr << "Skipping derivative of " << ii << endl;
			it++;
			continue;
		}
		//update maxlength
		tlen = it->size();
		maxlength = std::max(tlen, maxlength);

		//push corresponding derivative vector to back, then fill
		design.push_back(std::vector<double>(tlen, 0));

		//calculate the average of
		//each pair of time points and then replace the existing value with the
		//average. Thus timepoint 2 gets 0.5*(ts[2]-ts[1]) + 0.5(ts[3] - ts[2])
		for(int jj = 1 ; jj < tlen-1; jj++) {
			next = (*it)[jj+1];
			prev = (*it)[jj-1];
			design.back()[jj] = next-prev;
		}

		//first and last just get set to the first and last whole deltas
		design.back()[0] = (*it)[1] - (*it)[0];
		design.back()[tlen-1] = (*it)[tlen-1] - (*it)[tlen-2];

		it++;
	}
}

