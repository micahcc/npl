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

#include <tclap/CmdLine.h>

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

const double PI = acos(-1);

/***************************
 * Regression
 ***************************
*/

shared_ptr<MRImage> regressOut(shared_ptr<MRImage> inimg, 
        list<vector<double>> designs);

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
int extractLabelAvgTS(shared_ptr<const MRImage> fmri, 
        shared_ptr<const MRImage> labelmap, 
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
int extractLabelPcaTS(shared_ptr<const MRImage> fmri, 
        shared_ptr<const MRImage> labelmap, 
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
int extractLabelIcaTS(FImage4D::Pointer fmri, LImage3D::Pointer labelmap, 
		const set<int>& labels, size_t outsz, list<vector<double>>& design);

FImage4D::Pointer normaliseTS(FImage4D::Pointer fmri);

void computeAppendDerivs(list<vector<double>>& design, 
			int start, int number);

int main(int argc, char* argv[])
{
	try {
	/* 
	 * Command Line 
	 */
	TCLAP::CmdLine cmd("This is a program to perform general linear modeling on "
			"fMRI images. It can remove the effects of regressors (regress-out) "
			"and create maps of where the effects of regressors are significant. "
			"Regressors may be passed through CSV files (-r), or by providing a "
			"labeled image and label number to extract. The average of "
			"timeseries within the labeled regions may be used (-A) or the "
			"principal components (-P). Independent components (ICA) is also an "
			"option (-I). ", ' ', __version__ );

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
			" 3 column: onset duration value).", false,"*.csv", cmd);
	
	TCLAP::ValueArg<std::string> a_oregressors("R","outregress",
			"Output Regressors. Single file with all the regression timeseries "
			"used", false,"","*.csv", cmd);
	
	cmd.add(a_verbose);
	
	
	// parse arguments
	cmd.parse(argc, argv);

	/*
	 * Main Processing
	 */

	/* Initialize Data */
	FImage4D::Pointer fmri = NULL;
	LImage3D::Pointer labelmap = NULL;

	try{
		fmri = readImage<FImage4D>(a_input.getValue());
	} catch(...) {
		std::cerr << "Failed to load image" << a_input.getValue() << endl;
		return -1;
	}
	if(fmri->GetLargestPossibleRegion().GetSize()[3] <= 1) {
		std::cerr << "Error Image Input Not 4D" << endl;
		return -1;
	}

	
	int timepoints = fmri->GetLargestPossibleRegion().GetSize()[3];
	double tr = fmri->GetSpacing()[3];
	std::cout << "Time Points " << timepoints << endl;

	/* 
	 * Setting up Design Matrix X from:
	 * 	input files
	 * 	average of particular labels
	 * 	derivative of those
	 */
	//load design matrices from as many files are provided
	list<vector<double>> X;
	for(auto it = a_regressors.begin(); it != a_regressors.end(); it++) {
		vector<vector<string> > tmp;

		//note that this outputs a vector of rows, so we have
		//to take the transpose
		int maxwidth = smartReadCSV(*it, tmp);
		std::cout << "Read " << tmp.size() << " rows with max width " 
			<< maxwidth << endl;
		if((int)tmp.size() != timepoints) {
			std::cerr << "Error, " << *it << " has "
				<< tmp.size() << " timepoints, but the fmri image has "
				<< timepoints << " timepoints. These must be the same" << endl;
			return -5;
		}

		for(int jj = 0 ; jj < maxwidth; jj++) {
			X.push_back(vector<double>());
			X.back().resize(tmp.size(), 0);
		}
		
		list<vector<double> >::iterator first = X.end();
		for(int jj = 0; jj < maxwidth; jj++)
			first--;

		//iterate through rows
		for(size_t rr = 0; rr < tmp.size(); rr++) {
			list<vector<double> >::iterator cit = first;
			//iterate through columns
			for(size_t cc = 0 ; cc < tmp[rr].size() ; cit++, cc++) {
				(*cit)[rr] = atof((tmp[rr])[cc].c_str());
			}
		}
	}
	
	for(auto it = a_events.begin(); it != a_events.end(); it++) {
		vector<vector<string> > tmp;

		//note that this outputs a vector of rows, so we have
		//to take the transpose
		int maxwidth = smartReadCSV(*it, tmp);
		std::cout << "Read " << tmp.size() << " rows with max width " 
			<< maxwidth << endl;
		if(!(maxwidth == 3 || maxwidth == 1)) {
			std::cerr << "Error, " << *it << " is neither in the 1 or 3 column "
				<< "formats. Please provide stimulus files with 1 or 3 columns "
				"as specified in the -h page" << endl;
			return -5;
		}

		vector<vector<double>> tmp2(tmp.size());
		for(size_t rr=0; rr<tmp2.size(); rr++) {
			tmp2[rr].resize(maxwidth);
			if((int)tmp[rr].size() != maxwidth) {
				cerr << "Inconsistent row width in " << *it << endl;
				return -1;
			}
			for(int cc=0; cc<maxwidth; cc++) 
				tmp2[rr][cc] = atof(tmp[rr][cc].c_str());
		}

		X.push_back(getRegressor(tmp2, tr, timepoints, 0));
	}

	// Derivatives
	if(a_deriv.isSet()) {
		//Don't take the derivative of the input have input X 
		computeAppendDerivs(X, 0, X.size());
	}

	if(a_rplot.isSet()) {
		writePlot(a_rplot.getValue(), X);
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
				labelmap = readImage<LImage3D>(a_labelmap.getValue());
				cerr << "Labelmap voxels: " << labelmap->GetRequestedRegion().GetNumberOfPixels() << endl;
				auto tmp = extractTime(fmri, 0);
				labelmap = resampleTo<FImage3D, LImage3D>(tmp, labelmap, NEAREST);
				cerr << "Labelmap voxels: " << labelmap->GetRequestedRegion().GetNumberOfPixels() << endl;
			} catch(...) {
				std::cerr << "Failed to load image" << a_labelmap.getValue() << endl;
				return -1;
			}
		}

		// For Each group of labels
		std::vector<string> tmp;
		for(auto it=a_labels.begin(); it!=a_labels.end(); it++){
			parseLine(*it, ",", tmp);
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
		saveDesign(a_oregressors.getValue(), X);
	}

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

	if(a_output.isSet()) {
		writeImage<FImage4D>(a_output.getValue(), fmri);
	}
	
	// done, catch all argument errors
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}

/***************************
 * Regression
 ***************************
*/

FImage4D::Pointer regressOut(FImage4D::Pointer inimg, 
		list<vector<double> > designs)
{
	FImage4D::SizeType inSize = inimg->GetRequestedRegion().GetSize();
    FImage4D::IndexType index = {{0, 0, 0, 0}};
	size_t timepoints = inSize[3];
	size_t dsize = designs.size();

//	INPUT PARAMETERS:
//    XY          -   training set, array [0..NPoints-1,0..NVars]:
//                    * NVars columns - independent variables
//                    * last column - dependent variable
//    NPoints     -   training set size, NPoints>NVars+1
//    NVars       -   number of independent variables
//
//OUTPUT PARAMETERS:
//    Info        -   return code:
//                    * -255, in case of unknown internal error
//                    * -4, if internal SVD subroutine haven't converged
//                    * -1, if incorrect parameters was passed (NPoints<NVars+2, NVars<1).
//                    *  1, if subroutine successfully finished
//    LM          -   linear model in the ALGLIB format. Use subroutines of
//                    this unit to work with the model.
//    AR          -   additional results
//	const double tol = 0.0001;
//	size_t rank;
	
	// [0..NPoints-1,0..NVars]:
	// * NVars columns - independent variables
	// * last column - dependent variable
	alglib::real_2d_array xy;
	alglib::real_1d_array x;
	x.setlength(dsize);
	xy.setlength(timepoints, dsize+1);

	//fill the design matrix
	list<vector<double> >::iterator it = designs.begin();
	for(size_t ll = 0; it != designs.end(); ll++, it++){
		if(it->size() != timepoints) {
			std::cerr << "Error input design matrix time dimension must match"
				<< "image in time length." << endl;
			return NULL;
		}
		for(size_t tt = 0 ; tt < timepoints ; tt++) 
			xy(tt, ll) = (*it)[tt];
	}

	alglib::ae_int_t info;
	alglib::linearmodel lm;
	alglib::lrreport ar;

	//allocate output
	FImage4D::Pointer outimg = FImage4D::New();
	outimg->SetRegions(inimg->GetRequestedRegion());
	outimg->CopyInformation(inimg);
	outimg->Allocate();
    
	//apply to each voxel
	itk::ImageLinearIteratorWithIndex<FImage4D> iit(inimg, 
			inimg->GetRequestedRegion());
	itk::ImageLinearIteratorWithIndex<FImage4D> oit(outimg, 
			outimg->GetRequestedRegion());
	iit.SetDirection(3);
	oit.SetDirection(3);
	double avgRsq = 0;
	size_t nn = 0;
	for(index[0] = 0 ; index[0] < (int)inSize[0] ; index[0]++) {
        for(index[1] = 0 ; index[1] < (int)inSize[1] ; index[1]++) {
            for(index[2] = 0 ; index[2] < (int)inSize[2] ; index[2]++) {
				//copy from input to y
				iit.SetIndex(index);
				double min = std::numeric_limits<double>::max();
				double max = std::numeric_limits<double>::min();

				double mean = 0;
				double var = 0;
				iit.GoToBeginOfLine();
				for(size_t tt = 0 ; tt < inSize[3]; tt++, ++iit) {
					min = std::min(min, (double)iit.Get());
					max = std::max(max, (double)iit.Get());
					mean += iit.Get();
					var += pow(iit.Get(),2);
					xy(tt, dsize) = iit.Get();
				}

				var = sample_var(timepoints, mean, var);
				mean /= timepoints;

				if(fabs(max - min) > DELTA) {
					alglib::lrbuild(xy, timepoints, dsize,  info, lm, ar);
					double msqr = pow(ar.rmserror, 2);
	
					if(a_verbose.isSet()) {
						std::cout << index << " R-Squared: " << 1-(msqr/var) << "\n";
					}
					avgRsq += 1-(msqr/var);
					nn++;
					
					//copy to the output
					oit.SetIndex(index);
					iit.GoToBeginOfLine();
					for(size_t tt = 0 ; tt < inSize[3]; tt++, ++oit, ++iit) {
						for(size_t ii=0; ii<dsize; ii++)
							x[ii] = xy(tt,ii);

						oit.Set(iit.Get()-alglib::lrprocess(lm, x));
					}
				} else {
					//copy to the output
					oit.SetIndex(index);
					iit.GoToBeginOfLine();
					for(size_t tt = 0 ; tt < inSize[3]; tt++, ++oit) 
						oit.Set(iit.Get());
				}

			}
		}
	}
	std::cout << "Average R-Squared: " << avgRsq/nn << "\n";

	return outimg;
}

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
int extractLabelAvgTS(FImage4D::Pointer fmri, LImage3D::Pointer labelmap, 
		const set<int>& labels, list<vector<double>>& design)
{
	std::cout << "Average of: " ;
	for(auto it = labels.begin(); it != labels.end() ; it++) 
		std::cout << *it << ", " ;
	std::cout << endl;

	FImage4D::SizeType inSize = fmri->GetRequestedRegion().GetSize();
    FImage4D::IndexType findex = {{0, 0, 0, 0}};
    LImage3D::IndexType lindex = {{0, 0, 0}};
	int timepoints = inSize[3];
    
	design.push_back(vector<double>(timepoints,0));
	itk::ImageLinearIteratorWithIndex<FImage4D> iit(fmri, 
			fmri->GetRequestedRegion());
	iit.SetDirection(3);
	iit.GoToBegin();
	
	itk::ImageRegionIteratorWithIndex<LImage3D> lit(labelmap, 
			labelmap->GetRequestedRegion());
	lit.GoToBegin();

	size_t matchcount = 0;
	double mean;
	
	for(;!lit.IsAtEnd(); ++lit){
		if(labels.count(lit.Get()) > 0) {
			lindex = lit.GetIndex();
			for(int ii = 0 ; ii < 3 ; ii++)
				findex[ii] = lindex[ii];
			
			mean = 0;
			iit.SetIndex(findex);
			iit.GoToBeginOfLine();
			for(int tt = 0 ; !iit.IsAtEndOfLine(); tt++, ++iit) {
				mean += iit.Get();
			}

			//skip invalid timeseries
			if(isnan(mean) || isinf(mean)) 
				continue;
			mean /= timepoints;

			matchcount++;
			iit.SetIndex(findex);
			iit.GoToBeginOfLine();

			//since levels might vary across or over labels, subtract 
			//mean of voxels, then add to group mean
			for(int tt = 0 ; !iit.IsAtEndOfLine(); tt++, ++iit) 
				design.back()[tt] += (iit.Get() - mean);
		}
	}

	if(matchcount == 0) {
		//if there were no matching voxels, remove the added element
		design.pop_back();
	} else {
		for(int tt = 0 ; tt < timepoints ; tt++)
			design.back()[tt] /= matchcount;
	}
	return matchcount;
}

/**
 * @brief Computes the Principal Components of input matrix XT, 
 * NOTE: Data should already be de-meaned
 *
 * Outputs reduced dimension (fewer rows) output
 *
 * @param XT	rows x cols matrix where each column is a timepoint each row a dim
 * @param ncomp	number of output rows
 *
 * @return 		ncomp x cols matrix projected onto the principal components
 */
void pca(const alglib::real_2d_array& X, alglib::real_2d_array& Xr, int odim)
{
	alglib::real_2d_array U;
	alglib::real_2d_array VT;
	alglib::real_1d_array W;
	const double VARTHRESH = 1E-12;

	std::cout  << "svd: " << X.rows() << "x" << X.cols() << "...";
	alglib::rmatrixsvd(X, X.rows(), X.cols(), 1, 0, 2, W, U, VT);
	std::cout  << "  done" << endl;
	
	std::cout << "  Creating decorrelated timeseries...";
	if(odim <= 0 || odim > X.rows()) {
		//only keep dimensions with variance passing the threshold
		for(odim = 0; odim < std::min(X.cols(), X.rows()); odim++) {
			if(W[odim] < sqrt(VARTHRESH))
				break;
		}
	}

	std::cout << "odims = " << odim << "...";
	Xr.setlength(X.rows(), odim);
	for(int rr=0; rr<X.rows(); rr++) {
		for(int cc=0; cc<odim; cc++) {
			Xr(rr,cc) = U(rr, cc)*W[cc];
		}
	}
	std::cout  << "  Done" << endl;
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
int extractLabelPcaTS(FImage4D::Pointer fmri, LImage3D::Pointer labelmap, 
		const set<int>& labels, size_t outsz, list<vector<double>>& design)
{
	std::cout << "PCA of: " ;
	for(auto it = labels.begin(); it != labels.end() ; it++) 
		std::cout << *it << ", " << endl;
	std::cout << endl;

	FImage4D::SizeType inSize = fmri->GetRequestedRegion().GetSize();
    FImage4D::IndexType findex = {{0, 0, 0, 0}};
    LImage3D::IndexType lindex = {{0, 0, 0}};
	unsigned int timepoints = inSize[3];
    
	itk::ImageLinearIteratorWithIndex<FImage4D> iit(fmri, 
			fmri->GetRequestedRegion());
	iit.SetDirection(3);
	iit.GoToBegin();
	
	itk::ImageRegionIteratorWithIndex<LImage3D> lit(labelmap, 
			labelmap->GetRequestedRegion());
	lit.GoToBegin();

	unsigned int ndims = 0;
	
	for(;!lit.IsAtEnd(); ++lit){
		if(labels.count(lit.Get()) > 0) 
			ndims++;
	}

	std::cout << "Timeseries Matching Labels: " << ndims << endl;

	alglib::real_2d_array X;
	X.setlength(timepoints, ndims);

	//add every matching label to summary at given time
	lit.GoToBegin();
	for(int dd = 0; !lit.IsAtEnd(); ++lit){ 
		if(labels.count(lit.Get()) > 0) {
			lindex = lit.GetIndex();
			for(int ii = 0 ; ii < 3 ; ii++)
				findex[ii] = lindex[ii];
			
			double mean = 0;
			double stddev = 0;
			iit.SetIndex(findex);
			iit.GoToBeginOfLine();
			for(int tt = 0 ; !iit.IsAtEndOfLine(); tt++, ++iit)  {
				X(tt, dd) = iit.Get();
				mean += iit.Get();
				stddev += iit.Get()*iit.Get();
			}
			mean /= X.rows();
			stddev = sqrt(stddev/X.rows() - mean*mean);
			for(int tt = 0 ; tt < X.rows(); tt++) {
//				X(tt, dd) = (X(tt,dd)-mean)/stddev;
				X(tt, dd) = (X(tt,dd)-mean);
			}

			//only iterate through output if we find match
			dd++;
		}
	}

	alglib::real_2d_array ortho;
	std::cout << "PCA" << endl;
	pca(X, ortho, outsz);
	std::cout << "Done" << endl;

	std::cout << "Copying Reduced dimensions to output" << endl;
	//copy relevent timeseries:
	for(unsigned int dd = 0 ; dd < outsz; dd++) {
		design.push_back(vector<double>(timepoints));
		for(unsigned int tt = 0 ; tt < timepoints; tt++)  {
			design.back()[tt] = ortho(tt, dd);
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
int extractLabelIcaTS(FImage4D::Pointer fmri, LImage3D::Pointer labelmap, 
		const set<int>& labels, size_t outsz, list<vector<double>>& design)
{
	std::cout << "ICA of: " ;
	for(auto it = labels.begin(); it != labels.end() ; it++) 
		std::cout << *it << ", " << endl;
	std::cout << endl;

	FImage4D::SizeType inSize = fmri->GetRequestedRegion().GetSize();
    FImage4D::IndexType findex = {{0, 0, 0, 0}};
    LImage3D::IndexType lindex = {{0, 0, 0}};
	int timepoints = inSize[3];
    
	itk::ImageLinearIteratorWithIndex<FImage4D> iit(fmri, 
			fmri->GetRequestedRegion());
	iit.SetDirection(3);
	iit.GoToBegin();
	
	itk::ImageRegionIteratorWithIndex<LImage3D> lit(labelmap, 
			labelmap->GetRequestedRegion());
	lit.GoToBegin();

	int ndims = 0;
	
	for(;!lit.IsAtEnd(); ++lit){
		if(labels.count(lit.Get()) > 0) 
			ndims++;
	}

	std::cout << "Timeseries Matching Labels: " << ndims << endl;

	alglib::real_2d_array X;
	X.setlength(timepoints, ndims);

	//add every matching label to summary at given time
	lit.GoToBegin();
	for(int dd = 0; !lit.IsAtEnd(); ++lit){ 
		if(labels.count(lit.Get()) > 0) {
			lindex = lit.GetIndex();
			for(int ii = 0 ; ii < 3 ; ii++)
				findex[ii] = lindex[ii];
			
			double mean = 0;
			double stddev = 0;
			iit.SetIndex(findex);
			iit.GoToBeginOfLine();
			for(int tt = 0 ; !iit.IsAtEndOfLine(); tt++, ++iit)  {
				X(tt, dd) = iit.Get();
				mean += iit.Get();
				stddev += iit.Get()*iit.Get();
			}
			mean /= X.rows();
			stddev = sqrt(stddev/X.rows() - mean*mean);
			for(int tt = 0 ; tt < X.rows(); tt++) {
				X(tt, dd) = X(tt,dd)-mean;
//				X(tt, dd) = (X(tt,dd)-mean)/stddev;
			}

			//only iterate through output if we find match
			dd++;
		}
	}

	std::cout << "PCA...";
	alglib::real_2d_array ortho;
	pca(X, ortho, std::min((int)timepoints,ndims));
	std::cout << "Done " << endl;
	
	std::cout << "ICA...";
	ica(ortho, X, outsz);
	std::cout << "Done" << endl;

	std::cout << "Copying Reduced dimensions to output" << endl;
	//copy relevent timeseries:
	for(unsigned int dd = 0 ; dd < X.cols(); dd++) {
		design.push_back(vector<double>(timepoints, 0));
		for(int tt = 0 ; tt < X.rows(); tt++) 
			design.back()[tt] = X(tt, dd);
	}
	std::cout << "Done" << endl;

	return ndims;
}

FImage4D::Pointer normaliseTS(FImage4D::Pointer fmri)
{
	FImage4D::SizeType inSize = fmri->GetRequestedRegion().GetSize();
    //FImage4D::IndexType index = {{0, 0, 0, 0}};
	int timepoints = inSize[3];

	FImage4D::Pointer outimg = FImage4D::New();
	outimg->SetRegions(fmri->GetRequestedRegion());
	outimg->CopyInformation(fmri);
	outimg->Allocate();
	
	itk::ImageLinearIteratorWithIndex<FImage4D> iit(fmri, 
			fmri->GetRequestedRegion());
	itk::ImageLinearIteratorWithIndex<FImage4D> oit(outimg, 
			outimg->GetRequestedRegion());
	iit.SetDirection(3);
	oit.SetDirection(3);
	iit.GoToBegin();
	oit.GoToBegin();

	double mu = 0;
	double dev = 0;
	
	while(!iit.IsAtEnd() && !oit.IsAtEnd()) {
		mu = 0;
		dev = 0;
		while(!iit.IsAtEndOfLine()) {
			mu += iit.Get();
			dev += (iit.Get())*(iit.Get());
			++iit;
		}
		
		iit.GoToBeginOfLine();
		//var = 1/(n-1)(-N\bar{x}^2+sum(x^2))
		dev = sqrt(sample_var(timepoints, mu, dev));
		mu /= timepoints;
		if(dev == 0) {
			while(!oit.IsAtEndOfLine()) {
				oit.Set(0);
				++iit;
				++oit;
			}
		} else {
			while(!oit.IsAtEndOfLine()) {
				oit.Set((iit.Get()-mu)/dev);
				++iit;
				++oit;
			}
		}

		oit.NextLine();
		iit.NextLine();
	}

	return outimg;
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


