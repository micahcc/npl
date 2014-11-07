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
 * @file densitycluster.cpp Clustering based on local density.
 *
 *****************************************************************************/

#include <string>

#include <tclap/CmdLine.h>
#include "nplio.h"
#include "mrimage.h"
#include "iterators.h"
#include "accessors.h"
#include "ndarray_utils.h"
#include "statistics.h"
#include "version.h"
#include "macros.h"

using namespace std;
using namespace npl;
	
int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Segments image(s) based on intensity", ' ', 
			__version__ );

	TCLAP::MultiArg<string> a_in("i", "input", "Input image. If this is 4D "
			"then each of the volumes is treated as a dimension, the same as "
			"for multiple inputs.", true, "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_out("o", "out", "Output image (3D,INT32).",
			true, "", "*.nii.gz", cmd);
	TCLAP::MultiArg<int> a_dims("d", "dims", "Dimensions to use for "
			"classification. This is counted across inputs (-i), so if you "
			"give a 4D image with 3 timepoints and a 4D image with 2 "
			"timepoints then to select the first volume you would do -d 0. "
			"To also select the last you would do -d 0 -d 4",
			false, "*.nii.gz", cmd);
	TCLAP::ValueArg<double> a_valweight("w", "valweight", "Weight of values "
			"in terms of physical distance. The values are already "
			"standardized so this would be the number of mm equivalent to "
			"the entire range of values. So 5 means a 5mm distance is "
			"equivalent to the difference of min to max values.", 
			false, 15, "mm", cmd);
	TCLAP::ValueArg<double> a_winwidth("W", "kerne-window", 
			"Window size fo computing density, smaller values are somewhat"
			"faster and better as long as this is sufficiently large to capture"
			"several neighboring points.", false, 4, "mm", cmd);
	TCLAP::ValueArg<double> a_outthresh("T", "cluster-thresh", 
			"Threshold for determining whether something is a 'cluster'. This"
			"is the number of standard deviations from the mean to go when"
			"considering points as outliers and therefore deserving of their "
			"own cluster." , false, 8, "stddevs", cmd);

	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	// read inputs
	list<vector<double>> insamples;
	ptr<MRImage> outimg;
	size_t cdim = 0;
	size_t nrows = 0;
	vector<size_t> osize;
	VectorXd origin, spacing;
	MatrixXd direction;
	for(auto it=a_in.begin(); it!=a_in.end(); ++it) {
		auto inimg = readMRImage(*it);
		size_t tlen = inimg->tlen();
		size_t volsize = inimg->elements()/tlen;
		if(nrows == 0){
			nrows = volsize;
			osize.resize(min(inimg->ndim(),3UL));
			for(size_t dd=0; dd<osize.size(); dd++)
				osize[dd] = inimg->dim(dd);
			direction = inimg->getDirection();
			spacing = inimg->getSpacing();
			origin = inimg->getOrigin();
			outimg = dPtrCast<MRImage>(inimg->createAnother(min(3UL,
							inimg->ndim()), inimg->dim(), INT32));
		} else if(nrows != volsize) {
			cerr << "Input volumes must have same number of pixels" << endl;
			cerr << "Input: " << *it << " has different number from the rest!" 
				<< endl;
			return -1;
		}

		for(size_t tt=0; tt<tlen; tt++, cdim++) {
			bool use = !a_dims.isSet();
			for(auto dit = a_dims.begin(); dit != a_dims.end(); ++dit) {
				if(*dit == cdim) 
					use = true;
			}

			if(use) {
				cerr << "Including " << tt << " from " << *it << endl;
				// copy
				insamples.push_back(vector<double>());
				insamples.back().resize(nrows);
				size_t rr=0;
				for(Vector3DIter<double> it(inimg); !it.eof(); ++it, rr++) {
					insamples.back()[rr] = it[tt];
				}
			}
		}
	}

	if(insamples.size() == 0) {
		cerr << "No Data Selected!" << endl;
		return -1;
	}

	cerr << "Normalizing" << endl;
	// Normalize
	for(auto it = insamples.begin(); it != insamples.end(); it++) {
		double minv = INFINITY;
		double maxv = -INFINITY;
		for(size_t jj=0; jj<it->size(); jj++) {
			minv = min(minv, (*it)[jj]);
			maxv = max(maxv, (*it)[jj]);
		}

		for(size_t jj=0; jj<it->size(); jj++) 
			(*it)[jj] = a_valweight.getValue()*((*it)[jj]-minv)/
				(maxv-minv);
	}

	// Make each row of samples a pixel from the first input image. The highest
	// dimensions carry physical location, the lower cary pixel data
	Eigen::MatrixXf samples(nrows, insamples.size()+outimg->ndim());
	cerr << "Creating Samples (" << samples.rows() << "x" << samples.cols()
		<< ")" << endl;
	
	// add coordinates
	vector<double> pt(outimg->ndim());
	size_t rr = 0;
	for(NDIter<double> it(outimg); !it.eof(); ++it, ++rr) {
		// fill in samples from insamples
		size_t cc = 0;
		for(auto sit=insamples.begin(); sit!=insamples.end(); ++sit,++cc) {
			samples(rr,cc) = (*sit)[rr];
		}

		// fill in remaing from point locations
		it.index(pt.size(), pt.data());
		outimg->indexToPoint(pt.size(), pt.data(), pt.data());
		for(size_t kk=0; cc < samples.cols(); ++kk,++cc) {
			samples(rr, cc) = pt[kk];
		}
	}

	// free up memory
	insamples.clear();

	cerr << "Clustering" << endl;
	// Clustering By Fast Search and Find of Density Peaks
	Eigen::VectorXi labels;
	fastSearchFindDP(samples, a_winwidth.getValue(), a_outthresh.getValue(),
			labels,  false);
	cerr << "Done" << endl;

	/*
	 * Create, Fill Output
	 */ 
	size_t ii=0;
	for(FlatIter<int> it(outimg); !it.eof(); ++it, ++ii) 
		it.set(labels[ii]);

	// write 
	outimg->write(a_out.getValue());
	
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}



