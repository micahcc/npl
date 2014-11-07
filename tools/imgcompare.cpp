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
 * @file imgcompare.cpp Compare two images using some metric
 *
 *****************************************************************************/

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>

#include "tclap/CmdLine.h"
#include "version.h"
#include "mrimage_utils.h"
#include "ndarray_utils.h"
#include "mrimage.h"
#include "accessors.h"
#include "nplio.h"

using namespace npl;
using namespace std;

int main(int argc, char* argv[])
{
	cerr << "Version: " << __version__ << endl;
	try {
	
	TCLAP::CmdLine cmd("This program takes two images and directly compares "
			"them with mutual, information, normalized mutual information or "
			"correlation", 
			' ', __version__ );

	TCLAP::UnlabeledMultiArg<string> a_in("input", "MRI Image to Process.", 
			true, "*.nii.gz");
	cmd.add(a_in);
	
	TCLAP::ValueArg<string> a_mask("M", "mask", "Mask image", false, "", 
			"image", cmd);
	
	TCLAP::ValueArg<string> a_scatter("s", "scatter", "Scatter data file will "
			"have two columns, x,y ", false, "", "*.txt", cmd);

	TCLAP::ValueArg<int> a_bins("B", "bins", "Bins in marginal histograms "
			"(if MI is used)", false, 256, "#bins", cmd);
	TCLAP::ValueArg<int> a_radius("R", "radius", "Kernel radius during bin "
			"process (apprxomating histogram for MI)", false, 8, "radius", cmd);

	std::vector<string> allowed({"cor","mse", "mi", "nmi", "redund", "mim", 
				"dcor", "mid", "zcor"});
	TCLAP::ValuesConstraint<string> allowedVals( allowed );
	TCLAP::ValueArg<string> a_method("m", "metric", "Comparison metric. "
			"Options include correlation (cor), z-score transform "
			"correlation (zcor), mean-squared error (mse), "
			"mutual information (mi), normalized mutual information (nmi), "
			"redunancy (redund), mutual information metric (mim), "
			"dual correlation (dcor), and MI-based distance (mid).",
			false, "cor", &allowedVals, cmd);
	cmd.parse(argc, argv);
	
	if(a_in.getValue().size() != 2) {
		cerr << "Error, must provide two and only two images to compare" << endl;
		return -1;
	}
	auto img1 = readMRImage(a_in.getValue()[0]);
	auto img2 = readMRImage(a_in.getValue()[1]);
	size_t ndim = min(img1->ndim(), img2->ndim());
	img1 = dPtrCast<MRImage>(img1->extractCast(ndim, img1->dim()));
	img2 = dPtrCast<MRImage>(img2->extractCast(ndim, img2->dim()));

	for(size_t dd=0; dd<ndim; dd++) {
		if(img1->dim(dd) != img2->dim(dd)) {
			cerr << "Image dimension mismatch!" << endl; 
			return -1;
		}
	}
	// TODO FIX THIS, doesn't work for 4D images
//	{
//	auto img2r = dPtrCast<MRImage>(img1->createAnother(img2->type()));
//	//resample images together
//	vector<double> pt(ndim);
//	vector<double> cind(ndim);
//	LanczosInterpNDView<double> vw2(img2);
//	vw2.m_ras = true;
//	for(NDIter<double> it(img2r); !it.eof(); ++it) {
//		it.index(cind.size(), cind.data());
//		img2r->indexToPoint(cind.size(), cind.data(), pt.data());
//		cerr <<vw2.get(pt.size(), pt.data())<<endl;
//		it.set(vw2.get(pt.size(), pt.data()));
//	}
//	img2 = img2r;
//	}

	// Read Mask and Resample into Image 1 space
	ptr<MRImage> mask;
	if(a_mask.isSet()) {
		mask = readMRImage(a_mask.getValue());
		if(mask->ndim() < ndim) {
			cerr << "Mask is lower dimension that the other inputs!" << endl;
			return -1;
		}
		mask = dPtrCast<MRImage>(mask->extractCast(ndim, mask->dim()));
		auto maskr = dPtrCast<MRImage>(img1->createAnother(INT8));
		vector<double> indpt(ndim);
		
		//resample images together
		LinInterpNDView<double> vwm(mask);
		vwm.m_ras = true;
		for(NDIter<double> it(maskr); !it.eof(); ++it) {
			it.index(indpt.size(), indpt.data());
			maskr->indexToPoint(indpt.size(), indpt.data(), indpt.data());
			it.set(vwm.get(indpt.size(), indpt.data()) > 0.5);
		}
		mask = maskr;
	}

	// scatter
	if(a_scatter.isSet()) {
		ofstream ofs(a_scatter.getValue());
		if(!ofs.is_open()) {
			cerr << "Failed to open " << a_scatter.getValue() << endl;
			return -1;
		}

		FlatConstIter<int> mit;
		if(mask) {
			mit.setArray(mask);
			mit.goBegin();
		}
		for(FlatConstIter<double> it1(img1), it2(img2); !it1.eof(); ++it1, ++it2) {
			if(mask && *mit > 0) 
				ofs << it1.get() << ", " << it2.get() << endl;
			if(mask) 
				++mit;
		}
	}

	// Compare
	double sim = 0;
	cerr << "Perfoming Comparison" << endl;
	if(a_method.getValue() == "mse") 
		sim = mse(img1, img2, mask);
	else if(a_method.getValue() == "mi")
		sim = information(img1, img2, a_bins.getValue(),
				a_radius.getValue(), METRIC_MI, mask);
	else if(a_method.getValue() == "nmi")
		sim = information(img1, img2, a_bins.getValue(), a_radius.getValue(),
				METRIC_NMI, mask);
	else if(a_method.getValue() == "redund") 
		sim = information(img1, img2, a_bins.getValue(), a_radius.getValue(),
				METRIC_REDUNDANCY, mask);
	else if(a_method.getValue() == "mim")
		sim = information(img1, img2, a_bins.getValue(), a_radius.getValue(),
				METRIC_NMI, mask);
	else if(a_method.getValue() == "dcor")
		sim = information(img1, img2, a_bins.getValue(), a_radius.getValue(),
				METRIC_DUALTC, mask);
	else if(a_method.getValue() == "mid")
		sim = information(img1, img2, a_bins.getValue(), a_radius.getValue(),
				METRIC_VI, mask);
	else if(a_method.getValue() == "zcor") { 
		sim = corr(img1, img2, mask);
		sim = .5*log((1+sim)/(1-sim));
	} else if(a_method.getValue() == "cor")
		sim = corr(img1, img2, mask);
	else
		return 0;

	cout << sim << endl;

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }

	return 0;
}



