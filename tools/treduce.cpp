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
 * @file treduce.cpp Reduce 4D image to 3D image using one of the supplied
 * methods. 
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
#include <regex>

#include <tclap/CmdLine.h>

#include "version.h"
#include "mrimage.h"
#include "ndarray.h"
#include "nplio.h"
#include "iterators.h"
#include "utility.h"
#include "basic_functions.h"

using namespace npl;
using namespace std;

int main(int argc, char* argv[])
{
try {
	/* 
	 * Command Line 
	 */
	TCLAP::CmdLine cmd("This program takes an 4D volume, and outputs 3D"
			"volumes with the specified statistics.", ' ', __version__ );

	// arguments
	TCLAP::ValueArg<std::string> a_input("i","in","Input 4D Image",true,"",
			"4D Image", cmd);
	TCLAP::ValueArg<std::string> a_avg("a","avg","Output average of "
			"each timeseries", false,"", "img", cmd);
	TCLAP::ValueArg<std::string> a_sum("s","sum","Output sum of "
			"each timeseries", false,"", "img", cmd);
	TCLAP::ValueArg<std::string> a_variance("v","var","Output variance of "
			"each timeseries", false,"", "img", cmd);

	TCLAP::ValueArg<std::string> a_min("b","lowerbound","Minimum "
			"(lower bound) of each timeseries", false,"", "img", cmd);
	TCLAP::ValueArg<std::string> a_max("B","upperbound","Maximum "
			"(upper bound) of each timeseries", false,"", "img", cmd);
	TCLAP::ValueArg<std::string> a_median("m","median","Output median value "
			"of each timeseries", false,"", "img", cmd);

	// parse arguments
	cmd.parse(argc, argv);
	
	auto input = readMRImage(a_input.getValue());
	vector<size_t> osize(min(3UL, input->ndim()));
	for(size_t dd=0; dd<osize.size(); dd++) {
		osize[dd] = input->dim(dd);

		// compute sum, sum sqr
		auto avgimg = dPtrCast<MRImage>(input->createAnother(
					osize.size(), osize.data(), FLOAT32));
		auto varimg = dPtrCast<MRImage>(input->createAnother(
					osize.size(), osize.data(), FLOAT32));
		auto sumimg = dPtrCast<MRImage>(input->createAnother(
					osize.size(), osize.data(), FLOAT32));
		auto minimg = dPtrCast<MRImage>(input->createAnother(
					osize.size(), osize.data(), FLOAT32));
		auto maximg = dPtrCast<MRImage>(input->createAnother(
					osize.size(), osize.data(), FLOAT32));
		auto medianimg = dPtrCast<MRImage>(input->createAnother(
					osize.size(), osize.data(), FLOAT32));
		
		Vector3DIter<double> iit(input);
		NDIter<double> ait(avgimg);
		NDIter<double> vit(varimg);
		NDIter<double> sit(sumimg);
		NDIter<double> minit(minimg);
		NDIter<double> maxit(maximg);
		NDIter<double> medit(medianimg);
		vector<double> sorted(input->tlen());
		for(; !sit.eof() && !iit.eof(); ++iit) {
			double sum = 0;
			double ssq = 0;
			for(size_t tt=0; tt<input->tlen(); ++tt) {
				sum += iit[tt];
				ssq += iit[tt]*iit[tt];
				sorted[tt] = iit[tt];
			}

			sit.set(sum);
			vit.set(sample_var(input->tlen(), sum, ssq));
			ait.set(sum/input->tlen());
			
			std::sort(sorted.begin(), sorted.end());
			medit.set(sorted[input->tlen()/2]);
			maxit.set(sorted[input->tlen()-1]);
			minit.set(sorted[0]);

			++ait;
			++vit;
			++sit;
			++minit;
			++maxit;
			++medit;
		}

		if(a_avg.isSet())
			avgimg->write(a_avg.getValue());
		if(a_variance.isSet())
			varimg->write(a_variance.getValue());
		if(a_sum.isSet())
			sumimg->write(a_sum.getValue());
		if(a_min.isSet())
			minimg->write(a_min.getValue());
		if(a_max.isSet())
			maximg->write(a_max.getValue());
		if(a_median.isSet())
			medianimg->write(a_median.getValue());
	}
	
	// done, catch all argument errors
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
	
	return 0;
}


