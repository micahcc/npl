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
#include "statistics.h"
#include "mrimage.h"
#include "mrimage_utils.h"
#include "kernel_slicer.h"
#include "kdtree.h"
#include "iterators.h"
#include "accessors.h"
#include "utility.h"
#include "nplio.h"
#include "basic_plot.h"

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

	TCLAP::ValueArg<string> a_fmri("i", "input", "fMRI image.",
			true, "", "*.nii.gz", cmd);
	TCLAP::MultiArg<string> a_events("e", "event-reg", "Event-related regression "
			"variable. Three columns, ONSET DURATION VALUE. If these overlap, "
			"an error will be thrown. ", true, "*.txt", cmd);

    TCLAP::ValueArg<string> a_odir("o", "outdir", "Output directory.", false,
            ".", "dir", cmd);

	cmd.parse(argc, argv);
	
	// read fMRI image
	shared_ptr<MRImage> fmri = readMRImage(a_fmri.getValue());
	if(fmri->ndim() != 4) {
		cerr << "Input should be 4D!" << endl;
		return -1;
	}
	assert(fmri->tlen() == fmri->dim(3));
	int tlen = fmri->tlen();
	double TR = fmri->spacing(3);

	// read the event-related designs, will have rows to match time, and cols
	// to match number of regressors
	MatrixXd X(tlen, a_events.getValue().size());
	size_t regnum = 0;
	for(auto it=a_events.begin(); it != a_events.end(); it++, regnum++) {
		auto events = readNumericCSV(*it);
		auto v = getRegressor(events, TR, tlen, 0);

		// draw
		writePlot(a_odir.getValue()+"/ev1.svg", v);

		// copy to output
		for(size_t ii=0; ii<tlen; ii++) 
			X(ii, regnum) = v[ii];
	}

    // create output images
    std::list<ptr<MRImage>> tImgs;
    std::list<ptr<MRImage>> pImgs;
    std::list<NDAccess<double>> tAccs;
    std::list<NDAccess<double>> pAccs;
    for(size_t ii=0; ii<X.cols(); ii++) {
        tImgs.push_back(dPtrCast<MRImage>(fmri->extractCast(3, fmri->dim(), FLOAT64)));
        tAccs.push_back(NDAccess<double>(tImgs.back()));
        pImgs.push_back(dPtrCast<MRImage>(fmri->extractCast(3, fmri->dim(), FLOAT64)));
        pAccs.push_back(NDAccess<double>(pImgs.back()));
    }

    

    vector<int64_t> ind(3);

    // Cache Reused Vectors
    auto Xinv = pseudoInverse(X);
    auto covInv = pseudoInverse(X.transpose()*X);

    const double MAX_T = 100;
    const double STEP_T = 0.1;
    auto student_cdf = students_t_cdf(X.rows()-1, STEP_T, MAX_T);

    // regress each timesereies
    ChunkIter<double> it(fmri);
    it.setLineChunk(3);
    Eigen::VectorXd signal(tlen);
    for(it.goBegin(); !it.eof(); it.nextChunk()) {

        // copy to signal
        it.goChunkBegin();
        for(size_t tt=0; !it.eoc(); ++tt, ++it) 
            signal[tt] = *it;

        RegrResult ret = regress(signal, X, covInv, Xinv, student_cdf);

        auto t_it = tAccs.begin();
        auto p_it = pAccs.begin();
        for(size_t ii=0; ii<X.cols(); ii++) {
            (*t_it).set(ret.t[ii], ind);
            (*p_it).set(ret.p[ii], ind);
            ++t_it;
            ++p_it;
        }
    }

    auto t_it = tImgs.begin();
    auto p_it = pImgs.begin();
    for(size_t ii=0; ii<X.cols(); ii++) {
        (*t_it)->write(a_odir.getValue()+"/t_"+to_string(ii)+".nii.gz");
        (*p_it)->write(a_odir.getValue()+"/p_"+to_string(ii)+".nii.gz");
    }

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}


