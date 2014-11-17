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
 * @file motioncorr.cpp Perform motion correction on a 4D Image
 *
 *****************************************************************************/

#include <string>
#include <fstream>
#include <iterator>

#include <tclap/CmdLine.h>

#include "gradient.h"
#include "lbfgs.h"
#include "registration.h"

#include "nplio.h"
#include "mrimage.h"
#include "iterators.h"
#include "accessors.h"
#include "ndarray_utils.h"
#include "mrimage_utils.h"
#include "version.h"
#include "macros.h"

using namespace std;
using namespace npl;

/**
 * @brief Computes motion parameters from an fMRI image
 *
 * @param fmri Input fMRI 
 * @param reftime Volume to use for reference 
 * @param sigmas Standard deviation (in phyiscal space)
 * @param hardstops Lower bound on negative correlation  -1 would mean that it
 * stops when it hits -1
 *
 * @return Vector of motion parameters. Elements are (C = center, R = rotation
 * in radians, S = shift in index units): [CX, CY, CZ, RX, RY, RZ, SX, SY, SZ]
 */
vector<vector<double>> computeMotion(ptr<const MRImage> fmri, int reftime,
		const vector<double>& sigmas, double minstep, double maxstep, 
		int histsize, double beta, int padsize)
{
	assert(fmri->ndim());
    using namespace std::placeholders;
    using std::bind;

	// Initialize Variables
	if(padsize < 0)
		padsize = 0;

	Vector3DConstIter<double> iit(fmri); 
	vector<vector<double>> motion;
	double thresh = otsuThresh(fmri);
	
	// extract reference volumes and pre-smooth
	vector<size_t> vsize(fmri->dim(), fmri->dim()+fmri->ndim());
	for(size_t dd=0; dd<vsize.size(); dd++) {
		vsize[dd] += padsize;
	}
	vsize[3] = 0;

	vector<pair<int64_t,int64_t>> roi(fmri->ndim()-1);
	for(size_t dd=0; dd<vsize.size()-1; dd++) {
		roi[dd].first = padsize/2;
		roi[dd].second = roi[dd].first + fmri->dim(dd)-1;
	}

	// Registration Tools, create with placeholder images
	auto vol = dPtrCast<MRImage>(fmri->createAnother(3, vsize.data(), FLOAT32));
	RigidCorrComputer comp(true);

	// Pre-Compute Fixed Smoothing
	vector<ptr<MRImage>> fixed;
	for(size_t ii=0; ii<sigmas.size(); ii++) {
		NDIter<double> fit(vol);
		fit.setROI(roi);
		for(iit.goBegin(), fit.goBegin(); !fit.eof() && !iit.eof(); 
					++iit, ++fit) {
			double v = iit[reftime];
			if(v < thresh)
				fit.set(0);
			else
				fit.set(v);
		}

		// create another padded image, then smooth
		fixed.push_back(smoothDownsample(vol, sigmas[ii]));
	}

	// create value and gradient functions
	auto vfunc = bind(&RigidCorrComputer::value, &comp, _1, _2);
	auto vgfunc = bind(&RigidCorrComputer::valueGrad, &comp, _1, _2, _3);
	auto gfunc = bind(&RigidCorrComputer::grad, &comp, _1, _2);

	// initialize optimizer
	LBFGSOpt opt(6, vfunc, gfunc, vgfunc);
	opt.stop_Its = 10000000;
	opt.stop_X = minstep;
	opt.stop_G = 0;
	opt.stop_F = 0;
	opt.opt_histsize = histsize;
	opt.opt_ls_beta = beta;
	opt.opt_ls_s = maxstep;
	opt.state_x.setZero();

	Rigid3DTrans rigid;
	for(size_t tt=0; tt<fmri->tlen(); tt++) {
		cerr << "Time " << tt << " / " << fmri->tlen() << endl;
		if(tt == reftime) {
			motion.push_back(vector<double>());
			motion.back().resize(9, 0);
		} else {

			/****************************************************************
			 * Registration
			 ***************************************************************/
			for(size_t ii=0; ii<sigmas.size(); ii++) {

				/* 
				 * Extract, threshold and Smooth Moving Volume, Set Fixed in
				 * computer to Pre-Smoothed Version
				 */
				NDIter<double> mit(vol);
				mit.setROI(roi);
				for(iit.goBegin(), mit.goBegin(); !iit.eof() && !mit.eof();
								++iit, ++mit) {
					double v = iit[tt];
					if(v < thresh)
						mit.set(0);
					else
						mit.set(v);
				}
				auto sm_moving = smoothDownsample(vol, sigmas[ii]);

				comp.setFixed(fixed[ii]);
				comp.setMoving(sm_moving);

				// grab the parameters from the previous iteration (or initialized)
				rigid.toIndexCoords(sm_moving, true);
				for(size_t ii=0; ii<3; ii++) {
					opt.state_x[ii] = radToDeg(rigid.rotation[ii]);
					opt.state_x[ii+3] = rigid.shift[ii]*sm_moving->spacing(ii);
					assert(rigid.center[ii] == (sm_moving->dim(ii)-1.)/2.);
				}

				// run the optimizer
				opt.reset_history();
				StopReason stopr = opt.optimize();
				cerr << Optimizer::explainStop(stopr) << endl;

				// set values from parameters, and convert to RAS coordinate so that no
				// matter the sampling after smoothing the values remain
				for(size_t ii=0; ii<3; ii++) {
					rigid.rotation[ii] = degToRad(opt.state_x[ii]);
					rigid.shift[ii] = opt.state_x[ii+3]/sm_moving->spacing(ii);
					rigid.center[ii] = (sm_moving->dim(ii)-1)/2.;
				}

				rigid.toRASCoords(sm_moving);

			}

			/******************************************************************
			 * Results
			 *****************************************************************/
			motion.push_back(vector<double>());
			auto& m = motion.back();
			m.resize(9, 0);
			
			// Convert to RAS
			rigid.ras_coord = false;
			for(size_t dd=0; dd<3; dd++) {
				rigid.center[dd] = (vol->dim(dd)-1)/2.;
				rigid.rotation[dd] = opt.state_x[dd]*M_PI/180;
				rigid.shift[dd] = opt.state_x[dd+3]/vol->spacing(dd);
			}

			rigid.invert();
			rigid.toRASCoords(vol);
			for(size_t dd=0; dd<3; dd++) {
				m[dd] = rigid.center[dd];
				m[dd+3] = rigid.rotation[dd];
				m[dd+6] = rigid.shift[dd];
			}

			for(size_t dd=0; dd<9; dd++) { 
				if(dd != 0) cout << ", ";
				cout << m[dd];
			}
			cout << "\n==========================================" << endl;

		}
	}

	return motion;
}

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Motions corrects a 4D Image.", ' ', 
			__version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input 4D Image.", true, "",
			"*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_out("o", "output", "Output 4D motion-corrected "
			"Image.", false, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<int> a_ref("r", "ref", "Reference timepoint, < 0 values "
			"will result in reference at the middle volume, otherwise this "
			"indicates a timepoint (starting at 0)", false, -1, "T", cmd);

	TCLAP::ValueArg<string> a_motion("m", "motion", "Ouput motion as 9 column "
			"text file. Columns are Center (x,y,z), Rotation (in radians) "
			"about axes x/y/z through the center and shift (x,y,z).", 
			false, "", "*.txt", cmd);
	TCLAP::ValueArg<string> a_inmotion("a", "apply", "Apply motion from 9 "
			"column text file. Columns are Center (x,y,z), Rotation (in "
			"radians) through the center x/y/z and shift (x,y,z). "
			"If this is set, then instead of estimating motion, the inverse "
			"motion parameters are applied to the input timeseries.",
			false, "", "*.txt", cmd);
	TCLAP::MultiArg<double> a_sigmas("s", "sigmas", "Smoothing standard "
			"deviations. These are the steps of the registration.", false, 
			"sd", cmd);
	TCLAP::ValueArg<double> a_minstep("", "minstep", "Minimum step", 
			false, 1e-3, "float", cmd);
	TCLAP::ValueArg<double> a_maxstep("", "maxstep", "Maximum step", 
			false, 1, "float", cmd);
	TCLAP::ValueArg<int> a_lbfgs_hist("", "hist", "History for L-BFGS", 
			false, 4, "int", cmd);
	TCLAP::ValueArg<double> a_beta("", "reduction", "Reduction in step size", 
			false, 0.35, "float", cmd);
	TCLAP::ValueArg<int> a_padsize("", "pad", "Number of pixels to pad during "
			"registration. Reduces information lost, and stabilizes "
			"registration (slightly).", false, 3, "pixels", cmd);
	TCLAP::SwitchArg a_invert("I", "invert", "Invert motion parameters before "
			"applying them. This mostly just for simulating motion, don't do "
			"it unless you know what you are doing.", cmd);

	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	
	// read fMRI
	ptr<MRImage> fmri = readMRImage(a_in.getValue());
	if(!fmri->floatType())
		fmri = dPtrCast<MRImage>(fmri->copyCast(FLOAT32));

	if(fmri->tlen() == 1 || fmri->ndim() != 4) {
		cerr << "Warning input has " << fmri->ndim() << 
			" dimensions  and " << fmri->tlen() << " volumes." << endl;
	}
	
	// set reference volume
	int ref = a_ref.getValue();
	if(ref < 0 || ref >= fmri->tlen())
		ref = fmri->tlen()/2;

	// construct variables to get a particular volume
	vector<vector<double>> motion;

	// set up sigmas
	vector<double> sigmas({2,1,0.5});
	if(a_sigmas.isSet()) 
		sigmas.assign(a_sigmas.begin(), a_sigmas.end());
	
	if(a_inmotion.isSet()) {
		motion = readNumericCSV(a_inmotion.getValue());

		// Check Motion Results
		if(motion.size() != fmri->tlen()) {
			cerr << "Input motion rows doesn't match input fMRI timepoints!" 
				<< endl; 
			return -1;
		}

		for(size_t ll=0; ll < motion.size(); ll++) {
			if(motion[ll].size() != 9) {
				cerr << "On line " << ll << ", width should be 9 but is " 
					<< motion[ll].size() << endl;
				return -1;
			}
		}

	} else {
		// Compute Motion
		motion = computeMotion(fmri, ref, sigmas, a_minstep.getValue(),
				a_maxstep.getValue(), a_lbfgs_hist.getValue(),
				a_beta.getValue(), a_padsize.getValue());
	}

	// Write to Motion File
	if(a_motion.isSet()) {
		ofstream ofs(a_motion.getValue());
		if(!ofs.is_open()) {
			cerr<<"Error opening "<< a_motion.getValue()<<" for writing\n";
			return -1;
		}
		for(auto& line : motion) {
			assert(line.size() == 9);
			for(size_t ii=0; ii<9; ii++) {
				if(ii != 0)
					ofs << " ";
				ofs<<setw(15)<<setprecision(10)<<line[ii];
			}
			ofs << "\n";
		}
		ofs << "\n";
	}

	/*****************************************************
	 * apply motion parameters
	 ****************************************************/
	
	// Create working Buffer, iterators
	auto vol = dPtrCast<MRImage>(fmri->createAnother(3,fmri->dim(),FLOAT64));
	Vector3DIter<double> iit(fmri);
	NDIter<double> vit(vol);
	Rigid3DTrans rigid;

	// apply each time point then copy back into fMRI
	LanczosInterp3DView<double> interp(fmri);
	for(size_t tt=0; tt<fmri->tlen(); tt++) {
		
		// extract timepoint
		for(iit.goBegin(), vit.goBegin(); !iit.eof(); ++iit, ++vit) 
			vit.set(iit[tt]);
	
		// Convert from RAS to index
		rigid.ras_coord = true;
		for(size_t dd=0; dd<3; dd++) {
			rigid.center[dd] = motion[tt][dd];
			rigid.rotation[dd] = motion[tt][dd+3];
			rigid.shift[dd] = motion[tt][dd+6];
		}
		if(!a_invert.isSet())
			rigid.invert();
		
		cerr << "vol: " << endl << *vol << endl;
		cerr << "Rigid Transform: " << tt << "\n" << rigid <<endl;
		
		Matrix3d R = rigid.rotMatrix();
		Vector3d ind;
		for(vit.goBegin(); !vit.eof(); ++vit) {
			vit.index(3, ind.array().data());
			vol->indexToPoint(3, ind.array().data(), ind.array().data());
			ind = R*(ind-rigid.center) + rigid.center + rigid.shift;
			vol->pointToIndex(3, ind.array().data(), ind.array().data());
			vit.set(interp(ind[0], ind[1], ind[2], tt));
		}

		// Copy Result Back to input image
		for(iit.goBegin(), vit.goBegin(); !iit.eof(); ++iit, ++vit) 
			iit.set(tt, *vit);
	}

	if(a_out.isSet()) 
		fmri->write(a_out.getValue());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}



