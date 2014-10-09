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
#include "nplio.h"
#include "mrimage.h"
#include "iterators.h"
#include "accessors.h"
#include "ndarray_utils.h"
#include "registration.h"
#include "version.h"
#include "macros.h"

using namespace std;
using namespace npl;

std::ostream_iterator<double> doubleoit(std::cout, ", ");

int main(int argc, char** argv)
{
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
			"text file. Columns are Center (x,y,z), Shift in (x,y,z), and "
			"Rotation about (x,y,z).", false, "", "*.txt", cmd);
	TCLAP::ValueArg<string> a_inmotion("M", "inmotion", "Input motion as 9 "
			"column text file. Columns are Center (x,y,z), Shift in (x,y,z), "
			"and Rotation about (x,y,z). If this is set, then instead of "
			"estimating motion, the inverse motion parameters are just "
			"applied.", false, "", "*.txt", cmd);
	TCLAP::MultiArg<double> a_sigmas("s", "sigmas", "Smoothing standard "
			"deviations. These are the steps of the registration.", false, 
			"sd", cmd);

	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	
	// read fMRI
	ptr<MRImage> fmri = readMRImage(a_in.getValue());
	if(fmri->tlen() == 1 || fmri->ndim() != 4) {
		cerr << "Expect a 4D Image, but input had " << fmri->ndim() << 
			" dimensions  and " << fmri->tlen() << " volumes." << endl;
		return -1;
	}
	
	// set reference volume
	int ref = a_ref.getValue();
	if(ref < 0 || ref >= fmri->tlen())
		ref = fmri->tlen()/2;

	// construct variables to get a particular volume
	vector<vector<double>> motion;
	vector<int64_t> index(4, 0);
	vector<size_t> vsize(fmri->dim(), fmri->dim()+fmri->ndim());
	vsize[3] = 0;
	Rigid3DTrans rigid;

	// extract reference volume
	auto refvol = dPtrCast<MRImage>(fmri->extractCast(index.size(),
				index.data(), vsize.data(), FLOAT32));
	auto vol = dPtrCast<MRImage>(refvol->createAnother());
	Vector3DIter<double> iit(fmri);
	Vector3DIter<double> vit(vol);

	// set up sigmas
	vector<double> sigmas({3, 2, 1, 0});
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
		
		for(size_t tt=0; tt<fmri->tlen(); tt++) {
			if(tt == ref) {
				motion.push_back(vector<double>());
				motion.back().resize(9, 0);
			} else {
				// extract timepoint
				for(iit.goBegin(), vit.goBegin(); !iit.eof(); ++iit, ++vit) 
					vit.set(iit[tt]);

				// perform registration
				rigid = corReg3D(refvol, vol, sigmas);
				motion.push_back(vector<double>());
				auto& m = motion.back();
				m.resize(9, 0);

				copy(rigid.center.data(), rigid.center.data()+3, &m[0]);
				copy(rigid.shift.data(), rigid.shift.data()+3, &m[3]);
				copy(rigid.rotation.data(), rigid.rotation.data()+3, &m[6]);

#ifndef NDEBUG
				copy(m.begin(), m.end(), doubleoit);
#endif //NDEBUG
			}
		}
	}

	// Write to Motion File
	if(a_motion.isSet()) {
		ofstream ofs(a_motion.getValue());
		if(!ofs.is_open()) {
			cerr<<"Error opening "<< a_motion.getValue()<<" for writing\n";
			return -1;
		}
		for(auto& line : motion) {
			assert(line.size() == 0);
			for(size_t ii=0; ii<9; ii++) {
				if(ii != 0)
					ofs << " ";
				ofs<<setw(15)<<setprecision(10)<<line[ii];
			}
			ofs << "\n";
		}
		ofs << "\n";
	}

	// apply motion parameters
	NDIter<double> fit(fmri);
	for(size_t tt=0; tt<fmri->tlen(); tt++) {
		// extract timepoint
		for(iit.goBegin(), vit.goBegin(); !iit.eof(); ++iit, ++vit) 
			vit.set(iit[tt]);
		
		// convert motion parameters to centered rotation + translation
		copy(motion[tt].begin(), motion[tt].begin()+3, rigid.center.data());
		copy(motion[tt].begin()+3, motion[tt].begin()+6, rigid.shift.data());
		copy(motion[tt].begin()+6, motion[tt].end(), rigid.rotation.data());
		rigid.toIndexCoords(vol, true);
		cerr << "Rigid Transform: " << tt << "\n" << rigid <<endl;
		
		// Apply Rigid Transform
		rotateImageShearFFT(vol, rigid.rotation[0], rigid.rotation[1], 
				rigid.rotation[2]);
		vol->write("rotated"+to_string(tt)+".nii.gz");
		for(size_t dd=0; dd<3; dd++) 
			shiftImageFFT(vol, dd, rigid.shift[dd]);
		vol->write("shifted_rotated"+to_string(tt)+".nii.gz");

		// Copy Result Back to input image
		for(iit.goBegin(), vit.goBegin(); !iit.eof(); ++iit, ++vit) 
			iit.set(tt, vit[0]);
	}

	if(a_out.isSet()) 
		fmri->write(a_out.getValue());

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}



