/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file gettransform.cpp creates a transform from two image's orientations.
 * The transform is the rigid transform that would move from the moving
 * image space to the fixed image space.
 *
 *****************************************************************************/

#include <unordered_map>
#include <tclap/CmdLine.h>
#include <version.h>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <fstream>

#include "mrimage.h"
#include "nplio.h"
#include "mrimage_utils.h"
#include "ndarray_utils.h"
#include "kdtree.h"
#include "iterators.h"
#include "accessors.h"
#include "registration.h"
#include "lbfgs.h"

using namespace npl;
using namespace std;

#define VERYDEBUG
#include "macros.h"

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Creates a transform from two image's orientations. "
			"The transform is the rigid transform that would move from the "
			"moving image space to the fixed image space. Note that spacing "
			"cannot be taken into account and any flipping will screw this up, "
			"so the images should be identical. ", ' ', __version__ );

	TCLAP::ValueArg<string> a_fixed("f", "fixed", "Fixed image.", true, "",
			"*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_moving("m", "moving", "Moving image. ", true, 
			"", "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_moved("M", "moved", "Moving resampled into "
			"fixed space with identical transform. This is mostly for checking "
			" that the transform is correct.", false, "", "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_transform("t", "transform", "File to write "
			"transform parameters to. This will be a text file with 9 numbers "
			"indicating the center point, ", true, "", "*.rtm", cmd);

	cmd.parse(argc, argv);

	/*************************************************************************
	 * Read Inputs
	 *************************************************************************/

	// fixed image
	cout << "Reading Inputs...";
	ptr<MRImage> moving = readMRImage(a_moving.getValue());
	ptr<MRImage> fixed = readMRImage(a_fixed.getValue());
	cout << "Done" << endl << "Computing rotation..." << endl;
	Rigid3DTrans rigid;
	Matrix3d rot = fixed->getDirection().block<3,3>(0,0)*
		moving->getDirection().block<3,3>(0,0).inverse();
	rigid.setRotation(rot);
	rigid.shift = (fixed->getOrigin().head<3>() - rot*moving->getOrigin().head<3>());
	cout << "Done" << endl;
	cerr << rigid << endl;
	
	cout << "Writing output...";
	if(a_transform.isSet()) {
		ofstream ofs(a_transform.getValue().c_str());
		if(!ofs.is_open()) {
			cerr<<"Error opening "<< a_transform.getValue()<<" for writing\n";
			return -1;
		}
		for(size_t ii=0; ii<3; ii++) {
			if(ii != 0) ofs << " ";
			ofs << setw(15) << setprecision(10) << rigid.center[ii];
		}

		for(size_t ii=0; ii<3; ii++) {
			if(ii != 0) ofs << " ";
			ofs << setw(15) << setprecision(10) << rigid.rotation[ii];
		}

		for(size_t ii=0; ii<3; ii++) {
			if(ii != 0) ofs << " ";
			ofs << setw(15) << setprecision(10) << rigid.shift[ii];
		}
	}

	// Apply Rigid Transform
	if(a_moved.isSet()) {
		rigid.toIndexCoords(moving, true);
		rotateImageShearKern(moving, rigid.rotation[0], 
				rigid.rotation[1], rigid.rotation[2]);
		for(size_t dd=0; dd<3; dd++) 
			shiftImageKern(moving, dd, rigid.shift[dd]);
		moving->write(a_moved.getValue());
	}
	

} catch (TCLAP::ArgException &e)  // catch any exceptions
{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}


