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
 * @file skullstrip.cpp Experimental skull stripping algorithm based on point
 * cloud density
 *
 *****************************************************************************/

#include <string>

#include <tclap/CmdLine.h>
#include "nplio.h"
#include "mrimage.h"
#include "ndarray_utils.h"
#include "version.h"

using namespace std;
using namespace npl;

int main(int argc, char** argv)
{
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Removes the skull from a brain image.",
            ' ', __version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input image.",
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_out("o", "out", "Output image.",
			true, "", "*.nii.gz", cmd);

	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	std::shared_ptr<MRImage> inimg(readMRImage(a_in.getValue()));
	if(inimg->ndim() != 3) {
		cerr << "Expected input to be 3D Image!" << endl;
		return -1;
	}
	
    /*****************************
     * edge detection
     ****************************/
    auto deriv = sobelEdge(inimg);
    deriv->write("sobel.nii.gz");
    auto absderiv = collapseSum(deriv, 3);
    absderiv->write("sobel_abs.nii.gz");
//
//    /*****************************
//     * create point list from edges (based on top quartile of edges in each
//     * window) then extract points that meet local shape criteria 
//     ****************************/
//	MatrixXd points = genPoints(absderiv);
//    points = shapeFilter(points);
//    auto mask = pointsToMask(inimg, points);
//
    /*******************************************************
     * Propagate selected edges perpendicular to the edge
     ******************************************************/

    /***********************************
     * Watershed
     ***********************************/

    /************************************
     * Select Brain Watershed
     ***********************************/

    /************************************
     * Mask and Write 
     ***********************************/

    } catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}

