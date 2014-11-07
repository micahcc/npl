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
 * @file deformJacobianMap.cpp
 *
 *****************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include "basic_functions.h"
#include "mrimage.h"
#include "mrimage_utils.h"
#include "nplio.h"
#include "kdtree.h"
#include "iterators.h"
#include "accessors.h"

using std::string;
using namespace npl;
using std::shared_ptr;

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Computes a Jacobian determinant image from a deform",
			' ', __version__ );

	TCLAP::ValueArg<string> a_indef("i", "input", "Input deformation.",
			true, "", "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_out("o", "output", "Output image in "
			"deform space.", false, "", "*.nii.gz", cmd);

	cmd.parse(argc, argv);
	std::shared_ptr<MRImage> deform;
	std::shared_ptr<MRImage> atlas;
	std::shared_ptr<MRImage> mask;

	/**********
	 * Input
	 *********/
	deform = readMRImage(a_indef.getValue());
	if(deform->ndim() < 4 || deform->tlen() != 3) {
		cerr << "Expected dform to be 4D/5D Image, with 3 volumes!" << endl;
		return -1;
	}

	Vector3DView<double> dview(deform);
	int64_t index[3];
	int64_t tmpindex[3];
	Matrix3d jac;
	auto jacobian = dynamic_pointer_cast<MRImage>(deform->copyCast(3, deform->dim()));
	auto dfxdx = dynamic_pointer_cast<MRImage>(deform->copyCast(3, deform->dim()));
	auto dfxdy = dynamic_pointer_cast<MRImage>(deform->copyCast(3, deform->dim()));
	auto dfxdz = dynamic_pointer_cast<MRImage>(deform->copyCast(3, deform->dim()));
	auto dfydx = dynamic_pointer_cast<MRImage>(deform->copyCast(3, deform->dim()));
	auto dfydy = dynamic_pointer_cast<MRImage>(deform->copyCast(3, deform->dim()));
	auto dfydz = dynamic_pointer_cast<MRImage>(deform->copyCast(3, deform->dim()));
	auto dfzdx = dynamic_pointer_cast<MRImage>(deform->copyCast(3, deform->dim()));
	auto dfzdy = dynamic_pointer_cast<MRImage>(deform->copyCast(3, deform->dim()));
	auto dfzdz = dynamic_pointer_cast<MRImage>(deform->copyCast(3, deform->dim()));
	Vector3DView<double> view_dfxdx(dfxdx);
	Vector3DView<double> view_dfxdy(dfxdy);
	Vector3DView<double> view_dfxdz(dfxdz);
	Vector3DView<double> view_dfydx(dfydx);
	Vector3DView<double> view_dfydy(dfydy);
	Vector3DView<double> view_dfydz(dfydz);
	Vector3DView<double> view_dfzdx(dfzdx);
	Vector3DView<double> view_dfzdy(dfzdy);
	Vector3DView<double> view_dfzdz(dfzdz);

	for(OrderIter<double> it(jacobian); !it.eof(); ++it) {
		it.index(3, index);
		
		for(size_t dd=0; dd<3; dd++)
			tmpindex[dd] = index[dd];

		// dF1/dx dF1/dy dF1/dz
		// dF2/dx dF2/dy dF2/dz
		// dF3/dx dF3/dy dF3/dz
		for(size_t d1 = 0; d1 < 3; d1++) {

			for(size_t d2 = 0; d2 < 3; d2++) {
				
				// after
				tmpindex[d2] = clamp<int64_t>(0, deform->dim(d2)-1, index[d2]+1);
				jac(d1, d2) = .5*dview(tmpindex[0], tmpindex[1], tmpindex[2], d1);

				// before
				tmpindex[d2] = clamp<int64_t>(0, deform->dim(d2)-1, index[d2]-1);
				jac(d1, d2) -= .5*dview(tmpindex[0], tmpindex[1], tmpindex[2], d1);
			}
		}
		view_dfxdx.set(index[0], index[1], index[2], 0, jac(0,0));
		view_dfxdy.set(index[0], index[1], index[2], 0, jac(0,1));
		view_dfxdz.set(index[0], index[1], index[2], 0, jac(0,2));
		view_dfydx.set(index[0], index[1], index[2], 0, jac(1,0));
		view_dfydy.set(index[0], index[1], index[2], 0, jac(1,1));
		view_dfydz.set(index[0], index[1], index[2], 0, jac(1,2));
		view_dfzdx.set(index[0], index[1], index[2], 0, jac(2,0));
		view_dfzdy.set(index[0], index[1], index[2], 0, jac(2,1));
		view_dfzdz.set(index[0], index[1], index[2], 0, jac(2,2));

		it.set(jac.determinant());
	}
	
	dfxdx->write("dfxdx.nii.gz");
	dfxdy->write("dfxdy.nii.gz");
	dfxdz->write("dfxdz.nii.gz");
	dfydx->write("dfydx.nii.gz");
	dfydy->write("dfydy.nii.gz");
	dfydz->write("dfydz.nii.gz");
	dfzdx->write("dfzdx.nii.gz");
	dfzdy->write("dfzdy.nii.gz");
	dfzdz->write("dfzdz.nii.gz");

	// write
	jacobian->write(a_out.getValue());
	
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}


