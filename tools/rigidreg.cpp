/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file rigidreg.cpp Basic rigid registration tool. Supports correlation and
 * information (multimodal) metrics.
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

	TCLAP::CmdLine cmd("Computes a rigid transform to match a moving image to "
			"a fixed one. For a 4D input, the 0'th volume will be used. "
			"The outputs of this are either a transformed volume (-o) or a "
			"transform (-t). You can also apply an existing transform (-a). "
			"Two types of transform are recognized: FSL 4x4 *.mat files and "
			"my *.rtm (9 parameter: center rotation shift) format.", ' ',
			__version__ );

	TCLAP::ValueArg<string> a_fixed("f", "fixed", "Fixed image.", false, "",
			"*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_apply("a", "apply", "Apply transform. Instead "
			"of performing rigid registration apply the provided transform",
			false, "", "*.rtm", cmd);
	TCLAP::ValueArg<string> a_moving("m", "moving", "Moving image. ", true,
			"", "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_metric("M", "metric", "Metric to use. "
			"(NMI, MI, VI, COR)", false, "", "metric", cmd);
	TCLAP::ValueArg<string> a_out("o", "out", "Registered version of "
			"moving image. ", false, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_transform("t", "transform", "File to write "
			"transform parameters to. This will be a text file with 9 numbers "
			"indicating the center point, ", false, "", "*.rtm", cmd);
	TCLAP::SwitchArg a_resample("R", "resample", "Apply rigid  "
			"transform in pixel space instead of the default, which is to "
			"just modify the image's orientation", cmd);

	TCLAP::MultiArg<double> a_sigmas("s", "sigmas", "Smoothing standard "
			"deviations. These are the steps of the registration.", false,
			"sd", cmd);

	TCLAP::ValueArg<int> a_bins("b", "bins", "Bins to use in information "
			"metric to estimate the joint distribution. This is the "
			"the number of bins in the marginal distribution.", false, 200,
			"n", cmd);
	TCLAP::ValueArg<int> a_parzen("r", "radius", "Radius in parzen window "
			"for bins", false, 5, "n", cmd);


	cmd.parse(argc, argv);

	/*************************************************************************
	 * Read Inputs
	 *************************************************************************/

	// fixed image
	cout << "Reading Inputs...";

	// moving image, resample to fixed space
	ptr<MRImage> in_moving = readMRImage(a_moving.getValue());
	Rigid3DTrans rigid;
	cout << "Done" << endl;

	if(a_apply.isSet() && !a_fixed.isSet()) {
		cout << "Applying " << a_apply.getValue() << endl;
		ifstream ifs(a_apply.getValue().c_str());
		if(!ifs.is_open()) {
			cerr<<"Error opening "<< a_apply.getValue()<<" for reading\n";
			return -1;
		}

		// Read entire file
		list<double> vals;
		double v;
		ifs >> v;
		while(!ifs.fail()) {
			vals.push_back(v);
			ifs >> v;
		}

		if(vals.size() == 16) {
			return -1;
			cout << "FSL Matrix" << endl;
			Eigen::Matrix4d aff;
			auto it = vals.begin();
			for(size_t ii=0; ii<4; ii++) {
				for(size_t jj=0; jj<4; jj++, ++it) {
					aff(ii,jj) = *it;
				}
			}
			cerr << aff << endl;
			rigid.ras_coord = false;

			Eigen::JacobiSVD<MatrixXd> svd(aff.block<3,3>(0,0),
					Eigen::ComputeThinU|Eigen::ComputeThinV);
			rigid.setRotation(svd.matrixU()*svd.matrixV().transpose());
			rigid.shift = aff.block<3,1>(0, 3);
			cerr << rigid << endl;
			rigid.toRASCoords(in_moving);
		} else if(vals.size() == 9) {
			cout << "NPL Rigid" << endl;
			rigid.ras_coord = true;
			auto it = vals.begin();
			for(size_t ii=0; ii<3; it++, ii++)
				rigid.center[ii] = *it;

			for(size_t ii=0; ii<3; ++it, ii++)
				rigid.rotation[ii] = *it;

			for(size_t ii=0; ii<3; ++it, ii++)
				rigid.shift[ii] = *it;
		} else {
			cerr << "Unknown format of transform! " << a_apply.getValue() << endl;
			for(auto it=vals.begin(); it != vals.end(); ++it)
				cerr << *it << endl;
			return -1;
		}

		cout << "Read Transform: " << endl << rigid << endl;
	} else if(a_fixed.isSet() && !a_apply.isSet()) {
		ptr<MRImage> fixed = readMRImage(a_fixed.getValue());
		size_t ndim = 3;
		fixed = dPtrCast<MRImage>(fixed->copyCast(ndim, fixed->dim(), FLOAT32));

		// Downsample moving image
		auto moving = dPtrCast<MRImage>(fixed->createAnother());

		cout << "Done\nMoving Image to Fixed Space using Lanczos interp...";
		vector<int64_t> ind(fixed->ndim());
		vector<double> point(fixed->ndim());
		LanczosInterpNDView<double> interp(in_moving);
		interp.m_ras = true;
		for(NDIter<double> mit(moving); !mit.eof(); ++mit) {
			// get point of mit
			mit.index(ind.size(), ind.data());
			moving->indexToPoint(ind.size(), ind.data(), point.data());
			mit.set(interp.get(point));
		}

		// set up sigmas
		vector<double> sigmas({3,1.5,0});
		if(a_sigmas.isSet())
			sigmas.assign(a_sigmas.begin(), a_sigmas.end());

		/*
		 * Perform Registration
		 */
		if(a_metric.getValue() == "COR") {
			cout << "Done\nRigidly Registering with correlation..." << endl;
			rigid = corReg3D(fixed, moving, sigmas);
		} else {
			cout << "Done\nRigidly Registering with " << a_metric.getValue()
				<< "..." << endl;
			rigid = informationReg3D(fixed, moving, sigmas, a_bins.getValue(),
					a_parzen.getValue(), a_metric.getValue());
		}
		cout << "Finished\n.";
		rigid.invert();
	} else {
		cerr << "Either --fixed or --apply must be set but not both!" << endl;
		return -1;
	}

	cout << "Writing output..."<<endl;
	if(a_transform.isSet()) {
		cout<<"  Transform"<<endl;
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
			ofs <<" "<<setw(15) << setprecision(10) << rigid.rotation[ii];
		}

		for(size_t ii=0; ii<3; ii++) {
			ofs << " "<<setw(15) << setprecision(10) << rigid.shift[ii];
		}
	}

	if(a_out.isSet()) {
		cout<<"  Image"<<endl;
		if(a_resample.isSet()) {
			// Apply Rigid Transform
			rigid.toIndexCoords(in_moving, true);
			rotateImageShearKern(in_moving, rigid.rotation[0],
					rigid.rotation[1], rigid.rotation[2]);
			for(size_t dd=0; dd<3; dd++)
				shiftImageKern(in_moving, dd, rigid.shift[dd]);
		} else {
			VectorXd origin = rigid.rotMatrix()*
						(in_moving->getOrigin().head<3>() - rigid.center) +
						rigid.center + rigid.shift;
			MatrixXd dir = rigid.rotMatrix()*
						in_moving->getDirection().block<3,3>(0,0);
			cerr<<"Original Origin:\n"<<in_moving->getOrigin().transpose()<<endl;
			cerr<<"Original Direction:\n"<<in_moving->getDirection()<<endl;
			cerr<<"New Origin:\n"<<origin.transpose()<<endl;
			cerr<<"New Direction:\n"<<dir<<endl;
			in_moving->setOrigin(origin, false);
			in_moving->setDirection(dir, false);
		}
		in_moving->write(a_out.getValue());
	}
	cout << "Done" << endl;

} catch (TCLAP::ArgException &e)  // catch any exceptions
{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}

