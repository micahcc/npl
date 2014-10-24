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


/**
 * @brief Information based registration between two 3D volumes. note
 * that the two volumes should have identical sampling and identical
 * orientation. If that is not the case, an exception will be thrown.
 *
 * \todo make it v = Ru + s, then u = INV(R)*(v - s)
 *
 * @param fixed     Image which will be the target of registration. 
 * @param moving    Image which will be rotated then shifted to match fixed.
 * @param sigmas	Standard deviation of smoothing at each level
 * @param metric 	Type of information based metric to use
 *
 * @return rigid transform
 */
Rigid3DTrans inforeg(ptr<const MRImage> fixed, ptr<const MRImage> moving, 
		const std::vector<double>& sigmas, int bins, int parzrad, string metric);

/**
 * @brief Performs correlation based registration between two 3D volumes. note
 * that the two volumes should have identical sampling and identical
 * orientation. If that is not the case, an exception will be thrown.
 *
 * \todo make it v = Ru + s, then u = INV(R)*(v - s)
 *
 * @param fixed     Image which will be the target of registration. 
 * @param moving    Image which will be rotated then shifted to match fixed.
 * @param sigmas	Standard deviation of smoothing at each level
 *
 * @return rigid transform
 */
Rigid3DTrans correg(ptr<const MRImage> fixed, ptr<const MRImage> moving, 
		const std::vector<double>& sigmas);

int main(int argc, char** argv)
{
try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Computes a rigid transform to match a moving image to "
			"a fixed one. For a 4D input, the 0'th volume will be used", ' ',
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
	
	if(a_apply.isSet()) {
		ifstream ifs(a_apply.getValue().c_str());
		if(!ifs.is_open()) {
			cerr<<"Error opening "<< a_apply.getValue()<<" for reading\n";
			return -1;
		}
		for(size_t ii=0; ii<3; ii++) 
			ifs >> rigid.center[ii];

		for(size_t ii=0; ii<3; ii++) 
			ifs >> rigid.rotation[ii];

		for(size_t ii=0; ii<3; ii++) 
			ifs >> rigid.shift[ii];

		rigid.ras_coord = true;
		cout << "Read Transform: " << endl << rigid << endl;
	} else if(a_fixed.isSet()) {
		ptr<MRImage> fixed = readMRImage(a_fixed.getValue());
		fixed = dPtrCast<MRImage>(fixed->copyCast(min(fixed->ndim(),3UL), 
					fixed->dim(), FLOAT32));

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
			rigid = correg(fixed, moving, sigmas);
		} else {
			cout << "Done\nRigidly Registering with " << a_metric.getValue() 
				<< "..." << endl;
			rigid = informationReg3D(fixed, moving, sigmas, a_bins.getValue(),
					a_parzen.getValue(), a_metric.getValue()); 
		}
		cout << "Finished\n.";
	} else {
		cerr << "Either --fixed or --apply must be set, otherwise we can't "
			"create a transform!" << endl;
		return -1;
	}

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

	if(a_out.isSet()) {
		if(a_resample.isSet()) {
			// Apply Rigid Transform
			rigid.toIndexCoords(in_moving, true);
			rotateImageShearKern(in_moving, rigid.rotation[0], 
					rigid.rotation[1], rigid.rotation[2]);
			for(size_t dd=0; dd<3; dd++) 
				shiftImageKern(in_moving, dd, rigid.shift[dd]);
		} else {
			VectorXd origin = rigid.rotMatrix()*
						in_moving->getOrigin().head<3>() + rigid.shift;
			MatrixXd dir = rigid.rotMatrix()*
						in_moving->getDirection().block<3,3>(0,0);
			in_moving->setOrigin(origin, false);
			in_moving->setDirection(dir, false);
		}
		in_moving->write(a_out.getValue());
	}
	cout << "Done" << endl;

} catch (TCLAP::ArgException &e)  // catch any exceptions
{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}

Rigid3DTrans correg(ptr<const MRImage> fixed, ptr<const MRImage> moving, 
		const vector<double>& sigmas)
{
	using namespace std::placeholders;
	using std::bind;
	Rigid3DTrans rigid;
	rigid.center.setZero();
	rigid.rotation.setZero();
	rigid.shift.setZero();
	rigid.ras_coord = true;
	double met = 0;
	for(size_t ii=0; ii<sigmas.size(); ii++) {
		// smooth and downsample input images
		auto sm_fixed = smoothDownsample(fixed, sigmas[ii]);
		auto sm_moving = smoothDownsample(moving, sigmas[ii]);

		RigidCorrComputer comp(true);
		comp.setFixed(sm_fixed);
		comp.setMoving(sm_moving);

		// create value and gradient functions
		auto vfunc = bind(&RigidCorrComputer::value, &comp, _1, _2);
		auto vgfunc = bind(&RigidCorrComputer::valueGrad, &comp, _1, _2, _3);
		auto gfunc = bind(&RigidCorrComputer::grad, &comp, _1, _2);

		// initialize optimizer
		LBFGSOpt opt(6, vfunc, gfunc, vgfunc);
		opt.stop_Its = 10000;
		opt.stop_X = 0.00001;
		opt.stop_G = 0;
		opt.stop_F = 0;

		// grab the parameters from the previous iteration (or initialized)
		rigid.toIndexCoords(sm_moving, true);
		for(size_t ii=0; ii<3; ii++) {
			opt.state_x[ii] = radToDeg(rigid.rotation[ii]);
			opt.state_x[ii+3] = rigid.shift[ii]*sm_moving->spacing(ii);
		}

		// run the optimizer
		StopReason stopr = opt.optimize();
		cout << "Sigma: " << sigmas[ii] << endl;
		vfunc(opt.state_x, met);
		cout <<"Start Metric: " << met << endl;
		cerr << Optimizer::explainStop(stopr) << endl;
		vfunc(opt.state_x, met);
		cout << "Stop Metric: " << met << endl;

		// set values from parameters, and convert to RAS coordinate so that no
		// matter the sampling after smoothing the values remain
		for(size_t ii=0; ii<3; ii++) {
			rigid.rotation[ii] = degToRad(opt.state_x[ii]);
			rigid.shift[ii] = opt.state_x[ii+3]/sm_moving->spacing(ii);
			rigid.center[ii] = (sm_moving->dim(ii)-1)/2.;
		}

		rigid.toRASCoords(sm_moving);
	}

	return rigid;
};

//Rigid3DTrans inforeg(ptr<const MRImage> fixed, ptr<const MRImage> moving, 
//		const vector<double>& sigmas, int bins, int rad, string metric)
//{
//	using namespace std::placeholders;
//	using std::bind;
//	Rigid3DTrans rigid;
//	rigid.center.setZero();
//	rigid.rotation.setZero();
//	rigid.shift.setZero();
//
//	double met = 0;
//	for(size_t ii=0; ii<sigmas.size(); ii++) {
//		cout << "Sigma: " << sigmas[ii] << endl;
//		// smooth and downsample input images
//		auto sm_fixed = smoothDownsample(fixed, sigmas[ii]);
//		auto sm_moving = smoothDownsample(moving, sigmas[ii]);
//
//		RigidInformationComputer comp(true);
//		comp.setBins(bins, rad);
//		comp.setFixed(fixed);
//		comp.setMoving(fixed);
//
//		if(metric == "MI") 
//			comp.m_metric = METRIC_MI;
//		else if(metric == "NMI") 
//			comp.m_metric = METRIC_NMI;
//		else if(metric == "VI") {
//			comp.m_metric = METRIC_VI;
//		}
//
//		// create value and gradient functions
//		auto vfunc = bind(&RigidInformationComputer::value, &comp, _1, _2);
//		auto vgfunc = bind(&RigidInformationComputer::valueGrad, &comp, _1, _2, _3);
//		auto gfunc = bind(&RigidInformationComputer::grad, &comp, _1, _2);
//
//		// initialize optimizer
//		LBFGSOpt opt(6, vfunc, gfunc, vgfunc);
//		opt.stop_Its = 10000;
//		opt.stop_X = 0.00001;
//		opt.stop_G = 0;
//		opt.stop_F = 0;
//
//		// grab the parameters from the previous iteration (or initialized)
//		rigid.toIndexCoords(sm_moving, true);
//		for(size_t ii=0; ii<3; ii++) {
//			opt.state_x[ii] = radToDeg(rigid.rotation[ii]);
//			opt.state_x[ii+3] = rigid.shift[ii]*sm_moving->spacing(ii);
//		}
//
//		// run the optimizer
//		cout << "Sigma: " << sigmas[ii] << endl;
//		vfunc(opt.state_x, met);
//		cout <<"Start Metric: " << met << endl;
//		StopReason stopr = opt.optimize();
//		cerr << Optimizer::explainStop(stopr) << endl;
//		vfunc(opt.state_x, met);
//		cout <<"Stop Metric: " << met << endl;
//
//		// set values from parameters, and convert to RAS coordinate so that no
//		// matter the sampling after smoothing the values remain
//		for(size_t ii=0; ii<3; ii++) {
//			rigid.rotation[ii] = degToRad(opt.state_x[ii]);
//			rigid.shift[ii] = opt.state_x[ii+3]/sm_moving->spacing(ii);
//			rigid.center[ii] = (sm_moving->dim(ii)-1)/2.;
//		}
//
//		rigid.toRASCoords(sm_moving);
//	}
//
//	return rigid;
//};

