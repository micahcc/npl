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

#include <tclap/CmdLine.h>
#include <version.h>
#include <string>
#include <stdexcept>

#include "mrimage.h"
#include "nplio.h"
#include "mrimage_utils.h"
#include "ndarray_utils.h"
#include "iterators.h"
#include "accessors.h"
#include <Eigen/Eigenvalues>

using namespace npl;
using namespace std;

#define VERYDEBUG
#include "macros.h"

/**
 * @brief Computes the center of mass of the given image by average the weighted
 * position of points
 *
 * @param fixed Compute center of mass of the specified image 
 *
 * @return A vector which indicates the ND point where the center of mass
 * is located
 */
VectorXd computeCenterOfMass(ptr<const MRImage> in);

/**
 * @brief Information based registration between two 3D volumes. note
 * that the two volumes should have identical sampling and identical
 * orientation. If that is not the case, an exception will be thrown.
 *
 * \todo make it v = Ru + s, then u = INV(R)*(v - s)
 *
 * @param fixed Image which will be the target of registration. 
 * @param center The center of mass of the inmage (computed elsewhere)
 *
 * @return Matrix where each column is an axes in order of decreasing
 * eigenvalue. They are orthogonal
 */
MatrixXd computeAxes(ptr<const MRImage> in, const VectorXd& center);


int main(int argc, char** argv)
{
	try {

	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Computes the center of mass of the moving and fixed "
			"images and shifts the moving image's origin so that the center "
			"of masses line up. ONLY DO THIS ON SKULL-STRIPPED IMAGES "
			"OTHERWISE THE NECK OR OUTSIDE TISSUE COULD VASTLY IMPACT "
			"RESULTS. If the images are the same mode and subject then  you "
			"can also use -R to orient that images' (DONT DO THIS IF THE "
			"IMAGES OF DIFFERENT SUBJECTS OR MODALITIES)." ,' ', __version__ );

	TCLAP::ValueArg<string> a_fixed("f", "fixed", "Fixed image.", true, "",
			"*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_moving("m", "moving", "Moving image. ", true, 
			"", "*.nii.gz", cmd);

	TCLAP::ValueArg<string> a_out("o", "out", "Moving image with new "
			"orientation", true, "", "*.nii.gz", cmd);
	TCLAP::SwitchArg a_rotate("R", "rotation", "Attempt to rotate based on "
			"2nd moment of mass", cmd);
	TCLAP::SwitchArg a_minangle("M", "min-angle", "Minimize change in "
			"rotation matrix by finding matching eigenvectors based on their "
			"dot product, rather than the original eigenvalues", cmd);

	cmd.parse(argc, argv);

	/*************************************************************************
	 * Read Inputs
	 *************************************************************************/

	// fixed image
	cout << "Reading Inputs...";
	ptr<MRImage> origfixed = readMRImage(a_fixed.getValue());
	ptr<MRImage> origmoving = readMRImage(a_moving.getValue());
	ptr<MRImage> fixed = origfixed;
	ptr<MRImage> moving = origmoving;

	size_t ndim = std::min(fixed->ndim(), moving->ndim());
	if(ndim != fixed->ndim()) 
		fixed = dPtrCast<MRImage>(fixed->extractCast(ndim, fixed->dim()));
	if(ndim != moving->ndim()) 
		moving = dPtrCast<MRImage>(moving->extractCast(ndim, moving->dim()));
	cerr << "Done: " << endl;

	auto fixed_cent = computeCenterOfMass(fixed);
	cerr << "Fixed Center: " << fixed_cent.transpose() << endl;
	auto moving_cent = computeCenterOfMass(moving);
	cerr << "Moving Center: " << moving_cent.transpose() << endl;

	MatrixXd moving_axe, fixed_axe;
	if(a_rotate.isSet()) {
		cerr << "Fixed Axes:" << endl;
		fixed_axe = computeAxes(fixed, fixed_cent);
		for(size_t dd=0; dd<ndim; dd++)
			cerr << dd << ": " << fixed_axe.col(dd).transpose() << endl;
		cerr << endl;
		cerr << "Moving Axes:" << endl;
		moving_axe = computeAxes(moving, moving_cent);
		for(size_t dd=0; dd<ndim; dd++)
			cerr << dd << ": " << moving_axe.col(dd).transpose() << endl;
		cerr << endl;
	}

	/*************************************************************************
	 * Set Origin based on centers, point = rotation * spacing * index + origin
	 *************************************************************************/
	VectorXd neworigin = moving->getOrigin()+(fixed_cent-moving_cent);

	if(!a_rotate.isSet()) {
		origmoving->setOrigin(neworigin, false);
		origmoving->write(a_out.getValue());
		return 0;
	}

	/*************************************************************************
	 * set direction in moving image (R) so that the eigenvectors in the
	 * moving image (M) match those in the fixed image (F). 
	 * F = MR
	 * R => (M^-1F)
	 * F = M(M^-1F)
	 *************************************************************************/
	if(a_minangle.isSet()) {
		vector<int> corr(ndim);
		for(size_t ii=0; ii<ndim; ii++) {
			double maxd = 0;
			for(size_t jj=0; jj<ndim; jj++) {
				double d = moving_axe.row(ii).dot(fixed_axe.row(jj));
				if(fabs(d) > maxd) {
					corr[ii] = jj;
					maxd = fabs(d);
				}
			}
		}

		for(size_t dd=0; dd<ndim; dd++)
			cerr << dd << " -> " << corr[dd] << endl;

		cerr << "Pre-reorder:\n" << moving_axe << endl;
		MatrixXd tmp(ndim, ndim);
		for(size_t ii=0; ii<ndim; ii++) 
			tmp.row(corr[ii]) = moving_axe.row(ii);
		moving_axe = tmp;
		cerr << "Post-reorder:\n" << moving_axe << endl;
	}

	// Perform rotation
	MatrixXd newrotation = moving_axe.inverse()*fixed_axe;
	cerr << "Changing Direction:\n" << origmoving->getDirection() << endl << endl;
	origmoving->setDirection(newrotation, false);
	cerr << "To:\n" << origmoving->getDirection() << endl << endl;
	origmoving->write(a_out.getValue());

} catch (TCLAP::ArgException &e)  // catch any exceptions
{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}

/**
 * @brief Computes the center of mass of the given image by average the weighted
 * position of points
 *
 * @param fixed Compute center of mass of the specified image 
 *
 * @return A vector which indicates the ND point where the center of mass
 * is located
 */
VectorXd computeCenterOfMass(ptr<const MRImage> in)
{
	VectorXd sum(in->ndim());
	sum.setZero();
	double totalmass = 0;
	VectorXd pos(in->ndim());
	for(NDConstIter<double> it(in); !it.eof(); ++it) {
		it.index(pos.rows(), pos.array().data());
		totalmass += it.get();
		sum += it.get()*pos;
	}
	sum /= totalmass;

	in->indexToPoint(sum.rows(), sum.array().data(), sum.array().data());
	return sum;
}

/**
 * @brief Information based registration between two 3D volumes. note
 * that the two volumes should have identical sampling and identical
 * orientation. If that is not the case, an exception will be thrown.
 *
 * \todo make it v = Ru + s, then u = INV(R)*(v - s)
 *
 * @param fixed Image which will be the target of registration. 
 * @param center The center of mass of the inmage (computed elsewhere)
 *
 * @return Matrix where each column is an axes in order of decreasing
 * eigenvalue. They are orthogonal
 */
MatrixXd computeAxes(ptr<const MRImage> in, const VectorXd& center)
{
	size_t ndim = in->ndim();
	MatrixXd cov(in->ndim(), in->ndim());
	cov.setZero();
	double totalmass = 0;
	VectorXd pos(in->ndim());
	for(NDConstIter<double> it(in); !it.eof(); ++it) {
		it.index(pos.rows(), pos.array().data());
		totalmass += it.get();
		cov += (pos-center)*(pos-center).transpose()*it.get();
	}
	cov /= totalmass;

	Eigen::EigenSolver<MatrixXd> solver(cov, true);
	for(size_t d1=0; d1<ndim; d1++) {
		for(size_t d2=0; d2<ndim; d2++) {
			cov(d1, d2) = solver.eigenvectors()(d1,d2).real();
		}
	}
	return cov;
}
