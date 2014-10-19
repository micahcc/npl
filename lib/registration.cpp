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
 * @file registration.h Tools for registering 2D or 3D images.
 *
 *****************************************************************************/

#include "registration.h"
#include "lbfgs.h"
#include "mrimage_utils.h"
#include "ndarray_utils.h"
#include "macros.h"

#include <memory>
#include <stdexcept>
#include <functional>
#include <algorithm>

#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::Vector3d;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::AngleAxisd;

using std::cerr;
using std::endl;

#ifdef VERYDEBUG
#define DEBUGWRITE(FOO) FOO
#else
#define DEBUGWRITE(FOO)
#endif

namespace npl {

/******************************************************************
 * Registration Functions
 *****************************************************************/
/**
 * @brief Performs motion correction on a set of volumes. Each 3D volume is
 * extracted and linearly registered with the ref volume.
 *
 * @param input 3+D volume (set of 3D volumes, all higher dimensions are
 *              treated equally as separate volumes).
 * @param ref   Reference t, all images will be registered to the specified
 *              timepoint
 *
 * @return      Motion corrected volume.
 */
shared_ptr<MRImage> motionCorrect(shared_ptr<const MRImage> input, size_t ref)
{

	(void)input;
	(void)ref;
	return NULL;
};

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
 * @return Output rigid transform
 */
Rigid3DTrans corReg3D(shared_ptr<const MRImage> fixed,
		shared_ptr<const MRImage> moving, const std::vector<double>& sigmas)
{
	using namespace std::placeholders;
	using std::bind;

	// make sure the input image has matching properties
	if(!fixed->matchingOrient(moving, true, true))
		throw std::invalid_argument("Input images have mismatching pixels "
				"in\n" + __FUNCTION_STR__);

	// create value and gradient functions from RigiCorrComputer
	RigidCorrComputer comp(true);
	auto vfunc = bind(&RigidCorrComputer::value, &comp, _1, _2);
	auto vgfunc = bind(&RigidCorrComputer::valueGrad, &comp, _1, _2, _3);
	auto gfunc = bind(&RigidCorrComputer::grad, &comp, _1, _2);

	Rigid3DTrans rigid;
	for(size_t ii=0; ii<sigmas.size(); ii++) {
		// smooth and downsample input images
		auto sm_fixed = smoothDownsample(fixed, sigmas[ii]);
		auto sm_moving = smoothDownsample(moving, sigmas[ii]);
		DEBUGWRITE(sm_fixed->write("smooth_fixed_"+to_string(ii)+".nii.gz"));
		DEBUGWRITE(sm_moving->write("smooth_moving_"+to_string(ii)+".nii.gz"));
		comp.setFixed(sm_fixed);
		comp.setMoving(sm_moving);

		// initialize optimizer
		LBFGSOpt opt(6, vfunc, gfunc, vgfunc);
		opt.stop_Its = 10000;
		opt.stop_X = 0.00001;
		opt.stop_G = 0;
		opt.stop_F = 0;

		// grab the parameters from the previous iteration (or initialized)
		rigid.toIndexCoords(sm_moving, true);
		for(size_t ii=0; ii<3; ii++) {
			opt.state_x[ii] = rigid.rotation[ii]*180/M_PI;
			opt.state_x[ii+3] = rigid.shift[ii]/sm_moving->spacing(ii);
			assert(rigid.center[ii] == (sm_moving->dim(ii)-1.)/2.);
		}

		// run the optimizer
		StopReason stopr = opt.optimize();
		cerr << Optimizer::explainStop(stopr) << endl;

		// set values from parameters, and convert to RAS coordinate so that no
	// matter the sampling after smoothing the values remain
		for(size_t ii=0; ii<3; ii++) {
			rigid.rotation[ii] = opt.state_x[ii]*M_PI/180;
			rigid.shift[ii] = opt.state_x[ii+3]*sm_moving->spacing(ii);
			rigid.center[ii] = (sm_moving->dim(ii)-1)/2.;
		}

		rigid.toRASCoords(sm_moving);
	}

	return rigid;
};

/**
 * @brief Performs correlation based registration between two 3D volumes. note
 * that the two volumes should have identical sampling and identical
 * orientation. If that is not the case, an exception will be thrown.
 *
 * \todo make it v = Ru + s, then u = INV(R)*(v - s)
 *
 * @param fixed     Image which will be the target of registration.
 * @param moving    Image which will be rotated then shifted to match fixed.
 *
 * @return          4x4 Matrix, indicating rotation about the center then
 *                  shift. Rotation matrix is the first 3x3 and shift is the
 *                  4th column.
 */
Rigid3DTrans informationReg3D(shared_ptr<const MRImage> fixed,
		shared_ptr<const MRImage> moving, const std::vector<double>& sigmas,
		size_t nbins, size_t binradius)
{
	using namespace std::placeholders;
	using std::bind;

	Rigid3DTrans rigid;

	// make sure the input image has matching properties
	if(!fixed->matchingOrient(moving, true, true))
		throw std::invalid_argument("Input images have mismatching pixels "
				"in\n" + __FUNCTION_STR__);


	for(size_t ii=0; ii<sigmas.size(); ii++) {
		// smooth and downsample input images
		auto sm_fixed = smoothDownsample(fixed, sigmas[ii]);
		auto sm_moving = smoothDownsample(moving, sigmas[ii]);
		DEBUGWRITE(sm_fixed->write("smooth_fixed_"+to_string(ii)+".nii.gz"));
		DEBUGWRITE(sm_moving->write("smooth_moving_"+to_string(ii)+".nii.gz"));

		RigidInformationComputer comp(true);
		comp.setBins(nbins, binradius);
		comp.setFixed(sm_fixed);
		comp.setMoving(sm_moving);
		comp.m_metric = METRIC_MI;

		// create value and gradient functions
		auto vfunc = bind(&RigidInformationComputer::value, &comp, _1, _2);
		auto vgfunc = bind(&RigidInformationComputer::valueGrad, &comp, _1, _2, _3);
		auto gfunc = bind(&RigidInformationComputer::grad, &comp, _1, _2);

		// initialize optimizer
		LBFGSOpt opt(6, vfunc, gfunc, vgfunc);
		opt.stop_Its = 10000;
		opt.stop_X = 0.00001;
		opt.stop_G = 0;
		opt.stop_F = 0;

		cerr << "Init Rigid: " << ii << endl;
		cerr << "Rotation: " << rigid.rotation.transpose() << endl;
		cerr << "Center : " << rigid.center.transpose() << endl;
		cerr << "Shift : " << rigid.shift.transpose() << endl;

		// grab the parameters from the previous iteration (or initialized)
		rigid.toIndexCoords(sm_moving, true);
		for(size_t ii=0; ii<3; ii++) {
			opt.state_x[ii] = rigid.rotation[ii]*180/M_PI;
			opt.state_x[ii+3] = rigid.shift[ii]/sm_moving->spacing(ii);
			assert(rigid.center[ii] == (sm_moving->dim(ii)-1.)/2.);
		}

		cerr << "Init Rigid (Index Coord): " << ii << endl;
		cerr << "Rotation: " << rigid.rotation.transpose() << endl;
		cerr << "Center : " << rigid.center.transpose() << endl;
		cerr << "Shift : " << rigid.shift.transpose() << endl;

		// run the optimizer
		StopReason stopr = opt.optimize();
		cerr << Optimizer::explainStop(stopr) << endl;

		// set values from parameters, and convert to RAS coordinate so that no
	// matter the sampling after smoothing the values remain
		for(size_t ii=0; ii<3; ii++) {
			rigid.rotation[ii] = opt.state_x[ii]*M_PI/180;
			rigid.shift[ii] = opt.state_x[ii+3]*sm_moving->spacing(ii);
			rigid.center[ii] = (sm_moving->dim(ii)-1)/2.;
		}

		cerr << "Finished Rigid (Index Coord): " << ii << endl;
		cerr << "Rotation: " << rigid.rotation.transpose() << endl;
		cerr << "Center : " << rigid.center.transpose() << endl;
		cerr << "Shift : " << rigid.shift.transpose() << endl;

		rigid.toRASCoords(sm_moving);

		cerr << "Finished Rigid (RAS Coord): " << ii << endl;
		cerr << "Rotation: " << rigid.rotation.transpose() << endl;
		cerr << "Center : " << rigid.center.transpose() << endl;
		cerr << "Shift : " << rigid.shift.transpose() << endl;
	}

	return rigid;
};


/*****************************************************************************
 * Derivative Testers
 ****************************************************************************/

/**
 * @brief This function checks the validity of the derivative functions used
 * to optimize between-image corrlation.
 *
 * @param step Test step size
 * @param tol Tolerance in error between analytical and Numeric gratient
 * @param in1 Image 1
 * @param in2 Image 2
 *
 * @return 0 if success, -1 if failure
 */
int cor3DDerivTest(double step, double tol,
		shared_ptr<const MRImage> in1, shared_ptr<const MRImage> in2)
{
	using namespace std::placeholders;
	RigidCorrComputer comp(false);
	comp.setFixed(in1);
	comp.setMoving(in2);

	auto vfunc = std::bind(&RigidCorrComputer::value, &comp, _1, _2);
	auto gfunc = std::bind(&RigidCorrComputer::grad, &comp, _1, _2);

	double error = 0;
	VectorXd x = VectorXd::Ones(6);
	if(testgrad(error, x, step, tol, vfunc, gfunc) != 0) {
		return -1;
	}

	return 0;
}

/**
 * @brief This function checks the validity of the derivative functions used
 * to optimize between-image corrlation.
 *
 * @param step Test step size
 * @param tol Tolerance in error between analytical and Numeric gratient
 * @param in1 Image 1
 * @param in2 Image 2
 *
 * @return 0 if success, -1 if failure
 */
int information3DDerivTest(double step, double tol,
		shared_ptr<const MRImage> in1, shared_ptr<const MRImage> in2)
{
	using namespace std::placeholders;
	RigidInformationComputer comp(false);
	comp.setBins(128,4);
	comp.m_metric = METRIC_MI;
	comp.setFixed(in1);
	comp.setMoving(in2);

	auto vfunc = std::bind(&RigidInformationComputer::value, &comp, _1, _2);
	auto gfunc = std::bind(&RigidInformationComputer::grad, &comp, _1, _2);

	double error = 0;
	VectorXd x = VectorXd::Ones(6);
	if(testgrad(error, x, step, tol, vfunc, gfunc) != 0) {
		return -1;
	}

	return 0;
}


/*********************************************************************
 * Registration Class Implementations
 ********************************************************************/

/*********************
 * Correlation
 ********************/

/**
 * @brief Constructor for the rigid correlation class. Note that
 * rigid rotation is assumed to be about the center of the fixed
 * image space. If you make any changes to the internal images call
 * reinit() to reinitialize the derivative.
 *
 * @param fixed Fixed image. A copy of this will be made.
 * @param moving Moving image. A copy of this will be made.
 * @param compdiff Negate correlation to compute a difference measure
 */
RigidCorrComputer::RigidCorrComputer(bool compdiff) : m_compdiff(compdiff)
{
}

/**
 * @brief Computes the gradient and value of the correlation.
 *
 * @param x Paramters (Rx, Ry, Rz, Sx, Sy, Sz).
 * @param v Value at the given rotation
 * @param g Gradient at the given rotation
 *
 * @return 0 if successful
 */
int RigidCorrComputer::valueGrad(const VectorXd& params,
		double& val, VectorXd& grad)
{
	if(!m_fixed) throw INVALID_ARGUMENT("ERROR must set fixed image before "
				"computing value.");
	if(!m_moving) throw INVALID_ARGUMENT("ERROR must set moving image before "
				"computing value.");

	double rx = params[0]*M_PI/180.;
	double ry = params[1]*M_PI/180.;
	double rz = params[2]*M_PI/180.;
	double sx = params[3]*m_moving->spacing(0);
	double sy = params[4]*m_moving->spacing(1);
	double sz = params[5]*m_moving->spacing(2);

#if defined DEBUG || defined VERYDEBUG
	cerr << "Rotation: " << rx << ", " << ry << ", " << rz << ", Shift: "
		<< sx << ", " << sy << ", " << sz << endl;
#endif
	
	// for computing roted indices
	double ind[3];
	double cind[3];

	// Compute derivative Images (if debugging is enabled, otherwise just
	// compute the gradient)
	grad.setZero();
	size_t count = 0;
	double mov_sum = 0;
	double fix_sum = 0;
	double mov_ss = 0;
	double fix_ss = 0;
	double corr = 0;
	for(m_fit.goBegin(); !m_fit.eof(); ++m_fit) {
		m_fit.index(3, ind);
		// u = c + R^-1(v - s - c)
		// where u is the output index, v the input, c the center of rotation
		// and s the shift
		// cind = center + rInv*(ind-shift-center);

		double x = ind[0];
		double y = ind[1];
		double z = ind[2];

		double cx = m_center[0];
		double cy = m_center[1];
		double cz = m_center[2];

		cind[0] = cx + sx + (-cz + z)*sin(ry) + cos(ry)*((-cx + x)*cos(rz) +
				(cy - y)*sin(rz));
		cind[1] = cy + sy + (cz - z)*cos(ry)*sin(rx) + (-cx +
				x)*(cos(rz)*sin(rx)*sin(ry) + cos(rx)*sin(rz)) + (-cy +
					y)*(cos(rx)*cos(rz) - sin(rx)*sin(ry)*sin(rz));
		cind[2] = cz + sz + (-cz + z)*cos(rx)*cos(ry) + (cx -
				x)*(cos(rx)*cos(rz)*sin(ry) - sin(rx)*sin(rz)) + (-cy +
					y)*(cos(rz)*sin(rx) + cos(rx)*sin(ry)*sin(rz));

		// Here we compute dg(v(u,p))/dp, where g is the image, u is the
		// coordinate in the fixed image, and p is the param.
		// dg/dp = SUM_i dg/dv_i dv_i/dp, where v is the rotated coordinate, so
		// dg/dv_i is the directional derivative in original space,
		// dv_i/dp is the derivative of the rotated coordinate system with
		// respect to a parameter
		double dg_dx = m_dmove_get(cind[0], cind[1], cind[2], 0);
		double dg_dy = m_dmove_get(cind[0], cind[1], cind[2], 1);
		double dg_dz = m_dmove_get(cind[0], cind[1], cind[2], 2);

		double dx_dRx = 0;
		double dy_dRx = (cz - z)*cos(rx)*cos(ry) +
			(-cx + x)*(cos(rx)*cos(rz)*sin(ry) - sin(rx)*sin(rz)) +
			(cy - y)*(cos(rz)*sin(rx) + cos(rx)*sin(ry)*sin(rz));
		double dz_dRx = (cz - z)*cos(ry)*sin(rx) +
			(-cx + x)*(cos(rz)*sin(rx)*sin(ry) + cos(rx)*sin(rz)) +
			(-cy + y)*(cos(rx)*cos(rz) - sin(rx)*sin(ry)*sin(rz));

		double dx_dRy = (-cz + z)*cos(ry) + sin(ry)*((cx - x)*cos(rz) + (-cy + y)*sin(rz));
		double dy_dRy = sin(rx)*((-cz + z)*sin(ry) + cos(ry)*((-cx + x)*cos(rz) + (cy - y)*sin(rz)));
		double dz_dRy = cos(rx)*((cz - z)*sin(ry) + cos(ry)*((cx - x)*cos(rz) + (-cy + y)*sin(rz)));

		double dx_dRz = cos(ry)*((cy - y)*cos(rz) + (cx - x)*sin(rz));
		double dy_dRz = (cy - y)*(cos(rz)*sin(rx)*sin(ry) + cos(rx)*sin(rz)) +
			(-cx + x)*(cos(rx)*cos(rz) - sin(rx)*sin(ry)*sin(rz));
		double dz_dRz =  (-cy + y)*(cos(rx)*cos(rz)*sin(ry) - sin(rx)*sin(rz))
			+ (-cx + x)*(cos(rz)*sin(rx) + cos(rx)*sin(ry)*sin(rz));

		// derivative of coordinate system due
		const double dx_dSx = 1;
		const double dy_dSx = 0;
		const double dz_dSx = 0;

		const double dx_dSy = 0;
		const double dy_dSy = 1;
		const double dz_dSy = 0;

		const double dx_dSz = 0;
		const double dy_dSz = 0;
		const double dz_dSz = 1;

		// compute SUM_i dg/dv_i dv_i/dp
		double dgdRx = (dg_dx*dx_dRx + dg_dy*dy_dRx + dg_dz*dz_dRx);
		double dgdRy = (dg_dx*dx_dRy + dg_dy*dy_dRy + dg_dz*dz_dRy);
		double dgdRz = (dg_dx*dx_dRz + dg_dy*dy_dRz + dg_dz*dz_dRz);

		double dgdSx = (dg_dx*dx_dSx + dg_dy*dy_dSx + dg_dz*dz_dSx);
		double dgdSy = (dg_dx*dx_dSy + dg_dy*dy_dSy + dg_dz*dz_dSy);
		double dgdSz = (dg_dx*dx_dSz + dg_dy*dy_dSz + dg_dz*dz_dSz);

		// compute correlation, since it requires almost no additional work
		double g = m_move_get(cind[0], cind[1], cind[2]);
		double f = *m_fit;

		mov_sum += g;
		fix_sum += f;
		mov_ss += g*g;
		fix_ss += f*f;
		corr += g*f;

		grad[0] += (*m_fit)*dgdRx*M_PI/180.;
		grad[1] += (*m_fit)*dgdRy*M_PI/180.;
		grad[2] += (*m_fit)*dgdRz*M_PI/180.;
		grad[3] += (*m_fit)*dgdSx*m_moving->spacing(0);
		grad[4] += (*m_fit)*dgdSy*m_moving->spacing(1);
		grad[5] += (*m_fit)*dgdSz*m_moving->spacing(2);

	}

	count = m_fixed->elements();
	val = sample_corr(count, mov_sum, fix_sum, mov_ss, fix_ss, corr);
	double sd1 = sqrt(sample_var(count, mov_sum, mov_ss));
	double sd2 = sqrt(sample_var(count, fix_sum, fix_ss));
	grad /= (count-1)*sd1*sd2;

	if(m_compdiff) {
		grad = -grad;
		val = -val;
	}

#if defined VERYDEBUG || defined DEBUG
	cerr << "Value: " << val << endl;
	cerr << "Gradient: " << grad.transpose() << endl;
#endif
	
	return 0;
}

/**
 * @brief Computes the gradient of the correlation. Note that this
 * function just calls valueGrad because computing the
 * additional values are trivial
 *
 * @param params Paramters (Rx, Ry, Rz, Sx, Sy, Sz).
 * @param grad Gradient at the given rotation
 *
 * @return 0 if successful
 */
int RigidCorrComputer::grad(const VectorXd& params, VectorXd& grad)
{
	double v = 0;
	return valueGrad(params, v, grad);
}

/**
 * @brief Computes the correlation.
 *
 * @param params Paramters (Rx, Ry, Rz, Sx, Sy, Sz).
 * @param val Value at the given rotation
 *
 * @return 0 if successful
 */
int RigidCorrComputer::value(const VectorXd& params, double& val)
{
	if(!m_fixed) throw INVALID_ARGUMENT("ERROR must set fixed image before "
				"computing value.");
	if(!m_moving) throw INVALID_ARGUMENT("ERROR must set moving image before "
				"computing value.");

#if defined VERYDEBUG || DEBUG
	cerr << "Val()" << endl;
#endif

	assert(m_fixed->ndim() == 3);
	assert(m_moving->ndim() == 3);

	double rx = params[0]*M_PI/180.;
	double ry = params[1]*M_PI/180.;
	double rz = params[2]*M_PI/180.;
	double sx = params[3]*m_moving->spacing(0);
	double sy = params[4]*m_moving->spacing(1);
	double sz = params[5]*m_moving->spacing(2);
	//#if defined DEBUG || defined VERYDEBUG
	cerr << "Rotation: " << rx << ", " << ry << ", " << rz << ", Shift: "
		<< sx << ", " << sy << ", " << sz << endl;
	//#endif

	double ind[3];
	double cind[3];

	//  Resample Output and Compute Orientation. While actually resampling is
	//  optional, it helps with debugging
	double sum1 = 0;
	double sum2 = 0;
	double ss1 = 0;
	double ss2 = 0;
	double corr = 0;
	for(m_fit.goBegin(); !m_fit.eof(); ++m_fit) {
		m_fit.index(3, ind);

		// u = c + R^-1(v - s - c)
		// where u is the output index, v the input, c the center of rotation
		// and s the shift
		double x = ind[0];
		double y = ind[1];
		double z = ind[2];

		double cx = m_center[0];
		double cy = m_center[1];
		double cz = m_center[2];

		cind[0] = cx + sx + (-cz + z)*sin(ry) + cos(ry)*((-cx + x)*cos(rz) +
				(cy - y)*sin(rz));
		cind[1] = cy + sy + (cz - z)*cos(ry)*sin(rx) + (-cx +
				x)*(cos(rz)*sin(rx)*sin(ry) + cos(rx)*sin(rz)) + (-cy +
					y)*(cos(rx)*cos(rz) - sin(rx)*sin(ry)*sin(rz));
		cind[2] = cz + sz + (-cz + z)*cos(rx)*cos(ry) + (cx -
				x)*(cos(rx)*cos(rz)*sin(ry) - sin(rx)*sin(rz)) + (-cy +
					y)*(cos(rz)*sin(rx) + cos(rx)*sin(ry)*sin(rz));

		double a = m_move_get(cind[0], cind[1], cind[2]);
		double b = *m_fit;
		sum1 += a;
		ss1 += a*a;
		sum2 += b;
		ss2 += b*b;
		corr += a*b;
	}

	val = sample_corr(m_fixed->elements(), sum1, sum2, ss1, ss2, corr);
	if(m_compdiff)
		val = -val;

#if defined VERYDEBUG || defined DEBUG
	cerr << "Value: " << val << endl;
#endif
	return 0;
};

/**
 * @brief Set the fixed image for registration/comparison
 *
 * @param fixed Input fixed image (not modified)
 */
void RigidCorrComputer::setFixed(ptr<const MRImage> newfix)
{
	if(newfix->ndim() != 3)
		throw INVALID_ARGUMENT("Fixed image is not 3D!");
	
	m_fixed = newfix;
	m_fit.setArray(newfix);
};

/**
 * @brief Set the moving image for comparison, note that setting this
 * triggers a derivative computation and so is slower than setFixed.
 *
 * Note that modification of the moving image outside this class without
 * re-calling setMoving is undefined and will result in an out-of-date
 * moving image derivative. THIS WILL BREAK GRADIENT CALCULATIONS.
 *
 * @param moving Input moving image (not modified)
 */
void RigidCorrComputer::setMoving(ptr<const MRImage> newmove)
{
	if(newmove->ndim() != 3)
		throw INVALID_ARGUMENT("Moving image is not 3D!");
	
	m_moving = newmove;

	if(!m_dmoving || !newmove->matchingOrient(m_dmoving, false, true)) {
#ifndef NDEBUG
		cerr << "Allocating derivative image" << endl;
#endif //NDEBUG
		m_dmoving = dPtrCast<MRImage>(derivative(m_moving));
	} else {
		// Just fill in existing image
		derivative(m_moving, m_dmoving);
	}

	m_move_get.setArray(m_moving);
	m_dmove_get.setArray(m_dmoving);
	for(size_t ii=0; ii<3 && ii<m_moving->ndim(); ii++)
		m_center[ii] = (m_moving->dim(ii)-1)/2.;
}

/****************************************************************************
 * Mutual Information/Normalize Mutual Information/Variation of Information
 ****************************************************************************/

/**
 * @brief Constructor for the rigid correlation class. Note that
 * rigid rotation is assumed to be about the center of the fixed
 * image space. If necessary the input moving image will be resampled.
 * To the same space as the fixed image.
 *
 * @param fixed Fixed image. A copy of this will be made.
 * @param moving Moving image. A copy of this will be made.
 * @param compdiff negate MI and NMI to create an effective distance 
 */
RigidInformationComputer::RigidInformationComputer(bool compdiff) : 
	m_compdiff(compdiff), m_metric(METRIC_MI), m_gradHjoint(6), m_gradHmove(6)
{
	setBins(128, 4);
}

/**
 * @brief Reallocates histograms and if m_fixed has been set, regenerates 
 * histogram estimate of fixed pdf
 *
 * @param nbins Number of bins for marginal estimation
 * @param krad Number of bins in kernel radius
 */
void RigidInformationComputer::setBins(size_t nbins, size_t krad)
{
	// set number of bins
	m_bins = nbins;
	m_krad = krad;

	// reallocate memory
	m_pdfmove.resize({nbins});
	m_pdffix.resize({nbins});
	m_pdfjoint.resize({nbins, nbins});
	m_dpdfjoint.resize({6, nbins, nbins});
	m_dpdfmove.resize({6, nbins});

	if(m_fixed) {
		// if fixed hsa been set, recompute fixed entropy
		setFixed(m_fixed);
	}
}

/**
 * @brief Sets the moving image for Information-Based comparison. This sets
 * m_moving, m_move_get, d_moving, m_dmove_get and m_rangemove
 *
 * @param newmove New moving image
 */
void RigidInformationComputer::setMoving(ptr<const MRImage> newmove)
{
	if(newmove->ndim() != 3)
		throw INVALID_ARGUMENT("Moving image is not 3D!");
	
	//////////////////////////////////////
	// Setup accessors and Members Images
	//////////////////////////////////////
	m_moving = newmove;

	if(!m_dmoving || !newmove->matchingOrient(m_dmoving, false, true)) {
#ifndef NDEBUG
		cerr << "Allocating derivative image" << endl;
#endif //NDEBUG
		m_dmoving = dPtrCast<MRImage>(derivative(m_moving));
	} else {
		// Just fill in existing image
		derivative(m_moving, m_dmoving);
	}

	m_move_get.setArray(m_moving);
	m_dmove_get.setArray(m_dmoving);
	for(size_t ii=0; ii<3 && ii<m_moving->ndim(); ii++)
		m_center[ii] = (m_moving->dim(ii)-1)/2.;

	// must include 0 because outside values get mapped to 0
	m_rangemove[0] = 0;
	m_rangemove[1] = 0;
	for(NDConstIter<double> it(m_moving); !it.eof(); ++it) {
		m_rangemove[0] = std::min(m_rangemove[0], *it);
		m_rangemove[1] = std::max(m_rangemove[1], *it);
	}
}

/**
 * @brief Sets the fixed image for Information-Based comparison. This sets
 * m_fixed, m_fit, m_rangefix, m_wfix, fills the histogram m_pdffix, and 
 * computes m_Hfix
 *
 * @param newfixed New fixed image
 */
void RigidInformationComputer::setFixed(ptr<const MRImage> newfixed)
{
	if(newfixed->ndim() != 3)
		throw INVALID_ARGUMENT("Fixed image is not 3D!");
	
	//////////////////////////////////////
	// Set up Members
	//////////////////////////////////////
	m_fixed = newfixed;
	m_fit.setArray(m_fixed);
	
	//////////////////////////////////////
	// compute ranges, and bin widths
	//////////////////////////////////////
	m_rangefix[0] = INFINITY;
	m_rangefix[1] = -INFINITY;
	for(NDConstIter<double> it(m_fixed); !it.eof(); ++it) {
		m_rangefix[0] = std::min(m_rangefix[0], *it);
		m_rangefix[1] = std::max(m_rangefix[1], *it);
	}

	// fixed marginal pdf
	m_wfix = (m_rangefix[1]-m_rangefix[0])/(m_bins-2*m_krad-1);
	for(NDConstIter<double> it(m_fixed); !it.eof(); ++it) {
		// compute bins
		double cbin = ((*it)-m_rangefix[0])/m_wfix + m_krad;
		int bin = round(cbin);

		assert(bin+m_krad < m_bins);
		assert(bin-m_krad >= 0);

		for(int ii = bin-m_krad; ii <= bin+m_krad; ii++)
			m_pdffix[ii] += B3kern(ii-cbin, m_krad);
	}

	// fixed entropy
	m_Hfix = 0;
	double Nrecip = 1./m_fixed->elements();
	for(size_t ii=0; ii<m_bins; ii++) {
		m_pdffix[ii] *= Nrecip;
		m_Hfix -= m_pdffix[ii] > 0 ? m_pdffix[ii]*log(m_pdffix[ii]) : 0;
	}
}

/**
 * @brief Computes the gradient and value of the correlation.
 *
 * @param x Paramters (Rx, Ry, Rz, Sx, Sy, Sz).
 * @param v Value at the given rotation
 * @param g Gradient at the given rotation
 *
 * @return 0 if successful
 */
int RigidInformationComputer::valueGrad(const VectorXd& params,
		double& val, VectorXd& grad)
{
	if(!m_fixed) throw INVALID_ARGUMENT("ERROR must set fixed image before "
				"computing value.");
	if(!m_moving) throw INVALID_ARGUMENT("ERROR must set moving image before "
				"computing value.");
	if(!m_moving->matchingOrient(m_fixed, true, true))
		throw INVALID_ARGUMENT("ERROR input images must be in same index "
				"space");

	assert(m_pdfmove.dim(0) == m_bins);
	assert(m_pdffix.dim(0) == m_bins);
	assert(m_pdfjoint.dim(0) == m_bins);
	assert(m_pdfjoint.dim(1) == m_bins);

	assert(m_dpdfjoint.dim(0) == 6);
	assert(m_dpdfjoint.dim(1) == m_bins);
	assert(m_dpdfjoint.dim(1) == m_bins);

	assert(m_dpdfmove.dim(0) == 6);
	assert(m_dpdfmove.dim(1) == m_bins);

	double rx = params[0]*M_PI/180.;
	double ry = params[1]*M_PI/180.;
	double rz = params[2]*M_PI/180.;
	double sx = params[3]*m_moving->spacing(0);
	double sy = params[4]*m_moving->spacing(1);
	double sz = params[5]*m_moving->spacing(2);
//#if defined DEBUG || defined VERYDEBUG
	cerr << "Rotation: " << rx << ", " << ry << ", " << rz << ", Shift: "
		<< sx << ", " << sy << ", " << sz << endl;
//#endif

	// Zero Everything
	m_pdfmove.zero();
	m_pdfjoint.zero();
	m_dpdfmove.zero();
	m_dpdfjoint.zero();

	// compute updated moving width
	m_wmove = (m_rangemove[1]-m_rangemove[0])/(m_bins-2*m_krad-1);

	// for computing rotated indices
	double cbinmove, cbinfix; //continuous
	double binmove, binfix; // nearest int
	double ind[3];
	double cind[3];
	double dgdPhi[6];

	// Compute Probabilities
	for(m_fit.goBegin(); !m_fit.eof(); ++m_fit) {
		m_fit.index(3, ind);
		double x = ind[0];
		double y = ind[1];
		double z = ind[2];

		double cx = m_center[0];
		double cy = m_center[1];
		double cz = m_center[2];

		cind[0] = cx + sx + (-cz + z)*sin(ry) + cos(ry)*((-cx + x)*cos(rz) +
				(cy - y)*sin(rz));
		cind[1] = cy + sy + (cz - z)*cos(ry)*sin(rx) + (-cx +
				x)*(cos(rz)*sin(rx)*sin(ry) + cos(rx)*sin(rz)) + (-cy +
					y)*(cos(rx)*cos(rz) - sin(rx)*sin(ry)*sin(rz));
		cind[2] = cz + sz + (-cz + z)*cos(rx)*cos(ry) + (cx -
				x)*(cos(rx)*cos(rz)*sin(ry) - sin(rx)*sin(rz)) + (-cy +
					y)*(cos(rz)*sin(rx) + cos(rx)*sin(ry)*sin(rz));

		// Here we compute dg(v(u,p))/dp, where g is the image, u is the
		// coordinate in the fixed image, and p is the param.
		// dg/dp = SUM_i dg/dv_i dv_i/dp, where v is the rotated coordinate, so
		// dg/dv_i is the directional derivative in original space,
		// dv_i/dp is the derivative of the rotated coordinate system with
		// respect to a parameter
		double dg_dx = m_dmove_get(cind[0], cind[1], cind[2], 0);
		double dg_dy = m_dmove_get(cind[0], cind[1], cind[2], 1);
		double dg_dz = m_dmove_get(cind[0], cind[1], cind[2], 2);

		double dx_dRx = 0;
		double dy_dRx = (cz - z)*cos(rx)*cos(ry) +
			(-cx + x)*(cos(rx)*cos(rz)*sin(ry) - sin(rx)*sin(rz)) +
			(cy - y)*(cos(rz)*sin(rx) + cos(rx)*sin(ry)*sin(rz));
		double dz_dRx = (cz - z)*cos(ry)*sin(rx) +
			(-cx + x)*(cos(rz)*sin(rx)*sin(ry) + cos(rx)*sin(rz)) +
			(-cy + y)*(cos(rx)*cos(rz) - sin(rx)*sin(ry)*sin(rz));

		double dx_dRy = (-cz + z)*cos(ry) + sin(ry)*((cx - x)*cos(rz) + (-cy + y)*sin(rz));
		double dy_dRy = sin(rx)*((-cz + z)*sin(ry) + cos(ry)*((-cx + x)*cos(rz) + (cy - y)*sin(rz)));
		double dz_dRy = cos(rx)*((cz - z)*sin(ry) + cos(ry)*((cx - x)*cos(rz) + (-cy + y)*sin(rz)));

		double dx_dRz = cos(ry)*((cy - y)*cos(rz) + (cx - x)*sin(rz));
		double dy_dRz = (cy - y)*(cos(rz)*sin(rx)*sin(ry) + cos(rx)*sin(rz)) +
			(-cx + x)*(cos(rx)*cos(rz) - sin(rx)*sin(ry)*sin(rz));
		double dz_dRz =  (-cy + y)*(cos(rx)*cos(rz)*sin(ry) - sin(rx)*sin(rz))
			+ (-cx + x)*(cos(rz)*sin(rx) + cos(rx)*sin(ry)*sin(rz));

		// compute SUM_i dg/dv_i dv_i/dp
		dgdPhi[0] = (dg_dx*dx_dRx + dg_dy*dy_dRx + dg_dz*dz_dRx)*M_PI/180.;
		dgdPhi[1] = (dg_dx*dx_dRy + dg_dy*dy_dRy + dg_dz*dz_dRy)*M_PI/180.;
		dgdPhi[2] = (dg_dx*dx_dRz + dg_dy*dy_dRz + dg_dz*dz_dRz)*M_PI/180.;

		dgdPhi[3] = dg_dx*m_moving->spacing(0);
		dgdPhi[4] = dg_dy*m_moving->spacing(1);
		dgdPhi[5] = dg_dz*m_moving->spacing(2);

		// get actual values
		double valmove = m_move_get(cind[0], cind[1], cind[2]);
		double valfix = *m_fit;

		// compute bins
		cbinfix = (valfix-m_rangefix[0])/m_wfix + m_krad;
		cbinmove = (valmove-m_rangemove[0])/m_wmove + m_krad;
		binfix = round(cbinfix);
		binmove = round(cbinmove);

		assert(binfix+m_krad < m_bins);
		assert(binmove+m_krad < m_bins);
		assert(binfix-m_krad >= 0);
		assert(binmove-m_krad >= 0);

		// Value PDF
		for(int jj = binmove-m_krad; jj <= binmove+m_krad; jj++)
			m_pdfmove[jj] += B3kern(jj-cbinmove, m_krad);

		for(int ii = binfix-m_krad; ii <= binfix+m_krad; ii++) {
			for(int jj = binmove-m_krad; jj <= binmove+m_krad; jj++)
				m_pdfjoint[{ii,jj}] += B3kern(ii-cbinfix, m_krad)*
					B3kern(jj-cbinmove, m_krad);
		}

		// Derivatives
		for(int phi = 0; phi < 6; phi++) {
			for(int jj = binmove-m_krad; jj <= binmove+m_krad; jj++)
				m_dpdfmove[{phi,jj}] += dB3kern(jj-cbinmove, m_krad)*dgdPhi[phi];
		}

		for(int phi = 0; phi < 6; phi++) {
			for(int ii = binfix-m_krad; ii <= binfix+m_krad; ii++) {
				for(int jj = binmove-m_krad; jj <= binmove+m_krad; jj++) {
					m_dpdfjoint[{phi,ii,jj}] += B3kern(ii-cbinfix, m_krad)*
						dB3kern(jj-cbinmove, m_krad)*dgdPhi[phi];
				}
			}
		}
	}

	///////////////////////
	// Update Entropies
	///////////////////////

	// pdf's
	double scale = 1./(m_fixed->elements());
	for(size_t ii=0; ii<m_pdfmove.elements(); ii++)
		m_pdfmove[ii] *= scale;
	for(size_t ii=0; ii<m_pdfjoint.elements(); ii++)
		m_pdfjoint[ii] *= scale;

	// update m_Hmove
	m_Hmove = 0;
	for(int ii=0; ii<m_pdfmove.elements(); ii++)
		m_Hmove -= m_pdfmove[ii] > 0 ? m_pdfmove[ii]*log(m_pdfmove[ii]) : 0;

	// update m_Hjoint
	m_Hjoint = 0;
	for(int ii=0; ii<m_pdfjoint.elements(); ii++)
		m_Hjoint -= m_pdfjoint[ii] > 0 ? m_pdfjoint[ii]*log(m_pdfjoint[ii]) : 0;

	//////////////////////////////
	// Update Gradient Entropies
	//////////////////////////////

	// pdf's
	double dscale = -1./(m_wmove*m_fixed->elements());
	for(size_t ii=0; ii<m_dpdfmove.elements(); ii++)
		m_dpdfmove[ii] *= dscale;
	for(size_t ii=0; ii<m_dpdfjoint.elements(); ii++)
		m_dpdfjoint[ii] *= dscale;

	// Hmove
	for(int pp=0; pp<6; pp++) {
		m_gradHmove[pp] = 0;
		for(int jj=0; jj<m_bins; jj++) {
			double p = m_pdfmove[jj];
			double dp = m_dpdfmove[{pp,jj}];
			m_gradHmove[pp] -= p > 0 ? dp*log(p) : 0;
		}
	}

	// Hjoint
	for(int pp=0; pp<6; pp++) {
		m_gradHjoint[pp] = 0;
		for(int ii=0; ii<m_bins; ii++) {
			for(int jj=0; jj<m_bins; jj++) {
				double p = m_pdfjoint[{ii,jj}];
				double dp = m_dpdfjoint[{pp,ii,jj}];
				m_gradHjoint[pp] -= p > 0 ? dp*log(p) : 0;
			}
		}
	}


	// update value and grad
	if(m_metric == METRIC_MI) {
		val = m_Hfix+m_Hmove-m_Hjoint;
		for(size_t ii=0; ii<6; ii++)
			grad[ii] = m_gradHmove[ii]-m_gradHjoint[ii];

	} else if(m_metric == METRIC_VI) {
		val = 2*m_Hjoint-m_Hfix-m_Hmove;
		for(size_t ii=0; ii<6; ii++)
			grad[ii] = 2*m_gradHjoint[ii] - m_gradHmove[ii];

	} else if(m_metric == METRIC_NMI) {
		val =  (m_Hfix+m_Hmove)/m_Hjoint;
		for(size_t ii=0; ii<6; ii++)
			grad[ii] = m_gradHmove[ii]/m_Hjoint -
				m_gradHjoint[ii]*(m_Hfix+m_Hmove)/(m_Hjoint*m_Hjoint);
	}

//#if defined DEBUG || defined VERYDEBUG
	cerr << "ValueGrad() = " << val << " / " << grad.transpose() << endl;
//#endif

	// negate if we are using a similarity measure
	if(m_compdiff && (m_metric == METRIC_NMI || m_metric == METRIC_MI)) {
		grad = -grad;
		val = -val;
	}

	return 0;
}

/**
 * @brief Computes the gradient of the correlation. Note that this
 * function just calls valueGrad because computing the
 * additional values are trivial
 *
 * @param params Paramters (Rx, Ry, Rz, Sx, Sy, Sz).
 * @param grad Gradient at the given rotation
 *
 * @return 0 if successful
 */
int RigidInformationComputer::grad(const VectorXd& params, VectorXd& grad)
{
	double v = 0;
	return valueGrad(params, v, grad);
}

/**
 * @brief Computes the correlation.
 *
 * @param params Paramters (Rx, Ry, Rz, Sx, Sy, Sz).
 * @param val Value at the given rotation
 *
 * @return 0 if successful
 */
int RigidInformationComputer::value(const VectorXd& params, double& val)
{
	if(!m_fixed) 
		throw INVALID_ARGUMENT("ERROR must set fixed image before "
				"computing value.");
	if(!m_moving) 
		throw INVALID_ARGUMENT("ERROR must set moving image before "
				"computing value.");
	if(!m_moving->matchingOrient(m_fixed, true, true))
		throw INVALID_ARGUMENT("ERROR input images must be in same index "
				"space");

	assert(m_pdfmove.dim(0) == m_bins);
	assert(m_pdffix.dim(0) == m_bins);
	assert(m_pdfjoint.dim(0) == m_bins);
	assert(m_pdfjoint.dim(1) == m_bins);
	
	assert(m_dpdfjoint.dim(0) == 6);
	assert(m_dpdfjoint.dim(1) == m_bins);
	assert(m_dpdfjoint.dim(1) == m_bins);
	
	assert(m_dpdfmove.dim(0) == 6);
	assert(m_dpdfmove.dim(1) == m_bins);

	double rx = params[0]*M_PI/180.;
	double ry = params[1]*M_PI/180.;
	double rz = params[2]*M_PI/180.;
	double sx = params[3]*m_moving->spacing(0);
	double sy = params[4]*m_moving->spacing(1);
	double sz = params[5]*m_moving->spacing(2);
//#if defined DEBUG || defined VERYDEBUG
	cerr << "Rotation: " << rx << ", " << ry << ", " << rz << ", Shift: "
		<< sx << ", " << sy << ", " << sz << endl;
//#endif

	// Zero
	m_pdfmove.zero();
	m_pdfjoint.zero();

	// for computing rotated indices
	double cbinmove, cbinfix; //continuous
	double binmove, binfix; // nearest int
	double ind[3];
	double cind[3];

	// Compute Probabilities
	for(m_fit.goBegin(); !m_fit.eof(); ++m_fit) {
		m_fit.index(3, ind);

		double x = ind[0];
		double y = ind[1];
		double z = ind[2];

		double cx = m_center[0];
		double cy = m_center[1];
		double cz = m_center[2];

		cind[0] = cx + sx + (-cz + z)*sin(ry) + cos(ry)*((-cx + x)*cos(rz) +
				(cy - y)*sin(rz));
		cind[1] = cy + sy + (cz - z)*cos(ry)*sin(rx) + (-cx +
				x)*(cos(rz)*sin(rx)*sin(ry) + cos(rx)*sin(rz)) + (-cy +
					y)*(cos(rx)*cos(rz) - sin(rx)*sin(ry)*sin(rz));
		cind[2] = cz + sz + (-cz + z)*cos(rx)*cos(ry) + (cx -
				x)*(cos(rx)*cos(rz)*sin(ry) - sin(rx)*sin(rz)) + (-cy +
					y)*(cos(rz)*sin(rx) + cos(rx)*sin(ry)*sin(rz));

		// get actual values
		double valmove = m_move_get(cind[0], cind[1], cind[2]);
		double valfix = *m_fit;

		// compute bins
		cbinfix = (valfix-m_rangefix[0])/m_wfix + m_krad;
		cbinmove = (valmove-m_rangemove[0])/m_wmove + m_krad;
		binfix = round(cbinfix);
		binmove = round(cbinmove);

		assert(binfix+m_krad < m_bins);
		assert(binmove+m_krad < m_bins);
		assert(binfix-m_krad >= 0);
		assert(binmove-m_krad >= 0);

		//sum up kernel bins
		for(int jj = binmove-m_krad; jj <= binmove+m_krad; jj++)
			m_pdfmove[jj] += B3kern(jj-cbinmove, m_krad);

		for(int ii = binfix-m_krad; ii <= binfix+m_krad; ii++) {
			for(int jj = binmove-m_krad; jj <= binmove+m_krad; jj++)
				m_pdfjoint[{ii,jj}] += B3kern(ii-cbinfix, m_krad)*
					B3kern(jj-cbinmove, m_krad);
		}
	}

	// divide by constants
	double scale = 1./(m_fixed->elements());
	for(size_t ii=0; ii<m_pdfmove.elements();  ii++)
		m_pdfmove[ii] *= scale;
	for(size_t ii=0; ii<m_pdfjoint.elements(); ii++)
		m_pdfjoint[ii] *= scale;

	// update m_Hmove
	m_Hmove = 0;
	for(int ii=0; ii<m_pdfmove.elements(); ii++) {
		m_Hmove -= m_pdfmove[ii] > 0 ? m_pdfmove[ii]*log(m_pdfmove[ii]) : 0;
	}

	// update m_Hjoint
	m_Hjoint = 0;
	for(int ii=0; ii<m_pdfjoint.elements(); ii++) {
		m_Hjoint -= m_pdfjoint[ii] > 0 ? m_pdfjoint[ii]*log(m_pdfjoint[ii]) : 0;
	}

	// update value
	if(m_metric == METRIC_MI) {
		val = m_Hfix+m_Hmove-m_Hjoint;
	} else if(m_metric == METRIC_VI) {
		val = 2*m_Hjoint-m_Hfix-m_Hmove;
	} else if(m_metric == METRIC_NMI) {
		val =  (m_Hfix+m_Hmove)/m_Hjoint;
	}

//#if defined DEBUG || defined VERYDEBUG
	cerr << "Value() = " << val << endl;
//#endif

	// negate if we are using a similarity measure
	if(m_compdiff && (m_metric == METRIC_NMI || m_metric == METRIC_MI)) {
		val = -val;
	}

	return 0;
};


/****************************************************************************
 * Mutual Information/Normalize Mutual Information/Variation of Information
 * Distortion Correction
 ****************************************************************************/

/**
 * @brief Constructor for the rigid correlation class. Note that
 * rigid rotation is assumed to be about the center of the fixed
 * image space. If necessary the input moving image will be resampled.
 * To the same space as the fixed image.
 *
 * @param fixed Fixed image. A copy of this will be made.
 * @param moving Moving image. A copy of this will be made.
 * @param compdiff negate MI and NMI to create an effective distance 
 */
DistortionCorrectionInformationComputer::
			DistortionCorrectionInformationComputer(bool compdiff) : 
			m_compdiff(compdiff), m_metric(METRIC_MI), m_gradHjoint(6),
			m_gradHmove(6)
{
	setBins(128, 4);
}

/**
 * @brief Reallocates histograms and if m_fixed has been set, regenerates 
 * histogram estimate of fixed pdf
 *
 * @param nbins Number of bins for marginal estimation
 * @param krad Number of bins in kernel radius
 */
void DistortionCorrectionInformationComputer::setBins(size_t nbins, size_t krad)
{
	// set number of bins
	m_bins = nbins;
	m_krad = krad;

	// reallocate memory
	m_pdfmove.resize({nbins});
	m_pdffix.resize({nbins});
	m_pdfjoint.resize({nbins, nbins});

	m_dpdfjoint.resize({6, nbins, nbins});
	m_dpdfmove.resize({6, nbins});

	if(m_fixed) {
		// if fixed hsa been set, recompute fixed entropy
		setFixed(m_fixed);
	}
}

/**
 * @brief Sets the moving image for Information-Based comparison. This sets
 * m_moving, m_move_get, d_moving, m_dmove_get and m_rangemove
 *
 * @param newmove New moving image
 */
void DistortionCorrectionInformationComputer::setMoving(
		ptr<const MRImage> newmove)
{
	if(newmove->ndim() != 3)
		throw INVALID_ARGUMENT("Moving image is not 3D!");
	
	//////////////////////////////////////
	// Setup accessors and Members Images
	//////////////////////////////////////
	m_moving = newmove;

	if(!m_dmoving || !newmove->matchingOrient(m_dmoving, false, true)) {
#ifndef NDEBUG
		cerr << "Allocating derivative image" << endl;
#endif //NDEBUG
		m_dmoving = dPtrCast<MRImage>(derivative(m_moving));
	} else {
		// Just fill in existing image
		derivative(m_moving, m_dmoving);
	}

	m_move_get.setArray(m_moving);
	m_dmove_get.setArray(m_dmoving);
	for(size_t ii=0; ii<3 && ii<m_moving->ndim(); ii++)
		m_center[ii] = (m_moving->dim(ii)-1)/2.;

	// must include 0 because outside values get mapped to 0
	m_rangemove[0] = 0;
	m_rangemove[1] = 0;
	for(NDConstIter<double> it(m_moving); !it.eof(); ++it) {
		m_rangemove[0] = std::min(m_rangemove[0], *it);
		m_rangemove[1] = std::max(m_rangemove[1], *it);
	}
}

/**
 * @brief Computes the thin-plate spline regulization value and gradient
 *
 * @param val Output Value
 * @param grad Output Gradient
 */
void DistortionCorrectionInformationComputer::thinPlateSpline(
		double& val, VectorXd& grad)
{

}

/**
 * @brief Computes the thin-plate spline regulization value 
 *
 * @param val Output Value
 */
void DistortionCorrectionInformationComputer::thinPlateSpline(
		double& val)
{

}

/**
 * @brief Computes the sum of the determinant value and gradient
 *
 * @param val Output Value
 * @param grad Output Gradient
 */
void DistortionCorrectionInformationComputer::jacobianDet(
		double& val, VectorXd& grad)
{

}

/**
 * @brief Computes the sum of the jacobian determinant value 
 *
 * @param val Output Value
 */
void DistortionCorrectionInformationComputer::jacobianDet(
		double& val)
{

}

/**
 * @brief Computes the metric value and gradient
 *
 * @param val Output Value
 * @param grad Output Gradient
 */
void DistortionCorrectionInformationComputer::metric(double val, VectorXd& grad);

/**
 * @brief Computes the metric value
 *
 * @param val Output Value
 */
void DistortionCorrectionInformationComputer::metric(double val);


/**
 * @brief Sets the fixed image for Information-Based comparison. This sets
 * m_fixed, m_fit, m_rangefix, m_wfix, fills the histogram m_pdffix, and 
 * computes m_Hfix
 *
 * @param newfixed New fixed image
 */
void DistortionCorrectionInformationComputer::setFixed(ptr<const MRImage> newfixed)
{
	if(newfixed->ndim() != 3)
		throw INVALID_ARGUMENT("Fixed image is not 3D!");
	
	//////////////////////////////////////
	// Set up Members
	//////////////////////////////////////
	m_fixed = newfixed;
	m_fit.setArray(m_fixed);
	
	//////////////////////////////////////
	// compute ranges, and bin widths
	//////////////////////////////////////
	m_rangefix[0] = INFINITY;
	m_rangefix[1] = -INFINITY;
	for(NDConstIter<double> it(m_fixed); !it.eof(); ++it) {
		m_rangefix[0] = std::min(m_rangefix[0], *it);
		m_rangefix[1] = std::max(m_rangefix[1], *it);
	}

	// fixed marginal pdf
	m_wfix = (m_rangefix[1]-m_rangefix[0])/(m_bins-2*m_krad-1);
	for(NDConstIter<double> it(m_fixed); !it.eof(); ++it) {
		// compute bins
		double cbin = ((*it)-m_rangefix[0])/m_wfix + m_krad;
		int bin = round(cbin);

		assert(bin+m_krad < m_bins);
		assert(bin-m_krad >= 0);

		for(int ii = bin-m_krad; ii <= bin+m_krad; ii++)
			m_pdffix[ii] += B3kern(ii-cbin);
	}

	// fixed entropy
	m_Hfix = 0;
	double Nrecip = 1./m_fixed->elements();
	for(size_t ii=0; ii<m_bins; ii++) {
		m_pdffix[ii] *= Nrecip;
		m_Hfix -= m_pdffix[ii] > 0 ? m_pdffix[ii]*log(m_pdffix[ii]) : 0;
	}
}

/**
 * @brief Computes the gradient and value of the correlation.
 *
 * @param x Paramters (Rx, Ry, Rz, Sx, Sy, Sz).
 * @param v Value at the given rotation
 * @param g Gradient at the given rotation
 *
 * @return 0 if successful
 */
int DistortionCorrectionInformationComputer::valueGrad(const VectorXd& params,
		double& val, VectorXd& grad)
{
	/*
	 * Check That Everything Is Properly Initialized
	 */
	if(!m_fixed) throw INVALID_ARGUMENT("ERROR must set fixed image before "
				"computing value.");
	if(!m_moving) throw INVALID_ARGUMENT("ERROR must set moving image before "
				"computing value.");
	
	/*
	 * Compute the Gradient and Value
	 */
	VectorXd tmpgrad(grad.nrows()); // TODO CACHE

	double tmp = 0;
	grad.setZero();

	// Compute and add Thin Plate Spline
	if(m_tps_reg > 0) {
		if(thinPlateSpline(tmp, tmpgrad) != 0) return -1;
		value += m_tps_reg*tmp;
		grad += m_tps_reg*tmpgrad;
	}

	// Compute and add Jacobian
	if(m_jac_reg > 0) {
		if(jacobianDet(tmp, tmpgrad) != 0) return -1;
		value += m_jac_reg*tmp;
		grad += m_jac_reg*tmpgrad;
	}

	// Compute and add Metric
	if(metric(tmp, tmpgrad) != 0) return -1;
	value += tmp;
	grad += tmpgrad;

	return 0;
}

/**
 * @brief Computes the gradient of the correlation. Note that this
 * function just calls valueGrad because computing the
 * additional values are trivial
 *
 * @param params Paramters (Rx, Ry, Rz, Sx, Sy, Sz).
 * @param grad Gradient at the given rotation
 *
 * @return 0 if successful
 */
int DistortionCorrectionInformationComputer::grad(const VectorXd& params, VectorXd& grad)
{
	double v = 0;
	return valueGrad(params, v, grad);
}

/**
 * @brief Computes the correlation.
 *
 * @param params Paramters (Rx, Ry, Rz, Sx, Sy, Sz).
 * @param val Value at the given rotation
 *
 * @return 0 if successful
 */
int DistortionCorrectionInformationComputer::value(const VectorXd& params, double& val)
{
	/*
	 * Check That Everything Is Properly Initialized
	 */

	


	/*
	 * Compute Value
	 */
	double tmp = 0;

	// Compute and add Thin Plate Spline
	if(m_tps_reg > 0) {
		if(thinPlateSpline(tmp) != 0) return -1;
		value += m_tps_reg*tmp;
	}

	// Compute and add Jacobian
	if(m_jac_reg > 0) {
		if(jacobianDet(tmp) != 0) return -1;
		value += m_jac_reg*tmp;
	}

	// Compute and add Metric
	if(metric(tmp) != 0) return -1;
	value += tmp;

	return 0;
};


/*********************************************************************
 * Rigid Transform Struct
 *********************************************************************/

/**
 * @brief Inverts rigid transform, where:
 *
 * Original:
 * y = R(x-c)+s+c
 *
 * Inverse:
 * x = R^-1(y - s - c) + c
 * x = R^-1(y - c) - R^-1 s + c
 *
 * So the new parameters, interms of the old are:
 * New c = c
 * New s = -R^-1 s
 * New R = R^-1
 *
 */
void Rigid3DTrans::invert()
{
	Matrix3d Q = rotMatrix().inverse();

	auto tmp_shift = shift;
	shift = -Q*tmp_shift;

	rotation[1] = asin(Q(0,2));
	rotation[2] = -Q(0,1)/cos(rotation[1]);
	rotation[0] = -Q(1,2)/cos(rotation[1]);
}

/**
 * @brief Constructs and returns rotation Matrix.
 *
 * @return Rotation matrix
 */
Matrix3d Rigid3DTrans::rotMatrix()
{
	Matrix3d ret;
	ret = AngleAxisd(rotation[0], Vector3d::UnitX())*
		AngleAxisd(rotation[1], Vector3d::UnitY())*
		AngleAxisd(rotation[2], Vector3d::UnitZ());
	return ret;
};

/**
 * @brief Converts to world coordinates based on the orientation stored in
 * input image.
 *
 * The image center is just converted from index space to RAS space.
 *
 * For a rotation in RAS coordinates, with orientation \f$A\f$, origin \f$b\f$
 * rotation \f$Q\f$, shift \f$t\f$ and center \f$d\f$:
 *
 * \f[
 *   \hat u = Q(A\hat x + \hat b - \hat d) + \hat t + \hat d
 * \f]
 *
 * \f{eqnarray*}{
 *      R &=& Rotation matrix in index space \\
 * \hat c &=& center of rotation in index space \\
 * \hat s &=& shift vector in index space \\
 *      Q &=& Rotation matrix in RAS (physical) space \\
 * \hat d &=& center of rotation in RAS (physical) space \\
 * \hat t &=& shift in index space \\
 *      A &=& Direction matrix of input image * spacing \\
 *      b &=& Origin of input image.
 * }
 *
 * From a rotation in index space
 * \f{eqnarray*}{
 *  Q &=& ARA^{-1} \\
 *  \hat t &=& -Q\hat b + Q\hat d - \hat d - AR\hat c + A\hat s + A\hat c + \hat b \\
 *  \f}
 *
 * @param in Source of index->world transform
 */
void Rigid3DTrans::toRASCoords(shared_ptr<const MRImage> in)
{
	if(ras_coord) {
		throw INVALID_ARGUMENT("Rigid3DTrans is already in Index Coordinates");
		return;
	}

	ras_coord = true;

	Matrix3d R, Q, A;
	Vector3d c, s, d, t, b;

	////////////////////
  // Orientation
  ////////////////////
	A = in->getDirection(); // direction*spacing
	// premultiply direction matrix with spacing
	for(size_t rr=0; rr<A.rows(); rr++) {
		for(size_t cc=0; cc<A.cols(); cc++) {
			A(rr,cc) *= in->spacing(cc);
		}
	}
	b = in->getOrigin(); // origin

	////////////////////////
  // Index Space Rotation
  ////////////////////////
	R = AngleAxisd(rotation[0], Vector3d::UnitX())*
		AngleAxisd(rotation[1], Vector3d::UnitY())*
		AngleAxisd(rotation[2], Vector3d::UnitZ());
	s = shift; // shift in index space
	c = center; // center in index space

	////////////////////////
  // RAS Space Rotation
  ////////////////////////
	in->indexToPoint(3, c.data(), d.data()); // center
	Q = A*R*A.inverse(); // rotation in RAS

	// compute shift in RAS
	t = Q*(d-b)+A*(s+c-R*c)+b-d;

	shift = t;
	center = d;
	rotation[1] = asin(Q(0,2));
	rotation[2] = -Q(0,1)/cos(rotation[1]);
	rotation[0] = -Q(1,2)/cos(rotation[1]);
};

/**
 * @brief Converts from world coordinates to index coordinates based on the
 * orientation stored in input image.
 *
 * The image center is either converted from RAS space to index space or, if
 * forcegridcenter is true, then the center of rotation is set to the center of
 * the grid ((SIZE-1)/2 in each dimension).
 *
 * \f{eqnarray*}{
 *      R &=& Rotation matrix in index space \\
 * \hat c &=& center of rotation in index space \\
 * \hat s &=& shift vector in index space \\
 *      Q &=& Rotation matrix in RAS (physical) space \\
 * \hat d &=& center of rotation in RAS (physical) space \\
 * \hat t &=& shift in index space \\
 *      A &=& Direction matrix of input image * spacing \\
 *      b &=& Origin of input image.
 * }
 *
 * The Rotation (\f$R\f$) and Shift(\f$ \hat s \f$) are given by:
 * \f{eqnarray*}{
 * R &=& A^{-1}QA \\
 * \hat s &=& R\hat c -\hat c - A^{-1}(\hat b + Q\hat b - Q\hat d + \hat t + \hat d )
 * \f}
 *
 * where \f$A\f$ is the rotation of the grid, \f$\hat c\f$ is the center of the
 * grid, \f$\hat b \f$ is the origin of the grid, \f$ Q \f$ is the rotation
 * matrix in RAS coordinate space, \f$ \hat d \f$ is the given center of roation
 * (in RAS coordinates), and \f$ \hat t \f$ is the original shift in RAS
 * coordinates.
 *
 * @param in Source of index->world transform
 * @param forcegridcenter Force the center to be the center of the grid rather
 * than using the location corresponding to the current center
 */
void Rigid3DTrans::toIndexCoords(shared_ptr<const MRImage> in,
		bool forcegridcenter)
{
	if(!ras_coord) {
		throw INVALID_ARGUMENT("Rigid3DTrans is already in Index Coordinates");
		return;
	}

	ras_coord = false;

	Matrix3d R, Q, A;
	Vector3d c, s, d, t, b;

	////////////////////
  // Orientation
  ////////////////////
	A = in->getDirection(); // direction*spacing
	// premultiply direction matrix with spacing
	for(size_t rr=0; rr<A.rows(); rr++) {
		for(size_t cc=0; cc<A.cols(); cc++) {
			A(rr,cc) *= in->spacing(cc);
		}
	}
	b = in->getOrigin(); // origin

	////////////////////////
  // RAS Space Rotation
  ////////////////////////
	Q = AngleAxisd(rotation[0], Vector3d::UnitX())*
		AngleAxisd(rotation[1], Vector3d::UnitY())*
		AngleAxisd(rotation[2], Vector3d::UnitZ());
	t = shift; // shift in ras space
	d = center; // center in ras space

	////////////////////////
  // Index Space Rotation
  ////////////////////////
	if(forcegridcenter) {
		// make center the image center
		for(size_t dd=0; dd < 3; dd++)
			c[dd] = (in->dim(dd)-1.)/2.;
	} else {
		// change center to index space
		in->pointToIndex(3, d.data(), c.data());
	}

	R = A.inverse()*Q*A; // rotation in RAS

	// compute shift in RAS
	s = A.inverse()*(Q*(b+A*c-d) + t+d-b) - c;

	shift = s;
	center = c;
	rotation[1] = asin(R(0,2));
	rotation[2] = -R(0,1)/cos(rotation[1]);
	rotation[0] = -R(1,2)/cos(rotation[1]);
	//	rotation = R.eulerAngles(0,1,2);
};

}

