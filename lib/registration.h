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
 * @file registration.h Tools for registering 2D or 3D images.
 *
 *****************************************************************************/

#ifndef REGISTRATION_H
#define REGISTRATION_H

#include "mrimage.h"
#include "accessors.h"
#include "iterators.h"
#include <Eigen/Dense>

#include <memory>

namespace npl {

/** \defgroup ImageRegistration Image registration algorithms
 *
 * Registration functions are implemented with classes linked to optimization
 * functions. All registration algorithms ultimately may be performed with
 * a simple function call, but the Computer classes (of which there is
 * currently just RigidCorrComputer) are exposed in case you want to use them
 * for your own registration algorithms.
 *
 * A computer is needed for every pair of Metric and Transform type.
 *
 */

/** @{ */
	
/**
 * @brief Information-based Metric to use
 */
enum Metric {METRIC_MI, METRIC_VI, METRIC_NMI, METRIC_COR};

/**
 * @brief The Rigid MI Computer is used to compute the mutual information
 * and gradient of mutual information between two images. As the name implies,
 * it is designed for 6 parameter rigid transforms.
 *
 * Note, if you want to register you should set the m_mindiff variable, so that
 * the negative of mutual information will be computed. Eventually this
 * functional will be somewhat generalized for all information-based metrics.
 */
class RigidInformationComputer
{
public:

	/**
	 * @brief Constructor for the rigid correlation class. Note that
	 * rigid rotation is assumed to be about the center of the fixed
	 * image space.  * If changes are made to the moving image, then call
	 * reinit() to reinitialize the image derivative.
	 *
	 * @param fixed Fixed image. A copy of this will be made.
	 * @param moving Moving image. A copy of this will be made.
	 * minimize negative correlation using a gradient descent).
	 * @param bins Number of bins during marginal density estimation (joint
	 *                  with have nbins*nbins)
	 * @param kernrad During parzen window, the radius of the smoothing kernel
	 * @param mindiff Whether to use negative correlation (for instance to
	 * register images)
	 */
	RigidInformationComputer(bool mindiff);

	/**
	 * @brief Computes the gradient and value of the correlation.
	 *
	 * @param params Paramters (Rx, Ry, Rz, Sx, Sy, Sz).
	 * @param val Value at the given rotation
	 * @param grad Gradient at the given rotation
	 *
	 * @return 0 if successful
	 */
	int valueGrad(const Eigen::VectorXd& params, double& val,
			Eigen::VectorXd& grad);

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
	int grad(const Eigen::VectorXd& params, Eigen::VectorXd& grad);

	/**
	 * @brief Computes the correlation.
	 *
	 * @param params Paramters (Rx, Ry, Rz, Sx, Sy, Sz).
	 * @param val Value at the given rotation
	 *
	 * @return 0 if successful
	 */
	int value(const Eigen::VectorXd& params, double& val);

	/**
	 * @brief Compute the difference of images (negate MI and NMI)
	 */
	bool m_compdiff;

	/**
	 * @brief Metric to use
	 */
	Metric m_metric;

	/**
	 * @brief Reallocates histograms and if m_fixed has been set, regenerates 
	 * histogram estimate of fixed pdf
	 *
	 * @param nbins Number of bins for marginal estimation
	 * @param krad Number of bins in kernel radius
	 */
	void setBins(size_t nbins, size_t krad);

	/**
	 * @brief Set the fixed image for registration/comparison
	 *
	 * @param fixed Input fixed image (not modified)
	 */
	void setFixed(ptr<const MRImage> fixed);

	/**
	 * @brief Return the current fixed image.
	 *
	 */
	ptr<const MRImage> getFixed() { return m_fixed; };

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
	void setMoving(ptr<const MRImage> moving);

	/**
	 * @brief Return the current moving image.
	 *
	 * Note that modification of the moving image outside this class without
	 * re-calling setMoving is undefined and will result in an out-of-date
	 * moving image derivative. THIS WILL BREAK GRADIENT CALCULATIONS.
	 */
	ptr<const MRImage> getMoving() { return m_moving; };


private:

	ptr<const MRImage> m_fixed;
	ptr<const MRImage> m_moving;
	ptr<MRImage> m_dmoving;

	/**
	 * @brief Number of bins in marginal
	 */
	int m_bins;

	/**
	 * @brief Parzen Window (kernel) radius
	 */
	int m_krad;

	// for interpolating moving image, and iterating fixed
	LinInterp3DView<double> m_move_get;
	LinInterp3DView<double> m_dmove_get;
	NDConstIter<double> m_fit;

	double m_center[3];

	NDArrayStore<1, double> m_pdfmove;
	NDArrayStore<1, double> m_pdffix;

	NDArrayStore<2, double> m_pdfjoint;
	NDArrayStore<3, double> m_dpdfjoint;
	NDArrayStore<2, double> m_dpdfmove;

	vector<double> m_gradHjoint;
	vector<double> m_gradHmove;

	double m_Hfix;
	double m_Hmove;
	double m_Hjoint;

	double m_rangemove[2];
	double m_rangefix[2];
	double m_wmove;
	double m_wfix;
};

/**
 * @brief The Rigid Corr Computer is used to compute the correlation
 * and gradient of correlation between two images. As the name implies, it
 * is designed for 6 parameter rigid transforms.
 *
 * Requires that both setFixed and setMoving be set
 *
 * Note that if you want to use this for registration, you should set m_mindiff
 * to get the negative of correlation.
 */
class RigidCorrComputer
{
public:

	/**
	 * @brief Constructor for the rigid correlation class. Note that
	 * rigid rotation is assumed to be about the center of the fixed
	 * image space. Also note that changed to the input images by the outside
	 * will * be reflected in the registration images HOWEVER you need to call
	 * reinit() if you change the inputs, otherwise the image gradients will
	 * be incorrect.
	 *
	 * @param mindiff Whether to use negative correlation (for instance to
	 * minimize negative correlation using a gradient descent).
	 */
	RigidCorrComputer(bool mindiff);

	/**
	 * @brief Computes the gradient and value of the correlation.
	 *
	 * @param params Paramters (Rx, Ry, Rz, Sx, Sy, Sz).
	 * @param val Value at the given rotation
	 * @param grad Gradient at the given rotation
	 *
	 * @return 0 if successful
	 */
	int valueGrad(const Eigen::VectorXd& params, double& val,
			Eigen::VectorXd& grad);

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
	int grad(const Eigen::VectorXd& params, Eigen::VectorXd& grad);

	/**
	 * @brief Computes the correlation.
	 *
	 * @param params Paramters (Rx, Ry, Rz, Sx, Sy, Sz).
	 * @param val Value at the given rotation
	 *
	 * @return 0 if successful
	 */
	int value(const Eigen::VectorXd& params, double& val);

	/**
	 * @brief Set the fixed image for registration/comparison
	 *
	 * @param fixed Input fixed image (not modified)
	 */
	void setFixed(ptr<const MRImage> fixed);

	/**
	 * @brief Return the current fixed image.
	 *
	 */
	ptr<const MRImage> getFixed() { return m_fixed; };

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
	void setMoving(ptr<const MRImage> moving);

	/**
	 * @brief Return the current moving image.
	 *
	 * Note that modification of the moving image outside this class without
	 * re-calling setMoving is undefined and will result in an out-of-date
	 * moving image derivative. THIS WILL BREAK GRADIENT CALCULATIONS.
	 */
	ptr<const MRImage> getMoving() { return m_moving; };

	/**
	 * @brief Negative of correlation (which will make it work with most
	 * optimizers)
	 */
	bool m_compdiff;
private:

	ptr<const MRImage> m_fixed;
	ptr<const MRImage> m_moving;
	ptr<MRImage> m_dmoving;

	// for interpolating moving image, and iterating fixed
	LinInterp3DView<double> m_move_get;
	LinInterp3DView<double> m_dmove_get;
	NDConstIter<double> m_fit;

	double m_center[3];

};

/**
 * @brief The distortion correction MI Computer is used to compute the mutual
 * information and gradient of mutual information between two images using 
 * nonrigid, unidirectional, B-spline transform. 
 *
 * Note, if you want to register you should set the m_mindiff variable, so that
 * the negative of mutual information will be computed. Eventually this
 * functional will be somewhat generalized for all information-based metrics.
 */
class DistortionCorrectionInformationComputer
{
public:

	/**
	 * @brief Constructor for the rigid correlation class. Note that
	 * rigid rotation is assumed to be about the center of the fixed
	 * image space.  * If changes are made to the moving image, then call
	 * reinit() to reinitialize the image derivative.
	 *
	 * @param fixed Fixed image. A copy of this will be made.
	 * @param moving Moving image. A copy of this will be made.
	 * minimize negative correlation using a gradient descent).
	 * @param bins Number of bins during marginal density estimation (joint
	 *                  with have nbins*nbins)
	 * @param kernrad During parzen window, the radius of the smoothing kernel
	 * @param mindiff Whether to use negative correlation (for instance to
	 * register images)
	 */
	DistortionCorrectionInformationComputer(bool mindiff);

	/**
	 * @brief Computes the gradient and value of the correlation.
	 *
	 * @param params Paramters (Rx, Ry, Rz, Sx, Sy, Sz).
	 * @param val Value at the given rotation
	 * @param grad Gradient at the given rotation
	 *
	 * @return 0 if successful
	 */
	int valueGrad(const Eigen::VectorXd& params, double& val,
			Eigen::VectorXd& grad);

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
	int grad(const Eigen::VectorXd& params, Eigen::VectorXd& grad);

	/**
	 * @brief Computes the correlation.
	 *
	 * @param params Paramters (Rx, Ry, Rz, Sx, Sy, Sz).
	 * @param val Value at the given rotation
	 *
	 * @return 0 if successful
	 */
	int value(const Eigen::VectorXd& params, double& val);

	/**
	 * @brief Phase encode (distortion) dimensions
	 */
	int m_dir;

	/**
	 * @brief Compute the difference of images (negate MI and NMI)
	 */
	bool m_compdiff;

	/**
	 * @brief Thin-plate spline regularization weight
	 */
	double m_tps_reg;
	
	/**
	 * @brief Jacobian regularization weight
	 */
	double m_jac_reg;

	/**
	 * @brief Metric to use
	 */
	Metric m_metric;

	/**
	 * @brief Reallocates histograms and if m_fixed has been set, regenerates 
	 * histogram estimate of fixed pdf
	 *
	 * @param nbins Number of bins for marginal estimation
	 * @param krad Number of bins in kernel radius
	 */
	void setBins(size_t nbins, size_t krad);

	/**
	 * @brief Initializes knot spacing and if m_fixed has been set, then
	 * initializes the m_deform image.
	 *
	 * @param space Spacing between knots, in physical coordinates
	 */
	void setKnotSpacing(double space);

	/**
	 * @brief Set the fixed image for registration/comparison
	 *
	 * @param fixed Input fixed image (not modified)
	 */
	void setFixed(ptr<const MRImage> fixed);

	/**
	 * @brief Return the current fixed image.
	 *
	 */
	ptr<const MRImage> getFixed() { return m_fixed; };

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
	void setMoving(ptr<const MRImage> moving);

	/**
	 * @brief Return the current moving image.
	 *
	 * Note that modification of the moving image outside this class without
	 * re-calling setMoving is undefined and will result in an out-of-date
	 * moving image derivative. THIS WILL BREAK GRADIENT CALCULATIONS.
	 */
	ptr<const MRImage> getMoving() { return m_moving; };


private:

	/**
	 * @brief Computes the thin-plate spline regulization value and gradient
	 *
	 * @param val Output Value
	 * @param grad Output Gradient
	 */
	int thinPlateSpline(double& val, VectorXd& grad);

	/**
	 * @brief Computes the thin-plate spline regulization value 
	 *
	 * @param val Output Value
	 */
	int thinPlateSpline(double& val);

	/**
	 * @brief Computes the sum of the determinant value and gradient
	 *
	 * @param val Output Value
	 * @param grad Output Gradient
	 */
	int jacobianDet(double& val, VectorXd& grad);

	/**
	 * @brief Computes the sum of the jacobian determinant value 
	 *
	 * @param val Output Value
	 */
	int jacobianDet(double& val);

	/**
	 * @brief Computes the metric value and gradient
	 *
	 * @param val Output Value
	 * @param grad Output Gradient
	 */
	int metric(double& val, VectorXd& grad);

	/**
	 * @brief Computes the metric value
	 *
	 * @param val Output Value
	 */
	int metric(double& val);

	/* Variables: 
	 *
	 * m_bins, m_krad
	 *
	 */

	ptr<const MRImage> m_fixed;
	ptr<const MRImage> m_moving;
	ptr<MRImage> m_dmoving;
	ptr<MRImage> m_deform;

	/**
	 * @brief Number of bins in marginal
	 */
	int m_bins;

	/**
	 * @brief Parzen Window (kernel) radius
	 */
	int m_krad;

	/**
	 * @brief Spacing between knots in mm
	 */
	double m_knotspace;

	// for interpolating moving image, and iterating fixed
	VectorXd gradbuff;
	LinInterp3DView<double> m_move_get;
	LinInterp3DView<double> m_dmove_get;
	NDConstIter<double> m_fit;
	NDIter<double> m_dit;

	/**
	 * @brief Histogram of moving image (initialized by setBins)
	 */
	NDArrayStore<1, float> m_pdfmove;

	/**
	 * @brief Histogram of fixed image, (initialized by setBins)
	 */
	NDArrayStore<1, float> m_pdffix;

	/**
	 * @brief X - fixed PDF, Y - moving PDF (initialized by setBins)
	 */
	NDArrayStore<2, float> m_pdfjoint;

	/**
	 * @brief First 3 Dimensions match dimensions in m_field, last two match
	 * the dimensios of m_pdfjoint, (initialized by setKnotSpacing/setFixed)
	 */
	MRImageStore<5, float> m_dpdfjoint;
	
	/**
	 * @brief First 3 Dimensions match dimensions in m_field, last matches
	 * m_pdfmove, (initialized by setKnotSpacing/setFixed)
	 */
	MRImageStore<4, float> m_dpdfmove;

	/**
	 * @brief Gradient of joint entropy at each knot. 
	 * (initialized by setKnotSpacing/setFixed)
	 */
	MRImageStore<3, float> m_gradHjoint;
	
	/**
	 * @brief Gradient of marginal entropy at each knot
	 * (initialized by setKnotSpacing/setFixed)
	 */
	MRImageStore<3, float> m_gradHmove;

	/**
	 * @brief Entropy of fixed image (initialized by setFixed)
	 */
	double m_Hfix; 
	
	/**
	 * @brief Entropy of moving image (updated by metric())
	 */
	double m_Hmove;
	
	/**
	 * @brief Entropy of joint image (updated by metric())
	 */
	double m_Hjoint;

	/**
	 * @brief Min-Max values in moving image (updated in metric())
	 */
	double m_rangemove[2];

	/**
	 * @brief min-max values in fixed image (initialized in setFixed)
	 */
	double m_rangefix[2];

	/**
	 * @brief Width of bins in the marginal distribution for the moving image
	 * (updated in metric())
	 */
	double m_wmove;
	
	/**
	 * @brief Width of bins in the marginal distribution for the fixed image
	 * (initialized in setFixed)
	 */
	double m_wfix;
};


/**
 * @brief Struct for holding information about a rigid transform. Note that
 * rotation R = Rx*Ry*Rz, where Rx, Ry, and Rz are the rotations about x, y and
 * z aaxes, and the angles are stored (in radians) in the rotation member.
 *
 * \f$ \hat y = R(\hat x- \hat c)+ \hat s+ \hat c \f$
 *
 */
struct Rigid3DTrans
{
	Vector3d rotation; //Rx, Ry, Rz
	Vector3d shift;
	Vector3d center;

	/**
	 * @brief Indicates the stored transform is relative to physical coordintes
	 * rather than index coordinates
	 */
	bool ras_coord;

	Rigid3DTrans() {
		ras_coord = true;
		rotation.setZero();
		shift.setZero();
		center.setZero();
	};

	/**
	 * @brief Inverts rigid transform, where:
	 *
	 * Original:
	 * \f$ \hat y = R(\hat x- \hat c)+ \hat s+ \hat c \f$
	 *
	 * Inverse:
	 * \f$ \hat x = R^{-1}(\hat y - \hat s - \hat c) + \hat c \f$
	 *
	 * So the new parameters, interms of the old are:
	 * \f[ \hat c' = \hat s+ \hat c \f]
	 * \f[ \hat s' = -\hat s \f]
	 * \f[ \hat R' = R^{-1} \f]
	 *
	 */
	void invert();

	/**
	 * @brief Constructs and returns rotation Matrix.
	 *
	 * @return Rotation matrix
	 */
	Matrix3d rotMatrix();

	/**
	 * @brief Converts to world coordinates based on the orientation stored in
	 * input image.
	 *
	 * For a rotation in RAS coordinates, with rotation \f$Q\f$, shift \f$t\f$ and
	 * center \f$d\f$:
	 *
	 * \f[
	 *   \hat u = Q(A\hat x + \hat b - \hat d) + \hat t + \hat d
	 * \f]
	 *
	 * From a rotation in index space with rotation \f$R\f$, shift \f$s\f$ and
	 * center \f$c\f$:
	 * \f{eqnarray*}{
	 *  Q &=& A^{-1}AR \\
	 *  \hat t &=& -Q\hat b + Q\hat d - \hat d - AR\hat c + A\hat s + A\hat c + \hat b \\
	 * \f}
	 *
	 * @param in Source of index->world transform
	 */
	void toRASCoords(ptr<const MRImage> in);

	/**
	 * @brief Converts from world coordinates to index coordinates based on the
	 * orientation stored in input image.
	 *
	 * The center of rotation is assumed to be the center of the grid which is
	 * (SIZE-1)/2 in each dimension.
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
	 * @param forcegridcenter Force the center to be the center of the grid
	 * rather than using the location corresponding to the current center
	 */
	void toIndexCoords(ptr<const MRImage> in, bool forcegridcenter);

};

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
ptr<MRImage> motionCorrect(ptr<const MRImage> input, size_t ref);

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
Rigid3DTrans corReg3D(ptr<const MRImage> fixed, ptr<const MRImage> moving,
		const std::vector<double>& sigmas);

/**
 * @brief Performs information-based registration between two 3D volumes. note
 * that the two volumes should have identical sampling and identical
 * orientation. If that is not the case, an exception will be thrown.
 *
 * @param fixed     Image which will be the target of registration.
 * @param moving    Image which will be rotated then shifted to match fixed.
 * @param sigmas    Standard deviation of smoothing kernel at each level
 * @param nbins     Number of bins during marginal density estimation (joint
 *                  with have nbins*nbins)
 * @param binradius During parzen window, the radius of the smoothing kernel
 *
 * @return          Rigid transform.
 */
Rigid3DTrans informationReg3D(ptr<const MRImage> fixed,
		ptr<const MRImage> moving, const std::vector<double>& sigmas,
		size_t nbins = 128, size_t binradius = 4);

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
int cor3DDerivTest(double step, double tol, ptr<const MRImage> in1,
		ptr<const MRImage> in2);

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
		ptr<const MRImage> in1, ptr<const MRImage> in2);

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
int distcorDerivTest(double step, double tol,
		shared_ptr<const MRImage> in1, shared_ptr<const MRImage> in2);

/**
 * @brief Prints a rigid transform
 *
 * @param stream Output stream
 * @param rigid Rigid transform
 *
 * @return after this is inserted, stream
 */
std::ostream& operator<< (std::ostream& stream, const Rigid3DTrans& rigid)
{
	stream << "Rigid3DTrans ";
	if(rigid.ras_coord)
		stream << "(In RAS)\n";
	else
		stream << "(In Index)\n";

	stream << "Rotation: ";
	for(size_t ii=0; ii<2; ii++)
		stream << rigid.rotation[ii] << ", ";
	stream << rigid.rotation[2] << "\n";

	stream << "Center: ";
	for(size_t ii=0; ii<2; ii++)
		stream << rigid.center[ii] << ", ";
	stream << rigid.center[2] << "\n";

	stream << "Shift : ";
	for(size_t ii=0; ii<2; ii++)
		stream << rigid.shift[ii] << ", ";
	stream << rigid.shift[2] << "\n";

	return stream;
}

//// Standard Typedefs, because some the names are long /////
typedef RigidInformationComputer RigidInforComp;
typedef RigidCorrComputer RigidCorrComp;
typedef DistortionCorrectionInformationComputer DCInforComp;

/** @} */

} // npl
#endif  //REGISTRATION_H


