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

#define VERYDEBUG 1


using std::shared_ptr;

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
 * @{
 */

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
shared_ptr<MRImage> motionCorrect(shared_ptr<const MRImage> input, size_t ref);

/**
 * @brief Performs correlation based registration between two 3D volumes. note
 * that the two volumes should have identical sampling and identical
 * orientation. If that is not the case, an exception will be thrown.
 *
 * @param fixed     Image which will be the target of registration. 
 * @param moving    Image which will be rotated then shifted to match fixed.
 *
 * @return          4x4 Matrix, indicating rotation about the center then 
 *                  shift. Rotation matrix is the first 3x3 and shift is the
 *                  4th column.
 */
Eigen::Matrix4d corReg3D(shared_ptr<const MRImage> fixed, 
        shared_ptr<const MRImage> moving);


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
int cor3DDerivTest(double step, double tol, shared_ptr<const MRImage> in1,
        shared_ptr<const MRImage> in2);

/**
 * @brief The Rigid Corr Computer is used to compute the correlation
 * and gradient of correlation between two images. As the name implies, it 
 * is designed for 6 parameter rigid transforms.
 */
class RigidCorrComputer
{
    public:

    /**
     * @brief Constructor for the rigid correlation class. Note that 
     * rigid rotation is assumed to be about the center of the fixed 
     * image space. If necessary the input moving image will be resampled.
     * To the same space as the fixed image.
     *
     * @param fixed Fixed image. A copy of this will be made.
     * @param moving Moving image. A copy of this will be made.
     */
    RigidCorrComputer(shared_ptr<const MRImage> fixed,
            shared_ptr<const MRImage> moving);

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
     * @param value Value at the given rotation
     *
     * @return 0 if successful
     */
    int value(const Eigen::VectorXd& params, double& val);

    private:

    shared_ptr<MRImage> m_fixed;
    shared_ptr<MRImage> m_moving;
    shared_ptr<MRImage> m_dmoving;

    // for interpolating moving image, and iterating fixed
    LinInterp3DView<double> m_move_get;
    LinInterp3DView<double> m_dmove_get;
    NDConstIter<double> m_fit;

	double m_center[3];
    
#ifdef VERYDEBUG
    shared_ptr<MRImage> d_theta_x;
    shared_ptr<MRImage> d_theta_y;
    shared_ptr<MRImage> d_theta_z;
    shared_ptr<MRImage> d_shift_x;
    shared_ptr<MRImage> d_shift_y;
    shared_ptr<MRImage> d_shift_z;
    shared_ptr<MRImage> interpolated;
#endif

};

/** @} */

} // npl
#endif  //REGISTRATION_H


