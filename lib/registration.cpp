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

#include "registration.h"

using Eigen::VectorXd;

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
    
};


int computeRotationGrad(shared_ptr<const MRImage> fixed, 
        shared_ptr<const MRImage> fixed_deriv, 
        shared_ptr<const MRImage> moving, 
        shared_ptr<const MRImage> moving_deriv, 
        const VectorXd& params, VectorXd& grad)
{
    
};

int computeRotationValue(shared_ptr<const MRImage> fixed, 
        shared_ptr<const MRImage> moving, 
        const VectorXd& params, double& val)
{
    
};

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
Matrix4d corReg3D(shared_ptr<const MRImage> fixed, 
        shared_ptr<const MRImage> moving, size_t ref)
{
    // make sure the input image have matching properties
    if(!fixed->matchingOrient(moving, true))
        throw std::bad_argument("Input images have mismatching pixels in " +
                __NAME__);

    // compute input image derivatives
    

    // set up optimizer
    
    // run
    
    // apply parameters
};
