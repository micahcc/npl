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
#include "mrimage_utils.h"

using Eigen::VectorXd;

#define VERYDEBUG 1

#ifdef VERYDEBUG
#define DEBUGWRITE(FOO) FOO 
#else
#define DEBUGWRITE(FOO) 
#endif

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
    
    return NULL;
};


int computeRotationGrad(
        shared_ptr<const MRImage> fixed, 
        shared_ptr<const MRImage> moving, 
        shared_ptr<const MRImage> moving_deriv, 
        const VectorXd& params, VectorXd& grad)
{
    if(!workspace) 
        workspace = dynamic_pointer_cast<MRImage>(fixed->copy());

    assert(workspace->ndim() == 3);
    assert(fixed->ndim() == 3);
    assert(moving->ndim() == 3);

    double rx = params[0];
    double ry = params[1];
    double rz = params[2];
    double sx = params[3];
    double sy = params[4];
    double sz = params[5];

#ifdef VERYDEBUG
    auto d_theta_x = dynamic_pointer_cast<MRImage>(moving->copy());
    auto d_theta_y = dynamic_pointer_cast<MRImage>(moving->copy());
    auto d_theta_z = dynamic_pointer_cast<MRImage>(moving->copy());
    auto d_shift_x = dynamic_pointer_cast<MRImage>(moving->copy());
    auto d_shift_y = dynamic_pointer_cast<MRImage>(moving->copy());
    auto d_shift_z = dynamic_pointer_cast<MRImage>(moving->copy());
    NDAccess<double> d_ang_x(d_theta_x);
    NDAccess<double> d_ang_y(d_theta_y);
    NDAccess<double> d_ang_z(d_theta_z);
    NDAccess<double> d_shi_x(d_shift_x);
    NDAccess<double> d_shi_y(d_shift_y);
    NDAccess<double> d_shi_z(d_shift_z);
#endif

	LinInterp3DView<double> lin(moving);
	Matrix4d m;
    m.setZero();

	// negate because we are starting from the destination and mapping from
	// the source, since we need to invert, it is necessary to reverse the
    // real order
	m.block<3,3>(0,0) = 
                AngleAxisd(-rz, Vector3d::UnitZ())*
                AngleAxisd(-ry,Vector3d::UnitY())*
                AngleAxisd(-rx,Vector3d::UnitX());
    m(0, 3) = sx;
    m(1, 3) = sy;
    m(2, 3) = sz;

    DEBUGWRITE(cerr << "Params: " << params.transpose() << " Matrix: " 
            << m << endl;);

    // for interpolating moving image
    LinInterp3DView<double> get_dmove(moving_deriv);
    LinInterp3DView<double> get_move(moving);

    // for computing roted indices
	Vector4d ind; ind.setOnes();
	Vector4d cind; cind.setOnes();
	Vector4d center; center.setZero();

    // compute center
	for(size_t ii=0; ii<3 && ii<moving->ndim(); ii++) 
		center[ii] = (moving->dim(ii)-1)/2.;

    //  Compute derivative Images (if debugging is enabled, otherwise just
    //  compute the gradient)
    grad.setZero();
    for(OrderIter<double> it(fixed); !it.eof(); ++it) {
		it.index(3, ind.array().data());
		cind = m*(ind-center)+center;
        
        // Here we compute dg(v(u,p))/dp, where g is the image, u is the
        // coordinate in the fixed image, and p is the param. 
        // dg/dp = SUM_i dg/dv_i dv_i/dp, where v is the rotated coordinate, so
        // dg/dv_i is the directional derivative in original space,
        // dv_i/dp is the derivative of the rotated coordinate system with
        // respect to a parameter
        double x = ind[0];
        double y = ind[1];
        double z = ind[2];

        double dg_dx = get_dmove(cind[0], cind[1], cind[2], 0);
        double dg_dy = get_dmove(cind[0], cind[1], cind[2], 1);
        double dg_dz = get_dmove(cind[0], cind[1], cind[2], 2);

        double dx_dRx = 0;
        double dx_dRy = -(sin(rx)*(y*cos(rz) + x*sin(rz))) - cos(rx)*(z*cos(ry)
                + sin(ry)*(-(x*cos(rz)) + y*sin(rz)));
        double dx_dRz = cos(rx)*(y*cos(rz) + x*sin(rz)) - sin(rx)*(z*cos(ry) +
                sin(ry)*(-(x*cos(rz)) + y*sin(rz)));

        double dy_dRx = z*cos(ry) + sin(ry)*(-(x*cos(rz)) + y*sin(rz));
        double dy_dRy = sin(rx)*(z*sin(ry) + cos(ry)*(x*cos(rz) - y*sin(rz)));
        double dy_dRz = -(cos(rx)*(z*sin(ry) + cos(ry)*(x*cos(rz) - y*sin(rz))));
        
        double dz_dRx = -(cos(ry)*(y*cos(rz) + x*sin(rz)));
        double dz_dRy = -(sin(rx)*sin(ry)*(y*cos(rz) + x*sin(rz))) +
                cos(rx)*(x*cos(rz) - y*sin(rz));
        double dz_dRz = cos(rx)*sin(ry)*(y*cos(rz) + x*sin(rz)) +
                sin(rx)*(x*cos(rz) - y*sin(rz));

        // compute SUM_i dg/dv_i dv_i/dp
        double dCdRx = dg_dx*dx_dRx + dg_dy*dy_dRx + dg_dz*dz_dRx;
        double dCdRy = dg_dx*dx_dRy + dg_dy*dy_dRy + dg_dz*dz_dRy;
        double dCdRz = dg_dx*dx_dRz + dg_dy*dy_dRz + dg_dz*dz_dRz;
        double dCdSx = dg_dx;
        double dCdSy = dg_dy;
        double dCdSz = dg_dz;

#ifdef VERYDEBUG
        d_ang_x.set(dCdRx, index);
        d_ang_y.set(dCdRy, index);
        d_ang_z.set(dCdRz, index);
        d_shi_x.set(dCdSx, index);
        d_shi_y.set(dCdSy, index);
        d_shi_z.set(dCdSz, index);
#endif
     
        grad[0] += (*it)*dCdRx;
        grad[1] += (*it)*dCdRy;
        grad[2] += (*it)*dCdRz;
        grad[3] += (*it)*dCdSx;
        grad[4] += (*it)*dCdSy;
        grad[5] += (*it)*dCdSz;
    }

    grad /= fixed->elements();

#ifdef VERYDEBUG
    d_theta_x->write("d_theta_x.nii.gz");
    d_theta_y->write("d_theta_y.nii.gz");
    d_theta_z->write("d_theta_z.nii.gz");
    d_shift_x->write("d_shift_x.nii.gz");
    d_shift_y->write("d_shift_y.nii.gz");
    d_shift_z->write("d_shift_z.nii.gz");
#endif

};

int computeRotationValue(
        shared_ptr<const MRImage> fixed, 
        shared_ptr<const MRImage> moving, 
        const VectorXd& params, double& val)
{
#ifdef VERYDEBUG
    auto workspace = dynamic_pointer_cast<MRImage>(fixed->copy());
    NDAccess<double> acc(workspace);
#endif 

    assert(workspace->ndim() == 3);
    assert(fixed->ndim() == 3);
    assert(moving->ndim() == 3);

    double rx = params[0];
    double ry = params[1];
    double rz = params[2];
    double sx = params[3];
    double sy = params[4];
    double sz = params[5];

	LinInterp3DView<double> lin(moving);
	Matrix4d m;
    m.setZero();

	// negate because we are starting from the destination and mapping from
	// the source, since we need to invert, it is necessary to reverse the
    // real order
	m.block<3,3>(0,0) = 
                AngleAxisd(-rz, Vector3d::UnitZ())*
                AngleAxisd(-ry,Vector3d::UnitY())*
                AngleAxisd(-rx,Vector3d::UnitX());
    m(0, 3) = sx;
    m(1, 3) = sy;
    m(2, 3) = sz;

    DEBUGWRITE(cerr << "Params: " << params.transpose() << " Matrix: " 
            << m << endl;);

	Vector4d ind; ind.setOnes();
	Vector4d cind; cind.setOnes();
	Vector4d center; center.setZero();

    // compute center
	for(size_t ii=0; ii<3 && ii<moving->ndim(); ii++) 
		center[ii] = (moving->dim(ii)-1)/2.;

    //  Resample Output and Compute Orientation. While actually resampling is
    //  optional, it helps with debugging
    double sum1 = 0;
    double sum2 = 0;
    double ss1 = 0;
    double ss2 = 0;
    double corr = 0;
	for(OrderIter<double> it(fixed); !it.eof(); ++it) {
		it.index(3, ind.array().data());
		cind = m*(ind-center)+center;
        
        double a = lin(cind[0], cind[1], cind[2]);
#ifdef VERYDEBUG
        acc.set(a, index);
#endif
        double b = *it;
        sum1 += a;
        ss1 += a*a;
        sum2 += b;
        ss2 += b*b;
        corr += a*b;
    }

    val = sample_corr(fixed->elements(), sum1, sum2, ss1, ss2, corr);

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
        throw std::bad_argument("Input images have mismatching pixels in\n" +
                __FUNCTION_STR__);

    std::vector<double> sigma_schedule({8, 4, 2});

    for(size_t ii=0; ii<sigma_schedule.size(); ii++) {
        // smooth and downsample input images
        auto tmpfixed = smoothDownsample(fixed, sigma_schedule[ii]);
        auto tmpmoving = smoothDownsample(tmpmoving, sigma_schedule[ii]);

        // compute derivatives
        auto dfixed = derivative(tmpfixed);
        auto dmoving = derivative(tmpmoving);
        DEBUGWRITE(dfixed->write("dfixed.nii.gz");
        DEBUGWRITE(dmoving->write("dfixed.nii.gz");

        // initialize optimizer

    }


    // set up optimizer
    
    // run
    
    // apply parameters
};
