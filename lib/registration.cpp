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
    RigidCorrComputer comp(in1, in2, false);

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
Rigid3DTrans corReg3D(shared_ptr<const MRImage> fixed, 
        shared_ptr<const MRImage> moving, 
        const std::vector<double>& sigmas)
{
    using namespace std::placeholders;
    using std::bind;

    Rigid3DTrans rigid;

    // make sure the input image has matching properties
    if(!fixed->matchingOrient(moving, true))
        throw std::invalid_argument("Input images have mismatching pixels "
                "in\n" + __FUNCTION_STR__);

    
    for(size_t ii=0; ii<sigmas.size(); ii++) {
        // smooth and downsample input images
        auto sm_fixed = smoothDownsample(fixed, sigmas[ii]);
        auto sm_moving = smoothDownsample(moving, sigmas[ii]);
        DEBUGWRITE(sm_fixed->write("smooth_fixed_"+to_string(ii)+".nii.gz"));
        DEBUGWRITE(sm_moving->write("smooth_moving_"+to_string(ii)+".nii.gz"));

        RigidCorrComputer comp(sm_fixed, sm_moving, true);
        
        // create value and gradient functions
        auto vfunc = bind(&RigidCorrComputer::value, &comp, _1, _2);
        auto vgfunc = bind(&RigidCorrComputer::valueGrad, &comp, _1, _2, _3);
        auto gfunc = bind(&RigidCorrComputer::grad, &comp, _1, _2);

        // initialize optimizer
        LBFGSOpt opt(6, vfunc, gfunc, vgfunc);
        opt.stop_Its = 10000;
        opt.stop_X = 0.0001;
        opt.stop_G = 0;
        opt.stop_F = 0;
        
        cerr << "Init Rigid: " << ii << endl;
        cerr << "Rotation: " << rigid.rotation.transpose() << endl;
        cerr << "Center : " << rigid.center.transpose() << endl;
        cerr << "Shift : " << rigid.shift.transpose() << endl;

        // grab the parameters from the previous iteration (or initialized)
        rigid.toIndexCoords(sm_moving, true);
        for(size_t ii=0; ii<3; ii++) {
            opt.state_x[ii] = rigid.rotation[ii];
            opt.state_x[ii+3] = rigid.shift[ii];
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
            rigid.rotation[ii] = opt.state_x[ii];
            rigid.shift[ii] = opt.state_x[ii+3];
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


/*********************************************************************
 * Registration Class Implementations
 ********************************************************************/

/**
 * @brief Constructor for the rigid correlation class. Note that 
 * rigid rotation is assumed to be about the center of the fixed 
 * image space. If necessary the input moving image will be resampled.
 * To the same space as the fixed image.
 *
 * @param fixed Fixed image. A copy of this will be made.
 * @param moving Moving image. A copy of this will be made.
 * @param negate Whether to use negative correlation (for instance to
 * minimize negative correlation using a gradient descent).
 */
RigidCorrComputer::RigidCorrComputer(
        shared_ptr<const MRImage> fixed, shared_ptr<const MRImage> moving,
        bool negate) :
    m_fixed(dPtrCast<MRImage>(fixed->copy())),
    m_moving(dPtrCast<MRImage>(moving->copy())),
    m_dmoving(dPtrCast<MRImage>(derivative(moving))),
    m_move_get(m_moving),
    m_dmove_get(m_dmoving),
    m_fit(m_fixed),
    m_negate(negate)
{
    if(fixed->ndim() != 3)
        throw INVALID_ARGUMENT("Fixed image is not 3D!");
    if(moving->ndim() != 3)
        throw INVALID_ARGUMENT("Moving image is not 3D!");

#ifdef VERYDEBUG
    m_moving->write("init_moving.nii.gz");
    m_fixed->write("init_fixed.nii.gz");
    d_theta_x = dPtrCast<MRImage>(moving->copy());
    d_theta_y = dPtrCast<MRImage>(moving->copy());
    d_theta_z = dPtrCast<MRImage>(moving->copy());
    d_shift_x = dPtrCast<MRImage>(moving->copy());
    d_shift_y = dPtrCast<MRImage>(moving->copy());
    d_shift_z = dPtrCast<MRImage>(moving->copy());
    interpolated = dPtrCast<MRImage>(moving->copy());
    interpolated->write("init_interpolated.nii.gz");
    callcount = 0;
#endif
	
    for(size_t ii=0; ii<3 && ii<moving->ndim(); ii++) 
		m_center[ii] = (m_moving->dim(ii)-1)/2.;
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
    double rx = params[0];
    double ry = params[1];
    double rz = params[2];
    double sx = params[3];
    double sy = params[4];
    double sz = params[5];

#ifdef VERYDEBUG
    cerr << "ValGrad()" << endl;
    Pixel3DView<double> d_ang_x(d_theta_x);
    Pixel3DView<double> d_ang_y(d_theta_y);
    Pixel3DView<double> d_ang_z(d_theta_z);
    Pixel3DView<double> d_shi_x(d_shift_x);
    Pixel3DView<double> d_shi_y(d_shift_y);
    Pixel3DView<double> d_shi_z(d_shift_z);
    Pixel3DView<double> acc(interpolated);
#endif

    // for computing roted indices
	double ind[3];
	double cind[3];

    //  Compute derivative Images (if debugging is enabled, otherwise just
    //  compute the gradient)
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
        
        cind[0] = cx - (cx + sx - x)*cos(ry)*cos(rz) + cos(rz)*((cz + sz -
                    z)*cos(rx) - (cy + sy - y)*sin(rx))*sin(ry) - ((cy + sy -
                    y)*cos(rx) + (cz + sz - z)*sin(rx))*sin(rz);
        cind[1] = cy + cos(rz)*((-cy - sy + y)*cos(rx) - (cz + sz - z)*sin(rx))
                    + (cx + sx - x)*cos(ry)*sin(rz) + ((-cz - sz + z)*cos(rx) +
                    (cy + sy - y)*sin(rx))*sin(ry)*sin(rz);
        cind[2] = cz - (cz + sz - z)*cos(rx)*cos(ry) + (cy + sy -
                    y)*cos(ry)*sin(rx) - (cx + sx - x)*sin(ry);

        // Here we compute dg(v(u,p))/dp, where g is the image, u is the
        // coordinate in the fixed image, and p is the param. 
        // dg/dp = SUM_i dg/dv_i dv_i/dp, where v is the rotated coordinate, so
        // dg/dv_i is the directional derivative in original space,
        // dv_i/dp is the derivative of the rotated coordinate system with
        // respect to a parameter
        double dg_dx = m_dmove_get(cind[0], cind[1], cind[2], 0);
        double dg_dy = m_dmove_get(cind[0], cind[1], cind[2], 1);
        double dg_dz = m_dmove_get(cind[0], cind[1], cind[2], 2);

        double dx_dRx = -(cos(rz)*((cy + sy - y)*cos(rx) + (cz + sz -
                    z)*sin(rx))*sin(ry)) + ((-cz - sz + z)*cos(rx) + (cy +
                    sy - y)*sin(rx))*sin(rz);
        double dy_dRx =  cos(rz)*((-cz - sz + z)*cos(rx) + (cy + sy -
                    y)*sin(rx)) + ((cy + sy - y)*cos(rx) + (cz + sz -
                    z)*sin(rx))*sin(ry)*sin(rz);
        double dz_dRx = cos(ry)*((cy + sy - y)*cos(rx) + (cz + sz -
                    z)*sin(rx));

        double dx_dRy = cos(rz)*((cz + sz - z)*cos(rx)*cos(ry) - (cy + sy -
                    y)*cos(ry)*sin(rx) + (cx + sx - x)*sin(ry));
        double dy_dRy = (-((cz + sz - z)*cos(rx)*cos(ry)) + (cy + sy -
                    y)*cos(ry)*sin(rx) - (cx + sx - x)*sin(ry))*sin(rz);
        double dz_dRy = (-cx - sx + x)*cos(ry) + ((cz + sz - z)*cos(rx) - (cy +
                    sy - y)*sin(rx))*sin(ry);

        double dx_dRz = (-cz - sz + z)*cos(rz)*sin(rx) + ((cx + sx - x)*cos(ry)
                    + (cy + sy - y)*sin(rx)*sin(ry))*sin(rz) - cos(rx)*((cy +
                    sy - y)*cos(rz) + (cz + sz - z)*sin(ry)*sin(rz));
        double dy_dRz = (cx + sx - x)*cos(ry)*cos(rz) + cos(rz)*((-cz - sz +
                    z)*cos(rx) + (cy + sy - y)*sin(rx))*sin(ry) + ((cy + sy -
                    y)*cos(rx) + (cz + sz - z)*sin(rx))*sin(rz);
        double dz_dRz = 0;

        // derivative of coordinate system due 
        double dx_dSx = -(cos(ry)*cos(rz));
        double dy_dSx = cos(ry)*sin(rz);
        double dz_dSx = -sin(ry);
        
        double dx_dSy = -(cos(rz)*sin(rx)*sin(ry)) - cos(rx)*sin(rz);
        double dy_dSy = -(cos(rx)*cos(rz)) + sin(rx)*sin(ry)*sin(rz);
        double dz_dSy = cos(ry)*sin(rx);
        
        double dx_dSz = cos(rx)*cos(rz)*sin(ry) - sin(rx)*sin(rz);
        double dy_dSz = -(cos(rz)*sin(rx)) - cos(rx)*sin(ry)*sin(rz);
        double dz_dSz = -(cos(rx)*cos(ry));

        // compute SUM_i dg/dv_i dv_i/dp
        double dCdRx = dg_dx*dx_dRx + dg_dy*dy_dRx + dg_dz*dz_dRx;
        double dCdRy = dg_dx*dx_dRy + dg_dy*dy_dRy + dg_dz*dz_dRy;
        double dCdRz = dg_dx*dx_dRz + dg_dy*dy_dRz + dg_dz*dz_dRz;

        double dCdSx = dg_dx*dx_dSx + dg_dy*dy_dSx + dg_dz*dz_dSx;
        double dCdSy = dg_dx*dx_dSy + dg_dy*dy_dSy + dg_dz*dz_dSy;
        double dCdSz = dg_dx*dx_dSz + dg_dy*dy_dSz + dg_dz*dz_dSz;
        
        // compute correlation, since it requires almost no additional work
        double g = m_move_get(cind[0], cind[1], cind[2]);
        double f = *m_fit;
        
        mov_sum += g;
        fix_sum += f;
        mov_ss += g*g;
        fix_ss += f*f;
        corr += g*f;

#ifdef VERYDEBUG
        d_ang_x.set(dCdRx, ind[0], ind[1], ind[2]);
        d_ang_y.set(dCdRy, ind[0], ind[1], ind[2]);
        d_ang_z.set(dCdRz, ind[0], ind[1], ind[2]);
        d_shi_x.set(dCdSx, ind[0], ind[1], ind[2]);
        d_shi_y.set(dCdSy, ind[0], ind[1], ind[2]);
        d_shi_z.set(dCdSz, ind[0], ind[1], ind[2]);
        acc.set(g, ind[0], ind[1], ind[2]);
        assert(acc(ind[0], ind[1], ind[2]) == g);
#endif
     
        grad[0] += (*m_fit)*dCdRx;
        grad[1] += (*m_fit)*dCdRy;
        grad[2] += (*m_fit)*dCdRz;
        grad[3] += (*m_fit)*dCdSx;
        grad[4] += (*m_fit)*dCdSy;
        grad[5] += (*m_fit)*dCdSz;
     
    }

    count = m_fixed->elements();
    val = sample_corr(count, mov_sum, fix_sum, mov_ss, fix_ss, corr);
    double sd1 = sqrt(sample_var(count, mov_sum, mov_ss));
    double sd2 = sqrt(sample_var(count, fix_sum, fix_ss));
    grad /= (count-1)*sd1*sd2;

    if(m_negate) {
        grad = -grad;
        val = -val;
    }

#ifdef VERYDEBUG
    cerr << "Value: " << val << endl;
    cerr << "Gradient: " << grad.transpose() << endl;
    string sc = "_"+to_string(callcount);
    d_theta_x->write("d_theta_x"+sc+".nii.gz");
    d_theta_y->write("d_theta_y"+sc+".nii.gz");
    d_theta_z->write("d_theta_z"+sc+".nii.gz");
    d_shift_x->write("d_shift_x"+sc+".nii.gz");
    d_shift_y->write("d_shift_y"+sc+".nii.gz");
    d_shift_z->write("d_shift_z"+sc+".nii.gz");
    interpolated->write("interp"+sc+".nii.gz");
    callcount++;
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
#ifdef VERYDEBUG
    cerr << "Val()" << endl;
    Pixel3DView<double> acc(interpolated);
#endif 

    assert(m_fixed->ndim() == 3);
    assert(m_moving->ndim() == 3);

    double rx = params[0];
    double ry = params[1];
    double rz = params[2];
    double sx = params[3];
    double sy = params[4];
    double sz = params[5];

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
        
        cind[0] = cx - (cx + sx - x)*cos(ry)*cos(rz) + cos(rz)*((cz + sz -
                    z)*cos(rx) - (cy + sy - y)*sin(rx))*sin(ry) - ((cy + sy -
                    y)*cos(rx) + (cz + sz - z)*sin(rx))*sin(rz);
        cind[1] = cy + cos(rz)*((-cy - sy + y)*cos(rx) - (cz + sz - z)*sin(rx))
                    + (cx + sx - x)*cos(ry)*sin(rz) + ((-cz - sz + z)*cos(rx) +
                    (cy + sy - y)*sin(rx))*sin(ry)*sin(rz);
        cind[2] = cz - (cz + sz - z)*cos(rx)*cos(ry) + (cy + sy -
                    y)*cos(ry)*sin(rx) - (cx + sx - x)*sin(ry);
        
        double a = m_move_get(cind[0], cind[1], cind[2]);
#ifdef VERYDEBUG
        acc.set(a, ind[0], ind[1], ind[2]);
#endif
        double b = *m_fit;
        sum1 += a;
        ss1 += a*a;
        sum2 += b;
        ss2 += b*b;
        corr += a*b;
    }

    val = sample_corr(m_fixed->elements(), sum1, sum2, ss1, ss2, corr);
    if(m_negate)
        val = -val;

#ifdef VERYDEBUG
    DEBUGWRITE(cerr << "Value: " << val << endl);
    string sc = "_"+to_string(callcount);
    interpolated->write("interp"+sc+".nii.gz");
    callcount++;
#endif 
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
 *
 * So the new parameters, interms of the old are:
 * New c = s+c
 * New s = -s
 * New R = R^-1
 * 
 */
void Rigid3DTrans::invert() 
{
        auto tmp_shift = shift;
        auto tmp_center = center;
        auto tmp_rotation = rotation;
        shift = -tmp_shift;
        center = tmp_center+tmp_shift;
        rotation[2] = -tmp_rotation[0];
        rotation[1] = -tmp_rotation[1];
        rotation[0] = -tmp_rotation[2];
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
    rotation = Q.eulerAngles(0,1,2);
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
    rotation = R.eulerAngles(0,1,2);
};

}

