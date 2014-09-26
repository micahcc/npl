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
 * @file skullstrip.cpp Experimental skull stripping algorithm based on point
 * cloud density
 *
 *****************************************************************************/

#include <string>

#include <tclap/CmdLine.h>
#include "nplio.h"
#include "mrimage.h"
#include "iterators.h"
#include "accessors.h"
#include "ndarray_utils.h"
#include "version.h"
#include "macros.h"

#include <Eigen/SVD>

using namespace std;
using namespace npl;
using Eigen::JacobiSVD;
using Eigen::ComputeFullV;
using Eigen::ComputeFullU;

///**
// * @brief 
// *
// * @param size_t Number of dimensions
// * @param size_t Dimensions (size) of image
// * @param int64_t ND Index
// */
//typedef std::function<void(size_t, size_t*, int64_t*) NDFunction;
//
///**
// * @brief 
// *
// * @param size_t Number of dimensions
// * @param size_t Dimensions (size) of image
// * @param int64_t ND Index
// * @param int64_t Offset
// * @param int64_t Clamped Offset
// */
//typedef std::function<void(size_t, size_t*, int64_t*, int64_t*, int64_t*) NDKFunction;
//
//template <>
//int foreachND<3>(NDFunction foo, size_t* sz)
//{
//    int64_t index[3];
//    int64_t clamped[3];
//    for(int64_t index[0]=0; index[0] < sz[0]; index[0]++) {
//    for(int64_t index[1]=0; index[1] < sz[1]; index[1]++) {
//    for(int64_t index[2]=0; index[2] < sz[2]; index[2]++) {
//        foo(3, sz, index);
//    }
//    }
//    }
//}
//
//template <size_t D>
//int foreachKernND(NDKFunction foo, size_t* sz, int64_t radius)
//{
//}
//
//template <>
//int foreachNDKernel<3>(NDKFunction foo, size_t* sz, int64_t radius)
//{
//    int64_t index[1];
//    int64_t offset[1];
//    int64_t clamped[1];
//    for(int64_t index[0]=0; index[0] < sz[0]; index[0]++) {
//    for(int64_t index[1]=0; index[1] < sz[1]; index[1]++) {
//    for(int64_t index[2]=0; index[2] < sz[2]; index[2]++) {
//        for(int64_t offset[0]=-radius; offset[0] <= radius; offset[0]++) {
//        for(int64_t offset[1]=-radius; offset[1] <= radius; offset[1]++) {
//        for(int64_t offset[2]=-radius; offset[2] <= radius; offset[2]++) {
//            clamped[0] = clamp<int64_t>(0, sz[0]-1, index[0] + offset[0]);
//            clamped[1] = clamp<int64_t>(0, sz[1]-1, index[1] + offset[1]);
//            clamped[1] = clamp<int64_t>(0, sz[2]-1, index[2] + offset[2]);
//            foo(3, sz, index, offset, clamped);
//        }
//        }
//        }
//    }
//    }
//    }
//}
//
/**
 * @brief Generates a point cloud, stored in a KDTree based on locally large
 * scalar values, in the scale image. Coorresponding information in the vector
 * image vimg are stored with the points.
 *
 * @param scale Scalar image which will be neighborhood-thresheld to determine
 * points.
 * @param veimg Image which stores vector information for points
 * @param pct percentage 0-1 of points to keep. .1 would take the top 10%, .9 
 * would take the top 90%.
 *
 * @return KDTree storing the found points and their vector info.
 */
void genPoints(ptr<const MRImage> scale, ptr<const MRImage> vimg, double pct,
        size_t radius);

int main(int argc, char** argv)
{
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Removes the skull from a brain image.",
            ' ', __version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input image.",
			true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_out("o", "out", "Output image.",
			true, "", "*.nii.gz", cmd);

	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	std::shared_ptr<MRImage> inimg(readMRImage(a_in.getValue()));
	if(inimg->ndim() != 3) {
		cerr << "Expected input to be 3D Image!" << endl;
		return -1;
	}
	
    /*****************************
     * edge detection
     ****************************/
    auto deriv = dPtrCast<MRImage>(sobelEdge(inimg));
    cerr << *deriv << endl;
    deriv->write("sobel.nii.gz");
    auto absderiv = dPtrCast<MRImage>(collapseSum(deriv, 3, true));
    cerr << *absderiv << endl;
    absderiv->write("sobel_abs.nii.gz");

    /*****************************
     * create point list from edges (based on top quartile of edges in each
     * window) then extract points that meet local shape criteria 
     ****************************/
	genPoints(absderiv, deriv, .3, 3);
//    points = shapeFilter(points);
//    auto mask = pointsToMask(inimg, points);

    /*******************************************************
     * Propagate selected edges perpendicular to the edge
     ******************************************************/

    /***********************************
     * Watershed
     ***********************************/

    /************************************
     * Select Brain Watershed
     ***********************************/

    /************************************
     * Mask and Write 
     ***********************************/

    } catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}
	
/**
 * @brief Generates a point cloud, stored in a KDTree based on locally large
 * scalar values, in the scale image. Coorresponding information in the vector
 * image vimg are stored with the points.
 *
 * @param scale Scalar image which will be neighborhood-thresheld to determine
 * points.
 * @param veimg Image which stores vector information for points
 * @param pct percentage 0-1 of points to keep. .1 would take the top 10%, .9 
 * would take the top 90%.
 *
 * @return Image with 3 eigenvalues of neighbor positions, and 3 eigenvalues
 * of neighborhood directions (for a 4D image, size 6) and magnitude of the 
 * mean normed vector
 */
void genPoints(ptr<const MRImage> scale, 
        ptr<const MRImage> vimg, double pct, size_t radius)
{
    if(scale->ndim() != 3)
        throw INVALID_ARGUMENT("Scalar Input to genPoints must be 3D");
    if(vimg->ndim() != 4)
        throw INVALID_ARGUMENT("Vector Input to genPoints must be 4D");

    // create output image
    vector<size_t> osize(vimg->dim(), vimg->dim()+vimg->ndim());
    vector<size_t> sz(scale->dim(), scale->dim()+scale->ndim());
    osize[3] = 6;

    auto lambdas = vimg->createAnother(osize.size(), osize.data());
    auto grad_dir = vimg->createAnother(); 
    auto grad_scatter = vimg->createAnother(); 
    auto surface_norm = vimg->createAnother(); 
    Vector3DView<double> l_ac(lambdas);
    Vector3DView<double> sn_ac(surface_norm);
    Vector3DView<double> gd_ac(grad_dir);
    Vector3DView<double> gs_ac(grad_scatter);
    
    // theoretical metrics, not sure if they will work, so do them all
    auto scatter_metric = scale->createAnother(); 
    auto direction_mean = scale->createAnother(); 
    auto direction_var = scale->createAnother(); 
    Pixel3DView<double> sm_ac(scatter_metric);
    Pixel3DView<double> dm_ac(direction_mean);
    Pixel3DView<double> dv_ac(direction_var);

    // vimg accessor
    Pixel3DConstView<double> s_ac(scale);
    Vector3DConstView<double> v_ac(vimg);
    
    //split here
    size_t ksp = clamp<int64_t>(0, it.ksize()-1, it.ksize()*(1-pct)); 
    cerr << "Splitting at " << ksp << "/" << it.ksize() << endl;

    // create temporary arrays to handle all neighbors, and the selected 
    // ones
    vector<double> vals(it.ksize());
    Eigen::Vector3d tmp;
    Eigen::Matrix3d pcov; // position matrix
    Eigen::Vector3d pmean; 
    Eigen::Matrix3d vcov; // vector (gradient) matrix
    Eigen::Vector3d vmean; 
    Eigen::EigenSolver<Matrix3d> esolve;

    int count = 0;
    // go through every point, and the neighborhood of every point
    for(int64_t xx = 0; xx < sz[0]; ++xx){
    for(int64_t yy = 0; yy < sz[1]; ++yy){
    for(int64_t zz = 0; zz < sz[2]; ++zz){

        size_t k = 0;
        for(int64_t xxo = -radius; xxo <= radius; ++xxo){
            for(int64_t yyo = -radius; yyo <= radius; ++yyo){
                for(int64_t zzo = -radius; zzo <= radius; ++zzo, k++){
                    int64_t xeff = clamp(0, sz[0]-1, xx+xxo);
                    int64_t yeff = clamp(0, sz[1]-1, yy+yyo);
                    int64_t zeff = clamp(0, sz[2]-1, zz+zzo);

                    // take the values in the scalar image
                    vals[k] = s_ac(xeff, yeff, zeff);
                }
            }
        }

        // sort
        std::sort(vals.begin(), vals.end());
        
        if(count == 101 || count == 102) 
            cerr << "Computing Covariance/Mean For:" << endl;

        // Find Points Above Threshold, and construct mean and covariance
        // matrices for them
        pmean.setZero();
        vmean.setZero();
        pcov.setZero();
        vcov.setZero();
        for(int64_t xxo = -radius; xxo <= radius; ++xxo){
            for(int64_t yyo = -radius; yyo <= radius; ++yyo){
                for(int64_t zzo = -radius; zzo <= radius; ++zzo, k++){
                    int64_t xeff = clamp(0, sz[0]-1, xx+xxo);
                    int64_t yeff = clamp(0, sz[1]-1, yy+yyo);
                    int64_t zeff = clamp(0, sz[2]-1, zz+zzo);
                    
                    if(s_ac(xeff, yeff, zeff) > vals[ksp]) {
                        // compute weight from offset from center
                        double w = B3kern(xxo, radius)*B3kern(yyo,radius)*
                                B3kern(zzo, radius);

                        tmp[0] = xx*w;
                        tmp[1] = yy*w;
                        tmp[2] = zz*w;
                        pmean += tmp;
                        pcov += tmp*tmp.transpose();
                        if(count == 101) 
                            cerr << tmp.transpose() << endl;

                        // gradient (normalized), because we have already established
                        // this is a large gradient 
                        tmp[0] = w*vac(xeff, yeff, zeff, 0);
                        tmp[1] = w*vac(xeff, yeff, zeff, 1);
                        tmp[2] = w*vac(xeff, yeff, zeff, 2);
                        tmp.normalize();
                        vmean += tmp;
                        vcov += tmp*tmp.transpose();
                        if(count == 102) 
                            cerr << tmp.transpose() << endl;
                    }
                }
            }
        }

        /* 
         * Now Do Processing on the Covariance Matrices and Means 
         */

        // make x*xt into covariance
        pcov -= pmean*pmean.transpose();       
        vcov -= vmean*vmean.transpose();       

        if(count == 101) {
            cerr << "Covariance\n" << pcov << endl << endl;
            cerr << "Mean\n" << pmean.transpose() << endl << endl;
        }
        if(count == 102) {
            cerr << "Covariance\n" << vcov << endl << endl;
            cerr << "Mean\n" << vmean.transpose() << endl << endl;
        }

        /**********************************************************************
         * Scatter Metrics
         * for point location covariance we want the direction of minimal
         * scatter (lowest eigenvalue)
         *********************************************************************/

        esolve.compute(pcov, true);
        double bestlambda = INFINITY; //min
        double metric = 0;
        int bestlambda_i = -1;
        for(size_t dd=0; dd<3; dd++) {
            // set eigenvalues
            double lambda = esolve.eigenvalues()[dd].real();
            l_ac.set(xx, yy, zz, dd, lambda);

            if(lambda < bestlambda) {
                bestlambda = lambda;
                bestlambda_i = dd;
            }
        }

        for(size_t dd=0; dd<3; dd++) {
            // set direction of minimal eigenvalue
            double lambda = esolve.eigenvectors().col(bestlambda_i)[dd].real();
            sn_ac.set(xx, yy, zz, dd, lambda);

            // divide other two by minimum
            if(dd != bestlambda_i) 
                metric += lambda;
        }
        metric /= esolve.eigenvalues()[bestlambda_i].real();
        sm_ac.set(xx, yy, zz, metric);

        /**********************************************************************
         * Vector Metrics
         * We test two metrics, one with the average of the local gradients 
         * (after normalizing them). This could be called gradient coherence. 
         * The other is to take the scatter. If this is highly anisotropic you
         * are between two oppose facing edges.
         *********************************************************************/

        esolve.compute(vcov, ComputeFullU | ComputeFullV);
        bestlambda_i = -1;
        bestlambda = -INFINITY; // max
        for(size_t dd=0; dd<3; dd++) {
            double lambda = esolve.eigenvalues()[dd].real();
            l_ac.set(xx, yy, zz, dd+3, lambda);
            if(lambda > bestlambda) {
                bestlambda = lambda;
                bestlambda_i = dd;
            }
        }

        // set direction of maximum eigenvalue and mean
        metric = 0;
        for(size_t dd=0; dd<3; dd++) {
            // best eigenvector
            double ev = esolve.eigenvectors().col(bestlambda_i)[dd].real();
            gs_ac.set(xx, yy, zz, dd, ev);
        }

        // compute metric and save mean vector
        for(size_t dd=0; dd<3; dd++) {
            double lambda = esolve.eigenvalues()[dd].real();
            gd_ac.set(xx, yy, zz, dd, vmean[dd]);
            if(dd != bestlambda_i) 
                metric += lambda;
        }
        metric = esolve.eigenvalues()[bestlambda_i].real()/metric; // FA
        dv_ac.set(xx, yy, zz, metric);
        dm_ac.set(xx, yy, zz, vmean.norm());
    }
    }
    }
    }
    
    lambdas->write("lambdas.nii.gz");
    grad_dir->write("grad_dir.nii.gz");
    grad_scatter->write("grad_scatter.nii.gz");
    surface_norm->write("surface_norm.nii.gz");
    scatter_metric->write("scatter_metric.nii.gz");
    direction_mean->write("direction_mean.nii.gz");
    direction_var->write("direction_var.nii.gz");
}

