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
#include "mrimage_utils.h"
#include "ndarray_utils.h"
#include "statistics.h"
#include "version.h"
#include "macros.h"

#include <Eigen/SVD>

using namespace std;
using namespace npl;
using Eigen::JacobiSVD;
using Eigen::ComputeFullV;
using Eigen::ComputeFullU;
	
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
	cerr << "Version: " << __version__ << endl;
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
	TCLAP::ValueArg<double> a_spacing("s", "spacing", "Resample image to have "
			"the specified spacing. If you find that the skullstripping is too "
			"slow then you may want to increase this.", false, 1, "mm", cmd);
	TCLAP::ValueArg<double> a_lower("L", "freq-lower", "Lower freuqency for "
			"difference of gaussian. ", false, 2, "mm", cmd);
	TCLAP::ValueArg<double> a_upper("U", "freq-upper", "Upper freuqency for "
			"difference of gaussian. ", false, 8, "mm", cmd);
	TCLAP::ValueArg<double> a_fstep("S", "freq-step", "Upper freuqency for "
			"difference of gaussian. ", false, 1, "mm", cmd);

	cmd.parse(argc, argv);

	/**********
	 * Input
	 *********/
	std::shared_ptr<MRImage> inimg(readMRImage(a_in.getValue()));
	if(inimg->ndim() != 3) {
		cerr << "Expected input to be 3D Image!" << endl;
		return -1;
	}
	
	if(a_upper.getValue() <= a_lower.getValue()) {
		cerr << "Error upper frequency must be > lower frequency!" << endl;
		return -1;
	} else if(a_fstep.getValue() <= 0) {
		cerr << "Error, step must be > 0" << endl;
		return -1;
	}
	size_t nfreq = (a_upper.getValue()-a_lower.getValue())/a_fstep.getValue();
	cerr << "Number of frequency bins: " << nfreq << endl;
	
	cerr << "Computing Gaussian Steps..." << endl;
	vector<ptr<MRImage>> smoothed(nfreq);
	for(size_t ii=0; ii<nfreq; ii++) {
		double ff=a_lower.getValue()+ii*a_fstep.getValue();
		cerr << ff << endl;
		smoothed[ii] = smoothDownsample(inimg, ff, a_spacing.getValue());
		smoothed[ii]->write("smoothed_"+to_string(ff)+".nii.gz");
	}
	cerr << "Done" << endl;
	
	ptr<MRImage> maxvalue = dPtrCast<MRImage>(smoothed.front()->
				createAnother(FLOAT32));
	ptr<MRImage> maxfreq = dPtrCast<MRImage>(smoothed.front()->
				createAnother(FLOAT32));

	// fill max value with 0
	for(FlatIter<double> mit(maxvalue), fit(maxfreq); !mit.eof();  ++mit, ++fit) {
		mit.set(0);
		fit.set(-1);
	}

	cerr << "Computing Principal Frequency" << endl;
	// Compute Maximum Value
	for(size_t ii=1; ii<nfreq; ii++) {
		for(FlatIter<double> mit(maxvalue), lit(smoothed[ii-1]),
					uit(smoothed[ii]), fit(maxfreq); !mit.eof(); 
					++lit, ++uit, ++mit, ++fit) {
			double v = lit.get()-uit.get();
			if(v > mit.get()) {
				mit.set(v);
				fit.set((ii-1)*a_fstep.getValue()+a_lower.getValue());
			}
		}
	}
	cerr << "Done" << endl;

	maxvalue->write("maxvalue.nii.gz");
	maxfreq->write("maxfreq.nii.gz");

//    /*****************************
//     * edge detection
//     ****************************/
//    auto deriv = dPtrCast<MRImage>(sobelEdge(inimg));
//    cerr << *deriv << endl;
//    deriv->write("sobel.nii.gz");
//    auto absderiv = dPtrCast<MRImage>(collapseSum(deriv, 3, true));
//    cerr << *absderiv << endl;
//    absderiv->write("sobel_abs.nii.gz");
//
//    /*****************************
//     * create point list from edges (based on top quartile of edges in each
//     * window) then extract points that meet local shape criteria 
//     ****************************/
//	genPoints(absderiv, deriv, .3, 3);
////    points = shapeFilter(points);
////    auto mask = pointsToMask(inimg, points);
//
//    /*******************************************************
//     * Propagate selected edges perpendicular to the edge
//     ******************************************************/
//
//    /***********************************
//     * Watershed
//     ***********************************/
//
//    /************************************
//     * Select Brain Watershed
//     ***********************************/
//
//    /************************************
//     * Mask and Write 
//     ***********************************/
//
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
    osize[3] = 6;

    auto lambdas = vimg->createAnother(osize.size(), osize.data(), FLOAT64);
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

    // iterator
    KernelIter<double> it(scale); 
    it.setRadius(radius);

    // vimg accessor
    Vector3DConstView<double> vac(vimg);
    
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
    int order[3] = {0,1,2};

    int count = 0;
    // go through every point, and the neighborhood of every point
    vector<int64_t> index(4);
    for(it.goBegin(); !it.eof(); ++it) {
        count++;
        // take the neighborhood values
        for(size_t k = 0; k < it.ksize(); k++) 
            vals[k] = it[k];

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
        size_t n = 0; 
        for(size_t k = 0; k < it.ksize(); k++) {
            if(it[k] > vals[ksp]) {
                // note the exact number of matching could vary due to
                // enditical values
                
                // compute weight from offset from center
//                double w = 1./(it.ksize()-ksp);
                double w = 1.;
//                for(size_t dd=0; dd<3; dd++)
//                    w *= B3kern(it.offsetK(k,dd), radius);

                // get offset of found point
                it.offsetK(k, 3, index.data());

                // sample position
                for(size_t dd=0; dd<3; dd++) 
                    tmp[dd] = w*index[dd];
                pmean += tmp;
                pcov += tmp*tmp.transpose();
                if(count == 101) 
                    cerr << tmp.transpose() << endl;

                // gradient (normalized), because we have already established
                // this is a large gradient 
                it.indexK(k, 3, index.data());
                for(size_t dd=0; dd<3; dd++) {
                    tmp[dd] = w*vac(index[0], index[1], index[2], dd);
                }
                tmp.normalize();
                vmean += tmp;
                vcov += tmp*tmp.transpose();
                if(count == 102) {
                    cerr << tmp.transpose() << endl;
                }
             
                n += w;
            }
        }

        /* 
         * Now Do Processing on the Covariance Matrices and Means 
         */

        // make x*xt into covariance
        pmean /= n;
        pcov /= (n-1);
        pcov -= (n/(n-1.))*pmean*pmean.transpose();       
        
        vmean /= n;
        vcov /= (n-1);
        vcov -= (n/(n-1.))*vmean*vmean.transpose();       

        if(count == 101) {
            cerr << "Covariance\n" << pcov << endl << endl;
            cerr << "Mean\n" << pmean.transpose() << endl << endl;
        }
        if(count == 102) {
            cerr << n << endl;
            cerr << "Covariance\n" << vcov << endl << endl;
            cerr << "Mean\n" << vmean.transpose() << endl << endl;
        }

        it.indexC(3, index.data());

        /**********************************************************************
         * Scatter Metrics
         * for point location covariance we want the direction of minimal
         * scatter (lowest eigenvalue)
         *********************************************************************/

        esolve.compute(pcov, true);
        if(count == 101) {
            cerr << "Lambdas\n" << esolve.eigenvalues() << endl << endl;
            cerr << "Evs\n" << esolve.eigenvectors() << endl << endl;
        }

        // order eigenvectors
        std::sort(order, order+3, [&](int lhs, int rhs)
        { 
            return fabs(esolve.eigenvalues()[lhs].real()) < 
                    fabs(esolve.eigenvalues()[rhs].real());
        });

        for(size_t dd=0; dd<3; dd++) {
            // set eigenvalues
            double lambda = fabs(esolve.eigenvalues()[order[dd]].real());
            double ev = fabs(esolve.eigenvectors().col(order[0])[dd].real());
            l_ac.set(index[0], index[1], index[2], dd, lambda);
            sn_ac.set(index[0], index[1], index[2], dd, ev);
        }
        double metric = fabs(esolve.eigenvalues()[order[2]].real() / 
                    esolve.eigenvalues()[order[0]].real());
        sm_ac.set(index[0], index[1], index[2], metric);

        /**********************************************************************
         * Vector Metrics
         * We test two metrics, one with the average of the local gradients 
         * (after normalizing them). This could be called gradient coherence. 
         * The other is to take the scatter. If this is highly anisotropic you
         * are between two oppose facing edges.
         *********************************************************************/

        esolve.compute(vcov, ComputeFullU | ComputeFullV);
        if(count == 102) {
            cerr << "Lambdas\n" << esolve.eigenvalues() << endl << endl;
            cerr << "Evs\n" << esolve.eigenvectors() << endl << endl;
        }
        
        // order eigenvectors
        std::sort(order, order+3, [&](int lhs, int rhs)
        { 
            return fabs(esolve.eigenvalues()[lhs].real()) < 
                    fabs(esolve.eigenvalues()[rhs].real());
        });

        for(size_t dd=0; dd<3; dd++) {
            double lambda = fabs(esolve.eigenvalues()[order[dd]].real());
            double ev = esolve.eigenvectors().col(order[2])[dd].real();
            l_ac.set(index[0], index[1], index[2], dd+3, lambda);
            gs_ac.set(index[0], index[1], index[2], dd, ev);
            gd_ac.set(index[0], index[1], index[2], dd, vmean[dd]);
        }
        metric = fabs(esolve.eigenvalues()[order[2]].real()/
                (esolve.eigenvalues()[order[1]].real()+
                esolve.eigenvalues()[order[0]].real())); 
        dv_ac.set(index[0], index[1], index[2], metric);
        dm_ac.set(index[0], index[1], index[2], vmean.norm());
    }
    
    lambdas->write("lambdas.nii.gz");
    grad_dir->write("grad_dir.nii.gz");
    grad_scatter->write("grad_scatter.nii.gz");
    surface_norm->write("surface_norm.nii.gz");
    scatter_metric->write("scatter_metric.nii.gz");
    direction_mean->write("direction_mean.nii.gz");
    direction_var->write("direction_var.nii.gz");

	// classify
	KMeans classifier(lambdas->tlen(), 4);
	Eigen::Map<MatrixXd> samples((double*)lambdas->data(), scale->elements(),
				lambdas->tlen());
	classifier.compute(samples);
	Eigen::VectorXi labels = classifier.classify(samples);
	auto segmented = createMRImage(scale->ndim(), scale->dim(), INT32,
			labels.data(), [](void*){return;});
	segmented->write("segmented.nii.gz");
}

