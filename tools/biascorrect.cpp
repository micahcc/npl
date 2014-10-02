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
 * @file applyDeform.cpp Tool to apply a deformation field to another image. 
 * Not yet functional
 *
 *****************************************************************************/

#include <Eigen/SparseCore>
#include <Eigen/SparseQR>
#include <Eigen/SVD>
#include <unordered_map>
#include <tclap/CmdLine.h>
#include <version.h>
#include <string>
#include <stdexcept>
#include <iterator>

#include "mrimage.h"
#include "nplio.h"
#include "mrimage_utils.h"
#include "kdtree.h"
#include "iterators.h"
#include "accessors.h"

using namespace npl;
using namespace std;

#define VERYDEBUG
#include "macros.h"

std::ostream_iterator<double> vdstream (std::cout,", ");
std::ostream_iterator<int64_t> vistream (std::cout,", ");

typedef Eigen::SparseMatrix<double,Eigen::RowMajor> SparseMat;

ptr<MRImage> createBiasField(ptr<MRImage> in, double bspace)
{
	size_t ndim = in->ndim();
	VectorXd spacing(in->ndim());
	VectorXd origin(in->ndim());
	vector<size_t> osize(ndim, 0);

	// get spacing and size
	for(size_t dd=0; dd<osize.size(); ++dd) {
		osize[dd] = 4+in->dim(dd)*in->spacing(dd)/bspace;
		spacing[dd] = bspace;
	}

	auto biasfield = createMRImage(osize.size(), osize.data(), FLOAT64);
	biasfield->setDirection(in->getDirection(), false);
	biasfield->setSpacing(spacing, false);
	
	// compute center of input
	VectorXd indc(ndim); // center index
	for(size_t dd=0; dd<ndim; dd++) 
		indc[dd] = (in->dim(dd)-1.)/2.;
	VectorXd ptc(ndim); // point center
	in->indexToPoint(ndim, indc.array().data(), ptc.array().data());

	// compute origin from center index (x_c) and center of input (c): 
	// o = c-R(sx_c)
	for(size_t dd=0; dd<ndim; dd++) 
		indc[dd] = (osize[dd]-1.)/2.;
	origin = ptc - in->getDirection()*(spacing.asDiagonal()*indc);
	biasfield->setOrigin(origin, false);

	return biasfield;
}

//#define USE_SPARSE

double smoothwindow(double x, double a)
{
	return hannWindow(x, .8*a);
}

int main(int argc, char** argv)
{
try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Computes a bias-field from an image and a mask. ",
			' ', __version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input image.",
		true, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_mask("m", "mask", "Mask image. If this image is "
			"a scalar then values are taken as weights.", true, "", "*.nii.gz",
			cmd);
	TCLAP::ValueArg<double> a_spacing("s", "spacing", "Space between knots "
			"for bias-field estimation.", false, 40, "mm", cmd);
	TCLAP::ValueArg<double> a_downspace("d", "downsample", "Spacing in "
			"downsampled image. This is primarily to speed up processing. "
			"Because the problem is already overdetermined this doesn't impact "
			"results very much. So a 5 here would resample the image to 5x5x5 "
			"isotropic voxels.", false, 10, "pixsize", cmd);
	TCLAP::ValueArg<double> a_lambda("R", "regweight", "Regularization weight "
			"for ridge regression. Larger values will cause a smoother result",
			false, 1.e-3, "ratio", cmd);

	TCLAP::ValueArg<string> a_biasfield("o", "out", "Bias Field Image.",
			false, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_corimage("c", "corr", "Bias Field Corrected "
			"version of input image.", false, "", "*.nii.gz", cmd);

	cmd.parse(argc, argv);

	// read input, recast as double, with <= 3 dimensions
	ptr<MRImage> in = readMRImage(a_in.getValue());
	in = dPtrCast<MRImage>(in->copyCast(min(in->ndim(),3UL), in->dim(), FLOAT64));
	vector<double> dspace(in->ndim(), a_downspace.getValue());
	
	cout << "Downsampling input from [";
	copy(in->getSpacing().data(), in->getSpacing().data()+in->ndim(), vdstream);
	cout << "] spacing to [";
	copy(dspace.begin(), dspace.end(), vdstream);
	cout << "] spacing...";
	in = resample(in, dspace.data());
	cout << "Done\n";

	// read mask then downsample using nearest neighbor
	ptr<MRImage> mask;
	{
		auto tmpmask = readMRImage(a_mask.getValue());
		NNInterpNDView<double> mask_ac(tmpmask);
		mask_ac.m_ras = true;

		mask = dPtrCast<MRImage>(in->createAnother(FLOAT64));
		vector<double> pt(in->ndim());
		vector<int64_t> ind(in->ndim());
		for(NDIter<double> it(mask); !it.eof(); ++it) {
			it.index(ind);
			mask->indexToPoint(ind.size(), ind.data(), pt.data());
			it.set(mask_ac(pt));
		}
	}

#ifdef VERYDEBUG
	in->write("downsampled.nii.gz");
	mask->write("downsampled_mask.nii.gz");
#endif

	// Create Double Bias Field
	cout << "Creating Bias Field estimate with with " << a_spacing.getValue()
		<< "mm spacing...";
	auto biasfield = createBiasField(in, a_spacing.getValue());
	cout << "Done\n";
	
	size_t ndim = in->ndim();
	size_t nparams = 1; // number of Cubic-BSpline Parameters
	size_t npixels = 1; // Number of pixels we are comparing
	for(size_t ii=0; ii<ndim; ii++) {
		nparams *= biasfield->dim(ii);
		npixels *= in->dim(ii);
	}
#ifdef USE_SPARSE
	vector<Eigen::Triplet<double>> pixB;
	SparseMat Bmat(npixels, nparams);
#else
	MatrixXd Bmat(npixels, nparams);
	Bmat.setZero();
#endif
	
	Eigen::Map<VectorXd> pixels((double*)in->data(), npixels);
	Eigen::Map<VectorXd> params((double*)biasfield->data(), nparams);
	Eigen::Map<VectorXd> weights((double*)mask->data(), npixels);

	/***********************************************************************
	 * Take the log of the input image because Bias Fiels are multiplied by
	 * the image intensity
	 **********************************************************************/
	double minval = INFINITY;
	for(size_t ii=0; ii<pixels.rows(); ii++)
		minval = min(minval, pixels[ii]);
	if(minval > 0) minval = -1;
	
	for(size_t ii=0; ii<pixels.rows(); ii++)
		pixels[ii] = log(pixels[ii] - minval + 1);

	/************************************************************
	 * Calculate Parameter Weights at Each Pixel
	 * In the equation we are solving this would be B, p are the
	 * parameters and v are the voxel values:
	 * v = Bp
	 ***********************************************************/
	vector<size_t> roi_size(ndim);   // size of ROI in index space
	vector<int64_t> roi_start(ndim); // offset from center in index space
	vector<pair<int64_t,int64_t>> roi(ndim);
	for(size_t dd=0; dd<ndim; dd++) {
		roi_size[dd] = 5*biasfield->spacing(dd)/in->spacing(dd);
		roi_start[dd] = -2.5*biasfield->spacing(dd)/in->spacing(dd);
	}

	cout << "Filling Weights...";
	// for each parameter
	NDIter<double> iit(in); // input iterator
	vector<int64_t> bind(ndim); // bias field index
	vector<double> oind(ndim); // index in bias field, offset from center 
	vector<double> iind(ndim); // input image index
	vector<int64_t> iind_i(ndim); // input image index, integer
	vector<double> point(ndim); 
	for(NDIter<double> pit(biasfield); !pit.eof(); ++pit) {
		// compute index in input image
		pit.index(ndim, bind.data());
		size_t linbias = biasfield->getLinIndex(bind);
		biasfield->indexToPoint(ndim, bind.data(), point.data());
		in->pointToIndex(ndim, point.data(), iind.data());

		// create ROI
		for(size_t dd=0; dd<ndim; dd++) {
			roi[dd].first = roi_start[dd]+round(iind[dd]);
			roi[dd].second = roi[dd].first + roi_size[dd]-1;
		}
		// construct ROI, iterator for neighborhood of bias field parameter
		iit.setROI(roi);
		for(iit.goBegin(); !iit.eof(); ++iit) {
			// compute index in biasfield
			iit.index(ndim, iind_i.data());
			in->indexToPoint(ndim, iind_i.data(), point.data());
			biasfield->pointToIndex(ndim, point.data(), oind.data());

			// compute weight
			double w = 1;
			for(size_t dd=0; dd<ndim; dd++) 
				w *= B3kern(bind[dd] - oind[dd]);

			bool outside = false;
			for(size_t dd=0; dd<in->ndim(); dd++){
				if(iind_i[dd] < 0 || iind_i[dd] >= in->dim(dd))
					outside = true;
			}
			if(!outside && w > 0) {
				size_t lininput = in->getLinIndex(iind_i);
				assert(linbias < Bmat.cols());
				assert(lininput < Bmat.rows());
#ifdef USE_SPARSE
				pixB.push_back(Eigen::Triplet<double>(lininput, linbias, w)); 
#else
				Bmat(lininput, linbias) = w;
#endif
			}
		}
	}

#ifdef USE_SPARSE
	cout << "Done\nBuilding Sparse Matrix..." << endl;
	Bmat.setFromTriplets(pixB.begin(), pixB.end());
#endif

	// adjust with relative weighs from mask
	DBG1(void* ptr = pixels.data());
	pixels = weights.asDiagonal()*pixels;
	assert(ptr == pixels.data());
	
#ifdef USE_SPARSE
	Bmat.makeCompressed();
	Eigen::SparseQR<SparseMat,Eigen::COLAMDOrdering<int>> solver;
	cout << "Done\nComputing..." << endl;
	solver.compute(Bmat);
	cout << "Done\nSolving..." << endl;
	params = solver.solve(pixels);
#else
	// perform decomposition
	Eigen::JacobiSVD<MatrixXd> solver;
	cout << "Done\nComputing..." << endl;
	solver.compute(weights.asDiagonal()*Bmat, 
			Eigen::ComputeThinU | Eigen::ComputeThinV);

	double lambda = a_lambda.getValue();
	if(lambda < 0) {
		cerr << "Warning negative lambda given, setting to default" << endl;
		lambda = 1e-3;
	}
	cout << "Done\nSolving with lambda="<< lambda << "...";

	// adjust eigenvalues with regularization term
	// this is a form of Tikhonov Regularization
	VectorXd sigmas = solver.singularValues();
	for(size_t ii=0; ii<sigmas.rows(); ii++) {
		double is = sigmas[ii]/(sigmas[ii]*sigmas[ii]+lambda);
		if(std::isnan(is) || std::isinf(is) || is > 1e20)
			sigmas[ii] = 0;
		else
			sigmas[ii] = is;
	}
	params = solver.matrixV()*sigmas.asDiagonal()*solver.matrixU().transpose()*pixels;
#endif

	// estimate the bias field from the parameters
	pixels = Bmat*params;

	// find minimum in masked pixels and subtract that so that intensity
	// remains relatively the same after bias correction
	double avg = 0;
	double count = 0;
	for(size_t rr=0; rr<weights.rows(); rr++) {
		if(weights[rr] > 0) {
			avg += pixels[rr];
			count++;
		}
	}

#ifdef VERYDEBUG
	in->write("logbias.nii.gz");
#endif 
	avg /= count;
	for(size_t rr=0; rr<pixels.rows(); rr++) {
		pixels[rr] = exp(pixels[rr]-avg);
		if(std::isinf(pixels[rr]) || std::isnan(pixels[rr]))
			pixels[rr] = 1;
	}
#ifdef VERYDEBUG
	in->write("bias.nii.gz");
#endif 

	cout << "Done\nWriting..." << endl;
	if(a_biasfield.isSet())
		in->write(a_biasfield.getValue());
	if(a_corimage.isSet()) {
		LinInterpNDView<double> interp(in);
		interp.m_ras = true;

		auto reread = readMRImage(a_in.getValue());
		vector<double> pt(reread->ndim());
		for(NDIter<double> it(reread); !it.eof(); ++it) {
			it.index(pt.size(), pt.data());
			reread->indexToPoint(pt.size(), pt.data(), pt.data());
			it.set((*it)/interp.get(pt));
		}
		reread->write(a_corimage.getValue());
	}
	cout << "Done" << endl;

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}

