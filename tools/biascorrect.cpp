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
 * @file biascorrect.cpp Bias-field correction tool.
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
#include "ndarray_utils.h"
#include "kdtree.h"
#include "iterators.h"
#include "accessors.h"

using namespace npl;
using namespace std;

#define VERYDEBUG
#include "macros.h"

const double OUTSIDEWEIGHT = 1e-9;

std::ostream_iterator<double> vdstream (std::cout,", ");
std::ostream_iterator<int64_t> vistream (std::cout,", ");

double otsuThresh(ptr<const NDArray> in)
{
	vector<double> bins(sqrt(in->elements()));
	double minv = INFINITY;
	double maxv = -INFINITY;
	for(FlatConstIter<double> fit(in); !fit.eof(); ++fit) {
		minv = std::min(minv, *fit);
		maxv = std::max(maxv, *fit);
	}
	double bwidth = 0.99999999*bins.size()/(maxv-minv);
	for(FlatConstIter<double> fit(in); !fit.eof(); ++fit) 
		bins[floor((*fit-minv)*bwidth)]++;

	for(size_t bb=0; bb < bins.size(); bb++) {
		bins[bb] /= in->elements();
	}

	double prob1 = 0, prob2 = 0, mu1 = 0, mu2 = 0, sigma = 0;
	size_t tt =0;
	double max_sigma = -INFINITY;
	size_t max_t = 0;
	for(tt=0; tt<bins.size(); tt++) {
		prob1 = 0;
		for(size_t bb=0; bb<tt; bb++) 
			prob1 += bins[bb];
		mu1 = 0;
		for(size_t bb=0; bb<tt; bb++)
			mu1 += bins[bb]*(minv + bb/bwidth);
		mu1 /= prob1;

		prob2 = 0;
		for(size_t bb=tt; bb<bins.size(); bb++) 
			prob2 += bins[bb];
		mu2 = 0;
		for(size_t bb=tt; bb<bins.size(); bb++)
			mu2 += bins[bb]*(minv + bb/bwidth);
		mu2 /= prob2;

		sigma = prob1*prob2*(mu1-mu2)*(mu1-mu2);
		if(sigma > max_sigma) {
			max_t = tt;
			max_sigma = sigma;
		}
	}

	return max_t/bwidth + minv;
}

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

	auto biasparams = createMRImage(osize.size(), osize.data(), FLOAT64);
	biasparams->setDirection(in->getDirection(), false);
	biasparams->setSpacing(spacing, false);
	
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
	biasparams->setOrigin(origin, false);

	return biasparams;
}

//#define USE_SPARSE

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
			"a scalar then values are taken as weights. If not provided a "
			"mask based on the image mean will be used", false, "", "*.nii.gz",
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
			false, 1.e-5, "ratio", cmd);

	TCLAP::ValueArg<string> a_biasfield("b", "biasfield", "Bias Field Image.",
			false, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_biasparams("B", "bparams", "Bias Field "
			"parameters image. these are the knot value in the bias field.",
			false, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_corimage("c", "corr", "Bias Field Corrected "
			"version of input image.", false, "", "*.nii.gz", cmd);

	cmd.parse(argc, argv);

	// read input, recast as double, with <= 3 dimensions
	ptr<MRImage> fullres = readMRImage(a_in.getValue());
	fullres = dPtrCast<MRImage>(fullres->copyCast(min(fullres->ndim(),3UL), 
				fullres->dim(), FLOAT64));
	vector<double> dspace(fullres->ndim(), a_downspace.getValue());
	
	cout << "Downsampling input from [";
	copy(fullres->getSpacing().data(), fullres->getSpacing().data() +
			fullres->ndim(), vdstream);
	cout << "] spacing to [";
	copy(dspace.begin(), dspace.end(), vdstream);
	cout << "] spacing...";
	auto in = dPtrCast<MRImage>(resample(fullres, dspace.data()));
	cout << "Done\n";

	// read mask then downsample using nearest neighbor
	ptr<MRImage> fullmask;
	ptr<MRImage> mask;
	if(a_mask.isSet()) {
		fullmask = readMRImage(a_mask.getValue());

		// binarize
		for(FlatIter<double> fit(fullmask); !fit.eof(); ++fit) 
			fit.set(*fit > 0);

	} else {
		fullmask = dPtrCast<MRImage>(fullres->copyCast(FLOAT64));
		double thresh = otsuThresh(fullres);

		// binarize/threshold
		cerr << "Threshold: " << thresh << endl;
		for(FlatIter<double> fit(fullmask); !fit.eof(); ++fit) {
			fit.set(*fit > thresh);
		}
	}
	fullmask = dPtrCast<MRImage>(erode(fullmask, 1));
	fullmask->write("fullmask.nii.gz");
	
	{
		LinInterpNDView<double> mask_ac(fullmask);
		mask_ac.m_ras = true;

		mask = dPtrCast<MRImage>(in->createAnother(FLOAT64));
		vector<double> pt(in->ndim());
		vector<int64_t> ind(in->ndim());
		for(NDIter<double> it(mask); !it.eof(); ++it) {
			it.index(ind);
			mask->indexToPoint(ind.size(), ind.data(), pt.data());
			it.set(mask_ac(pt) > .5);
		}
	}
	fullmask.reset();

#ifdef VERYDEBUG
	in->write("downsampled.nii.gz");
	mask->write("downsampled_mask.nii.gz");
#endif

	// Create Double Bias Field
	cout << "Creating Bias Field estimate with with " << a_spacing.getValue()
		<< "mm spacing...";
	auto biasparams = createBiasField(in, a_spacing.getValue());
	cout << "Done\n";
	
	size_t ndim = in->ndim();
	size_t nparams = 1; // number of Cubic-BSpline Parameters
	size_t npixels = 1; // Number of pixels we are comparing
	for(size_t ii=0; ii<ndim; ii++) {
		nparams *= biasparams->dim(ii);
		npixels *= in->dim(ii);
	}
#ifdef USE_SPARSE
	vector<Eigen::Triplet<double>> pixB;
	pixB.reserve(nparams*5);
#else
	MatrixXd Bmat(npixels, nparams);
	Bmat.setZero();
#endif
	
	Eigen::Map<VectorXd> pixels((double*)in->data(), npixels);
	Eigen::Map<VectorXd> params((double*)biasparams->data(), nparams);
	Eigen::Map<VectorXd> weights((double*)mask->data(), npixels);

	/***********************************************************************
	 * Take the log of the input image because Bias Fiels are multiplied by
	 * the image intensity
	 **********************************************************************/
	// first average the masked regions and divide by avg
	double avg = 0;
	size_t count = 0;
	double minval = INFINITY;
	for(size_t ii=0; ii<pixels.rows(); ii++) {
		if(weights[ii] > 0) {
			avg += pixels[ii];
			minval = min(minval, pixels[ii]);
			count++;
		}
	}
	avg /= count;
	
	// divide by mean and compute minimum value
	for(size_t ii=0; ii<pixels.rows(); ii++) {
		if(weights[ii] > 0) { 
			pixels[ii] = (pixels[ii]-minval+1)/(avg - minval-1);
		} else {
			pixels[ii] = 1;
		}
	}

#ifdef VERYDEBUG
	in->write("normalized.nii.gz");
#endif 

	// take the log
	for(size_t ii=0; ii<pixels.rows(); ii++)
		pixels[ii] = log(pixels[ii]);
	
	// make outside mask values 1 (0 in log space)
	for(size_t ii=0; ii<weights.rows(); ii++) {
		if(weights[ii] <= 0) {
			weights[ii] = OUTSIDEWEIGHT;
		}
	}

#ifdef VERYDEBUG
	in->write("lognormalized.nii.gz");
#endif 

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
		roi_size[dd] = 5*biasparams->spacing(dd)/in->spacing(dd);
		roi_start[dd] = -2.5*biasparams->spacing(dd)/in->spacing(dd);
	}

	cout << "Filling Weights...";
	// for each parameter
	NDIter<double> iit(in); // input iterator
	vector<int64_t> bind(ndim); // bias field index
	vector<double> oind(ndim); // index in bias field, offset from center 
	vector<double> iind(ndim); // input image index
	vector<int64_t> iind_i(ndim); // input image index, integer
	vector<double> point(ndim); 
	for(NDIter<double> pit(biasparams); !pit.eof(); ++pit) {
		// compute index in input image
		pit.index(ndim, bind.data());
		size_t linbias = biasparams->getLinIndex(bind);
		biasparams->indexToPoint(ndim, bind.data(), point.data());
		in->pointToIndex(ndim, point.data(), iind.data());

		// create ROI
		for(size_t dd=0; dd<ndim; dd++) {
			roi[dd].first = roi_start[dd]+round(iind[dd]);
			roi[dd].second = roi[dd].first + roi_size[dd]-1;
		}
		// construct ROI, iterator for neighborhood of bias field parameter
		iit.setROI(roi);
		for(iit.goBegin(); !iit.eof(); ++iit) {
			// compute index in biasparams
			iit.index(ndim, iind_i.data());
			in->indexToPoint(ndim, iind_i.data(), point.data());
			biasparams->pointToIndex(ndim, point.data(), oind.data());

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

	/*****************************************************************
	 * Least Squares Fit
	 ****************************************************************/
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
	cout << "Done\nComputing...";
	solver.compute(Bmat);
	cout << "Done\nSolving..." << endl;
	params = solver.solve(pixels);
#else
	// perform decomposition
	Eigen::JacobiSVD<MatrixXd> solver;
	cout << "Done\nComputing...";
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
		if(std::isnan(is) || std::isinf(is) || is > 1e20) {
			cerr << "Warning odd singular values (inf/nan/>1e20)" << endl;
			sigmas[ii] = 0;
		} else
			sigmas[ii] = is;
	}
	params = solver.matrixV()*sigmas.asDiagonal()*solver.matrixU().transpose()*pixels;
#endif

	/*************************************************************************
	 * estimate the bias field from the parameters
	 ************************************************************************/
	cout << "Estimating Bias Field From Parameters";
	auto biasfield = dPtrCast<MRImage>(fullres->createAnother(FLOAT64));
	NDConstIter<double> bp_it(biasparams); // iterator of biasparams
	for(NDIter<double> it(biasfield); !it.eof(); ++it) {
		// get index/point
		it.index(ndim, iind_i.data());
		biasfield->indexToPoint(ndim, iind_i.data(), point.data());
		biasparams->pointToIndex(ndim, point.data(), oind.data());
		for(size_t dd=0; dd<ndim; dd++) 
			bind[dd] = round(oind[dd]);
		
		// create ROI
		for(size_t dd=0; dd<ndim; dd++) {
			roi[dd].first = bind[dd]-2;
			roi[dd].second = bind[dd]+2;
		}

		// construct ROI, iterator for neighborhood of bias field parameter
		bp_it.setROI(roi);
		double sum = 0;
		for(bp_it.goBegin(); !bp_it.eof(); ++bp_it) {
			double w = 1;
			bp_it.index(ndim, bind.data());
			for(size_t dd=0; dd<ndim; dd++)
				w *= B3kern(bind[dd]-oind[dd]);
			sum += w*(*bp_it);
		}
		it.set(sum);
	}
	cout << "Done" << endl;

#ifdef VERYDEBUG
	biasfield->write("logbias.nii.gz");
#endif 
	for(FlatIter<double> it(biasfield); !it.eof(); ++it) {
		it.set(exp(*it));
		if(std::isinf(*it) || std::isnan(*it)) {
			cerr << "Unusual bias field value (nan/inf) found" << endl;
			it.set(1);
		}
	}
#ifdef VERYDEBUG
	biasfield->write("bias.nii.gz");
#endif 

	cout << "Done\nWriting..." << endl;
	if(a_biasparams.isSet()) 
		biasparams->write(a_biasparams.getValue());
	if(a_biasfield.isSet())
		biasfield->write(a_biasfield.getValue());
	if(a_corimage.isSet()) {
		FlatIter<double> bit(biasfield);
		FlatIter<double> it(fullres);
		for(; !it.eof() && !bit.eof(); ++bit, ++it) {
			it.set((*it)/(*bit));
		}
		fullres->write(a_corimage.getValue());
	}
	cout << "Done" << endl;

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}

