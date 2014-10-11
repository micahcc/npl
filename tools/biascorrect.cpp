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

/**
 * @brief This function does a couple things (which I know violates the rule of
 * functions): it normalizes, logs and and downsampls the input (this is placed
 * into * downimg) and it downsamples the input mask (placing the result in
 * downmask)
 *
 * @param input Input image to downsample
 * @param mask Input mask to downsample (also needed to normalize masked values)
 * @param spacing Spacing of output images
 * @param outside_weight Value of masked points OUTSIDE original mask
 * @param downimg Output: Downsampled and normalized version of input
 * @param downmask Output: Downsampled version of mask (with 0 values changed
 * outside_weight)
 */
void preprocInputs(ptr<const MRImage> input, 
		ptr<const MRImage> mask, double spacing, double outside_weight,
		ptr<MRImage>& downimg, ptr<MRImage>& downmask);

/**
 * @brief Creates a bias field with the specified spacing, that overlaps
 * with the input image.
 *
 * @param in Template image, same direction will be used, and the bias field
 * will have an overlapping grid
 * @param bspace Spacing of bias field
 *
 * @return Bias field
 */
ptr<MRImage> createBiasField(ptr<const MRImage> in, double bspace);

/**
 * @brief Computes a threshold based on OTSU.
 *
 * @param in Input image.
 *
 * @return Threshold 
 */
double otsuThresh(ptr<const NDArray> in);

ptr<MRImage> estBiasParams(ptr<const MRImage> in, ptr<const MRImage> mask, 
		double spacing, double lambda);

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
	TCLAP::ValueArg<double> a_oweight("O", "oweight", "Weight of pixels "
			"outside the mask (which prevents numerical instability",
			false, 1.e-5, "ratio", cmd);

	TCLAP::ValueArg<string> a_biasfield("b", "biasfield", "Bias Field Image.",
			false, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_biasparams("B", "bparams", "Bias Field "
			"parameters image. these are the knot value in the bias field.",
			false, "", "*.nii.gz", cmd);
	TCLAP::ValueArg<string> a_corimage("c", "corr", "Bias Field Corrected "
			"version of input image.", false, "", "*.nii.gz", cmd);

	cmd.parse(argc, argv);

	/****************************************************
	 * Read a Single 3D Volume from Input Image
	 ****************************************************/
	ptr<MRImage> fullres = readMRImage(a_in.getValue());
	fullres = dPtrCast<MRImage>(fullres->copyCast(min(fullres->ndim(),3UL), 
				fullres->dim(), FLOAT64));
	size_t ndim = fullres->ndim();

	/****************************************************
	 * Read or Create Mask
	 ****************************************************/
	cout << "Loading/Creating Mask...";
	ptr<MRImage> fullmask;
	if(a_mask.isSet()) {
		fullmask = readMRImage(a_mask.getValue());

		// binarize
		for(FlatIter<double> fit(fullmask); !fit.eof(); ++fit) {
			double v = *fit;
			if(v > 0)
				fit.set(1);
			else
				fit.set(a_oweight.getValue());
		}

	} else {
		fullmask = dPtrCast<MRImage>(fullres->copyCast(FLOAT64));
		double thresh = otsuThresh(fullres);

		// binarize/threshold
		cerr << "Threshold: " << thresh << endl;
		for(FlatIter<double> fit(fullmask); !fit.eof(); ++fit) {
			fit.set(*fit > thresh);
		}
		fullmask = dPtrCast<MRImage>(erode(fullmask, 1));
	}
	cout << "Done...";
	fullmask->write("fullmask.nii.gz");
	
	/****************************************************
	 * Create Downsampled, logged Versions if Inputs
	 ****************************************************/
	cout << "Downsampling/Normalizing...";
	ptr<MRImage> dinput;
	ptr<MRImage> dmask;
	preprocInputs(fullres, fullmask, a_downspace.getValue(),
			a_oweight.getValue(), dinput, dmask);
	fullmask.reset();
	cout << "Done" << endl;
	
	/********************************************************************
	 * Estimate Bias Field Parameters from Pixels and Weights 
	 ********************************************************************/
	cout << "Estimating Bias Field...";
	auto biasparams = estBiasParams(dinput, dmask, a_spacing.getValue(),
			a_lambda.getValue());
	cout << "Done...";
	if(a_biasparams.isSet()) 
		biasparams->write(a_biasparams.getValue());

	/*************************************************************************
	 * estimate the bias field from the parameters
	 ************************************************************************/
	cout << "Estimating Bias Field From Parameters";
	vector<pair<int64_t,int64_t>> roi(ndim);
	NDConstIter<double> bp_it(biasparams); // iterator of biasparams
	vector<int64_t> ind(ndim); // index
	vector<double> pt(ndim);   // point
	vector<double> cind(ndim); // continuous index
	for(NDIter<double> it(fullres); !it.eof(); ++it) {

		// get continuous index of voxel
		it.index(ind.size(), ind.data());
		fullres->indexToPoint(ind.size(), ind.data(), pt.data());
		biasparams->pointToIndex(pt.size(), pt.data(), cind.data());
		for(size_t dd=0; dd<ndim; dd++) 
			ind[dd] = round(cind[dd]);
		
		// create ROI from nearest index to continuous one
		for(size_t dd=0; dd<ndim; dd++) {
			roi[dd].first = ind[dd]-2;
			roi[dd].second = ind[dd]+2;
		}

		// construct ROI, iterator for neighborhood of bias field parameter
		bp_it.setROI(roi);
		double sum = 0;
		for(bp_it.goBegin(); !bp_it.eof(); ++bp_it) {
			double w = 1;
			bp_it.index(ind.size(), ind.data());
			for(size_t dd=0; dd<ndim; dd++)
				w *= B3kern(ind[dd]-cind[dd]);
			sum += w*(*bp_it);
		}
		it.set(sum);
	}
	cout << "Done" << endl;

#ifdef VERYDEBUG
	fullres->write("logbias.nii.gz");
#endif 
	for(FlatIter<double> it(fullres); !it.eof(); ++it) {
		it.set(exp(*it));
		if(std::isinf(*it) || std::isnan(*it)) {
			cerr << "Unusual bias field value (nan/inf) found" << endl;
			it.set(1);
		}
	}
	if(a_biasfield.isSet())
		fullres->write(a_biasfield.getValue());

	/*************************************************************************
	 * Finally Write the Output
	 ************************************************************************/
	cout << "Done\nWriting..." << endl;
	if(a_corimage.isSet()) {
		// Re-Read Input (even if it is 4D)
		auto input = readMRImage(a_in.getValue());

		NDView<double> b_vw(fullres);
		for(NDIter<double> iit(input); !iit.eof(); ++iit) {
			iit.index(ind);
			iit.set(iit.get()/b_vw[ind]);
		}
		input->write(a_corimage.getValue());
	}
	cout << "Done" << endl;

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}

/**
 * @brief Given an input image and mask, this estimates the bias-field
 * parameters. 
 *
 * @param in Input scalar image, should already be log-transformed
 * @param weight Input weight image  
 * @param spacing
 * @param lambda
 *
 * @return 
 */
ptr<MRImage> estBiasParams(ptr<const MRImage> in, ptr<const MRImage> weight, 
		double spacing, double lambda)
{
	// Create Double Bias Field
	cout << "Creating Bias Field estimate with with " << spacing 
		<< "mm spacing...";
	auto biasparams = createBiasField(in, spacing);
	cout << "Done\n";
	
	size_t ndim = in->ndim();
	size_t nparams = 1; // number of Cubic-BSpline Parameters
	size_t npixels = 1; // Number of pixels we are comparing
	for(size_t ii=0; ii<ndim; ii++) {
		nparams *= biasparams->dim(ii);
		npixels *= in->dim(ii);
	}
	vector<Eigen::Triplet<double>> pixB;
	pixB.reserve(nparams*6);
	
	// add in augmented parameters for regularization, hence the npixels+pp
	for(size_t pp=0; pp<nparams; pp++)
		pixB.push_back(Eigen::Triplet<double>(npixels+pp, pp, lambda));
	
	/********************************************************************
	 * Create Pixel, Parameter and Weight Vectors
	 ********************************************************************/
	VectorXd pixels(npixels+nparams);
	Eigen::Map<VectorXd> params((double*)biasparams->data(), nparams);
	VectorXd weights(npixels+nparams);
	FlatConstIter<double> iit(in);
	FlatConstIter<double> mit(weight);

	// b (main part)
	for(size_t ii=0; ii<npixels; ii++, ++iit, ++mit) {
		pixels[ii] = *iit;
		weights[ii] = *mit;
	}
	
	// gamma (regularization part)
	for(size_t ii=npixels; ii<npixels+nparams; ii++) {
		pixels[ii] = 0;
		weights[ii] = 1;
	}

	/************************************************************
	 * Calculate Parameter Weights at Each Pixel
	 * In the equation we are solving this would be B, p are the
	 * parameters and v are the voxel values:
	 * v = Bp
	 ***********************************************************/
	vector<pair<int64_t,int64_t>> roi(ndim);
	NDConstIter<double> bit(biasparams); // iterator of biasparams
	vector<int64_t> ind(ndim); // index
	vector<double> pt(ndim);   // point
	vector<double> cind(ndim); // continuous index
	size_t linparam, linpix;
	for(NDConstIter<double> iit(in); !iit.eof(); ++iit) {

		// get continuous index of voxel
		iit.index(ind.size(), ind.data());
		linpix = in->getLinIndex(ind);

		in->indexToPoint(ind.size(), ind.data(), pt.data());
		biasparams->pointToIndex(pt.size(), pt.data(), cind.data());
		for(size_t dd=0; dd<ndim; dd++) 
			ind[dd] = round(cind[dd]);
		
		// create ROI from nearest index to continuous one
		for(size_t dd=0; dd<ndim; dd++) {
			roi[dd].first = ind[dd]-2;
			roi[dd].second = ind[dd]+2;
		}

		// construct ROI, iterator for neighborhood of bias field parameter
		bit.setROI(roi);
		for(bit.goBegin(); !bit.eof(); ++bit) {
			bit.index(ind.size(), ind.data());
			linparam = biasparams->getLinIndex(ind);

			// compute weight
			double w = 1;
			for(size_t dd=0; dd<ndim; dd++)
				w *= B3kern(ind[dd]-cind[dd]);

			assert(linparam < nparams);
			assert(linpix < npixels);
			pixB.push_back(Eigen::Triplet<double>(linpix, linparam, w)); 
		}
	}

	/*****************************************************************
	 * Least Squares Fit
	 ****************************************************************/
	cout << "Done\nBuilding Sparse Matrix...";
	Eigen::SparseMatrix<double,Eigen::ColMajor> Bmat(npixels+nparams,nparams);
	Bmat.setFromTriplets(pixB.begin(), pixB.end());
	Bmat.makeCompressed();
	Eigen::SparseQR<Eigen::SparseMatrix<double,Eigen::ColMajor>,
			Eigen::COLAMDOrdering<int>> solver;

	// adjust with relative weighs from weight
	pixels = weights.asDiagonal()*pixels;
	
	cout << "Done\nComputing...";
	solver.compute(weights.asDiagonal()*Bmat);
	cout << "Done\nSolving...";
	params = solver.solve(pixels);
	cout << "Done" << endl;

	return biasparams;
}

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

ptr<MRImage> createBiasField(ptr<const MRImage> in, double bspace)
{
	size_t ndim = in->ndim();
	VectorXd spacing(in->ndim());
	VectorXd origin(in->ndim());
	vector<size_t> osize(ndim, 0);

	// get spacing and size
	for(size_t dd=0; dd<osize.size(); ++dd) {
		osize[dd] = 2+in->dim(dd)*in->spacing(dd)/bspace;
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

/**
 * @brief This function does a couple things (which I know violates the rule of
 * functions): it normalizes, logs and and downsampls the input (this is placed
 * into * downimg) and it downsamples the input mask (placing the result in
 * downmask)
 *
 * @param input Input image to downsample
 * @param mask Input mask to downsample (also needed to normalize masked values)
 * @param spacing Spacing of output images
 * @param outside_weight Value of masked points OUTSIDE original mask
 * @param downimg Output: Downsampled and normalized version of input
 * @param downmask Output: Downsampled version of mask (with 0 values changed
 * outside_weight)
 */
void preprocInputs(ptr<const MRImage> input, 
		ptr<const MRImage> mask, double spacing, double outside_weight,
		ptr<MRImage>& downimg, ptr<MRImage>& downmask)
{
	auto normed = dPtrCast<MRImage>(input->copy());
	
	vector<double> pt(normed->ndim());
	vector<int64_t> ind(normed->ndim());
	LinInterpNDView<double> m_interp(mask);
	m_interp.m_ras = true;
	
	/************************************************************************
	 * first normalize
	 ************************************************************************/
	double mean = 0;
	size_t count = 0;
	for(NDIter<double> iit(normed); !iit.eof(); ++iit) {
		iit.index(ind);
		normed->indexToPoint(ind.size(), ind.data(), pt.data());
		double m = m_interp(pt);
		if(m > 0.5) {
			mean += *iit;
			count++;
		}
	}
	mean /= count;
	cerr << "Masked Mean: " << mean;
	
	for(NDIter<double> iit(normed); !iit.eof(); ++iit) {
		iit.index(ind);
		normed->indexToPoint(ind.size(), ind.data(), pt.data());
		double m = m_interp(pt);
		if(m > 0.5) {
			iit.set(*iit/mean);
		} else {
			iit.set(1);
		}
	}
	normed->write("normed.nii.gz");

	for(FlatIter<double> iit(normed); !iit.eof(); ++iit) {
		if(*iit < 0)
			iit.set(0);
		else
			iit.set(log(*iit));
	}
	normed->write("logged.nii.gz");

	/************************************************************************
	 * Now Downsample
	 ***********************************************************************/
	vector<double> dspace(normed->ndim(), spacing);
	cout << "Downsampling input from [";
	for(size_t ii=0; ii<3; ii++) {
		if(ii!=0) cout << ", ";
		cout << normed->spacing(ii);
	}
	cout << "] spacing to [";
	downimg = dPtrCast<MRImage>(resample(normed, dspace.data()));
	for(size_t ii=0; ii<3; ii++) {
		if(ii!=0) cout << ", ";
		cout << downimg->spacing(ii);
	}
	cout << "] spacing...";
	cout << "Done\n";

	/************************************************************************
	 * And Downsample Mask
	 ***********************************************************************/
	downmask = dPtrCast<MRImage>(downimg->createAnother());
	for(NDIter<double> mit(downmask); !mit.eof(); ++mit) {
		mit.index(ind);
		downmask->indexToPoint(ind.size(), ind.data(), pt.data());
		double m = m_interp(pt);
		if(m > 0.5) {
			mit.set(1);
		} else {
			mit.set(outside_weight);
		}
	}
}
