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
#include "macros.h"

using namespace npl;
using namespace std;

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
		ptr<const MRImage> mask, double spacing,
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
 * @brief Performs bias field estimation, the mask will be used as weights
 *
 * @param in MRImage with bias field
 * @param mask Weights of respective pixels
 * @param spacing Spacing of bias field image
 * @param lambda Regularization weight
 *
 * @return
 */
ptr<MRImage> estBiasParams(ptr<const MRImage> in, ptr<const MRImage> mask,
		double spacing, double lambda);

/**
 * @brief Constructs a bias field in the space of input, based on the
 * parameters in biasparams
 *
 * @param biasparams Parameters of bias field
 * @param input Input spacing image
 *
 * @return Bias field sampled on the input grid
 */
ptr<MRImage> reconstructBiasField(ptr<const MRImage> biasparams,
		ptr<const MRImage> input);

/**
 * @brief The main function
 *
 * @param argc
 * @param argv
 *
 * @return
 */
int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
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
			"for bias-field estimation.", false, 30, "mm", cmd);
	TCLAP::ValueArg<double> a_downspace("d", "downsample", "Spacing in "
			"downsampled image. This is primarily to speed up processing. "
			"Because the problem is already overdetermined this doesn't impact "
			"results very much. So a 5 here would resample the image to 5x5x5 "
			"isotropic voxels.", false, 10, "pixsize", cmd);
	TCLAP::ValueArg<double> a_lambda("R", "regweight", "Regularization weight "
			"for ridge regression. Larger values will force a smoother result,"
			" and values closer to 1.",
			false, 1.e-1, "real", cmd);
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

	/****************************************************
	 * Read or Create Mask
	 ****************************************************/
	cout << "Loading/Creating Mask"<<endl;
	ptr<MRImage> fullmask;
	if(a_mask.isSet()) {
		fullmask = readMRImage(a_mask.getValue());

		// binarize
		for(FlatIter<double> fit(fullmask); !fit.eof(); ++fit) {
			double v = *fit;
			if(v > 0)
				fit.set(1);
			else
				fit.set(0);
		}

	} else {
		fullmask = dPtrCast<MRImage>(fullres->copyCast(FLOAT64));
		double thresh = otsuThresh(fullres);

		// binarize/threshold
		cout << " (Threshold: " << thresh << ") " << endl;
		for(FlatIter<double> fit(fullmask); !fit.eof(); ++fit) {
			fit.set(*fit > thresh);
		}
		fullmask = dPtrCast<MRImage>(erode(fullmask, 1));
	}
#if defined VERYDEBUG || DEBUG
	fullmask->write("fullmask.nii.gz");
#endif

	/****************************************************
	 * Create Downsampled, logged Versions if Inputs
	 ****************************************************/
	cout << "Downsampling/Normalizing" << endl;
	ptr<MRImage> dinput;
	ptr<MRImage> dmask;
	preprocInputs(fullres, fullmask, a_downspace.getValue(), dinput, dmask);
	fullmask.reset();

	/********************************************************************
	 * Estimate Bias Field Parameters from Pixels and Weights
	 ********************************************************************/
	cout << "Estimating Bias Field" << endl;
	auto biasparams = estBiasParams(dinput, dmask, a_spacing.getValue(),
			a_lambda.getValue());
	if(a_biasparams.isSet())
		biasparams->write(a_biasparams.getValue());

	/*************************************************************************
	 * estimate the bias field from the parameters
	 ************************************************************************/

	fullres = reconstructBiasField(biasparams, fullres);
#if defined VERYDEBUG || DEBUG
	fullres->write("logbias.nii.gz");
#endif
	for(FlatIter<double> it(fullres); !it.eof(); ++it) {
		double v = exp(*it);
		if(std::isinf(*it) || std::isnan(*it)) {
//			cerr << "Unusual bias field value (nan/inf) found" << endl;
			it.set(1);
		} else {
			it.set(v);
		}
	}
	if(a_biasfield.isSet())
		fullres->write(a_biasfield.getValue());

	/*************************************************************************
	 * Finally Write the Output
	 ************************************************************************/
	cout << "Writing output" << endl;
	if(a_corimage.isSet()) {
		// Re-Read Input (even if it is 4D)
		auto input = readMRImage(a_in.getValue());

		vector<int64_t> ind(fullres->ndim());
		NDView<double> b_vw(fullres);
		for(NDIter<double> iit(input); !iit.eof(); ++iit) {
			iit.index(ind);
			iit.set(iit.get()/b_vw[ind]);
		}
		input->write(a_corimage.getValue());
	}

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
	const double ABSLIMIT = 5;

	// Create Double Bias Field
	cout << "Creating Bias Field estimate with with " << spacing
		<< "mm spacing" << endl;
	auto biasparams = createBiasField(in, spacing);

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

			// Apply Weighting By Row, since we want to multiply the diagonal
			// weighting matrix by the B matrix from the left
			assert(linparam < nparams);
			assert(linpix < npixels+nparams);
			pixB.push_back(Eigen::Triplet<double>(linpix, linparam,
						w*weights[linpix]));
		}
	}

	/*****************************************************************
	 * Least Squares Fit
	 ****************************************************************/
	Eigen::SparseMatrix<double,Eigen::ColMajor> Bmat(npixels+nparams,nparams);
	Bmat.setFromTriplets(pixB.begin(), pixB.end());

	Bmat.makeCompressed();
	Eigen::SparseQR<Eigen::SparseMatrix<double,Eigen::ColMajor>,
			Eigen::COLAMDOrdering<int>> solver;
	solver.compute(Bmat);

	// adjust with relative weighs from weight
	cout << "Solving Params"<<endl;
	params = solver.solve(weights.asDiagonal()*pixels);

	// Find Values that are < 1/5 or more thant 5x
	for(size_t cc=0; cc<params.rows(); cc++) {
		if(params[cc] > log(ABSLIMIT) || params[cc] < log(1./ABSLIMIT))
			params[cc] = 0;
	}
	return biasparams;
}

/**
 * @brief Reconstruct a bias field in the spacing of the input image
 *
 * @param input Image to use sa the template for the bias field parameters
 *
 * @return (log) Bias field image
 */
ptr<MRImage> reconstructBiasField(ptr<const MRImage> biasparams,
		ptr<const MRImage> input)
{
	cout << "Estimating Bias Field From Parameters";
	if(biasparams->getDirection() != input->getDirection()) {
		throw INVALID_ARGUMENT("Input bias parameters and sample image do "
				"not have identical direction matrices!");
	}

	auto out = dPtrCast<MRImage>(input->createAnother());

	// for each kernel, iterate over the points in the neighborhood
	size_t ndim = input->ndim();
	vector<pair<int64_t,int64_t>> roi(ndim);
	NDIter<double> pit(out); // iterator of pixels
	vector<int64_t> pind(ndim); // index of pixel
	vector<int64_t> ind(ndim); // index
	vector<double> pt(ndim);   // point
	vector<double> cind(ndim); // continuous index

	vector<int> winsize(ndim);
	vector<vector<double>> karray(ndim);
	vector<vector<int>> iarray(ndim);
	for(size_t dd=0; dd<ndim; dd++) {
		winsize[dd] = 1+4*ceil(biasparams->spacing(dd)/out->spacing(dd));
		karray[dd].resize(winsize[dd]);
	}

	// we go through each parameter, and compute the weight of the B-spline
	// parameter at each pixel within the range (2 indexes in parameter
	// space, 2*S_B/S_I indexs in pixel space)
	for(NDConstIter<double> bit(biasparams); !bit.eof(); ++bit) {

		// get continuous index of pixel
		bit.index(ind.size(), ind.data());
		biasparams->indexToPoint(ind.size(), ind.data(), pt.data());
		out->pointToIndex(pt.size(), pt.data(), cind.data());

		// construct weights / construct ROI
		double dist = 0;
		for(size_t dd=0; dd<ndim; dd++) {
			pind[dd] = round(cind[dd]); //pind is the center
			for(int ww=-winsize[dd]/2; ww<=winsize[dd]/2; ww++) {
				dist = (pind[dd]+ww-cind[dd])*out->spacing(dd)/biasparams->spacing(dd);
				karray[dd][ww+winsize[dd]/2] = B3kern(dist);
			}
			roi[dd].first = pind[dd]-winsize[dd]/2;
			roi[dd].second = pind[dd]+winsize[dd]/2;
		}

		pit.setROI(roi);
		for(pit.goBegin(); !pit.eof(); ++pit) {
			pit.index(ind);
			double w = 1;
			for(size_t dd=0; dd<ndim; dd++)
				w *= karray[dd][ind[dd]-pind[dd]+winsize[dd]/2];
			pit.set(*pit + w*(*bit));
		}
	}

	return out;
}

ptr<MRImage> createBiasField(ptr<const MRImage> in, double bspace)
{
	size_t ndim = in->ndim();
	VectorXd spacing(in->ndim());
	VectorXd origin(in->ndim());
	vector<size_t> osize(ndim, 0);

	// get spacing and size
	for(size_t dd=0; dd<osize.size(); ++dd) {
		osize[dd] = ceil(in->dim(dd)*in->spacing(dd)/bspace);
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

	for(FlatIter<double> it(biasparams); !it.eof(); ++it)
		it.set(0);

	return biasparams;
}

double avg(ptr<const MRImage> input, ptr<const MRImage> mask)
{
	LinInterpNDView<double> m_interp(mask);
	m_interp.m_ras = true;
	vector<double> pt(input->ndim());
	vector<int64_t> ind(input->ndim());

	size_t count = 0;
	double sum = 0;
	for(NDConstIter<double> fit(input); !fit.eof(); ++fit) {
		fit.index(ind);
		input->indexToPoint(ind.size(), ind.data(), pt.data());
		double m = m_interp(pt);
		if(m > 0.5) {
			sum += *fit;
			count++;
		}
	}

	return sum/count;
}

double mode(ptr<const MRImage> input, ptr<const MRImage> mask)
{
	LinInterpNDView<double> m_interp(mask);
	m_interp.m_ras = true;

	vector<double> bins(sqrt(input->elements()));
	vector<double> pt(input->ndim());
	vector<int64_t> ind(input->ndim());
	double minv = INFINITY;
	double maxv = -INFINITY;
	for(NDConstIter<double> fit(input); !fit.eof(); ++fit) {
		fit.index(ind);
		input->indexToPoint(ind.size(), ind.data(), pt.data());
		double m = m_interp(pt);
		if(m > 0.5) {
			minv = std::min(minv, *fit);
			maxv = std::max(maxv, *fit);
		}
	}
	double bwidth = 0.99999999*bins.size()/(maxv-minv);
	for(NDConstIter<double> fit(input); !fit.eof(); ++fit) {
		fit.index(ind);
		input->indexToPoint(ind.size(), ind.data(), pt.data());
		double m = m_interp(pt);
		if(m > 0.5) {
			bins[floor((*fit-minv)*bwidth)]++;
		}
	}

	maxv = -INFINITY;
	int maxb = -1;
	for(size_t bb=0; bb < bins.size(); bb++) {
		if(bins[bb] > maxv) {
			maxv = bins[bb];
			maxb = bb;
		}
	}

	return maxb/bwidth + minv;
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
 * to outside_weight)
 */
void preprocInputs(ptr<const MRImage> input,
		ptr<const MRImage> mask, double spacing,
		ptr<MRImage>& downimg, ptr<MRImage>& downmask)
{
	auto normed = dPtrCast<MRImage>(input->copy());

	vector<double> pt(normed->ndim());
	vector<int64_t> ind(normed->ndim());
	LinInterpNDView<double> m_interp(mask);
	m_interp.m_ras = true;

	/************************************************************************
	 * first normalize, dividing by the mode
	 ************************************************************************/
//	double normval = normval(normed, mask);
//	cout  << " (Mode for normalizing: " << normval << " ) "; ;
	double normval = avg(normed, mask);
	cout  << " (Average for normalizing: " << normval << " ) "; ;

	for(NDIter<double> iit(normed); !iit.eof(); ++iit) {
		iit.index(ind);
		normed->indexToPoint(ind.size(), ind.data(), pt.data());
		double m = m_interp(pt);
		if(m > 0.5) {
			iit.set(*iit/normval);
		} else {
			iit.set(1);
		}
	}
#if defined VERYDEBUG || DEBUG
	normed->write("normed.nii.gz");
#endif

	for(FlatIter<double> iit(normed); !iit.eof(); ++iit) {
		if(*iit < 0)
			iit.set(0);
		else
			iit.set(log(*iit));
	}
#if defined VERYDEBUG || DEBUG
	normed->write("logged.nii.gz");
#endif

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
			mit.set(0);
		}
	}
}
