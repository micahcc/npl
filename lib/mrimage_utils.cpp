/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file mrimage_utils.cpp Utilities for operating on MR Images. These are
 * functions which are sensitive to MR variables, such as spacing, orientation,
 * slice timing etc. ndarray_utils contains more general purpose tools.
 *
 *****************************************************************************/

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "mrimage.h"
#include "iterators.h"
#include "accessors.h"
#include "ndarray_utils.h"
#include "mrimage_utils.h"
#include "registration.h"
#include "byteswap.h"
#include "macros.h"

#include "fftw3.h"

#include <string>
#include <iostream>
#include <string>
#include <iomanip>
#include <cassert>
#include <memory>
#include <cstring>
#include <cmath>
#include <random>

namespace npl {

using std::vector;

#define VERYDEBUG

#ifdef VERYDEBUG
#define DEBUGWRITE(FOO) FOO
#else
#define DEBUGWRITE(FOO)
#endif

/**
 * @brief Writes a pair of images, one real, one imaginary or if absPhase is
 * set to true then an absolute image and a phase image.
 *
 * @param basename Base filename _abs.nii.gz and _phase.nii.gz or _re.nii.gz
 * and _im.nii.gz will be appended, depending on absPhase
 * @param in Input image
 * @param absPhase Whether the break up into absolute and phase rather than
 * re/imaginary
 */
void writeComplex(std::string basename, ptr<const MRImage> in,
        bool absPhase)
{
	auto img1 = dPtrCast<MRImage>(in->copyCast(FLOAT64));
	auto img2 = dPtrCast<MRImage>(in->copyCast(FLOAT64));

	OrderIter<double> it1(img1);
	OrderIter<double> it2(img2);
	OrderConstIter<cdouble_t> init(in);
	for(; !init.eof(); ++init, ++it1, ++it2) {
		if(absPhase) {
			it1.set(abs(*init));
			it2.set(arg(*init));
		} else {
			it1.set((*init).real());
			it2.set((*init).imag());
		}
	}

	if(absPhase) {
		img1->write(basename + "_abs.nii.gz");
		img2->write(basename + "_ang.nii.gz");
	} else {
		img1->write(basename + "_re.nii.gz");
		img2->write(basename + "_im.nii.gz");
	}
}

/**
 * @brief Performs forward FFT transform in N dimensions.
 *
 * @param in Input image
 * @param in_osize Size of output image (will be padded up to this prior to
 * FFT)
 *
 * @return Frequency domain of input. Note the output will be
 * COMPLEX128/CDOUBLE type
 */
ptr<MRImage> fft_forward(ptr<const MRImage> in,
		const std::vector<size_t>& in_osize)
{

	// make sure osize matches input dimensions
	vector<size_t> osize(in_osize);
	osize.resize(in->ndim(), 1);
	size_t ndim = osize.size();

	// create padded NDArray, allocated with fftw
	size_t opixels = 1;
	vector<int> osize32(ndim);
	for(size_t ii=0; ii<ndim; ii++) {
		opixels *= osize[ii];
		osize32[ii] = osize[ii];

		if(osize[ii] < in->dim(ii))
			throw std::invalid_argument("Input image larger than output size!"
					" In\n" + __FUNCTION_STR__);
	}

	auto outbuff = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*opixels);
	auto output = createMRImage(osize.size(), osize.data(), CDOUBLE,
			outbuff, [](void* ptr) {fftw_free(ptr);});
	output->copyMetadata(in);

	// create ND FFTW Plan
	auto fwd = fftw_plan_dft((int)ndim, osize32.data(), outbuff, outbuff,
			FFTW_FORWARD, FFTW_MEASURE);
	for(size_t ii=0; ii<opixels; ii++) {
		outbuff[ii][0] = 0;
		outbuff[ii][1] = 0;
	}

	// fill padded from input
	OrderConstIter<cdouble_t> iit(in);
	OrderIter<cdouble_t> pit(output);
	pit.setROI(ndim, in->dim());
	pit.setOrder(iit.getOrder());
	for(iit.goBegin(), pit.goBegin(); !iit.eof() && !pit.eof(); ++pit, ++iit)
		pit.set(*iit);
	assert(iit.eof() && pit.eof());

	DEBUGWRITE(writeComplex("forward_prefft", output));

	// fourier transform
	fftw_execute(fwd);

#ifndef NDEBUG
	OrderIter<cdouble_t> it(output);;
	for(size_t ii=0; !it.eof(); ii++, ++it) {
		cdouble_t tmp(*it);
		assert(tmp.real() == outbuff[ii][0]);
		assert(tmp.imag() == outbuff[ii][1]);
	}
#endif

	// normalize
	double normf = 1./opixels;
	for(size_t ii=0; ii<opixels; ii++) {
		outbuff[ii][0] = normf*outbuff[ii][0];
		outbuff[ii][1] = normf*outbuff[ii][1];
	}

	DEBUGWRITE(writeComplex("forward_postfft", output));

	return output;
}

/**
 * @brief Performs inverse FFT transform in N dimensions.
 *
 * @param in Input image
 * @param in_osize Size of output image. If this is smaller than the input then
 * the frequency domain will be trunkated, if it is larger then the fourier
 * domain will be padded ( output upsampled )
 *
 * @return Frequency domain of input. Note the output will be
 * COMPLEX128/CDOUBLE type
 */
ptr<MRImage> fft_backward(ptr<const MRImage> in,
		const std::vector<size_t>& in_osize)
{

	// make sure osize matches input dimensions
	vector<size_t> osize(in_osize);
	osize.resize(in->ndim(), 1);
	size_t ndim = osize.size();

	// create output NDArray, allocated with fftw
	size_t opixels = 1;
	vector<int> osize32(ndim);
	for(size_t ii=0; ii<ndim; ii++) {
		opixels *= osize[ii];
		osize32[ii] = osize[ii];
	}

	auto outbuff = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*opixels);
	auto output = createMRImage(osize.size(), osize.data(), CDOUBLE,
			outbuff, [](void* ptr) {fftw_free(ptr);});
	output->copyMetadata(in);

	// create ND FFTW Plan
	auto plan = fftw_plan_dft((int)ndim, osize32.data(), outbuff, outbuff,
			FFTW_BACKWARD, FFTW_MEASURE);
	for(size_t ii=0; ii<opixels; ii++) {
		outbuff[ii][0] = 0;
		outbuff[ii][1] = 0;
	}

	// fill padded from input
	NDConstView<cdouble_t> iacc(in);
	OrderIter<cdouble_t> it(output);
	vector<int64_t> iindex(ndim);
	vector<int64_t> oindex(ndim);
	for(it.goBegin(); !it.eof(); ++it) {
		it.index(oindex);

		// if the curent oindex doesn't exist in the input (due to output size
	// being larger than input size), then leave as 0
		bool skip = false;

		// compute input index, handling frequency unrwrapping
		int64_t ilen, olen;
		for(size_t dd=0; dd<ndim; dd++) {
			ilen = in->dim(dd);
			olen = output->dim(dd);

			if(oindex[dd] < olen/2) {
				iindex[dd] = oindex[dd];
				if(iindex[dd] >= ilen/2) {
					skip = true;
					break;
				}
			} else  {
				// negative frequencies
				iindex[dd] = ilen-(olen-oindex[dd]);
				if(iindex[dd] < ilen/2) {
					skip = true;
					break;
				}
			}

		}

		if(skip)
			continue;

		it.set(iacc[iindex]);
	}

	// fourier transform
	fftw_execute(plan);

	return output;
}

/**
 * @brief Performs fourier resampling using fourier transform and the provided
 * window function.
 *
 * TODO fix for upsampling
 *
 * Given Lv (input length), Lz (input pad), and Lu (output size),
 * Padded size = Lv + Lz
 * Truncated/Padded Fourier domain = Lv + Lz + Ly (Ly may be negative)
 * Output size Lu = (Ly+Lz+Lz)*Lv/(Lv+Lz)
 * The padding in fourier domain Ly = (Lv+Lz)(Lu-Lv)/Lv
 * Ly may be negative in case of downsampling
 *
 * @param in Input image
 * @param spacing Desired output spacing
 * @param window Window function  to reduce ringing
 *
 * @return  Smoothed and downsampled image
 */
ptr<MRImage> resample(ptr<const MRImage> in, double* spacing,
		double(*window)(double, double))
{
	size_t ndim = in->ndim();

	// create downsampled image
	vector<int64_t> isize(in->dim(), in->dim()+ndim); //input size
	vector<int64_t> psize(in->ndim()); // padsize
	vector<int64_t> rsize(in->ndim()); // truncated/padded frequency domain length
	vector<int64_t> osize(ndim); // output size

	int64_t linelen = 0;
	for(size_t dd=0; dd<ndim; dd++) {
		// compute ratio
		double ratio = in->spacing(dd)/spacing[dd];
		psize[dd] = round2(2*isize[dd]);
		osize[dd] = ceil(isize[dd]*ratio);
		rsize[dd] = psize[dd]*osize[dd]/isize[dd];

		linelen = max(linelen, rsize[dd]);
		linelen = max(linelen, psize[dd]);
	}

	vector<size_t> roi(in->dim(), in->dim()+ndim);
	auto working = dPtrCast<MRImage>(in->copyCast(COMPLEX128));
	//	writeComplex("workinginit", working);
	auto ibuffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*linelen*2);
	auto obuffer = &ibuffer[linelen];
	for(size_t dd=0; dd<ndim; dd++) {
		auto fwd = fftw_plan_dft_1d((int)psize[dd], ibuffer, obuffer,
				FFTW_FORWARD, FFTW_MEASURE);
		auto bwd = fftw_plan_dft_1d((int)rsize[dd], ibuffer, obuffer,
				FFTW_BACKWARD, FFTW_MEASURE);

		// extract line
		ChunkIter<cdouble_t> it(working);
		it.setROI(roi.size(), roi.data());
		it.setLineChunk(dd);
		for(it.goBegin(); !it.eof(); it.nextChunk()) {
			int64_t ii=0;
			for(it.goChunkBegin(), ii=0; !it.eoc(); ++it, ++ii) {
				ibuffer[ii][0] = (*it).real();
				ibuffer[ii][1] = (*it).imag();
			}
			for(; ii<psize[dd]; ii++){
				ibuffer[ii][0] = 0;
				ibuffer[ii][1] = 0;
			}

			// fourier tansform line
			fftw_execute(fwd);

			double normf = 1./psize[dd];
			// zero all
			for(ii=0; ii<rsize[dd]; ii++) {
				ibuffer[ii][0] = 0;
				ibuffer[ii][1] = 0;
			}
			// positive frequencies
			for(ii=0; ii<(min(rsize[dd],psize[dd])+1)/2; ii++) {
				double w = window(ii, psize[dd]/2.);
				ibuffer[ii][0] = obuffer[ii][0]*w*normf;
				ibuffer[ii][1] = obuffer[ii][1]*w*normf;
			}
			// negative frequencies
			for(ii=1; ii<=(min(rsize[dd],psize[dd]))/2; ii++) {
				double w = window(ii, psize[dd]/2.);
				ibuffer[rsize[dd]-ii][0] = obuffer[psize[dd]-ii][0]*w*normf;
				ibuffer[rsize[dd]-ii][1] = obuffer[psize[dd]-ii][1]*w*normf;
			}

			// inverse fourier tansform
			fftw_execute(bwd);

			// write out (ignore zero extra area)
			for(it.goChunkBegin(), ii=0; ii<osize[dd]; ++it, ++ii) {
				cdouble_t tmp(obuffer[ii][0], obuffer[ii][1]);
				it.set(tmp);
			}
		}

		// update ROI
		roi[dd] = osize[dd];
		DBG3(cerr << isize[dd] << "->" << osize[dd] << endl);
//		writeComplex("working"+to_string(dd), working);
	}

	// copy roi into output
	vector<size_t> trueosize(in->ndim());
	for(size_t dd=0; dd<in->ndim(); dd++) trueosize[dd] = osize[dd];
	auto out = dPtrCast<MRImage>(working->copyCast(osize.size(),
				trueosize.data(), FLOAT64));

	// set spacing
	for(size_t dd=0; dd<in->ndim(); dd++)
		out->spacing(dd) *= ((double)psize[dd])/((double)rsize[dd]);

	fftw_free(ibuffer);
	return out;
}

/**
 * @brief Performs smoothing in each dimension, then downsamples so that pixel
 * spacing is roughly equal to FWHM.
 *
 * /todo less memory allocation, reuse fftw_alloc data
 *
 * @param in    Input image
 * @param sigma Standard deviation for smoothing
 * @param spacing Ouptut image spacing (isotropic). If this is <= 0, then sigma
 * will be used, which is a very conservative downsampling.
 *
 * @return  Smoothed and downsampled image
 */
ptr<MRImage> smoothDownsample(ptr<const MRImage> in, double sigma, double spacing)
{
	size_t ndim = in->ndim();

	// create downsampled image
	vector<int64_t> isize(in->dim(), in->dim()+ndim); //input size
	vector<int64_t> psize(in->ndim()); // padsize
	vector<int64_t> rsize(in->ndim()); // truncated/padded frequency domain length
	vector<int64_t> osize(ndim); // output size

	if(spacing <= 0)
		spacing = sigma;

	// Enforce Isotropic Pixels
	for(size_t dd=0; dd<in->ndim(); dd++)
		spacing = max(spacing, in->spacing(dd));

	int64_t linelen = 0;
	for(size_t dd=0; dd<ndim; dd++) {
		// compute ratio
		double ratio = in->spacing(dd)/spacing;
		assert(ratio <= 1 && ratio > 0);
		psize[dd] = round2(2*isize[dd]);
		osize[dd] = ceil(isize[dd]*ratio);
		rsize[dd] = psize[dd]*osize[dd]/isize[dd];

		linelen = max(linelen, rsize[dd]);
		linelen = max(linelen, psize[dd]);
	}

	vector<size_t> roi(in->dim(), in->dim()+ndim);
	auto working = dPtrCast<MRImage>(in->copyCast(COMPLEX128));
	//	writeComplex("workinginit", working);
	auto ibuffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*linelen*2);
	auto obuffer = &ibuffer[linelen];
	for(size_t dd=0; dd<ndim; dd++) {
		auto fwd = fftw_plan_dft_1d((int)psize[dd], ibuffer, obuffer,
				FFTW_FORWARD, FFTW_MEASURE);
		auto bwd = fftw_plan_dft_1d((int)rsize[dd], ibuffer, obuffer,
				FFTW_BACKWARD, FFTW_MEASURE);

		double sd = sigma/in->spacing(dd);

		// extract line
		ChunkIter<cdouble_t> it(working);
		it.setROI(roi.size(), roi.data());
		it.setLineChunk(dd);
		for(it.goBegin(); !it.eof(); it.nextChunk()) {
			int64_t ii=0;
			for(it.goChunkBegin(), ii=0; !it.eoc(); ++it, ++ii) {
				ibuffer[ii][0] = (*it).real();
				ibuffer[ii][1] = (*it).imag();
			}
			for(; ii<psize[dd]; ii++){
				ibuffer[ii][0] = 0;
				ibuffer[ii][1] = 0;
			}

			// fourier tansform line
			fftw_execute(fwd);

			double normf = 1./psize[dd];
			// zero all
			for(ii=0; ii<rsize[dd]; ii++) {
				ibuffer[ii][0] = 0;
				ibuffer[ii][1] = 0;
			}
			// positive frequencies
			for(ii=0; ii<(min(rsize[dd],psize[dd])+1)/2; ii++) {
				double ff = 2.*ii/psize[dd];
				double w = exp(-M_PI*M_PI*ff*ff*2*sd*sd);
				ibuffer[ii][0] = obuffer[ii][0]*w*normf;
				ibuffer[ii][1] = obuffer[ii][1]*w*normf;
			}
			// negative frequencies
			for(ii=1; ii<=(min(rsize[dd],psize[dd]))/2; ii++) {
				double ff = 2.*ii/psize[dd];
				double w = exp(-M_PI*M_PI*ff*ff*2*sd*sd);
				ibuffer[rsize[dd]-ii][0] = obuffer[psize[dd]-ii][0]*w*normf;
				ibuffer[rsize[dd]-ii][1] = obuffer[psize[dd]-ii][1]*w*normf;
			}

			// inverse fourier tansform
			fftw_execute(bwd);

			// write out (ignore zero extra area)
			for(it.goChunkBegin(), ii=0; ii<osize[dd]; ++it, ++ii) {
				cdouble_t tmp(obuffer[ii][0], obuffer[ii][1]);
				it.set(tmp);
			}
		}

		// update ROI
		roi[dd] = osize[dd];
		DBG3(cerr << isize[dd] << "->" << osize[dd] << endl);
//		writeComplex("working"+to_string(dd), working);
	}

	// copy roi into output
	vector<size_t> trueosize(in->ndim());
	for(size_t dd=0; dd<in->ndim(); dd++) trueosize[dd] = osize[dd];
	auto out = dPtrCast<MRImage>(working->copyCast(osize.size(),
				trueosize.data(), FLOAT64));

	// set spacing
	for(size_t dd=0; dd<in->ndim(); dd++) {
		out->spacing(dd) *= ((double)psize[dd])/((double)rsize[dd]);
	}

	fftw_free(ibuffer);
	return out;
}

/**
 * @brief Smooths an image in 1 dimension
 *
 * @param inout Input/output image to smooth
 * @param dim dimensions to smooth in. If you are smoothing individual volumes
 * of an fMRI you would provide dim={0,1,2}
 * @param stddev standard deviation in physical units index*spacing
 *
 */
void gaussianSmooth1D(ptr<MRImage> inout, size_t dim,
		double stddev)
{
	if(stddev <= 0)
		return;

	const auto gaussKern = [](double x)
	{
	  const double PI = acos(-1);
	  const double den = 1./sqrt(2*PI);
	  return den*exp(-x*x/(2));
  };

	if(dim >= inout->ndim()) {
		throw std::out_of_range("Invalid dimension specified for 1D gaussian "
				"smoothing");
	}

	std::vector<int64_t> index(dim, 0);
	stddev /= inout->spacing(dim);
	std::vector<double> buff(inout->dim(dim));

	// for reading have the kernel iterator
	KernelIter<double> kit(inout);
	std::vector<size_t> radius(inout->ndim(), 0);
	for(size_t dd=0; dd<inout->ndim(); dd++) {
		if(dd == dim)
			radius[dd] = round(2*stddev);
	}
	kit.setRadius(radius);
	kit.goBegin();

	// calculate normalization factor
	double normalize = 0;
	int64_t rad = radius[dim];
	for(int64_t ii=-rad; ii<=rad; ii++)
		normalize += gaussKern(ii/stddev);

	// for writing, have the regular iterator
	OrderIter<double> it(inout);
	it.setOrder(kit.getOrder());
	it.goBegin();
	while(!it.eof()) {

		// perform kernel math, writing to buffer
		for(size_t ii=0; ii<inout->dim(dim); ii++, ++kit) {
			double tmp = 0;
			for(size_t kk=0; kk<kit.ksize(); kk++) {
				double dist = kit.offsetK(kk, dim);
				double nval = kit[kk];
				double stddist = dist/stddev;
				double weighted = gaussKern(stddist)*nval/normalize;
				tmp += weighted;
			}
			buff[ii] = tmp;
		}

		// write back out
		for(size_t ii=0; ii<inout->dim(dim); ii++, ++it)
			it.set(buff[ii]);

	}
}

//
///**
// * @brief Uses fourier shift theorem to shift an image
// *
// * @param in Input image to shift
// * @param len length of dx array
// * @param dx movement in physical coordinates
// *
// * @return shifted image
// */
//ptr<MRImage> shiftImageFFT(ptr<MRImage> in, size_t len, double* dx)
//{
//
//	auto out = dPtrCast<MRImage>(in->copy());
//	std::vector<double> shift(len);
//	in->disOrientVector(len, dx, shift.data());
//
//	// for each dimension
//	for(size_t ii=0; ii<len && ii<in->ndim(); ii++) {
//		shiftImageFFT(out, ii, dx[ii]);
//	}
//
//	return out;
//}


/**
 * @brief Rotates an image around the center using shear decomposition followed
 * by kernel-based shearing. Rotation order is Rz, Ry, Rx, and about the center
 * of the image. This means that 1D interpolation will be used.
 *
 * @param inout Input/output image
 * @param rx Rotation about x axis
 * @param ry Rotation about y axis
 * @param rz Rotation about z axis
 */
int rotateImageShearKern(ptr<MRImage> inout, double rx, double ry, double rz,
		double(*kern)(double,double))
{
	if(!inout->isIsotropic(true, 0.01))
		cerr << "Warning Shear Rotation with non-isotropic voxels "
			"experimental!" << endl;

	const double PI = acos(-1);
	if(fabs(rx) > PI/4. || fabs(ry) > PI/4. || fabs(rz) > PI/4.) {
		cerr << "Fast large rotations not yet implemented" << endl;
		return -1;
	}

	std::list<Matrix3d> shears;

	// decompose into shears
	clock_t c = clock();
	if(shearDecompose(shears, rx, ry, rz, inout->spacing(0),
				inout->spacing(1), inout->spacing(2)) != 0) {
		cerr << "Failed to find valid shear matrices" << endl;
		return -1;
	}
	c = clock() - c;
    shears.reverse();
	DBG3(cerr << "Shear Decompose took " << c << " ticks " << endl);

	// perform shearing
	double shearvals[3];
	for(const Matrix3d& shmat: shears) {
		int64_t sheardim = -1;;
		for(size_t rr = 0 ; rr < 3 ; rr++) {
			for(size_t cc = 0 ; cc < 3 ; cc++) {
				if(rr != cc && shmat(rr,cc) != 0) {
					if(sheardim != -1 && sheardim != rr) {
						cerr << "Error, multiple shear dimensions!" << endl;
						return -1;
					}
					sheardim = rr;
					shearvals[cc] = shmat(rr,cc);
				}
			}
		}

		// perform shear
		if(sheardim != -1) {
			shearImageKern(dPtrCast<NDArray>(inout), sheardim, 3, shearvals, kern);
		}

	}

	return 0;
}

/**
 * @brief Rotates an image around the center using shear decomposition followed
 * by FFT-based shearing. Rotation order is Rz, Ry, Rx, and about the center of
 * the image.
 *
 * @param inout Input/output image
 * @param rx Rotation about x axis
 * @param ry Rotation about y axis
 * @param rz Rotation about z axis
 * @param rz Rotation about z axis
 * @param window Window function to apply in fourier domain
 */
int rotateImageShearFFT(ptr<MRImage> inout, double rx, double ry, double rz,
		double(*window)(double,double))
{
	if(!inout->isIsotropic(true, 0.01))
		cerr << "Warning Shear Rotation with non-isotropic voxels "
			"experimental!" << endl;

	const double PI = acos(-1);
	if(fabs(rx) > PI/4. || fabs(ry) > PI/4. || fabs(rz) > PI/4.) {
		cerr << "Fast large rotations not yet implemented" << endl;
		return -1;
	}

	std::list<Matrix3d> shears;

	// decompose into shears
	clock_t c = clock();
	if(shearDecompose(shears, rx, ry, rz, inout->spacing(0), inout->spacing(1),
				inout->spacing(2)) != 0) {
		cerr << "Failed to find valid shear matrices" << endl;
		return -1;
	}
	c = clock() - c;
    shears.reverse();
	DBG3(cerr << "Shear Decompose took " << c << " ticks " << endl);

	// perform shearing
	double shearvals[3];
	for(const Matrix3d& shmat: shears) {
		int64_t sheardim = -1;;
		for(size_t rr = 0 ; rr < 3 ; rr++) {
			for(size_t cc = 0 ; cc < 3 ; cc++) {
				if(rr != cc && shmat(rr,cc) != 0) {
					if(sheardim != -1 && sheardim != rr) {
						cerr << "Error, multiple shear dimensions!" << endl;
						return -1;
					}
					sheardim = rr;
					shearvals[cc] = shmat(rr,cc);
				}
			}
		}

		// perform shear (if there is one - if there isn't do nothing)
		if(sheardim != -1) {
			shearImageFFT(dPtrCast<NDArray>(inout), sheardim, 3, shearvals, window);
		}
	}

	return 0;
}

/**
 * @brief Rigid Transforms an image in RAS coordinates. This isn't that fast
 * but it is definitely RIGHT
 *
 * @param inout Input/output image
 * @param rx Rotation about x axis
 * @param ry Rotation about y axis
 * @param rz Rotation about z axis
 * @param sx shift along x axis (mm)
 * @param sy shift along y axis (mm)
 * @param sz shift along z axis (mm)
 */
ptr<MRImage> rigidTransform(ptr<MRImage> in, double rx, double ry, double rz,
		double sx, double sy, double sz)
{
	// Set Rotation/Shift
	Matrix3d R, Rinv;
	R = Eigen::AngleAxisd(rx, Vector3d::UnitX())*
		Eigen::AngleAxisd(ry, Vector3d::UnitY())*
		Eigen::AngleAxisd(rz, Vector3d::UnitZ());

	Vector3d shift, ishift, center;
	shift[0] = sx;
	shift[1] = sy;
	shift[2] = sz;

	// Set Center
	for(size_t dd=0; dd<3; dd++)
		center[dd] = (in->dim(dd)-1)/2.;
	in->indexToPoint(3, center.array().data(), center.array().data());

	// Invert
	Rinv = R.inverse();
	ishift = -R*shift;

	LanczosInterp3DView<double> vw(in);
	vw.m_ras = true;
	auto out = dPtrCast<MRImage>(in->createAnother());
	Vector3d pt;
	for(Vector3DIter<double> it(out); !it.eof(); ++it) {
		it.index(3, pt.array().data());
		out->indexToPoint(3, pt.array().data(), pt.array().data());
		pt = Rinv*(pt-center) + center + ishift;
		it.set(vw(pt[0], pt[1], pt[2]));
	}

	return out;
}

/**
 * @brief Computes difference of gaussians.
 *
 * @param in Gaussian smooths an image twice and subtracts
 * @param sd Standard deviation in each dimension
 *
 * @return Difference of two different gaussians
 */
ptr<MRImage> diffOfGauss(ptr<const MRImage> in, double sd1, double sd2)
{
	auto tmp = dPtrCast<MRImage>(in->copyCast(FLOAT32));
	auto out = dPtrCast<MRImage>(in->copyCast(FLOAT32));

	for(size_t ii=0; ii<in->ndim(); ii++) {
		gaussianSmooth1D(tmp, ii, sd1/in->spacing(ii));
		gaussianSmooth1D(out, ii, sd2/in->spacing(ii));
	}

	for(FlatIter<double> it1(tmp), it2(out); !it1.eof(); ++it1, ++it2)
		it2.set(it1.get()-it2.get());

	return out;
}

/**
 * @brief Computes the overlap of the two images' in 3-space.
 *
 * @param a Image
 * @param b Image
 *
 * @return Ratio of b that overlaps with a's grid
 */
double overlapRatio(shared_ptr<MRImage> a, shared_ptr<MRImage> b)
{
	int64_t index[3];
	double point[3];
	size_t incount = 0;
	size_t maskcount = 0;
	for(OrderIter<int64_t> it(a); !it.eof(); ++it) {
		it.index(3, index);
		a->indexToPoint(3, index, point);
		++maskcount;
		incount += (b->pointInsideFOV(3, point));
	}
	return (double)(incount)/(double)(maskcount);
}

template <typename T>
void resampleNN_help(ptr<const MRImage> in, ptr<MRImage> target)
{
	NDIter<T> it(target);
	NNInterpNDView<T> vw(in);
	vw.m_ras = true;

	vector<int64_t> ind(in->ndim(), 0);
	vector<double> pt(in->ndim(), 0);
	for(it.goBegin(); !it.eof(); ++it) {
		it.index(ind.size(), ind.data());
		target->indexToPoint(ind.size(), ind.data(), pt.data());
		it.set(vw.get(pt.size(), pt.data()));
	}
}

/**
 * @brief Performs nearest neighbor resasmpling of input to atlas
 *
 * @param in Input image
 * @param newspace new spacing
 * @param type pixel type of output (defaults to input)
 *
 * @return Input image resampled into atlas space
 */
ptr<MRImage> resampleNN(ptr<const MRImage> in, double* newspace,
		PixelT type)
{
	// Get the new image size (use ceil so we don't lose anything)
	vector<size_t> newsize(in->ndim());
	for(size_t ii=0; ii<in->ndim(); ii++)
		newsize[ii] = ceil(in->dim(ii)*in->spacing(ii)/newspace[ii]);

	// Set the Default Type to input
	if(type == UNKNOWN_TYPE)
		type = in->type();

	// Create Output Image, Zero
	auto out = createMRImage(in->ndim(), newsize.data(), type);
	for(FlatIter<int> it(out); !it.eof(); ++it) it.set(0);

	// Set Orient
	VectorXd spacing(in->ndim());
	for(size_t ii=0; ii<in->ndim(); ii++)
		spacing[ii] = newspace[ii];
	out->setOrient(in->getOrigin(), spacing, in->getDirection());

	switch(type) {
		case UINT8:
			resampleNN_help<uint8_t>(in, out);
			break;
		case INT16:
			resampleNN_help<int16_t>(in, out);
			break;
		case INT32:
			resampleNN_help<int32_t>(in, out);
			break;
		case FLOAT32:
			resampleNN_help<float>(in, out);
			break;
		case CFLOAT:
			resampleNN_help<cfloat_t>(in, out);
			break;
		case FLOAT64:
			resampleNN_help<double>(in, out);
			break;
		case RGB24:
			resampleNN_help<rgb_t>(in, out);
			break;
		case INT8:
			resampleNN_help<int8_t>(in, out);
			break;
		case UINT16:
			resampleNN_help<uint16_t>(in, out);
			break;
		case UINT32:
			resampleNN_help<uint32_t>(in, out);
			break;
		case INT64:
			resampleNN_help<int64_t>(in, out);
			break;
		case UINT64:
			resampleNN_help<uint64_t>(in, out);
			break;
		case FLOAT128:
			resampleNN_help<long double>(in, out);
			break;
		case CDOUBLE:
			resampleNN_help<cdouble_t>(in, out);
			break;
		case CQUAD:
			resampleNN_help<cquad_t>(in, out);
			break;
		case RGBA32:
			resampleNN_help<rgba_t>(in, out);
			break;
		default:
			throw INVALID_ARGUMENT("Unknown type set!");
	}

	return out;
}

/**
 * @brief Performs nearest neighbor resasmpling of input to atlas
 *
 * @param in Input image
 * @param atlas
 * @param type pixel type of output (defaults to input)
 *
 * @return Input image resampled into atlas space
 */
ptr<MRImage> resampleNN(ptr<const MRImage> in, ptr<const MRImage> atlas,
		PixelT type)
{
	// Set the Default Type to input
	if(type == UNKNOWN_TYPE)
		type = in->type();

	auto out = dPtrCast<MRImage>(atlas->createAnother(type));

	switch(type) {
		case UINT8:
			resampleNN_help<uint8_t>(in, out);
			break;
		case INT16:
			resampleNN_help<int16_t>(in, out);
			break;
		case INT32:
			resampleNN_help<int32_t>(in, out);
			break;
		case FLOAT32:
			resampleNN_help<float>(in, out);
			break;
		case CFLOAT:
			resampleNN_help<cfloat_t>(in, out);
			break;
		case FLOAT64:
			resampleNN_help<double>(in, out);
			break;
		case RGB24:
			resampleNN_help<rgb_t>(in, out);
			break;
		case INT8:
			resampleNN_help<int8_t>(in, out);
			break;
		case UINT16:
			resampleNN_help<uint16_t>(in, out);
			break;
		case UINT32:
			resampleNN_help<uint32_t>(in, out);
			break;
		case INT64:
			resampleNN_help<int64_t>(in, out);
			break;
		case UINT64:
			resampleNN_help<uint64_t>(in, out);
			break;
		case FLOAT128:
			resampleNN_help<long double>(in, out);
			break;
		case CDOUBLE:
			resampleNN_help<cdouble_t>(in, out);
			break;
		case CQUAD:
			resampleNN_help<cquad_t>(in, out);
			break;
		case RGBA32:
			resampleNN_help<rgba_t>(in, out);
			break;
		default:
			throw INVALID_ARGUMENT("Unknown type set!");
	}

	return out;
}

/**
 * @brief Create random image, with gaussian distribution
 *
 * @param type Type of pixels
 * @param mean mean of Gaussian
 * @param sd Standard deviation of distribution
 * @param x X size
 * @param y Y size
 * @param z Z size
 * @param t Time size, default 0 time
 *
 * @return Output MRImage
 */
ptr<MRImage> randImage(PixelT type, double mean, double sd,
		size_t x, size_t y, size_t z, size_t t)
{
	std::random_device rd;
	std::default_random_engine rng(rd());
	std::normal_distribution<double> dist(mean, sd);

	size_t sz[4] = {x,y,z,t};
	ptr<MRImage> out;
	out = createMRImage(t==0 ? 3 : 4, sz, type);
	for(FlatIter<double> it(out); !it.eof(); ++it) {
		it.set(dist(rng));
	}

	return out;
}


} // npl


