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
 * @file ndarray_utils.cpp
 *
 *****************************************************************************/

/******************************************************************************
 * @file ndarray_utils.cpp
 * @brief This file contains common functions which are useful for processing
 * of N-dimensional arrays and their derived counterparts (MRImage for
 * example). All of these functions return pointers to NDArray types, however
 * if an image is passed in, then the output will also be an image, you just
 * need to cast the output using std::dynamic_pointer_cast<MRImage>(out).
 * mrimage_utils.h is for more specific image-processing algorithm, this if for
 * generally data of any dimension, without regard to orientation.
 ******************************************************************************/

#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include "ndarray_utils.h"
#include "ndarray.h"
#include "npltypes.h"
#include "matrix.h"
#include "utility.h"
#include "iterators.h"
#include "basic_functions.h"

#include <Eigen/Geometry> 
#include <Eigen/Dense> 

#include "fftw3.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <memory>
#include <stdexcept>

namespace npl {

using std::vector;
using std::shared_ptr;

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::AngleAxisd;

int hob (int num)
{
	if (!num)
		return 0;

	int ret = 1;

	while (num >>= 1)
		ret <<= 1;

	return ret;
}

int64_t round2(int64_t in)
{
	int64_t just_hob = hob(in);
	if(just_hob == in)
		return in;
	else
		return (in<<1);
}

/**
 * @brief Performs in-place fft. Note that the input should already be padded
 * and a complex type
 *
 * @param in
 * @param dim
 */
//void fft1d(shared_ptr<NDArray> in, size_t dd, bool inverse)
//{
//	// plan and execute
//	double* indata = fftw_alloc_real(in->dim(dd));
//	fftw_complex *data = fftw_alloc_complex(in->dim(dd));
//	fftw_plan plan;
//	if(inverse) {
//		plan = fftw_plan_dft_1d(in->dim(dd), data, data, FFTW_BACKWARD,
//				FFTW_MEASURE);
//	} else {
//		plan = fftw_plan_dft_1d(in->dim(dd), data, data, FFTW_FORWARD,
//				FFTW_MEASURE);
//	}
//
//	OrderIter<cdouble_t> oit(in);
//	oit.setOrder({dd});
//	OrderConstIter<cdouble_t> iit(in);
//	iit.setOrder(oit.getOrder()); // make dd the fastest
//	while(!iit.eof() && !oit.eof()) {
//
//		// fill array
//		for(size_t ii=0; ii<in->dim(dd); ++iit, ii++) {
//			data[ii][0] = (*iit).real();
//			data[ii][1] = (*iit).imag();
//		}
//		
//		// execute
//		fftw_execute(plan);
//		
//		// write array
//		cdouble_t tmp;
//		for(size_t ii=0; ii<in->dim(dd); ++oit, ii++) {
//			tmp.real(data[ii][0]);
//			tmp.real(data[ii][1]);
//			data[ii][0] = (*iit).real();
//			data[ii][1] = (*iit).imag();
//			oit.set(tmp);
//		}
//		
//	}
//
//}

/**
 * @brief Perform fourier transform on the dimensions specified. Those
 * dimensions will be padded out. The output of this will be a double.
 * If len = 0 or dim == NULL, then ALL dimensions will be transformed.
 *
 * @param in Input image to inverse fourier trnasform
 * @param len length of input dimension array
 * @param dim dimensions to transform
 *
 * @return Image with specified dimensions in the real domain. Image will
 * differ in size from input.
 */
shared_ptr<NDArray> ifft_c2r(shared_ptr<const NDArray> in)
{
	if(in->type() != COMPLEX128 && in->type() != COMPLEX64 && in->type() !=
				COMPLEX256) {
		std::invalid_argument("Input to fft_c2r is not complex!");
	}
	
	size_t ndim = in->ndim();;

	/*
	 * pad out the given dimensions
	 */

	// start with original size and then round up specified dimensions
	cerr << "Input Dimensions: [";
	for(size_t dd=0; dd<in->ndim(); dd++)
		cerr << in->dim(dd) << ",";
	cerr << "]" << endl;
	size_t inpixel = 1;
	size_t outpixel = 1;
	std::vector<int> insize(in->dim(), in->dim()+ndim);
	std::vector<size_t> outsize(in->dim(), in->dim()+ndim);
	for(size_t ii=0; ii<ndim; ii++) {
		insize[ii] = in->dim(ii);
		inpixel *= insize[ii];
		outsize[ii] = insize[ii];
		outpixel *= outsize[ii];
	}

	auto idata = fftw_alloc_complex(inpixel);
	auto odata = fftw_alloc_complex(outpixel);

	// plan
	fftw_plan plan = fftw_plan_dft((int)insize.size(), insize.data(), idata,
				odata, FFTW_BACKWARD, FFTW_MEASURE);

	// copy data into idata
	OrderConstIter<cdouble_t> iit(in);
	for(size_t ii=0; !iit.eof(); ++iit, ii++) {
		idata[ii][0] = (*iit).real();
		idata[ii][1] = (*iit).imag();
	}

	// fourier transform
	fftw_execute(plan);
	fftw_free(idata);

	// copy data out
	auto out = in->copyCast(outsize.size(), outsize.data(), FLOAT64);

	OrderIter<cdouble_t> oit(out);
	cdouble_t tmp;
	for(size_t ii=0; !oit.eof(); ii++, ++oit) {
		tmp.real(odata[ii][0]);
		tmp.imag(odata[ii][1]);
		oit.set(tmp);;
	}

	cerr << "Out Dimensions: [";
	for(size_t dd=0; dd<out->ndim(); dd++)
		cerr << out->dim(dd) << ",";
	cerr << "]" << endl;
	
	fftw_free(odata);
	return out;
}

/**
 * @brief Perform fourier transform on the dimensions specified. Those
 * dimensions will be padded out. The output of this will be a complex double.
 * If len = 0 or dim == NULL, then ALL dimensions will be transformed.
 *
 * @param in Input image to fourier transform
 *
 * @return Complex image, which is the result of inverse fourier transforming
 * the (Real) input image. Note that the last dimension only contains the real
 * frequencies, but all other dimensions contain both
 */
shared_ptr<NDArray> fft_r2c(shared_ptr<const NDArray> in)
{
	if(in->type() == COMPLEX128 || in->type() == COMPLEX64 || in->type() ==
				COMPLEX256 || in->type() == RGB24 || in->type() == RGBA32) {
		std::invalid_argument("Input to fft_r2c is not scalar!");
	}
	
	size_t ndim = in->ndim();;

	/*
	 * pad out the given dimensions
	 */

	// start with original size and then round up specified dimensions
	cerr << "Input Dimensions: [";
	for(size_t dd=0; dd<in->ndim(); dd++)
		cerr << in->dim(dd) << ",";
	cerr << "]" << endl;
	size_t inpixel = 1;
	size_t outpixel = 1;
	std::vector<int> padsize(in->dim(), in->dim()+ndim);
	std::vector<size_t> outsize(in->dim(), in->dim()+ndim);
	for(size_t ii=0; ii<ndim; ii++) {
		padsize[ii] = round2(in->dim(ii));
		inpixel *= padsize[ii];
		outsize[ii] = padsize[ii];
		outpixel *= outsize[ii];
	}

	auto idata = fftw_alloc_complex(inpixel);
	auto odata = fftw_alloc_complex(outpixel);

	// plan
	fftw_plan plan = fftw_plan_dft((int)padsize.size(), padsize.data(),
				idata, odata, FFTW_FORWARD, FFTW_MEASURE);

	// copy data into idata
	OrderConstIter<cdouble_t> iit(in);
	for(size_t ii=0; !iit.eof(); ++iit, ii++) {
		idata[ii][0] = (*iit).real();
		idata[ii][1] = (*iit).imag();
	}

	// fourier transform
	fftw_execute(plan);
	fftw_free(idata);

	// copy data out
	auto out = in->copyCast(outsize.size(), outsize.data(), COMPLEX128);

	OrderIter<cdouble_t> oit(out);
	cdouble_t tmp;
	for(size_t ii=0; !oit.eof(); ii++, ++oit) {
		tmp.real(odata[ii][0]/inpixel);
		tmp.imag(odata[ii][1]/inpixel);
		oit.set(tmp);;
	}

	cerr << "Out Dimensions (with hermitian symmetry): [";
	for(size_t dd=0; dd<out->ndim(); dd++)
		cerr << out->dim(dd) << ",";
	cerr << "]" << endl;
	
	fftw_free(odata);
	return out;
}

//shared_ptr<NDArray> ppfft(shared_ptr<NDArray> in, size_t len, size_t* dims)
//{
//	// compute the 3D foureir transform for the entire image (or at least the
//	// dimensions specified)
//	auto fourier = fft_r2c(in);
//
//	// compute for each fan
//	for(size_t ii=0; ii<len; ii++) {
//
//	}
//}


/**
 * @brief Returns whether two NDArrays have the same dimensions, and therefore
 * can be element-by-element compared/operated on. elL is set to true if left
 * is elevatable to right (ie all dimensions match or are missing or are unary).
 * elR is the same but for the right.
 *
 * Strictly R is elevatable if all dimensions that don't match are missing or 1
 * Strictly L is elevatable if all dimensions that don't match are missing or 1
 *
 * Examples of *elR = true (return false):
 *
 * left = [10, 20, 1]
 * right = [10, 20, 39]
 *
 * left = [10]
 * right = [10, 20, 39]
 *
 * Examples where neither elR or elL (returns true):
 *
 * left = [10, 20, 39]
 * right = [10, 20, 39]
 *
 * Examples where neither elR or elL (returns false):
 *
 * left = [10, 20, 9]
 * right = [10, 20, 39]
 *
 * left = [10, 1, 9]
 * right = [10, 20, 1]
 *
 * @param left	NDArray input
 * @param right NDArray input
 * @param elL Whether left is elevatable to right (see description of function)
 * @param elR Whether right is elevatable to left (see description of function)
 *
 * @return
 */
bool comparable(const NDArray* left, const NDArray* right, bool* elL, bool* elR)
{
	bool ret = true;

	bool rightEL = true;
	bool leftEL = true;

	for(size_t ii = 0; ii < left->ndim(); ii++) {
		if(ii < right->ndim()) {
			if(right->dim(ii) != left->dim(ii)) {
				ret = false;
				// if not 1, then R is not elevateable
				if(right->dim(ii) != 1)
					rightEL = false;
			}
		}
	}
	
	for(size_t ii = 0; ii < right->ndim(); ii++) {
		if(ii < left->ndim()) {
			if(right->dim(ii) != left->dim(ii)) {
				ret = false;
				// if not 1, then R is not elevateable
				if(left->dim(ii) != 1)
					leftEL = false;
			}
		}
	}
	
	if(ret) {
		leftEL = false;
		rightEL = false;
	}

	if(elL) *elL = leftEL;
	if(elR) *elR = rightEL;

	return ret;
}

/**
 * @brief Erode an binary array repeatedly
 *
 * @param in Input to erode
 * @param reps Number of radius-1 kernel erosions to perform
 *
 * @return Eroded Image
 */
shared_ptr<NDArray> erode(shared_ptr<NDArray> in, size_t reps)
{
	std::vector<int64_t> index1(in->ndim(), 0);
	std::vector<int64_t> index2(in->ndim(), 0);
	auto prev = in->copy();
	auto out = in->copy();
	for(size_t rr=0; rr<reps; ++rr) {
		std::swap(prev, out);
		
		KernelIter<int> it(prev);
		it.setRadius(1);
		OrderIter<int> oit(out);
		oit.setOrder(it.getOrder());
		// for each pixels neighborhood, smooth neightbors
		for(oit.goBegin(), it.goBegin(); !it.eof(); ++it, ++oit) {
			oit.index(index1.size(), index1.data());
			it.center_index(index2.size(), index2.data());

			// if any of the neighbors are 0, then set to 0
			bool erodeme = false;
			for(size_t ii=0; ii<it.ksize(); ++ii) {
				if(it.offset(ii) == 0) {
					erodeme = true;
				}
			}

			if(erodeme)
				oit.set(0);
		}
	}

	return out;
}

/**
 * @brief Dilate an binary array repeatedly
 *
 * @param in Input to dilate
 * @param reps Number of radius-1 kernel dilations to perform
 *
 * @return Dilated Image
 */
shared_ptr<NDArray> dilate(shared_ptr<NDArray> in, size_t reps)
{
	std::vector<int64_t> index1(in->ndim(), 0);
	std::vector<int64_t> index2(in->ndim(), 0);
	auto prev = in->copy();
	auto out = in->copy();
	for(size_t rr=0; rr<reps; ++rr) {
		std::swap(prev, out);
		
		KernelIter<int> it(prev);
		it.setRadius(1);
		OrderIter<int> oit(out);
		oit.setOrder(it.getOrder());
		// for each pixels neighborhood, smooth neightbors
		for(oit.goBegin(), it.goBegin(); !it.eof(); ++it, ++oit) {
			oit.index(index1.size(), index1.data());
			it.center_index(index2.size(), index2.data());
			for(size_t ii=0; ii<in->ndim(); ++ii) {
				if(index1[ii] != index2[ii]) {
					throw std::logic_error("Error differece in iteration!");
				}
			}

			// if any of the neighbors are 0, then set to 0
			bool dilateme = false;
			int dilval = 0;
			for(size_t ii=0; ii<it.ksize(); ++ii) {
				if(it.offset(ii) != 0) {
					dilval = it.offset(ii);
					dilateme = true;
				}
			}

			if(dilateme)
				oit.set(dilval);
		}
	}

	return out;
}

/**
 * @brief Uses fourier shift theorem to rotate an image, using shears
 *
 * @param inout Input/output image to shift
 * @param dd dimension of image to shift
 * @param dist Distance (index's) to shift
 *
 * @return shifted image
 */
void shiftImageKern(shared_ptr<NDArray> inout, size_t dd, double dist)
{
	assert(dd < inout->ndim());

	const int64_t RADIUS = 3;
	const std::complex<double> I(0, 1);
	std::vector<double> buf(inout->dim(dd), 0);

	// need copy data into center of buffer, create iterator that moves
	// in the specified dimension fastest
	OrderIter<double> oit(inout);
	OrderIter<double> iit(inout);
	iit.setOrder({dd});
	oit.setOrder({dd});

	for(iit.goBegin(), oit.goBegin(); !iit.isEnd() ; ){

		// fill buffer 
		for(size_t tt=0; tt<inout->dim(dd); ++iit, tt++) 
			buf[tt] = *iit;

		// fill from line
		for(size_t tt=0; tt<inout->dim(dd); ++oit, tt++) {
			double tmp = 0;
			double source = (double)tt-dist;
			int64_t isource = round(source);
			for(int64_t oo = -RADIUS; oo <= RADIUS; oo++) {
				int64_t ind = clamp<int64_t>(0, inout->dim(dd)-1, isource+oo);
				tmp += lanczosKernel(oo+isource-source, RADIUS)*buf[ind];
			}

			oit.set(tmp);
		}
	}
}

/**
 * @brief Uses fourier shift theorem to shift an image, using shears
 *
 * @param inout Input/output image to shift
 * @param dd dimension of image to shift
 * @param dist Distance (index's) to shift
 *
 * @return shifted image
 */
void shiftImageFFT(shared_ptr<NDArray> inout, size_t dd, double dist,
		double(*window)(double, double))
{
	assert(dd < inout->ndim());

	const std::complex<double> I(0, 1);
	const double PI = acos(-1);
	size_t padsize = round2(inout->dim(dd));
	size_t paddiff = padsize-inout->dim(dd);
	auto buffer = fftw_alloc_complex(padsize);
	fftw_plan fwd = fftw_plan_dft_1d((int)padsize, buffer, buffer, 
			FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan rev = fftw_plan_dft_1d((int)padsize, buffer, buffer, 
			FFTW_BACKWARD, FFTW_MEASURE);

	// need copy data into center of buffer, create iterator that moves
	// in the specified dimension fastest
	OrderConstIter<cdouble_t> iit(inout);
	OrderIter<cdouble_t> oit(inout);
	iit.setOrder({dd});
	oit.setOrder({dd});

	for(iit.goBegin(), oit.goBegin(); !iit.isEnd() ;) {
		// zero buffer 
		for(size_t tt=0; tt<padsize; tt++) {
			buffer[tt][0] = 0;
			buffer[tt][1] = 0;
		}

		// fill from line
		for(size_t tt=0; tt<inout->dim(dd); ++iit, tt++) {
			buffer[tt+paddiff/2][0] = (*iit).real();
			buffer[tt+paddiff/2][1] = (*iit).imag();
		}

		// fourier transform
		fftw_execute(fwd);

		// fourier shift
		double normf = pow(padsize,-1);
		for(size_t tt=0; tt<padsize/2; tt++) {
			double ff = (double)tt/(double)padsize;
			cdouble_t tmp(buffer[tt][0]*normf, buffer[tt][1]*normf);
			tmp *= window(ff, .5)*std::exp(-2.*PI*I*dist*ff);
			buffer[tt][0] = tmp.real();
			buffer[tt][1] = tmp.imag();
		}
		for(size_t tt=padsize/2; tt<padsize; tt++) {
			double ff = -(double)(padsize-tt)/(double)padsize;
			cdouble_t tmp(buffer[tt][0]*normf, buffer[tt][1]*normf);
			tmp *= window(ff, .5)*std::exp(-2.*PI*I*dist*ff);
			buffer[tt][0] = tmp.real();
			buffer[tt][1] = tmp.imag();
		}

		// inverse fourier transform
		fftw_execute(rev);

		// fill line from buffer
		for(size_t tt=0; tt<inout->dim(dd); ++oit, tt++) {
			cdouble_t tmp(buffer[tt+paddiff/2][0], buffer[tt+paddiff/2][1]);
			oit.set(tmp); 
		}
	}
}

void shearImageKern(shared_ptr<NDArray> inout, size_t dd, size_t len, double* dist)
{
	assert(dd < inout->ndim());

	const int64_t RADIUS = 3;
	std::vector<double> buf(inout->dim(dd), 0);
	std::vector<double> center(inout->ndim());
	for(size_t ii=0; ii<center.size(); ii++) {
		center[ii] = inout->dim(ii)/2.;
	}

	// need copy data into center of buffer, create iterator that moves
	// in the specified dimension fastest
	OrderIter<double> oit(inout);
	OrderIter<double> iit(inout);
	iit.setOrder({dd});
	oit.setOrder({dd});

	std::vector<int64_t> index(inout->ndim());
	for(iit.goBegin(), oit.goBegin(); !iit.isEnd() ; ){
		iit.index(index.size(), index.data());
		
		// calculate line shift
		double lineshift = 0;
		for(size_t ii=0; ii<len; ii++) {
			if(ii != dd)
				lineshift += dist[ii]*(index[ii]-center[ii]);
		}

		// fill buffer 
		for(size_t tt=0; tt<inout->dim(dd); ++iit, tt++) 
			buf[tt] = *iit;

		// fill from line
		for(size_t tt=0; tt<inout->dim(dd); ++oit, tt++) {
			double tmp = 0;
			double source = (double)tt-lineshift;
			int64_t isource = round(source);
			for(int64_t oo = -RADIUS; oo <= RADIUS; oo++) {
				int64_t ind = clamp<int64_t>(0, inout->dim(dd)-1, isource+oo);
				tmp += lanczosKernel(oo+isource-source, RADIUS)*buf[ind];
			}

			oit.set(tmp);
		}
	}
}


/**
 * @brief Performs a shear on the image where the sheared dimension (dim) will
 * be shifted depending on the index in other dimensions (dist). 
 * (in units of pixels), using FFT.
 *
 * @param inout Input/output image
 * @param dim Dimension to shift/shear
 * @param len Length of dist array
 * @param dist Distance terms to travel. Shift[dim] = x0*dist[0]+x1*dist[1] ...
 * @param window Windowing function of fourier domain (default sinc)
 */
void shearImageFFT(shared_ptr<NDArray> inout, size_t dd, size_t len, double* dist,
		double(*window)(double,double))
{
	assert(dd < inout->ndim());

	const std::complex<double> I(0, 1);
	const double PI = acos(-1);
	size_t padsize = round2(2*inout->dim(dd));
	size_t paddiff = padsize-inout->dim(dd);
	auto buffer = fftw_alloc_complex(padsize);
	fftw_plan fwd = fftw_plan_dft_1d((int)padsize, buffer, buffer, 
			FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan rev = fftw_plan_dft_1d((int)padsize, buffer, buffer, 
			FFTW_BACKWARD, FFTW_MEASURE);
	std::vector<double> center(inout->ndim());
	for(size_t ii=0; ii<center.size(); ii++) {
		center[ii] = inout->dim(ii)/2.;
	}

	// need copy data into center of buffer, create iterator that moves
	// in the specified dimension fastest
	ChunkIter<cdouble_t> it(inout);
	it.setLineChunk(dd);
	std::vector<int64_t> index(inout->ndim());
	for(it.goBegin(); !it.isEnd() ; it.nextChunk()) {
		it.index(index.size(), index.data());

		double lineshift = 0;
		for(size_t ii=0; ii<len; ii++) {
			if(ii != dd)
				lineshift += dist[ii]*(index[ii]-center[ii]);
		}

		// zero buffer 
		for(size_t tt=0; tt<padsize; tt++) {
			buffer[tt][0] = 0;
			buffer[tt][1] = 0;
		}

		// fill from line
		for(size_t tt=0; !it.isChunkEnd(); ++it, tt++) {
			buffer[tt+paddiff/2][0] = (*it).real();
			buffer[tt+paddiff/2][1] = (*it).imag();
		}

		// fourier transform
		fftw_execute(fwd);

		// fourier shift
		double normf = pow(padsize,-1);
		for(size_t tt=0; tt<padsize/2; tt++) {
			double ff = (double)tt/(double)padsize;
			cdouble_t tmp(buffer[tt][0]*normf, buffer[tt][1]*normf);
			tmp *= window(ff, .5)*std::exp(-2.*PI*I*lineshift*ff);
			buffer[tt][0] = tmp.real();
			buffer[tt][1] = tmp.imag();
		}
		for(size_t tt=padsize/2; tt<padsize; tt++) {
			double ff = -(double)(padsize-tt)/(double)padsize;
			cdouble_t tmp(buffer[tt][0]*normf, buffer[tt][1]*normf);
			tmp *= window(ff, .5)*std::exp(-2.*PI*I*lineshift*ff);
			buffer[tt][0] = tmp.real();
			buffer[tt][1] = tmp.imag();
		}

		// inverse fourier transform
		fftw_execute(rev);

		// fill line from buffer
		it.goChunkBegin();
		for(size_t tt=0; !it.isChunkEnd(); ++it, tt++) {
			cdouble_t tmp(buffer[tt+paddiff/2][0], buffer[tt+paddiff/2][1]);
			it.set(tmp); 
		}
	}
}

///**
// * @brief Rotates an image around the center using shear decomposition.
// * Rotation order is Rz, Ry, Rx.
// *
// * @param inout Input/output image
// * @param axis Rotation axis (through the middle)
// * @param theta angle of rotation
// */
//void rotateImageFFT(shared_ptr<NDArray> inout, double axis[3], double theta)
//{
//	// convert to euler angles
//	Vector3d axisv(axis);
//	Matrix3d rotation = AngleAxisd(theta, axisv);
//	rx = rotation(0,0); //TODO
//	ry = rotation(0,0);
//	rz = rotation(0,0);
//
//	// decompose into shears
//	
//	// perform shearing
//}

int shearYZXY(double terms[4][2], double* err, double* maxshear,
		double Rx, double Ry, double Rz)
{
	cerr << "Shear YZXY" << endl;
	terms[0][0] = -cos(Ry)*sin(Rz);
	terms[0][1] = -csc(Rx)*sin(Rz)+cot(Rx)*sec(Ry)*sin(Rz)+cos(Rz)*tan(Ry);
	terms[1][0] = csc(Rx)*tan(Ry)+sec(Ry)*(csc(Rz)-cot(Rz)*sec(Ry)-cot(Rx)*tan(Ry));
	terms[1][1] = cot(Rx)-csc(Rx)*sec(Ry);
	terms[2][0] = (csc(Rz)-cot(Rz)*sec(Ry))*sin(Rx)-cos(Rx)*tan(Ry);
	terms[2][1] = cos(Ry)*sin(Rx);
	terms[3][0] = -cot(Rz)+csc(Rz)*sec(Ry);
	terms[3][1] = -csc(Rz)*tan(Ry)+sec(Ry)*(-csc(Rx)+cot(Rx)*sec(Ry)+cot(Rz)*tan(Ry));
	
	// construct matrices 
	Matrix3d rotation;
	rotation = AngleAxisd(Rx, Vector3d::UnitX())*
				AngleAxisd(Ry, Vector3d::UnitY())*
				AngleAxisd(Rz, Vector3d::UnitZ());
//	rotation = AngleAxisd(Rz, Vector3d::UnitZ())*
//				AngleAxisd(Ry, Vector3d::UnitY())*
//				AngleAxisd(Rx, Vector3d::UnitX());
	Matrix3d sy1 = Matrix3d::Identity();
	sy1(1,0) = terms[0][0];
	sy1(1,2) = terms[0][1];
	cerr << "Shear Y:\n" << sy1 << endl;

	Matrix3d sz = Matrix3d::Identity();
	sz(2,0) = terms[1][0];
	sz(2,1) = terms[1][1];
	cerr << "Shear Z:\n" << sz << endl;

	Matrix3d sx = Matrix3d::Identity();
	sx(0,1) = terms[2][0];
	sx(0,2) = terms[2][1];
	cerr << "Shear X:\n" << sx << endl;

	Matrix3d sy2 = Matrix3d::Identity();
	sy2(1,0) = terms[3][0];
	sy2(1,2) = terms[3][1];
	cerr << "Shear Y:\n" << sy2 << endl;

	auto shearProduct = sy1*sz*sx*sy2;
	cerr << "Rotation:\n" << rotation << endl;
	cerr << "Sheared Rotation:\n" << shearProduct<< endl;
	cerr << "Error:\n" << (rotation-shearProduct).cwiseAbs() << endl;
	if(err)
		*err = (rotation-shearProduct).cwiseAbs().sum();

	if(maxshear) {
		*maxshear = 0;
		for(size_t ii=0; ii<4; ii++){
			for(size_t jj=0; jj<2; jj++){
				if(*maxshear < terms[ii][jj])
					*maxshear = terms[ii][jj];
			}
		}
	}
	
	return 0;
}

///**
// * @brief Decomposes a rotation about Z,Y,X into shears, in order of X shears 
// * then Z shears then Y sehars the X shears. 
// *
// * @param terms[8][2] Shears, 4 sets of shear terms
// * @param rx Rotation about x axis (last)
// * @param ry Rotation about y axis (middl)
// * @param rz Rotation about z axis (first)
// *
// * @return Error when reconstructing the rotation matrxi
// */
//double shearXYZXDecompose(double terms[4][2], double rx, double ry, double rz)
//{
//	// calculate terms
//	
//	// construct matrices 
//	Matrix3d rotation = AngleAxisd(rx, Vector3d::UnitX())*
//				AngleAxisd(ry, Vector3d::UnitY())*
//				AngleAxisd(rz, Vector3d::UnitZ());
//	Matrix3d sx1 = Identity();
//	sx1(0,1) = terms[0][0]
//	sx1(0,2) = terms[0][1]
//
//	Matrix3d sy = Identity();
//	sy(1,1) = terms[1][0]
//	sy(1,2) = terms[1][1]
//
//	Matrix3d sz = Identity();
//	sz(2,1) = terms[2][0]
//	sz(2,2) = terms[2][1]
//
//	Matrix3d sx2 = Identity();
//	sx2(0,1) = terms[3][0]
//	sx2(0,2) = terms[3][1]
//
//	auto shearProduct = sx1*sy*sz*sx2;
//	
//	return (rotation - shearProduct).abs().sub();
//}
//
/**
 * @brief Rotates an image around the center using shear decomposition.
 * Rotation order is Rz, Ry, Rx, and about the center of the image.
 *
 * @param inout Input/output image
 * @param rx Rotation about x axis (applied last)
 * @param ry Rotation about y axis (applied second)
 * @param rz Rotation about z axis (applied first)
 */
void rotateImageFFT(shared_ptr<NDArray> inout, double rx, double ry, double rz)
{
	double MAXERR = 0.001;
	double MAXSHEAR = 0.1;
	// decompose into shears
	
	// perform shearing
}


} // npl
#endif  //IMAGE_PROCESSING_H


