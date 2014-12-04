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
 * @brief This file contains common functions which are useful for processing
 * of N-dimensional arrays and their derived counterparts (MRImage for
 * example). All of these functions return pointers to NDArray types, however
 * if an image is passed in, then the output will also be an image, you just
 * need to cast the output using dPtrCast<MRImage>(out).
 * mrimage_utils.h is for more specific image-processing algorithm, this if for
 * generally data of any dimension, without regard to orientation.
 *
 ******************************************************************************/

#include "ndarray_utils.h"
#include "ndarray.h"
#include "npltypes.h"
#include "utility.h"
#include "iterators.h"
#include "basic_functions.h"
#include "chirpz.h"

#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "fftw3.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <memory>
#include <unordered_map>
#include <stdexcept>

#include "mrimage.h"
#include "accessors.h"
#include "macros.h"

namespace npl {

using std::vector;

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::AngleAxisd;

/**
 * @brief Computes the derivative of the image in the specified direction.
 * The output will be the same size as the input.
 *
 * @param in    Input image/NDArray
 * @param dir   Specify the dimension
 *
 * @return      Image storing the directional derivative of in
 */
ptr<NDArray> derivative(ptr<const NDArray> in, size_t dir)
{
	if(dir >= in->ndim())
		throw std::invalid_argument("Input direction is outside range of "
				"input dimensions in\n" + __FUNCTION_STR__);

	auto out = in->copy();
	vector<int64_t> index(in->ndim());
	NDConstView<double> inGet(in);

	NDIter<double> oit(out);
	for(oit.goBegin(); !oit.eof(); ++oit) {
		// get index
		oit.index(index.size(), index.data());

		double dx = 0;
		double dy = 0;

		// before
		if(index[dir] == 0) {
			dy -= inGet[index];
		} else {
			index[dir]--;
			dy -= inGet[index];
			dx++;
			index[dir]++;
		}

		// after
		if(index[dir] == in->dim(dir)-1) {
			dy += inGet[index];
		} else {
			index[dir]++;
			dy += inGet[index];
			dx++;
		}

		// set
		if(fabs(dy) < 0.00000000001)
			oit.set(0);
		else
			oit.set(dy/dx);
	}

	return out;
}

/**
 * @brief Computes the derivative of the image. Computes all
 * directional derivatives of the input image and the output
 * image will have 1 higher dimension with derivative of 0 in the first volume
 * 1 in the second and so on.
 *
 * Thus a 2D image will produce a [X,Y,2] image and a 3D image will produce a
 * [X,Y,Z,3] sized image.
 *
 * @param in    Input image/NDArray
 * @param out	Derivative of input
 *
 * @return 0 if successful
 */
int derivative(ptr<const NDArray> in, ptr<NDArray> out)
{
	if(out->ndim() != in->ndim()+1)
		throw INVALID_ARGUMENT("Output (derivative) should have 1 extra dimension "
				"compared to input.");
	for(size_t dd=0; dd<in->ndim(); dd++) {
		if(out->dim(dd) != in->dim(dd))
			throw INVALID_ARGUMENT("Input and Output sizes differ");
	}

	vector<int64_t> index(in->ndim());
	NDConstView<double> inGet(in);

	Vector3DIter<double> oit(out);
	for(oit.goBegin(); !oit.eof(); ++oit) {
		// get index
		oit.index(index.size(), index.data());

		// compute derivative in each direction
		for(size_t dd=0; dd<in->ndim(); dd++) {
			double dx = 0;
			double dy = 0;

			// before
			if(index[dd] == 0) {
				dy -= inGet[index];
			} else {
				index[dd]--;
				dy -= inGet[index];
				dx++;
				index[dd]++;
			}

			// after
			if(index[dd] == in->dim(dd)-1) {
				dy += inGet[index];
			} else {
				index[dd]++;
				dy += inGet[index];
				dx++;
				index[dd]--;
			}

			// set
			if(fabs(dy) < 0.00000000001)
				oit.set(dd, 0);
			else
				oit.set(dd, dy/dx);
		}
	}

	return 0;
}


/**
 * @brief Computes the derivative of the image. Computes all
 * directional derivatives of the input image and the output
 * image will have 1 higher dimension with derivative of 0 in the first volume
 * 1 in the second and so on.
 *
 * Thus a 2D image will produce a [X,Y,2] image and a 3D image will produce a
 * [X,Y,Z,3] sized image.
 *
 * @param in    Input image/NDArray
 *
 * @return
 */
ptr<NDArray> derivative(ptr<const NDArray> in)
{
	vector<size_t> osize(in->dim(), in->dim()+in->ndim());
	osize.push_back(in->ndim());
	auto out = in->createAnother(osize.size(), osize.data());

	derivative(in, out);
	return out;
}

//ptr<NDArray> ppfft(ptr<NDArray> in, size_t len, size_t* dims)
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

/*************************
 * Basic Kernel Functions
 *************************/

/**
 * @brief Smooths an image in 1 dimension
 *
 * @param inout Input/output image to smooth
 * @param dim dimensions to smooth in. If you are smoothing individual volumes
 * of an fMRI you would provide dim={0,1,2}
 * @param stddev standard deviation in physical units index*spacing
 *
 */
void gaussianSmooth1D(ptr<NDArray> inout, size_t dim,
		double stddev)
{
	if(stddev <= 0)
		return;

    const auto gaussKern = [](double x)
    {
        const double den = 1./sqrt(2*M_PI);
        return den*exp(-x*x/(2));
    };

	if(dim >= inout->ndim()) {
		throw std::out_of_range("Invalid dimension specified for 1D gaussian "
				"smoothing");
	}

	std::vector<int64_t> index(dim, 0);
	std::vector<double> ibuff(inout->dim(dim));
	std::vector<double> obuff(inout->dim(dim));

	// calculate normalization factor
	double normalize = 0;
	int rad = 3*stddev;
	for(int ii=-rad; ii<=rad; ii++)
		normalize += gaussKern(ii/stddev);

	// for writing, have the regular iterator
	ChunkIter<double> it(inout);
	it.setLineChunk(dim);
	for(it.goBegin(); !it.eof(); it.nextChunk()) {
		it.goChunkBegin();
		for(size_t ii=0; !it.eoc(); ++it, ++ii)
			ibuff[ii] = *it;

		// perform kernel math, writing to buffer
		for(int ii=0; ii<(int)inout->dim(dim); ii++) {
			double sum = 0;
			for(int kk=-rad; kk<=rad; kk++) {
				if(kk+ii >= 0 && kk+ii < inout->dim(dim))
					sum += ibuff[kk+ii]*gaussKern(kk/stddev);
			}
			obuff[ii] = sum/normalize;
		}

		it.goChunkBegin();
		for(size_t ii=0; !it.eoc(); ++it, ++ii)
			it.set(obuff[ii]);
	}
}


/**
 * @brief Erode an binary array repeatedly
 *
 * @param in Input to erode
 * @param reps Number of radius-1 kernel erosions to perform
 *
 * @return Eroded Image
 */
ptr<NDArray> erode(ptr<NDArray> in, size_t reps)
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
			it.indexC(index2.size(), index2.data());

			// if any of the neighbors are 0, then set to 0
			bool erodeme = false;
			for(size_t ii=0; ii<it.ksize(); ++ii) {
				if(it.getK(ii) == 0) {
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
ptr<NDArray> dilate(ptr<NDArray> in, size_t reps)
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
			it.indexC(index2.size(), index2.data());
			for(size_t ii=0; ii<in->ndim(); ++ii) {
				if(index1[ii] != index2[ii]) {
					throw std::logic_error("Error differece in iteration!");
				}
			}

			// if any of the neighbors are 0, then set to 0
			bool dilateme = false;
			int dilval = 0;
			for(size_t ii=0; ii<it.ksize(); ++ii) {
				if(it.getK(ii) != 0) {
					dilval = it.getK(ii);
					dilateme = true;
				}
			}

			if(dilateme)
				oit.set(dilval);
		}
	}

	return out;
}

/********************
 * Image Shifting
 ********************/

/**
 * @brief Uses fourier shift theorem to rotate an image, using shears
 *
 * @param inout Input/output image to shift
 * @param dd dimension of image to shift
 * @param dist Distance (index's) to shift
 *
 * @return shifted image
 */
void shiftImageKern(ptr<NDArray> inout, size_t dd, double dist,
		double(*kern)(double,double))
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
				int64_t ind =  isource+oo;
				if(ind >= 0 && ind < inout->dim(dd))
					tmp += kern(oo+isource-source, RADIUS)*buf[ind];
			}

			oit.set(tmp);
		}
	}
}

/**
 * @brief Uses fourier shift theorem to shift an image, using shears
 *
 * @param inout Input/output image to shift
 * @param dim dimension of image to shift
 * @param dist Distance (index's) to shift
 * @param window Windowing function to apply in fourier domain
 *
 * @return shifted image
 */
void shiftImageFFT(ptr<NDArray> inout, size_t dim, double dist,
		double(*window)(double, double))
{
	assert(dim < inout->ndim());

	const std::complex<double> I(0, 1);
	const double PI = acos(-1);
	size_t padsize = round2(inout->dim(dim));
	size_t paddiff = padsize-inout->dim(dim);
	auto ibuffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*padsize);
	auto obuffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*padsize);
	fftw_plan fwd = fftw_plan_dft_1d((int)padsize, ibuffer, obuffer,
			FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan rev = fftw_plan_dft_1d((int)padsize, ibuffer, obuffer,
			FFTW_BACKWARD, FFTW_MEASURE);

	// need copy data into center of buffer, create iterator that moves
	// in the specified dimension fastest
	OrderConstIter<cdouble_t> iit(inout);
	OrderIter<cdouble_t> oit(inout);
	iit.setOrder({dim});
	oit.setOrder({dim});

	for(iit.goBegin(), oit.goBegin(); !iit.isEnd() ;) {
		// zero buffer
		for(size_t tt=0; tt<padsize; tt++) {
			ibuffer[tt][0] = 0;
			ibuffer[tt][1] = 0;
		}

		// fill from line
		for(size_t tt=0; tt<inout->dim(dim); ++iit, tt++) {
			ibuffer[tt+paddiff/2][0] = (*iit).real();
			ibuffer[tt+paddiff/2][1] = (*iit).imag();
		}

		// fourier transform
		fftw_execute(fwd);

		// fourier shift
		double normf = pow(padsize,-1);
		for(size_t tt=0; tt<padsize/2; tt++) {
			double ff = (double)tt/(double)padsize;
			cdouble_t tmp(obuffer[tt][0]*normf, obuffer[tt][1]*normf);
			tmp *= window(ff, .5)*std::exp(-2.*PI*I*dist*ff);
			ibuffer[tt][0] = tmp.real();
			ibuffer[tt][1] = tmp.imag();
		}
		for(size_t tt=padsize/2; tt<padsize; tt++) {
			double ff = -(double)(padsize-tt)/(double)padsize;
			cdouble_t tmp(obuffer[tt][0]*normf, obuffer[tt][1]*normf);
			tmp *= window(ff, .5)*std::exp(-2.*PI*I*dist*ff);
			ibuffer[tt][0] = tmp.real();
			ibuffer[tt][1] = tmp.imag();
		}

		// inverse fourier transform
		fftw_execute(rev);

		// fill line from buffer
		for(size_t tt=0; tt<inout->dim(dim); ++oit, tt++) {
			cdouble_t tmp(obuffer[tt+paddiff/2][0], obuffer[tt+paddiff/2][1]);
			oit.set(tmp);
		}
	}
}

/********************
 * Image Shearing
 ********************/

/**
 * @brief Performs a shear on the image where the sheared dimension (dim) will
 * be shifted depending on the index in other dimensions (dist).
 * (in units of pixels). Uses Lanczos interpolation.
 *
 * @param inout Input/output image
 * @param dim Dimension to shift/shear
 * @param len Length of dist array
 * @param dist Distance terms to travel. Shift[dim] = x0*dist[0]+x1*dist[1] ...
 * @param kern 1D interpolation kernel
 */
void shearImageKern(ptr<NDArray> inout, size_t dim, size_t len,
        double* dist, double(*kern)(double,double))
{
	assert(dim < inout->ndim());

	const int64_t RADIUS = 5;
	std::vector<double> buf(inout->dim(dim), 0);
	std::vector<double> center(inout->ndim());
	for(size_t ii=0; ii<center.size(); ii++) {
		center[ii] = (inout->dim(ii)-1)/2.;
	}

	// need copy data into center of buffer, create iterator that moves
	// in the specified dimension fastest
	OrderIter<double> oit(inout);
	OrderIter<double> iit(inout);
	iit.setOrder({dim});
	oit.setOrder({dim});

	std::vector<int64_t> index(inout->ndim());
	for(iit.goBegin(), oit.goBegin(); !iit.isEnd() ; ){
		iit.index(index.size(), index.data());

		// calculate line shift
		double lineshift = 0;
		for(size_t ii=0; ii<len; ii++) {
			if(ii != dim)
				lineshift += dist[ii]*(index[ii]-center[ii]);
		}

		// fill buffer
		for(size_t tt=0; tt<inout->dim(dim); ++iit, tt++)
			buf[tt] = *iit;

		// fill from line
		for(size_t tt=0; tt<inout->dim(dim); ++oit, tt++) {
			double tmp = 0;
			double source = (double)tt-lineshift;
			int64_t isource = round(source);
			for(int64_t oo = -RADIUS; oo <= RADIUS; oo++) {
				int64_t ind = isource+oo;
				if(ind >= 0 && ind < inout->dim(dim))
					tmp += kern(oo+isource-source, RADIUS)*buf[ind];
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
void shearImageFFT(ptr<NDArray> inout, size_t dim, size_t len, double* dist,
		double(*window)(double,double))
{
	assert(dim < inout->ndim());

	const std::complex<double> I(0, 1);
	const double PI = acos(-1);
	size_t padsize = round2(2*inout->dim(dim));
	size_t paddiff = padsize-inout->dim(dim);
	auto buffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*padsize);
	fftw_plan fwd = fftw_plan_dft_1d((int)padsize, buffer, buffer,
			FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan rev = fftw_plan_dft_1d((int)padsize, buffer, buffer,
			FFTW_BACKWARD, FFTW_MEASURE);
	std::vector<double> center(inout->ndim());
	for(size_t ii=0; ii<center.size(); ii++) {
		center[ii] = (inout->dim(ii)-1)/2.;
	}

	// need copy data into center of buffer, create iterator that moves
	// in the specified dimension fastest
	ChunkIter<cdouble_t> it(inout);
	it.setLineChunk(dim);
	std::vector<int64_t> index(inout->ndim());
	for(it.goBegin(); !it.isEnd() ; it.nextChunk()) {
		it.index(index.size(), index.data());

		double lineshift = 0;
		for(size_t ii=0; ii<len; ii++) {
			if(ii != dim)
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

double getMaxShear(const Matrix3d& in)
{
	double mval = 0;
	for(size_t ii=0; ii<3; ii++) {
		for(size_t jj=0; jj<3; jj++) {
			if(ii != jj) {
				if(fabs(in(ii,jj)) > mval)
					mval = fabs(in(ii,jj));
			}
		}
	}

	return mval;
}

/*****************************************************************************
 * Single Rotation Shears, which end up being 3 shears rather than four.
 ****************************************************************************/

int shearYZXY(std::list<Matrix3d>& terms, double* err, double* maxshear,
		double x, double y, double z)
{
	DBG3(cerr << "Shear YZXY" << endl);
	Matrix3d sy1 = Matrix3d::Identity();
	Matrix3d sz = Matrix3d::Identity();
	Matrix3d sx = Matrix3d::Identity();
	Matrix3d sy2 = Matrix3d::Identity();
	Matrix3d shearProduct = Matrix3d::Identity();

	sy1(1,0) = csc(x)*tan(y)+sec(y)*(csc(z)-cot(z)*sec(y)-cot(x)*tan(y));
	sy1(1,2) = cot(x)-csc(x)*sec(y);
	sz (2,0) = (csc(z)-cot(z)*sec(y))*sin(x)-cos(x)*tan(y);
	sz (2,1) = cos(y)*sin(x);
	sx (0,1) = -cos(y)*sin(z);
	sx (0,2) = -csc(x)*sin(z)+cot(x)*sec(y)*sin(z)+cos(z)*tan(y);
	sy2(1,0) = -cot(z)+csc(z)*sec(y);
	sy2(1,2) = -csc(z)*tan(y)+sec(y)*(-csc(x)+cot(x)*sec(y)+cot(z)*tan(y));

	terms.clear();
	terms.push_back(sy1);
	terms.push_back(sz);
	terms.push_back(sx);
	terms.push_back(sy2);

	if(maxshear) {
		*maxshear = 0;
		for(auto v:terms) {
			DBG3(cerr << "Shear:\n" << v << endl);
			shearProduct *= v;
			double mv = getMaxShear(v);
			if(mv > *maxshear)
				*maxshear = mv;
		}
	}

	//*construct*matrices*
	Matrix3d rotation;
	rotation = AngleAxisd(x, Vector3d::UnitX())*
				AngleAxisd(y, Vector3d::UnitY())*
				AngleAxisd(z, Vector3d::UnitZ());

	DBG3(cerr << "Rotation:\n" << rotation << endl);
	DBG3(cerr << "Sheared Rotation:\n" << shearProduct<< endl);
	DBG3(cerr << "Error:\n" << (rotation-shearProduct).cwiseAbs() << endl);
	if(err)
		*err = (rotation-shearProduct).cwiseAbs().sum();

	return 0;
}

int shearXYZX(std::list<Matrix3d>& terms, double* err, double* maxshear,
		double x, double y, double z)
{
	DBG3(cerr << "Shear XYZX" << endl);
	Matrix3d sx1 = Matrix3d::Identity();
	Matrix3d sy = Matrix3d::Identity();
	Matrix3d sz = Matrix3d::Identity();
	Matrix3d sx2 = Matrix3d::Identity();
	Matrix3d shearProduct = Matrix3d::Identity();

	sx1(0,1) = cos(x)*cot(z)*sec(y)-csc(z)*sec(y)-sin(x)*tan(y);
	sx1(0,2) = (sin(x)*(sin(x)-cos(z)*sec(y)*sin(x)-cot(z)*tan(y))+cos(x)*(-sec(y)+cos(2*z)*csc(z)*sin(x)*tan(y))+cos(x)*cos(x)*(1+cos(z)*sin(y)*tan(y)))/(cos(x)*cos(z)*sin(y)-sin(x)*sin(z));
	sy (1,0) = cos(y)*sin(z);
	sy (1,2) = (-cos(z)*sin(x)*sin(y)+(-cos(x)+cos(y))*sin(z))/(cos(x)*cos(z)*sin(y)-sin(x)*sin(z));
	sz (2,0) = -cos(x)*cos(z)*sin(y)+sin(x)*sin(z);
	sz (2,1) = sec(y)*sin(x)+(-cos(x)*cot(z)+csc(z))*tan(y);
	sx2(0,1) = (2*sec(y)*sin(x)+cos(z)*(-2*sin(x)+cos(x)*cot(z)*sin(y))-cos(x)*sin(y)*(csc(z)+sin(z))+2*(-cos(x)*cot(z)+csc(z))*tan(y))/(2*cos(x)*cos(z)*sin(y)-2*sin(x)*sin(z));
	sx2(0,2) = (-1+cos(x)*cos(y))/(-cos(x)*cos(z)*sin(y)+sin(x)*sin(z));

	terms.clear();
	terms.push_back(sx1);
	terms.push_back(sy);
	terms.push_back(sz);
	terms.push_back(sx2);

	if(maxshear) {
		*maxshear = 0;
		for(auto v:terms) {
			DBG3(cerr << "Shear:\n" << v << endl);
			shearProduct *= v;
			double mv = getMaxShear(v);
			if(mv > *maxshear)
				*maxshear = mv;
		}
	}

	//*construct*matrices*
	Matrix3d rotation;
	rotation = AngleAxisd(x, Vector3d::UnitX())*
				AngleAxisd(y, Vector3d::UnitY())*
				AngleAxisd(z, Vector3d::UnitZ());

	DBG3(cerr << "Rotation:\n" << rotation << endl);
	DBG3(cerr << "Sheared Rotation:\n" << shearProduct<< endl);
	DBG3(cerr << "Error:\n" << (rotation-shearProduct).cwiseAbs() << endl);
	if(err)
		*err = (rotation-shearProduct).cwiseAbs().sum();

	return 0;
}

int shearXZYX(std::list<Matrix3d>& terms, double* err, double* maxshear,
		double x, double y, double z)
{
	DBG3(cerr << "Shear XZYX" << endl);
	Matrix3d sx1 = Matrix3d::Identity();
	Matrix3d sz = Matrix3d::Identity();
	Matrix3d sy = Matrix3d::Identity();
	Matrix3d sx2 = Matrix3d::Identity();
	Matrix3d shearProduct = Matrix3d::Identity();

	sx1(0,1) = (-2+2*cos(x)*cos(z)-cos(x)*cos(x)*cos(y)*cos(z)+cos(y)*cos(z)*(1+sin(x)*sin(x))-2*csc(y)*sin(x)*sin(z)+cot(y)*sin(2*x)*sin(z))/(2*(cos(z)*sin(x)*sin(y)+cos(x)*sin(z)));
	sx1(0,2) = -cos(x)*cot(y)+csc(y);
	sz (2,0) = -sin(y);
	sz (2,1) = (sin(y)-cos(x)*cos(z)*sin(y)+sin(x)*sin(z))/(cos(z)*sin(x)*sin(y)+cos(x)*sin(z));
	sy (1,0) = cos(z)*sin(x)*sin(y)+cos(x)*sin(z);
	sy (1,2) = -cos(z)*sin(x)+(-cos(x)+cos(y))*csc(y)*sin(z);
	sx2(0,1) = (-1+cos(x)*cos(z)-sin(x)*sin(y)*sin(z))/(cos(z)*sin(x)*sin(y)+cos(x)*sin(z));
	sx2(0,2) = ((-cos(y)+cos(z))*sin(x)+(-cot(y)+cos(x)*csc(y))*sin(z))/(cos(z)*sin(x)*sin(y)+cos(x)*sin(z));

	terms.clear();
	terms.push_back(sx1);
	terms.push_back(sz);
	terms.push_back(sy);
	terms.push_back(sx2);

	if(maxshear) {
		*maxshear = 0;
		for(auto v:terms) {
			DBG3(cerr << "Shear:\n" << v << endl);
			shearProduct *= v;
			double mv = getMaxShear(v);
			if(mv > *maxshear)
				*maxshear = mv;
		}
	}

	//*construct*matrices*
	Matrix3d rotation;
	rotation = AngleAxisd(x, Vector3d::UnitX())*
				AngleAxisd(y, Vector3d::UnitY())*
				AngleAxisd(z, Vector3d::UnitZ());

	DBG3(cerr << "Rotation:\n" << rotation << endl);
	DBG3(cerr << "Sheared Rotation:\n" << shearProduct<< endl);
	DBG3(cerr << "Error:\n" << (rotation-shearProduct).cwiseAbs() << endl);
	if(err)
		*err = (rotation-shearProduct).cwiseAbs().sum();

	return 0;
}

int shearZXYZ(std::list<Matrix3d>& terms, double* err, double* maxshear,
		double x, double y, double z)
{
	DBG3(cerr << "Shear ZXYZ" << endl);
	Matrix3d sz1 = Matrix3d::Identity();
	Matrix3d sx = Matrix3d::Identity();
	Matrix3d sy = Matrix3d::Identity();
	Matrix3d sz2 = Matrix3d::Identity();
	Matrix3d shearProduct = Matrix3d::Identity();

	sz1(2,0) = (-1+cos(y)*cos(z))/(cos(x)*cos(z)*sin(y)-sin(x)*sin(z));
	sz1(2,1) = ((cos(x)-sec(y))*sin(z)-csc(x)*tan(y)+cos(z)*(sin(x)*sin(y)+cot(x)*tan(y)))/(cos(x)*cos(z)*sin(y)-sin(x)*sin(z));
	sx (0,1) = -sec(y)*sin(z)+(cos(z)*cot(x)-csc(x))*tan(y);
	sx (0,2) = cos(x)*cos(z)*sin(y)-sin(x)*sin(z);
	sy (1,0) = ((-cos(y)+cos(z))*sin(x)+cos(x)*sin(y)*sin(z))/(cos(x)*cos(z)*sin(y)-sin(x)*sin(z));
	sy (1,2) = -cos(y)*sin(x);
	sz2(2,0) = (-4+cos(x-y)+cos(x+y)+(4*cos(z)+cos(x)*(-3+cos(2*y))*cos(2*z))*sec(y)+4*cot(x)*sin(z)*tan(y)-2*cos(2*x)*csc(x)*sin(2*z)*tan(y))/(4*cos(x)*cos(z)*sin(y)-4*sin(x)*sin(z));
	sz2(2,1) = (-cos(z)*cot(x)+csc(x))*sec(y)+sin(z)*tan(y);

	terms.clear();
	terms.push_back(sz1);
	terms.push_back(sx);
	terms.push_back(sy);
	terms.push_back(sz2);

	if(maxshear) {
		*maxshear = 0;
		for(auto v:terms) {
			DBG3(cerr << "Shear:\n" << v << endl);
			shearProduct *= v;
			double mv = getMaxShear(v);
			if(mv > *maxshear)
				*maxshear = mv;
		}
	}

	//*construct*matrices*
	Matrix3d rotation;
	rotation = AngleAxisd(x, Vector3d::UnitX())*
				AngleAxisd(y, Vector3d::UnitY())*
				AngleAxisd(z, Vector3d::UnitZ());

	DBG3(cerr << "Rotation:\n" << rotation << endl);
	DBG3(cerr << "Sheared Rotation:\n" << shearProduct<< endl);
	DBG3(cerr << "Error:\n" << (rotation-shearProduct).cwiseAbs() << endl);
	if(err)
		*err = (rotation-shearProduct).cwiseAbs().sum();

	return 0;
}

int shearZYXZ(std::list<Matrix3d>& terms, double* err, double* maxshear,
		double x, double y, double z)
{
	DBG3(cerr << "Shear ZYXZ" << endl);
	Matrix3d sz1 = Matrix3d::Identity();
	Matrix3d sy = Matrix3d::Identity();
	Matrix3d sx = Matrix3d::Identity();
	Matrix3d sz2 = Matrix3d::Identity();
	Matrix3d shearProduct = Matrix3d::Identity();

	sz1(2,0) = ((cos(y)-cos(z))*csc(y)*sin(x)+(-cos(x)+cos(y))*sin(z))/(cos(z)*sin(x)+cos(x)*sin(y)*sin(z));
	sz1(2,1) = (1-cos(x)*cos(z)+sin(x)*sin(y)*sin(z))/(cos(z)*sin(x)+cos(x)*sin(y)*sin(z));
	sy (1,0) = -cot(y)*sin(x)+cos(z)*csc(y)*sin(x)+cos(x)*sin(z);
	sy (1,2) = -cos(z)*sin(x)-cos(x)*sin(y)*sin(z);
	sx (0,1) = -((sin(y)-cos(x)*cos(z)*sin(y)+sin(x)*sin(z))/(cos(z)*sin(x)+cos(x)*sin(y)*sin(z)));
	sx (0,2) = sin(y);
	sz2(2,0) = (-1+cos(y)*cos(z))*csc(y);
	sz2(2,1) = -((-4+cos(x-y)+cos(x+y)+4*cos(x)*cos(z)-2*cos(x)*cos(y)*cos(2*z)+4*(cos(z)*cot(y)-csc(y))*sin(x)*sin(z))/(4*(cos(z)*sin(x)+cos(x)*sin(y)*sin(z))));

	terms.clear();
	terms.push_back(sz1);
	terms.push_back(sy);
	terms.push_back(sx);
	terms.push_back(sz2);

	if(maxshear) {
		*maxshear = 0;
		for(auto v:terms) {
			DBG3(cerr << "Shear:\n" << v << endl);
			shearProduct *= v;
			double mv = getMaxShear(v);
			if(mv > *maxshear)
				*maxshear = mv;
		}
	}

	//*construct*matrices*
	Matrix3d rotation;
	rotation = AngleAxisd(x, Vector3d::UnitX())*
				AngleAxisd(y, Vector3d::UnitY())*
				AngleAxisd(z, Vector3d::UnitZ());

	DBG3(cerr << "Rotation:\n" << rotation << endl);
	DBG3(cerr << "Sheared Rotation:\n" << shearProduct<< endl);
	DBG3(cerr << "Error:\n" << (rotation-shearProduct).cwiseAbs() << endl);
	if(err)
		*err = (rotation-shearProduct).cwiseAbs().sum();

	return 0;
}

int shearYXZY(std::list<Matrix3d>& terms, double* err, double* maxshear,
		double x, double y, double z)
{
	DBG3(cerr << "Shear YXZY" << endl);
	Matrix3d sy1 = Matrix3d::Identity();
	Matrix3d sx = Matrix3d::Identity();
	Matrix3d sz = Matrix3d::Identity();
	Matrix3d sy2 = Matrix3d::Identity();
	Matrix3d shearProduct = Matrix3d::Identity();

	sy1(1,0) = (1-cos(y)*cos(z))/(cos(z)*sin(x)*sin(y)+cos(x)*sin(z));
	sy1(1,2) = (-8*cos(z)*sin(x)*sin(y)+4*cos(2*z)*sin(2*x)*sin(y)+8*(-cos(x)+cos(y))*sin(z)+(-1+3*cos(2*x)-2*cos(x)*cos(x)*cos(2*y))*sin(2*z))/(8*(cos(z)*sin(x)*sin(y)+cos(x)*sin(z))*(cos(z)*sin(x)+cos(x)*sin(y)*sin(z)));
	sx (0,1) = -cos(z)*sin(x)*sin(y)-cos(x)*sin(z);
	sx (0,2) = (cos(z)*sin(x)*sin(y)+(cos(x)-cos(y))*sin(z))/(cos(z)*sin(x)+cos(x)*sin(y)*sin(z));
	sz (2,0) = ((cos(y)-cos(z))*sin(x)-cos(x)*sin(y)*sin(z))/(cos(z)*sin(x)*sin(y)+cos(x)*sin(z));
	sz (2,1) = cos(z)*sin(x)+cos(x)*sin(y)*sin(z);
	sy2(1,0) = (-cos(z)*sin(x)*(-1+cos(z)*(cos(y)+cos(x)*sin(y)*sin(y)))+(cos(x)-cos(2*x)*cos(z))*sin(y)*sin(z)+(cos(x)-cos(y))*sin(x)*sin(z)*sin(z))/((cos(z)*sin(x)*sin(y)+cos(x)*sin(z))*(cos(z)*sin(x)+cos(x)*sin(y)*sin(z)));
	sy2(1,2) = (-1+cos(x)*cos(y))/(cos(z)*sin(x)+cos(x)*sin(y)*sin(z));

	terms.clear();
	terms.push_back(sy1);
	terms.push_back(sx);
	terms.push_back(sz);
	terms.push_back(sy2);

	if(maxshear) {
		*maxshear = 0;
		for(auto v:terms) {
			DBG3(cerr << "Shear:\n" << v << endl);
			shearProduct *= v;
			double mv = getMaxShear(v);
			if(mv > *maxshear)
				*maxshear = mv;
		}
	}

	//*construct*matrices*
	Matrix3d rotation;
	rotation = AngleAxisd(x, Vector3d::UnitX())*
				AngleAxisd(y, Vector3d::UnitY())*
				AngleAxisd(z, Vector3d::UnitZ());

	DBG3(cerr << "Rotation:\n" << rotation << endl);
	DBG3(cerr << "Sheared Rotation:\n" << shearProduct<< endl);
	DBG3(cerr << "Error:\n" << (rotation-shearProduct).cwiseAbs() << endl);
	if(err)
		*err = (rotation-shearProduct).cwiseAbs().sum();

	return 0;
}

/*****************************************************************************
 * Single Rotation Shears, which end up being 3 shears rather than four.
 ****************************************************************************/

int shearYXY(std::list<Matrix3d>& terms, double* err, double* maxshear,
		double Rx, double Ry, double Rz)
{
	DBG3(cerr << "Shear YXY" << endl);

	double MINANG = 0.00000001;
	if(fabs(Rx) > MINANG || fabs(Ry) > MINANG) {
		if(err)
			*err = NAN;
		if(maxshear)
			*maxshear= NAN;
		return -1;
	}

	Matrix3d sy1 = Matrix3d::Identity();
	Matrix3d sx = Matrix3d::Identity();
	Matrix3d sy2 = Matrix3d::Identity();
	Matrix3d shearProduct = Matrix3d::Identity();

	sy1(1,0) = tan(Rz/2.);
	sy1(1,2) = 0;
	sx (0,1) = -sin(Rz);
	sx (0,2) = 0;
	sy2(1,0) = tan(Rz/2.);
	sy2(1,2) = 0;

	terms.clear();
	terms.push_back(sy1);
	terms.push_back(sx);
	terms.push_back(sy2);

	//*construct*matrices*
	Matrix3d rotation;
	rotation = AngleAxisd(Rx, Vector3d::UnitX())*
				AngleAxisd(Ry, Vector3d::UnitY())*
				AngleAxisd(Rz, Vector3d::UnitZ());

	if(maxshear) {
		*maxshear = 0;
		for(auto v:terms) {
			DBG3(cerr << "Shear:\n" << v << endl);
			shearProduct *= v;
			double mv = getMaxShear(v);
			if(mv > *maxshear)
				*maxshear = mv;
		}
	}

	DBG3(cerr << "Rotation:\n" << rotation << endl);
	DBG3(cerr << "Sheared Rotation:\n" << shearProduct<< endl);
	DBG3(cerr << "Error:\n" << (rotation-shearProduct).cwiseAbs() << endl);
	if(err)
		*err = (rotation-shearProduct).cwiseAbs().sum();


	return 0;
}

int shearXZX(std::list<Matrix3d>& terms, double* err, double* maxshear,
		double Rx, double Ry, double Rz)
{
	DBG3(cerr << "Shear XZX" << endl);

	double MINANG = 0.00000001;
	if(fabs(Rx) > MINANG || fabs(Rz) > MINANG) {
		if(err)
			*err = NAN;
		if(maxshear)
			*maxshear= NAN;
		return -1;
	}

	Matrix3d sx1 = Matrix3d::Identity();
	Matrix3d sz  = Matrix3d::Identity();
	Matrix3d sx2 = Matrix3d::Identity();
	Matrix3d shearProduct = Matrix3d::Identity();

	sx1(0,1) = 0;
	sx1(0,2) = tan(Ry/2.);
	sz (2,0) = -sin(Ry);
	sz (2,1) = 0;
	sx2(0,1) = 0;
	sx2(0,2) = tan(Ry/2.);

	terms.clear();
	terms.push_back(sx1);
	terms.push_back(sz);
	terms.push_back(sx2);

	if(maxshear) {
		*maxshear = 0;
		for(auto v:terms) {
			DBG3(cerr << "Shear:\n" << v << endl);
			shearProduct *= v;
			double mv = getMaxShear(v);
			if(mv > *maxshear)
				*maxshear = mv;
		}
	}

	//*construct*matrices*
	Matrix3d rotation;
	rotation = AngleAxisd(Rx, Vector3d::UnitX())*
				AngleAxisd(Ry, Vector3d::UnitY())*
				AngleAxisd(Rz, Vector3d::UnitZ());

	DBG3(cerr << "Rotation:\n" << rotation << endl);
	DBG3(cerr << "Sheared Rotation:\n" << shearProduct<< endl);
	DBG3(cerr << "Error:\n" << (rotation-shearProduct).cwiseAbs() << endl);
	if(err)
		*err = (rotation-shearProduct).cwiseAbs().sum();

	return 0;
}

int shearZYZ(std::list<Matrix3d>& terms, double* err, double* maxshear,
		double Rx, double Ry, double Rz)
{
	DBG3(cerr << "Shear ZYZ" << endl);

	double MINANG = 0.00000001;
	if(fabs(Ry) > MINANG || fabs(Rz) > MINANG) {
		if(err)
			*err = NAN;
		if(maxshear)
			*maxshear= NAN;
		return -1;
	}

	Matrix3d sz1 = Matrix3d::Identity();
	Matrix3d sy  = Matrix3d::Identity();
	Matrix3d sz2 = Matrix3d::Identity();
	Matrix3d shearProduct = Matrix3d::Identity();

	sz1(2,0) = 0;
	sz1(2,1) = tan(Rx/2.);
	sy (1,0) = 0;
	sy (1,2) = -sin(Rx);
	sz2(2,0) = 0;
	sz2(2,1) = tan(Rx/2.);

	terms.clear();
	terms.push_back(sz1);
	terms.push_back(sy);
	terms.push_back(sz2);

	if(maxshear) {
		*maxshear = 0;
		for(auto v:terms) {
			DBG3(cerr << "Shear:\n" << v << endl);
			shearProduct *= v;
			double mv = getMaxShear(v);
			if(mv > *maxshear)
				*maxshear = mv;
		}
	}

	//*construct*matrices*
	Matrix3d rotation;
	rotation = AngleAxisd(Rx, Vector3d::UnitX())*
				AngleAxisd(Ry, Vector3d::UnitY())*
				AngleAxisd(Rz, Vector3d::UnitZ());

	DBG3(cerr << "Rotation:\n" << rotation << endl);
	DBG3(cerr << "Sheared Rotation:\n" << shearProduct<< endl);
	DBG3(cerr << "Error:\n" << (rotation-shearProduct).cwiseAbs() << endl);
	if(err)
		*err = (rotation-shearProduct).cwiseAbs().sum();

	return 0;
}

/**
 * @brief Decomposes a euler angle rotation using the rotation matrix made up
 * of R = Rx*Ry*Rz. Note that this would be multiplying the input vector by Rz
 * then Ry, then Rx. This does not support angles > PI/4. To do that, you
 * should first do bulk rotation using 90 degree rotations (which requires not
 * interpolation).
 *
 * @param bestshears    List of the best fitting shears, should be applied in
 *                      forward order
 * @param Rx            Rotation about X axis
 * @param Ry            Rotation about Y axis
 * @param Rz            Rotation about Z axis
 *
 * @return              Success if 0
 */
int shearDecompose(std::list<Matrix3d>& bestshears, double Rx, double Ry, double Rz)
{
	double ERRTOL = 0.0001;
	double SHEARMAX = 1;
	double ANGMIN = 0.000001;
	double err, maxshear;
	double bestmshear = SHEARMAX;
	std::list<Matrix3d> ltmp;

	// single angles first
	if(fabs(Rx) < ANGMIN && fabs(Ry) < ANGMIN) {
        DBG3(cerr << "Chose YXY" << endl);
		shearYXY(bestshears, &err, &maxshear, Rx, Ry, Rz);
		if(err < ERRTOL && maxshear < SHEARMAX) {
			return 0;
		} else {
			return -1;
		}
	}

    if(fabs(Rx) < ANGMIN && fabs(Rz) < ANGMIN) {
        DBG3(cerr << "Chose XZX" << endl);
        shearXZX(bestshears, &err, &maxshear, Rx, Ry, Rz);
        if(err < ERRTOL && maxshear < SHEARMAX) {
            return 0;
        } else {
            return -1;
        }
    }

    if(fabs(Rz) < ANGMIN && fabs(Ry) < ANGMIN) {
        DBG3(cerr << "Chose ZYZ" << endl);
        shearZYZ(bestshears, &err, &maxshear, Rx, Ry, Rz);
        if(err < ERRTOL && maxshear < SHEARMAX) {
            return 0;
        } else {
            return -1;
		}
	}

	shearYZXY(ltmp, &err, &maxshear, Rx, Ry, Rz);
	if(err < ERRTOL && maxshear < bestmshear) {
		bestmshear = maxshear;
		bestshears.clear();
		bestshears.splice(bestshears.end(), ltmp);
	}

	shearYXZY(ltmp, &err, &maxshear, Rx, Ry, Rz);
	if(err < ERRTOL && maxshear < bestmshear) {
		bestmshear = maxshear;
		bestshears.clear();
		bestshears.splice(bestshears.end(), ltmp);
	}

	shearXYZX(ltmp, &err, &maxshear, Rx, Ry, Rz);
	if(err < ERRTOL && maxshear < bestmshear) {
		bestmshear = maxshear;
		bestshears.clear();
		bestshears.splice(bestshears.end(), ltmp);
	}

	shearXZYX(ltmp, &err, &maxshear, Rx, Ry, Rz);
	if(err < ERRTOL && maxshear < bestmshear) {
		bestmshear = maxshear;
		bestshears.clear();
		bestshears.splice(bestshears.end(), ltmp);
	}

	shearZYXZ(ltmp, &err, &maxshear, Rx, Ry, Rz);
	if(err < ERRTOL && maxshear < bestmshear) {
		bestmshear = maxshear;
		bestshears.clear();
		bestshears.splice(bestshears.end(), ltmp);
	}

	shearZXYZ(ltmp, &err, &maxshear, Rx, Ry, Rz);
	if(err < ERRTOL && maxshear < bestmshear) {
		bestmshear = maxshear;
		bestshears.clear();
		bestshears.splice(bestshears.end(), ltmp);
	}

	if(bestmshear <= SHEARMAX) {
		DBG3(cerr << "Best Shear:" << bestmshear << endl);
		return 0;
	}

	return -1;
}

/**
 * @brief Tests all the shear decomposition functions.
 * Given a rotation, it computes shear versions of the rotation matrix and
 * the error. The error should be very near 0 unless the reconstruction is
 * NAN.
 *
 * @param Rx Rotation about x axis (first)
 * @param Ry Rotation about y axis (middle)
 * @param Rz Rotation about z axis (last)
 *
 * @return 0 if successful.
 */
int shearTest(double Rx, double Ry, double Rz)
{
	cerr << "------------------------------" << endl;
	cerr << Rx << "," << Ry << "," << Rz << endl;
	double ERRTOL = 0.0001;
	std::list<Matrix3d> terms;
	double err, maxshear;

	// single angles first
	shearYXY(terms, &err, &maxshear, Rx, Ry, Rz);
	cerr << "YXY maxshear: " << maxshear << endl;
	if(isnormal(err) && err > ERRTOL) {
		cerr << "Err: " << err << endl;
		return -1;
	}

	shearXZX(terms, &err, &maxshear, Rx, Ry, Rz);
	cerr << "XZX maxshear: " << maxshear << endl;
	if(isnormal(err) && err > ERRTOL) {
		cerr << "Err: " << err << endl;
		return -1;
	}

	shearZYZ(terms, &err, &maxshear, Rx, Ry, Rz);
	cerr << "ZYZ maxshear: " << maxshear << endl;
	if(isnormal(err) && err > ERRTOL) {
		cerr << "Err: " << err << endl;
		return -1;
	}

	shearYZXY(terms, &err, &maxshear, Rx, Ry, Rz);
	cerr << "YZXY maxshear: " << maxshear << endl;
	if(isnormal(err) && err > ERRTOL) {
		cerr << "Err: " << err << endl;
		return -1;
	}

	shearYXZY(terms, &err, &maxshear, Rx, Ry, Rz);
	cerr << "YXZY maxshear: " << maxshear << endl;
	if(isnormal(err) && err > ERRTOL) {
		cerr << "Err: " << err << endl;
		return -1;
	}

	shearXYZX(terms, &err, &maxshear, Rx, Ry, Rz);
	cerr << "XYZX maxshear: " << maxshear << endl;
	if(isnormal(err) && err > ERRTOL) {
		cerr << "Err: " << err << endl;
		return -1;
	}

	shearXZYX(terms, &err, &maxshear, Rx, Ry, Rz);
	cerr << "XZYX maxshear: " << maxshear << endl;
	if(isnormal(err) && err > ERRTOL) {
		cerr << "Err: " << err << endl;
		return -1;
	}

	shearZYXZ(terms, &err, &maxshear, Rx, Ry, Rz);
	cerr << "ZYXZ maxshear: " << maxshear << endl;
	if(isnormal(err) && err > ERRTOL) {
		cerr << "Err: " << err << endl;
		return -1;
	}

	shearZXYZ(terms, &err, &maxshear, Rx, Ry, Rz);
	cerr << "ZXYZ maxshear: " << maxshear << endl;
	if(isnormal(err) && err > ERRTOL) {
		cerr << "Err: " << err << endl;
		return -1;
	}

	return 0;
}


/*****************************************
 * Image Rotation
 ****************************************/

/**
 * @brief Performs a rotation of the image first by rotating around z, then
 * around y, then around x.
 *
 * @param rx Rotation around x, radians
 * @param ry Rotation around y, radians
 * @param rz Rotation around z, radians
 * @param in Input image
 *
 * @return
 */
ptr<NDArray> linearRotate(double rx, double ry, double rz,
		ptr<const NDArray> in)
{
	Matrix3d m;
	// negate because we are starting from the destination and mapping from
	// the source, since we need to invert, it is necessary to reverse the
    // real order
	m = AngleAxisd(-rz, Vector3d::UnitZ())*AngleAxisd(-ry,Vector3d::UnitY())*
                AngleAxisd(-rx,Vector3d::UnitX());
	LinInterp3DView<double> lin(in);
	auto out = in->copy();
	Vector3d ind;
	Vector3d cind;
	Vector3d center;
	for(size_t ii=0; ii<3 && ii<in->ndim(); ii++)
		center[ii] = (in->dim(ii)-1)/2.;

	for(Vector3DIter<double> it(out); !it.isEnd(); ++it) {
		it.index(3, ind.array().data());
		cind = m*(ind-center)+center;

		// set for each t
		for(size_t tt = 0; tt<in->tlen(); tt++)
			it.set(tt, lin(cind[0], cind[1], cind[2], tt));
	}

	return out;
}


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
int rotateImageShearKern(ptr<NDArray> inout, double rx, double ry, double rz,
		double(*kern)(double,double))
{
	const double PI = acos(-1);
	if(fabs(rx) > PI/4. || fabs(ry) > PI/4. || fabs(rz) > PI/4.) {
		cerr << "Fast large rotations not yet implemented" << endl;
		return -1;
	}

	std::list<Matrix3d> shears;

	// decompose into shears
	clock_t c = clock();
	if(shearDecompose(shears, rx, ry, rz) != 0) {
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
		if(sheardim != -1)
			shearImageKern(inout, sheardim, 3, shearvals, kern);

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
int rotateImageShearFFT(ptr<NDArray> inout, double rx, double ry, double rz,
		double(*window)(double,double))
{
	const double PI = acos(-1);
	if(fabs(rx) > PI/4. || fabs(ry) > PI/4. || fabs(rz) > PI/4.) {
		cerr << "Fast large rotations not yet implemented" << endl;
		return -1;
	}

	std::list<Matrix3d> shears;

	// decompose into shears
	clock_t c = clock();
	if(shearDecompose(shears, rx, ry, rz) != 0) {
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
		if(sheardim != -1)
			shearImageFFT(inout, sheardim, 3, shearvals, window);
	}

	return 0;
}

/**********************************
 * Radial Fourier Transforms
 *********************************/

// upsample in anglular directions
ptr<NDArray> pphelp_padFFT(ptr<const NDArray> in,
		const std::vector<double>& upsamp)
{
	std::vector<size_t> osize(in->ndim(), 0);
	size_t maxbsize = 0;
	for(size_t ii=0; ii<in->ndim(); ii++) {
		osize[ii] = round2(in->dim(ii)*upsamp[ii]);
		maxbsize = std::max(maxbsize, osize[ii]);
	}

	auto buffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*maxbsize);
	auto oimg = in->copyCast(osize.size(), osize.data(), COMPLEX128);
	std::vector<int64_t> index(in->ndim());

	// fourier transform
	for(size_t dd = 0; dd < oimg->ndim(); dd++) {
		fftw_plan fwd = fftw_plan_dft_1d((int)osize[dd], buffer, buffer,
				FFTW_FORWARD, FFTW_MEASURE);

		ChunkIter<cdouble_t> it(oimg);
		it.setLineChunk(dd);
		for(it.goBegin(); !it.eof() ; it.nextChunk()) {
			it.index(index.size(), index.data());

			// fill from line
			for(size_t tt=0; !it.eoc(); ++it, tt++) {
				buffer[tt][0] = (*it).real();
				buffer[tt][1] = (*it).imag();
			}

			// fourier transform
			fftw_execute(fwd);

			// copy/shift
			// F += N/2 (even), for N = 4:
			// 0 -> 2 (f =  0)
			// 1 -> 3 (f = +1)
			// 2 -> 0 (f = -2)
			// 3 -> 1 (f = -1)
			it.goChunkBegin();
			double norm = 1./sqrt(osize[dd]);
			for(size_t tt=osize[dd]/2; !it.isChunkEnd(); ++it) {
				cdouble_t tmp(buffer[tt][0]*norm, buffer[tt][1]*norm);
				it.set(tmp);
				tt=(tt+1)%osize[dd];
			}
		}

		fftw_destroy_plan(fwd);
	}

	fftw_free(buffer);

	return oimg;
}

/**
 * @brief Computes the pseudopolar-gridded fourier transform on the input
 * image, with prdim as the pseudo-radius direction. To sample the whole space
 * you would need to call this once for each of the dimensions, or use the
 * other function which does not take this argument, and returns a vector.
 * This function skips the chirpz transform by interpolation-zooming.
 *
 * @param inimg	Input image to compute pseudo-polar fourier transform on
 * @param prdim	Dimension to be the pseudo-radius in output
 *
 * @return 		Pseudo-polar sample fourier transform
 */
ptr<NDArray> pseudoPolarZoom(ptr<const NDArray> inimg, size_t prdim)
{
	// create output
	std::vector<double> upsample(inimg->ndim(), 2);
	upsample[prdim] = 1;

	ptr<NDArray> out = pphelp_padFFT(inimg, upsample);

	// write out padded/FFT image
	{
		auto absimg = out->copyCast(FLOAT64);
		auto angimg = out->copyCast(FLOAT64);

		OrderIter<double> rit(absimg);
		OrderIter<double> iit(angimg);
		OrderConstIter<cdouble_t> init(out);
		while(!init.eof()) {
			rit.set(abs(*init));
			iit.set(arg(*init));
			++init;
			++rit;
			++iit;
		}

		DBG3(absimg->write("fft"+to_string(prdim)+"_abs.nii.gz"));
		DBG3(angimg->write("fft"+to_string(prdim)+"_ang.nii.gz"));
	}

	// declare variables
	std::vector<int64_t> index(out->ndim());

	// compute/initialize buffer
	size_t buffsize = [&]
	{
		size_t m = 0;
		for(size_t dd=0; dd<out->ndim(); dd++) {
			if(dd != prdim)
				m = max(out->dim(dd), m);
		}
		return m*2;
	}();

	auto buffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*buffsize);


	for(size_t dd=0; dd<out->ndim(); dd++) {
		if(dd == prdim)
			continue;

		size_t isize = out->dim(dd);

		ChunkIter<cdouble_t> it(out);
		it.setLineChunk(dd);
		it.setOrder({prdim}, true); // make pseudoradius slowest
		for(it.goBegin(); !it.eof(); it.nextChunk()) {
			it.index(index);

			// recompute chirps if alpha changed
			double alpha = 2*(index[prdim]/(out->dim(prdim)-1.)) - 1;

			// copy from input image
			it.goChunkBegin();
			for(size_t ii=0; !it.eoc(); ++it, ii++) {
				buffer[ii][0] = (*it).real();
				buffer[ii][1] = (*it).imag();
			}

			// zoom
			assert(buffsize >= isize*2);
			zoom(isize, &buffer[0], &buffer[isize], alpha);

			// copy from buffer back to output
			it.goChunkBegin();
			for(size_t ii=0; !it.eoc(); ii++, ++it) {
				cdouble_t tmp(buffer[isize+ii][0], buffer[isize+ii][1]);
				it.set(tmp);
			}
		}
	}
	fftw_free(buffer);

	return out;
}

/**
 * @brief Computes the pseudopolar-gridded fourier transform on the input
 * image, with prdim as the pseudo-radius direction. To sample the whole space
 * you would need to call this once for each of the dimensions, or use the
 * other function which does not take this argument, and returns a vector.
 *
 * @param in	Input image to compute pseudo-polar fourier transform on
 * @param prdim	Dimension to be the pseudo-radius in output
 *
 * @return 		Pseudo-polar sample fourier transform
 */
ptr<NDArray> pseudoPolar(ptr<const NDArray> in, size_t prdim)
{
	// create output
	std::vector<double> upsample(in->ndim(), 2);
	upsample[prdim] = 1;

	ptr<NDArray> out = pphelp_padFFT(in, upsample);

	// declare variables
	std::vector<int64_t> index(out->ndim());

	// compute/initialize buffer
	size_t buffsize = [&]
	{
		size_t m = 0;
		for(size_t dd=0; dd<in->ndim(); dd++) {
			if(dd != prdim)
				m = std::max(out->dim(dd), m);
		}
		return m*30;
	}();

	auto buffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*buffsize);

	for(size_t dd=0; dd<out->ndim(); dd++) {
		if(dd == prdim)
			continue;

		int64_t usize = out->dim(dd);
		int64_t uppadsize = usize*2;
		fftw_complex* current = &buffer[0];
		fftw_complex* prechirp = &buffer[usize+uppadsize];
		fftw_complex* postchirp = &buffer[usize+2*uppadsize];
		fftw_complex* convchirp = &buffer[usize+3*uppadsize];
		fftw_plan plan = fftw_plan_dft_1d((int)usize, current, current,
				FFTW_BACKWARD, FFTW_MEASURE);

		assert(buffsize >= usize+3*uppadsize);

		ChunkIter<cdouble_t> it(out);
		it.setLineChunk(dd);
		it.setOrder({prdim}, true); // make pseudoradius slowest
		double alpha, prevAlpha = NAN;
		for(it.goBegin(); !it.eof(); it.nextChunk()) {
			it.index(index);

			// recompute chirps if alpha changed
			alpha = 2*(index[prdim]/(out->dim(prdim)-1)) - 1;
			if(alpha != prevAlpha) {
				createChirp(uppadsize, prechirp, usize, 1, alpha,
						false, false);
				createChirp(uppadsize, postchirp, usize, 1, alpha,
						true, false);
				createChirp(uppadsize, convchirp, usize, 1, -alpha,
						true, true);
			}

			// copy from input image, shift
			it.goChunkBegin();
			for(size_t ii=usize/2; !it.eoc(); ++it) {
				current[ii][0] = (*it).real();
				current[ii][1] = (*it).imag();
				ii=(ii+1)%usize;
			}

			fftw_execute(plan);
			double norm = 1./sqrt(usize*usize);
			for(size_t ii=0; ii<usize; ii++) {
				current[ii][0] *= norm;
				current[ii][1] *= norm;
			}

			// compute chirpz transform
			chirpzFFT(usize, usize, current, uppadsize, &buffer[usize],
						prechirp, convchirp, postchirp);

			// copy from buffer back to output
			it.goChunkBegin();
			for(size_t ii=0; !it.eoc(); ii++, ++it) {
				cdouble_t tmp(current[ii][0], current[ii][1]);
				it.set(tmp);
			}
			prevAlpha = alpha;
		}

		fftw_destroy_plan(plan);
	}
	fftw_free(buffer);

	return out;
}

/**
 * @brief Computes the pseudopolar-gridded fourier transform on the input
 * image returns a vector of pseudo-polar sampled image, one for each dimension
 * as the pseudo-radius.
 *
 * @param in	Input image to compute pseudo-polar fourier transform on
 *
 * @return 		Vector of Pseudo-polar sample fourier transforms, one for each
 * dimension
 */
vector<ptr<NDArray>> pseudoPolar(ptr<const NDArray> in)
{
	std::vector<ptr<NDArray>> out(in->ndim());
	for(size_t dd=0; dd < in->ndim(); dd++) {
		out[dd] = pseudoPolar(in, dd);
	}

	return out;
}

/**
 * @brief Sets the middle of the image += radius (in index space) to 1,
 * everything else to 0
 *
 * @param inout Input/output image.
 * @param radius Radius (distance from center) to set to 1
 * @param alphis the what the distance is raised to in each dimension (2 is
 * euclidian distance)
 */
void fillCircle(ptr<NDArray> inout, double radius, double alpha)
{
    double rsqr = pow(radius, alpha);
    vector<int64_t> index(inout->ndim(), 0);
    for(NDIter<double> it(inout); !it.eof(); ++it) {
        it.index(index.size(), index.data());
        double dist = 0;
        for(size_t dd=0; dd<inout->ndim(); dd++) {
            dist += fabs(pow(index[dd]-(inout->dim(dd)-1.)/2.,alpha));
        }
        if(dist <= rsqr)
            it.set(1);
        else
            it.set(0);
    }
}

/**
 * @brief Fills image with the linear index at each pixel
 *
 * @param inout input/output image, will be filled with linear index
 *
 */
void fillLinear(ptr<NDArray> inout)
{
    FlatIter<float> it(inout);
    for(size_t ii=0; !it.eof(); ii++, ++it) {
        it.set(ii);
    }
}

/**
 * @brief Fills image with the linear index at each pixel
 *
 * @param inout input/output image, will be filled with gaussian white noise
 *
 */
void fillGaussian(ptr<NDArray> inout)
{
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::normal_distribution<double> gauss(0, 1);

    for(FlatIter<float> it(inout); !it.eof(); ++it) {
        it.set(gauss(rng));
    }
}

/**
 * @brief Concatinates image in the direction specified by dir. So if dir
 * is 0, and two images, sized [32, 32, 34] and [12, 32, 34] were passed
 * in the input vector, then the output would be [44, 32, 34].
 *
 * @param images Input images, will be placed in order of input vector
 * @param dir Direction to concatinate, all dimesnions other than dir
 * must match in size
 *
 * @return New image that has had the images pasted together
 */
ptr<NDArray> concat(const vector<ptr<NDArray>>& images, size_t dir)
{
    if(images.size() == 0)
        throw INVALID_ARGUMENT("Input image array had size zero!");

    if(images[0]->type() == RGBA32 || images[0]->type() == RGB24 ||
            images[0]->type() == COMPLEX128 || images[0]->type() == COMPLEX64
            || images[0]->type() == COMPLEX256) {
        throw INVALID_ARGUMENT("Concatination of tuple-types has not yet "
                "been implemented");
    }

    if(dir >= images[0]->ndim()) {
        throw INVALID_ARGUMENT("Input direction dim is greater than input "
                "images dimensions!");
    }

    size_t ndim = images[0]->ndim();
    vector<size_t> osize(images[0]->dim(), images[0]->dim()+ndim);

    // figure output size
    osize[dir] = 0;
    for(const auto& v: images) {
        // check sizes
        for(size_t dd=0; dd<ndim; dd++) {
            if(dd != dir && v->dim(dd) != osize[dd])
                throw INVALID_ARGUMENT("Input image have different sizes!");
        }
        osize[dir] += v->dim(dir);
    }

    // create output image
    ptr<NDArray> oimg = images[0]->copyCast(osize.size(), osize.data());

    // iterate through, make highest dimension slowest
    NDIter<double> oit(oimg);
    vector<size_t> order;
    for(int64_t ii=ndim; ii>= 0; ii--) {
        if(ii != dir)
            order.push_back(ii);
    }
    order.push_back(dir);
    oit.setOrder(order);
    oit.goBegin();

    // that dimension doesn't exist in these images
    size_t imgii = 0;
    NDConstIter<double> iit;

    // go through input images one by one
    while(!oit.eof()) {
        // switch to the next input image
        iit.setArray(images[imgii++]);
        iit.setOrder(order);
        iit.goBegin();

        while(!iit.eof() && !oit.eof()) {
            oit.set(*iit);
            ++oit;
            ++iit;
        }
    }

    assert(imgii == images.size());
    assert(iit.eof());

    return oimg;
}

/**
 * @brief Concatinates images/arrays. 1 Extra dimension will be added, all the
 * lower dimensions of the images must match. An example with lastdim = false
 * would be 3 [32,32,32] images, which would result in 1 [32,32,32,3] image
 * with the orienation matching from the first image.
 *
 * @param images Array of images to concatinate
 *
 * @return New image with 1 extra dimension
 */
ptr<NDArray> concatElevate(const vector<ptr<NDArray>>& images)
{
    if(images.size() == 0)
        throw INVALID_ARGUMENT("Input image array had size zero!");

    if(images[0]->type() == RGBA32 || images[0]->type() == RGB24 ||
            images[0]->type() == COMPLEX128 || images[0]->type() == COMPLEX64
            || images[0]->type() == COMPLEX256) {
        throw INVALID_ARGUMENT("Concatination of tuple-types has not yet "
                "been implemented");
    }

    // concatinate in a new dimension
    size_t ndim = images[0]->ndim()+1;
    vector<size_t> osize(images[0]->dim(), images[0]->dim()+images[0]->ndim());
    osize.push_back(images.size());

    // check sizes
    for(const auto& v: images) {
        for(size_t dd=0; dd<ndim-1; dd++) {
            if(v->dim(dd) != osize[dd])
                throw INVALID_ARGUMENT("Input image have different sizes!");
        }
    }

    // create output image
    ptr<NDArray> oimg = images[0]->copyCast(osize.size(), osize.data());

    // iterate through, make highest dimension slowest
    NDIter<double> oit(oimg);
    vector<size_t> order;
    for(int64_t ii=ndim-2; ii>= 0; ii--)
        order.push_back(ii);
    order.push_back(ndim-1);

    oit.setOrder(order);
    oit.goBegin();

    // that dimension doesn't exist in these images
    size_t imgii = 0;
    NDConstIter<double> iit;

    // go through input images one by one
    while(!oit.eof()) {
        // switch to the next input image
        iit.setArray(images[imgii++]);
        iit.goBegin();

        while(!iit.eof() && !oit.eof()) {
            oit.set(*iit);
            ++oit;
            ++iit;
        }
    }

    assert(imgii == images.size());
    assert(iit.eof());

    return oimg;
}

/**
 * @brief Increases the number of dimensions by 1 then places the edges
 * in each dimension at indexes matching the direction of edge detection.
 * So an input 3D image will produce a 4D image with volume 0 the x edges,
 * volume 1 the y edges and volume 2 the z edges.
 *
 * @param img Input image ND
 *
 * @return Output image N+1D
 */
ptr<NDArray> laplacian(ptr<const NDArray> img)
{
	// create output
	size_t ndim = img->ndim();
	vector<size_t> osize(img->dim(), img->dim()+ndim);
	osize.push_back(ndim);
	auto out = img->copyCast(osize.size(), osize.data());

	vector<double> der_profile({1, -2, 1});
	vector<double> avg_profile({0.25, 0.5, 0.25});

	//////////////////
	// iterate through
	//////////////////

	// kernel iterator to get neighbors of the coordesponding output point
	KernelIter<double> kit(img);
	kit.setRadius(1);

	// chunk up by volumes
	ChunkIter<double> oit(out);
	oit.setChunkSize(ndim, img->dim(), true);
	oit.setOrder(kit.getOrder());
	vector<int64_t> index(ndim+1);
	for(oit.goBegin(); !oit.eof(); oit.nextChunk()) {
		oit.index(index);
		size_t graddir = index[ndim];

		// apply kernel in dimension of graddir
		for(kit.goBegin(); !kit.eof() && !oit.eoc(); ++kit, ++oit) {
			double sum = 0;
			for(size_t kk=0; kk<kit.ksize(); kk++) {
				kit.offsetK(kk, index.size(), index.data());

				// compute weight of kernel element, note that because
				// from_center is the offset from center, we need to add 1
				double w = 1;
				for(size_t dd=0; dd<ndim; dd++) {
					if(dd == graddir)
						w *= der_profile[index[dd]+1];
					else
						w *= avg_profile[index[dd]+1];
				}

				sum += w*kit[kk];
			}

			oit.set(sum);
		}

		assert(kit.eof() && oit.eoc());
	}
	assert(kit.eof() && oit.eof());

	return out;
}

/**
 * @brief Increases the number of dimensions by 1 then places the edges
 * in each dimension at indexes matching the direction of edge detection.
 * So an input 3D image will produce a 4D image with volume 0 the x edges,
 * volume 1 the y edges and volume 2 the z edges.
 *
 * @param img Input image ND
 *
 * @return Output image N+1D
 */
ptr<NDArray> sobelEdge(ptr<const NDArray> img)
{
    // create output
    size_t ndim = img->ndim();
    vector<size_t> osize(img->dim(), img->dim()+ndim);
    osize.push_back(ndim);
    auto out = img->copyCast(osize.size(), osize.data());

    // sobel is the combination of 1 derivative with averaging in the other
    // dimensions
    vector<double> der_profile({-0.5, 0, 0.5});
    vector<double> avg_profile({0.25, 0.5, 0.25});

    //////////////////
    // iterate through
    //////////////////

    // kernel iterator to get neighbors of the coordesponding output point
    KernelIter<double> kit(img);
    kit.setRadius(1);

    // chunk up by volumes
    ChunkIter<double> oit(out);
    oit.setChunkSize(ndim, img->dim(), true);
    oit.setOrder(kit.getOrder());
    vector<int64_t> index(ndim+1);
    for(oit.goBegin(); !oit.eof(); oit.nextChunk()) {
        oit.index(index);
        size_t graddir = index[ndim];

        // apply kernel in dimension of graddir
        for(kit.goBegin(); !kit.eof() && !oit.eoc(); ++kit, ++oit) {
            double sum = 0;
            for(size_t kk=0; kk<kit.ksize(); kk++) {
                kit.offsetK(kk, index.size(), index.data());

                // compute weight of kernel element, note that because
                // from_center is the offset from center, we need to add 1
                double w = 1;
                for(size_t dd=0; dd<ndim; dd++) {
                    if(dd == graddir)
                        w *= der_profile[index[dd]+1];
                    else
                        w *= avg_profile[index[dd]+1];
                }

                sum += w*kit[kk];
            }

            oit.set(sum);
        }

        assert(kit.eof() && oit.eoc());
    }
    assert(kit.eof() && oit.eof());

    return out;
}

/**
 * @brief Creates a new image with the specified dimension collapsed and the
 * values in each output point set to the sum of the values in the collapsed
 * dimension
 *
 * @param img Input image
 *
 * @return
 */
ptr<NDArray> collapseSum(ptr<const NDArray> img, size_t dim, bool doabs)
{
    if(dim >= img->ndim()) {
        throw INVALID_ARGUMENT("Input dim is >= number of dimensions!");
    }

    vector<size_t> osize(img->ndim()-1);
    for(size_t ii=0, jj=0; ii<img->ndim(); ii++) {
        if(ii != dim)
            osize[jj++] = img->dim(ii);
    }

    auto out = img->createAnother(img->ndim()-1, osize.data());

    vector<int64_t> index1(img->ndim());
    vector<int64_t> index2(img->ndim()-1);
    NDView<double> oac(out);
    ChunkConstIter<double> iit(img);
    iit.setLineChunk(dim);
    for(iit.goBegin(); !iit.eof(); iit.nextChunk()) {
        double sum = 0;
        for(; !iit.eoc(); ++iit)
            sum += doabs ? fabs(*iit) : *iit;

        iit.index(index1);

        // convert index
        for(size_t ii=0, jj=0; ii<img->ndim(); ii++) {
            if(ii != dim)
                index2[jj++] = index1[ii];
        }
        oac.set(index2, sum);
    }

    return out;
}

/**
 * @brief Fills an image with zeros.
 *
 * @param inout input/output image. Will be all zeros
 */
inline
void zero(ptr<NDArray> inout)
{
	for(FlatIter<int> it(inout); !it.eof(); ++it)
		it.set(0);
}

/**
 * @brief Performs relabeling based on connected component using the two pass
 * algorithm.
 *
 * @param input Input labelmap image
 *
 * @return Relabeled image with connected components labeled together, and
 * non-connected components labeled separately
 */
ptr<NDArray> relabelConnected(ptr<NDArray> input)
{
	size_t ndim = input->ndim();
	auto output = input->createAnother();
	zero(output);

	// accessors and iterator
	NDConstView<int> oac(output);
	NDConstView<int> iac(input);
	NDIter<int> oit(output);
	NDConstIter<int> iit(input);

	// maps for checking equivalence
	std::unordered_map<int, int> equivalent;
	std::unordered_map<int, int> remap;
	vector<int64_t> ind(input->ndim());

	//connected component
	int maxlabel = 1;
	for(oit.goBegin(); !oit.eof(); ++oit) {
		oit.index(ind);

		//check before in each dimension
		int pval, npval;
		int newlabel = 0;
		for(int dd = 0; dd < ndim;  dd++) {
			ind[dd]--;

			//skip points outside the image
			if(ind[dd] < 0) {
				ind[dd]++;
				continue;
			}

			pval = iac[ind];
			npval = oac[ind];

			// found a connection
			if(pval == *iit) {

				// pixel unclaimed so far, so claim with newlabel
				// (which must have been labeled because it is before us)
				if(newlabel == 0) {
					newlabel = oac[ind];

				// this pixel has already been claimed by another
				} else if(npval != newlabel) {
					auto ret1 = equivalent.insert({newlabel, npval});
					auto ret2 = equivalent.insert({npval, newlabel});

					newlabel = min(newlabel, npval);
					newlabel = min(newlabel, ret1.first->second);
					newlabel = min(newlabel, ret2.first->second);

					// point both of the lookups to the minimum value,
					// to reduce the number of hops to the minimum
					ret1.first->second = newlabel;
					ret2.first->second = newlabel;

				}
			}

			//return to middle
			ind[dd]++;
		}

		//no existing neighbor label found, use a new label
		if(newlabel == 0) {
			newlabel = maxlabel++;
		}

		oit.set(newlabel);
	}

	//second pass
	maxlabel = 1;
	for(oit.goBegin() ;!oit.eof(); ++oit) {
		int newlabel = *oit;
		int prev = newlabel;

		//find minimal equivalence
		do {
			prev = newlabel;
			newlabel = equivalent[newlabel];
		} while(prev != newlabel);

		//try insertion into remap table
		auto ret1 = remap.insert({newlabel, maxlabel});

		if(ret1.second) {
			// if it was successful insertion, then use the maxlabel and increment
			oit.set(maxlabel++);
		} else {
			// insertion failed, so use the existing remap
			oit.set(ret1.first->second);
		}
	}

	return output;
};

/**
 * @brief Computes a histogram, then spaces out intensities so that each
 * intensity has equal volume/area in the image.
 *
 * @param in Input image.
 *
 * @return Image that has been histogram equalized
 */
ptr<NDArray> histEqualize(ptr<const NDArray> in)
{
	cerr << "Histogram equalizer is beta!" << endl;
	auto out = in->copy();
	vector<double> bins(sqrt(in->elements()));
	double minv = INFINITY;
	double maxv = -INFINITY;
	for(FlatConstIter<double> iit(in); !iit.eof(); ++iit) {
		minv = std::min(minv, *iit);
		maxv = std::max(maxv, *iit);
	}
	double bwidth = 0.99999999*bins.size()/(maxv-minv);
	for(FlatConstIter<double> iit(in); !iit.eof(); ++iit)
		bins[floor((*iit-minv)*bwidth)]++;

	double cumsum = 0;
	for(size_t bb=0; bb < bins.size(); bb++) {
		bins[bb] /= in->elements();
		double p = cumsum;
		cumsum += bins[bb];
		bins[bb] = p;
	}

	FlatIter<double> oit(out);
	for(FlatConstIter<double> iit(in); !oit.eof(); ++oit, ++iit) {
		double cbin = (*iit-minv)*bwidth;
		int bin = floor(cbin);
		double v;
		if(bin == bins.size()-1)
			v = (1-cbin+bin)*bins[bin] + (cbin-bin);
		else
			v = (1-cbin+bin)*bins[bin] + (cbin-bin)*bins[bin+1];
		oit.set(v);
	}


	return out;
}

/**
 * @brief Thresholds the image, changing everything below t to 0
 *
 * @param in Input image.
 * @param t Threshold to apply to the image.
 */
void binarizeIP(ptr<NDArray> in, double t)
{
	for(FlatIter<double> it(in); !it.eof(); ++it) {
		if(*it < t)
			it.set(0);
		else
			it.set(1);
	}
}

/**
 * @brief Thresholds the image, changing everything below t to 0
 *
 * @param in Input image.
 * @param t Threshold to apply to the image.
 *
 * @return Threshold image
 */
ptr<NDArray> binarize(ptr<const NDArray> in, double t)
{
	auto out = in->copy();
	binarizeIP(out, t);
	return out;
}

/**
 * @brief Thresholds the image, changing everything below t to 0
 *
 * @param in Input image.
 * @param t Threshold to apply to the image.
 *
 * @return Threshold image (INT16 type)
 */
ptr<NDArray> threshold(ptr<const NDArray> in, double t)
{
	auto out = in->copyCast(INT16);
	thresholdIP(out, t);
	return out;
}

/**
 * @brief Thresholds the image, changing everything below t to 0
 *
 * @param in Input image.
 * @param t Threshold to apply to the image.
 */
void thresholdIP(ptr<NDArray> in, double t)
{
	for(FlatIter<double> it(in); !it.eof(); ++it) {
		if(*it < t)
			it.set(0);
	}
}

/**
 * @brief Computes a threshold based on OTSU.
 *
 * @param in Input image.
 *
 * @return Threshold
 */
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

/**
 * @brief Computes the mean-squared-error between two images. They should
 * be identically gridded.
 *
 * @param a Input 1
 * @param b Input 2
 * @param mask If not null then only compare areas within the masked region
 *
 * @return Error between images
 */
double mse(ptr<const NDArray> a, ptr<const NDArray> b, ptr<const NDArray> mask)
{
	assert(a->ndim() == b->ndim());
	for(size_t ii=0; ii<a->ndim(); ii++)
		assert(a->dim(ii) == b->dim(ii));
	if(mask) {
		assert(mask->ndim() == b->ndim());
		for(size_t ii=0; ii<b->ndim(); ii++)
			assert(mask->dim(ii) == b->dim(ii));
	}

	FlatConstIter<double> it1(a);
	FlatConstIter<double> it2(b);
	FlatConstIter<int64_t> itm;
	if(mask) {
		itm.setArray(mask);
		itm.goBegin();
	}

	double error = 0;
	size_t count = 0;
	for(it1.goBegin(), it2.goBegin(); !it1.eof() && !it2.eof(); ++it1, ++it2) {
		if(!mask || *itm > 0) {
			error += ((*it1)-(*it2))*((*it1)-(*it2));
			count++;
		}
		if(mask)
			++itm;
	}
	error /= count;

	return error;
}

/**
 * @brief Computes the correlation between two images. They should
 * be identically gridded.
 *
 * @param a Input 1
 * @param b Input 2
 * @param mask If not null then only compare areas within the masked region
 *
 * @return Correlation of images
 */
double corr(ptr<const NDArray> a, ptr<const NDArray> b, ptr<const NDArray> mask)
{
	assert(a->ndim() == b->ndim());
	for(size_t ii=0; ii<a->ndim(); ii++)
		assert(a->dim(ii) == b->dim(ii));
	if(mask) {
		assert(mask->ndim() == b->ndim());
		for(size_t ii=0; ii<b->ndim(); ii++)
			assert(mask->dim(ii) == b->dim(ii));
	}

	FlatConstIter<double> it1(a);
	FlatConstIter<double> it2(b);
	FlatConstIter<int64_t> itm;
	if(mask) {
		itm.setArray(mask);
		itm.goBegin();
	}

	double m1 = 0;
	double m2 = 0;
	int count = 0;
	double s1 = 0;
	double s2 = 0;
	double cor = 0;
	for(it1.goBegin(), it2.goBegin(); !it1.eof() && !it2.eof(); ++it1, ++it2) {
		if(!mask || *itm > 0) {
			cor += it1.get()*it2.get();
			s1 += it1.get()*it1.get();
			s2 += it2.get()*it2.get();
			m1 += it1.get();
			m2 += it2.get();
			count++;
		}
		if(mask)
			++itm;
	}
	s1 = sqrt(sample_var(count, m1, s1));
	s2 = sqrt(sample_var(count, m2, s2));
	m1 /= count;
	m2 /= count;
	cerr << "Count: " << count << endl;
	cerr << "Mean 1: " << m1 << endl;
	cerr << "Mean 2: " << m2 << endl;
	cerr << "Std 1: " << s1 << endl;
	cerr << "Std 2: " << s2 << endl;

	cor = (cor-count*m1*m2)/((count-1)*s1*s2);
	return cor;
}

/**
 * @brief Computes the dice coefficent between two labelmap images. They should
 * be identically gridded.
 *
 * @param a Input 1
 * @param b Input 2
 * @param mask If not null then only compare areas within the masked region
 *
 * @return Similarity of labels
 */
double dice(ptr<const NDArray> a, ptr<const NDArray> b, ptr<const NDArray> mask)
{
	assert(a->ndim() == b->ndim());
	for(size_t ii=0; ii<a->ndim(); ii++)
		assert(a->dim(ii) == b->dim(ii));
	if(mask) {
		assert(mask->ndim() == b->ndim());
		for(size_t ii=0; ii<b->ndim(); ii++)
			assert(mask->dim(ii) == b->dim(ii));
	}

	FlatConstIter<int64_t> it1(a);
	FlatConstIter<int64_t> it2(b);
	FlatConstIter<int64_t> itm;
	if(mask) {
		itm.setArray(mask);
		itm.goBegin();
	}

	int same = 0;
	int count = 0;
	for(it1.goBegin(), it2.goBegin(); !it1.eof() && !it2.eof(); ++it1, ++it2) {
		if(!mask || *itm > 0) {
			if(*it1 == *it2)
				same++;
			count++;
		}
		if(mask) ++itm;
	}
	return (same)/(double)count;
}

/**
 * @brief Computes the information based difference/similarity between two
 * images. They should be identically gridded.
 *
 * @param a Input 1
 * @param b Input 2
 * @param bins Number of bins in histogram PDF estimation (marginal, joint is
 * squared of this)
 * @param krad the radius of the parzen window
 * @param m metric to use (ie MI or NMI)
 * @param mask If not null then only compare areas within the masked region
 *
 * @return Information-based metric of two images
 */
double information(ptr<const NDArray> a, ptr<const NDArray> b,
		int m_bins, int m_krad, Metric m, ptr<const NDArray> mask)
{
	assert(a->ndim() == b->ndim());
	for(size_t ii=0; ii<a->ndim(); ii++)
		assert(a->dim(ii) == b->dim(ii));
	if(mask) {
		assert(mask->ndim() == b->ndim());
		for(size_t ii=0; ii<b->ndim(); ii++)
			assert(mask->dim(ii) == b->dim(ii));
	}

	FlatConstIter<double> it1(a);
	FlatConstIter<double> it2(b);
	FlatConstIter<int64_t> itm;
	if(mask) {
		itm.setArray(mask);
		itm.goBegin();
	}

	double range1[2] = {INFINITY, -INFINITY};
	double range2[2] = {INFINITY, -INFINITY};
	for(it1.goBegin(), it2.goBegin(); !it1.eof() && !it2.eof(); ++it1, ++it2) {
		if(!mask || *itm > 0) {
			range1[0] = std::min(range1[0], *it1);
			range2[0] = std::min(range2[0], *it2);
			range1[1] = std::max(range1[1], *it1);
			range2[1] = std::max(range2[1], *it2);
		}
	}

	//continuous bin locations
	double cbin1, cbin2;

	//set bin widths
	double wid1 = (range1[1]-range1[0])/(m_bins-2*m_krad-1);
	double wid2 = (range2[1]-range2[0])/(m_bins-2*m_krad-1);

	std::vector<double> joint(m_bins*m_bins, 0);
	if(mask) itm.goBegin();
	for(it1.goBegin(), it2.goBegin(); !it1.eof() && !it2.eof(); ++it1, ++it2) {
		if(mask && itm.get() > 0) {
			++itm;
			continue;
		}

		//bin Fa, Fc
		double f1 = *it1;
		double f2 = *it2;

		cbin1 = (f1-range1[0])/wid1 + m_krad;
		cbin2 = (f2-range2[0])/wid2 + m_krad;
		int bin1 = round(cbin1);
		int bin2 = round(cbin2);

		//sum up kernel bins
		for(int ii = bin1-m_krad; ii <= bin1+m_krad; ii++) {
			for(int jj = bin2-m_krad; jj <= bin2+m_krad; jj++)
				joint[ii*m_bins+jj] += B3kern(ii-cbin1, m_krad)*
					B3kern(jj-cbin2, m_krad);
		}
	}

	//normalize
	double total = 0;
	for(int ii = 0 ; ii < (int)joint.size(); ii++)
		total += joint[ii];

	for(int ii = 0 ; ii < (int)joint.size(); ii++)
		joint[ii] /= total;

	std::vector<double> marg1(m_bins, 0);
	std::vector<double> marg2(m_bins, 0);

	//marginalize
	for(int ii = 0 ; ii < (int)marg1.size(); ii++) {
		for(int jj = 0 ; jj < (int)marg2.size(); jj++) {
			marg1[ii] += joint[ii*m_bins+jj];
			marg2[jj] += joint[ii*m_bins+jj];
		}
	}

	double entj = 0;
	for(int ii = 0 ; ii < m_bins*m_bins; ii++) {
		if(joint[ii] > 0)
			entj -= joint[ii]*log(joint[ii]);
	}

	double ent1 = 0;
	for(int ii = 0 ; ii < m_bins; ii++) {
		if(marg1[ii] > 0)
			ent1 -= marg1[ii]*log(marg1[ii]);
	}

	double ent2 = 0;
	for(int ii = 0 ; ii < m_bins; ii++) {
		if(marg2[ii] > 0)
			ent2 -= marg2[ii]*log(marg2[ii]);
	}

	double mi = 0;
	for(int ii = 0 ; ii < m_bins; ii++) {
		for(int jj = 0 ; jj < m_bins; jj++) {
			if(joint[ii*m_bins+jj] > 0)
				mi += joint[ii*m_bins+jj]*log(joint[ii*m_bins+jj]/
						(marg1[ii]*marg2[jj]));
		}
	}

	double maxent = std::max(ent1, ent2);

	switch(m) {
		case METRIC_NMI:
			if(ent1+ent2 == entj || entj == 0)
				return 1;
			else
				return (ent1+ent2)/entj;
			break;

		case METRIC_REDUNDANCY:
			return mi/(ent1+ent2);
			break;

		case METRIC_NMIMETRIC:
			return 1-mi/maxent;
			break;
		case METRIC_DUALTC:
			return mi/entj;
			break;
		case METRIC_VI:
			return entj-mi;
			break;
		case METRIC_MI:
		default:
			return mi;
	}
	return 0;
}

} // npl


