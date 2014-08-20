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
 * @file fft_rotate_test.cpp A test of automatica computation of the rotation 
 * axis, based on the pseudo-polar fourier transform
 *
 *****************************************************************************/

#include <version.h>
#include <string>
#include <stdexcept>

#include <Eigen/Geometry> 

#define DEBUG 1

#include "mrimage.h"
#include "ndarray_utils.h"
#include "iterators.h"
#include "accessors.h"
#include "basic_functions.h"
#include "basic_plot.h"
#include "fract_fft.h"

#include "fftw3.h"


using namespace npl;
using namespace std;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::AngleAxisd;

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
shared_ptr<MRImage> bruteForceRotate(Vector3d axis, double theta,
		shared_ptr<const MRImage> in)
{
	Matrix3d m;
	// negate because we are starting from the destination and mapping from
	// the source
	m = AngleAxisd(-theta, axis);
	LinInterp3DView<double> lin(in);
	auto out = dynamic_pointer_cast<MRImage>(in->copy());
	Vector3d ind;
	Vector3d cind;
	Vector3d center;
	for(size_t ii=0; ii<3 && ii<in->ndim(); ii++) {
		center[ii] = (in->dim(ii)-1)/2.;
	}

	for(Vector3DIter<double> it(out); !it.isEnd(); ++it) {
		it.index(3, ind.array().data());
		cind = m*(ind-center)+center;

		// set for each t
		for(size_t tt = 0; tt<in->tlen(); tt++) 
			it.set(tt, lin(cind[0], cind[1], cind[2], tt));
	}

	return out;
}

int closeCompare(shared_ptr<const MRImage> a, shared_ptr<const MRImage> b)
{
	if(a->ndim() != b->ndim()) {
		cerr << "Error image dimensionality differs" << endl;
		return -1;
	}
	
	for(size_t dd=0; dd<a->ndim(); dd++) {
		if(a->dim(dd) != b->dim(dd)) {
			cerr << "Image size in the " << dd << " direction differs" << endl;
			return -1;
		}
	}

	OrderConstIter<double> ita(a);
	OrderConstIter<double> itb(b);
	itb.setOrder(ita.getOrder());
	for(ita.goBegin(), itb.goBegin(); !ita.eof() && !itb.eof(); ++ita, ++itb) {
		double diff = fabs(*ita - *itb);
		if(diff > 1E-10) {
			cerr << "Images differ!" << endl;
			return -1;
		}
	}

	return 0;
}

shared_ptr<MRImage> padFFT(shared_ptr<const MRImage> in, size_t ldim)
{
	if(in->ndim() != 3) {
		throw std::invalid_argument("Error, input image should be 3D!");
	}

	std::vector<size_t> osize(3, 0);
	for(size_t ii=0; ii<3; ii++) {
		osize[ii] = round2(in->dim(ii));
		if(ldim != ii)
			osize[ldim] += in->dim(ii);
	}
	osize[ldim] = round2(osize[ldim]+1);

	auto oimg = createMRImage(3, osize.data(), COMPLEX128);
	cerr << "Input:\n" << in << "\nPadded:\n" << oimg << endl;
	
	std::vector<int64_t> shift(3,0);
	for(size_t dd=0; dd<3; dd++) 
		shift[dd] = (osize[dd]-in->dim(dd))/2;
	
	// copy data
	NDConstAccess<cdouble_t> inview(in);
	std::vector<int64_t> index(in->ndim());
	for(OrderIter<cdouble_t> it(oimg); !it.isEnd(); ++it) {
		it.index(index.size(), index.data());
		
		bool outside = false;
		for(size_t dd=0; dd<3; dd++) {
			index[dd] -= shift[dd];
			if(index[dd] < 0 || index[dd] >= in->dim(dd))
				outside = true;
		}

		if(outside) 
			it.set(0);
		else
			it.set(inview[index]);
	}

	in->write("prepadded.nii.gz");
	oimg->write("padded.nii.gz");

	// fourier transform
	for(size_t dd = 0; dd < 3; dd++) {
		auto buffer = fftw_alloc_complex((int)osize[dd]);
		fftw_plan fwd = fftw_plan_dft_1d((int)osize[dd], buffer, buffer, 
				FFTW_FORWARD, FFTW_MEASURE);
		
		ChunkIter<cdouble_t> it(oimg);
		it.setLineChunk(dd);
		for(it.goBegin(); !it.isEnd() ; it.nextChunk()) {
			it.index(index.size(), index.data());

			// fill from line
			for(size_t tt=0; !it.isChunkEnd(); ++it, tt++) {
				buffer[tt][0] = (*it).real();
				buffer[tt][1] = (*it).imag();
			}

			// fourier transform
			fftw_execute(fwd);
			
			// copy/shift
			double normf = 1./osize[dd];
			for(size_t tt=osize[dd]/2; !it.isChunkEnd(); ++it) {
				cdouble_t tmp(buffer[tt][0]*normf, buffer[tt][1]*normf);
				it.set(tmp);
				tt=(tt+1)%osize[dd];
			}
		}

		fftw_free(buffer);
		fftw_destroy_plan(fwd);
	}
	oimg->write("padded_fftd.nii.gz");
	
	return oimg;
}


void writeAngle(string filename, shared_ptr<const MRImage> in)
{
	OrderConstIter<cdouble_t> iit(in);
	auto out = dynamic_pointer_cast<MRImage>(in->copyCast(FLOAT64));
	OrderIter<double> oit(out);
	for(iit.goBegin(), oit.goBegin(); !iit.eof() && !oit.eof(); ++iit, ++oit) {
		oit.set(std::arg(*iit));
	}

	out->write(filename);
}

void writeAbs(string filename, shared_ptr<const MRImage> in)
{
	OrderConstIter<cdouble_t> iit(in);
	auto out = dynamic_pointer_cast<MRImage>(in->copyCast(FLOAT64));
	OrderIter<double> oit(out);
	for(iit.goBegin(), oit.goBegin(); !iit.eof() && !oit.eof(); ++iit, ++oit) {
		oit.set(std::abs(*iit));
	}

	out->write(filename);
}

/**
 * @brief Fills the input array (chirp) with a chirp of the specified type
 *
 * @param sz 		Size of output array
 * @param chirp 	Output array
 * @param origsz 	Original size, decides maximum frequency reached
 * @param upratio 	Ratio of upsampling performed. This may be different than 
 * 					sz/origsz
 * @param alpha 	Positive term in exp
 * @param beta 		Negative term in exp
 * @param fft 		Whether to fft the output (put it in frequency domain)
 */
void createChirp(int64_t sz, fftw_complex* chirp, int64_t origsz,
		double upratio, double alpha, bool fft)
{
//	assert(sz%2==1);
	const double PI = acos(-1);
	const complex<double> I(0,1);
	
	auto fwd_plan = fftw_plan_dft_1d((int)sz, chirp, chirp, FFTW_FORWARD,
				FFTW_MEASURE | FFTW_PRESERVE_INPUT);

	cerr << "Upsample: " << upratio << endl;
	for(int64_t ii=-sz/2; ii<=sz/2; ii++) {
		double xx = ((double)ii)/upratio;
		auto tmp = std::exp(I*PI*alpha*xx*xx/(double)origsz);
		chirp[ii+sz/2][0] = tmp.real();
		chirp[ii+sz/2][1] = tmp.imag();
	}
	
	if(fft) {
		fftw_execute(fwd_plan);
		double norm = sqrt(1./sz);
		for(size_t ii=0; ii<sz; ii++) {
			chirp[ii][0] *= norm;
			chirp[ii][1] *= norm;
		}
	}

	fftw_destroy_plan(fwd_plan);
}

void powerFFT_help(int64_t isize, int64_t usize, int64_t uppadsize,
		fftw_complex* inout, fftw_complex* buffer, double alpha)
{
//	assert(usize%2 == 1);
//	assert(uppadsize%2 == 1);

	const complex<double> I(0,1);

	// zero
	for(size_t ii=0; ii<uppadsize*3; ii++) {
		buffer[ii][0] = 0;
		buffer[ii][1] = 0;
	}

	fftw_complex* sigbuff = &buffer[0]; // note the overlap with upsampled
	fftw_complex* upsampled = &buffer[uppadsize/2-usize/2];
	fftw_complex* posa_chirp = &buffer[uppadsize];
	fftw_complex* nega_chirp = &buffer[uppadsize*2];

	// create buffers and plans
	createChirp(uppadsize, nega_chirp, isize, (double)usize/(double)isize, -alpha, false);
	createChirp(uppadsize, posa_chirp, isize, (double)usize/(double)isize, alpha, true);

#ifdef DEBUG
	writePlotReIm("fft_negachirp.svg", uppadsize, nega_chirp);
#endif //DEBUG

#ifdef DEBUG
	writePlotReIm("fft_posachirp.svg", uppadsize, posa_chirp);
#endif //DEBUG

	fftw_plan sigbuff_plan_fwd = fftw_plan_dft_1d(uppadsize, sigbuff, sigbuff,
			FFTW_FORWARD, FFTW_MEASURE);
	fftw_plan sigbuff_plan_rev = fftw_plan_dft_1d(uppadsize, sigbuff, sigbuff,
			FFTW_BACKWARD, FFTW_MEASURE);

#ifdef DEBUG
	writePlotReIm("fft_in.svg", isize, inout);
#endif //DEBUG
	// upsample input
	interp(isize, inout, usize, upsampled);
#ifdef DEBUG
	writePlotReIm("fft_upin.svg", usize, upsampled);
#endif //DEBUG
	
	// pre-multiply
	for(int64_t nn = -usize/2; nn<=usize/2; nn++) {
		complex<double> tmp1(nega_chirp[nn+uppadsize/2][0],
				nega_chirp[nn+uppadsize/2][1]);
		complex<double> tmp2(upsampled[nn+usize/2][0],
				upsampled[nn+usize/2][1]);
		tmp1 *= tmp2;
		upsampled[nn+usize/2][0] = tmp1.real();
		upsampled[nn+usize/2][1] = tmp1.imag();
	}
#ifdef DEBUG
	writePlotReIm("fft_premult.svg", usize, upsampled);
#endif //DEBUG

	/*
	 * convolve
	 */
	fftw_execute(sigbuff_plan_fwd);
	double normfactor = sqrt(1./uppadsize);
	for(size_t ii=0; ii<uppadsize; ii++) {
		sigbuff[ii][0] *= normfactor;
		sigbuff[ii][1] *= normfactor;
	}

	for(size_t ii=0; ii<uppadsize; ii++) {
		complex<double> tmp1(sigbuff[ii][0], sigbuff[ii][1]);
		complex<double> tmp2(posa_chirp[ii][0], posa_chirp[ii][1]);
		tmp1 *= tmp2;
		sigbuff[ii][0] = tmp1.real();
		sigbuff[ii][1] = tmp1.imag();
	}
	fftw_execute(sigbuff_plan_rev);
	
	for(size_t ii=0; ii<uppadsize; ii++) {
		sigbuff[ii][0] *= normfactor;
		sigbuff[ii][1] *= normfactor;
	}

#ifdef DEBUG
	writePlotReIm("fft_convolve.svg", uppadsize, sigbuff);
#endif //DEBUG

	// circular shift
	std::rotate(&sigbuff[0][0], &sigbuff[(uppadsize-1)/2][0],
			&sigbuff[uppadsize][0]);
#ifdef DEBUG
	writePlotReIm("fft_rotated.svg", uppadsize, sigbuff);
#endif //DEBUG
	
	// post-multiply
	for(int64_t ii=-usize/2; ii<=usize/2; ii++) {
		complex<double> tmp1(nega_chirp[ii+uppadsize/2][0],
				nega_chirp[ii+uppadsize/2][1]);
		complex<double> tmp2(upsampled[ii+usize/2][0],
				upsampled[ii+usize/2][1]);
		tmp1 = tmp1*tmp2;
		upsampled[ii+usize/2][0] = tmp1.real();
		upsampled[ii+usize/2][1] = tmp1.imag();
	}

#ifdef DEBUG
	writePlotReIm("fft_postmult.svg", uppadsize, sigbuff);
#endif //DEBUG
	
	interp(usize, upsampled, isize, inout);
	
#ifdef DEBUG
	writePlotReIm("fft_out.svg", isize, inout);
#endif //DEBUG

	fftw_destroy_plan(sigbuff_plan_rev);
	fftw_destroy_plan(sigbuff_plan_fwd);
}

void powerFT_brute2_help(int64_t isize, int64_t usize, int64_t uppadsize,
		fftw_complex* inout, fftw_complex* buffer, double alpha)
{
//	assert(usize%2 == 1);
//	assert(uppadsize%2 == 1);

	const complex<double> I(0,1);
	const double PI = acos(-1);

	// zero
	for(size_t ii=0; ii<uppadsize*3; ii++) {
		buffer[ii][0] = 0;
		buffer[ii][1] = 0;
	}

	fftw_complex* sigbuff = &buffer[0]; // note the overlap with upsampled
	fftw_complex* upsampled = &buffer[usize];

#ifdef DEBUG
	writePlotReIm("brute2_in.svg", isize, inout);
#endif //DEBUG
	// upsample input
	interp(isize, inout, usize, upsampled);
#ifdef DEBUG
	writePlotReIm("brute2_upin.svg", usize, upsampled);
#endif //DEBUG
	
	// pre-multiply
	for(int64_t nn = -usize/2; nn<=usize/2; nn++) {
		double ff = nn*isize/(double)usize;
		complex<double> tmp1 = std::exp(-PI*I*alpha*ff*ff/(double)isize);
		complex<double> tmp2(upsampled[nn+usize/2][0], upsampled[nn+usize/2][1]);
		tmp1 *= tmp2;
		upsampled[nn+usize/2][0] = tmp1.real();
		upsampled[nn+usize/2][1] = tmp1.imag();
	}
#ifdef DEBUG
	writePlotReIm("brute2_premult.svg", usize, upsampled);
#endif //DEBUG

	/*
	 * convolve
	 */

	for(int64_t ii=-usize/2; ii<=usize/2; ii++) {
		double xx = ii*isize/(double)usize;
		sigbuff[ii+usize/2][0] = 0;
		sigbuff[ii+usize/2][1] = 0;
		for(int64_t jj=-usize/2; jj<=usize/2; jj++) {
			double ww = jj*isize/(double)usize;
			complex<double> tmp1 = std::exp(PI*I*alpha*(ww-xx)*(ww-xx)/(double)isize);
			complex<double> tmp2(upsampled[jj+usize/2][0], upsampled[jj+usize/2][1]);
			tmp1 *= tmp2;
			sigbuff[ii+usize/2][0] += tmp1.real();
			sigbuff[ii+usize/2][1] += tmp1.imag();
		}
		complex<double> tmp3(sigbuff[ii+usize/2][0], sigbuff[ii+usize/2][1]);
	}
		
#ifdef DEBUG
	writePlotReIm("brute2_convolve.svg", usize, sigbuff);
#endif //DEBUG
	
	// post-multiply
	for(int64_t nn = -usize/2; nn<=usize/2; nn++) {
		double ff = nn*isize/(double)usize;
		complex<double> tmp1 = std::exp(-PI*I*alpha*ff*ff/(double)isize);
		complex<double> tmp2(sigbuff[nn+usize/2][0], sigbuff[nn+usize/2][1]);
		tmp1 *= tmp2;
		upsampled[nn+usize/2][0] = tmp1.real();
		upsampled[nn+usize/2][1] = tmp1.imag();
	}

#ifdef DEBUG
	writePlotReIm("brute2_postmult.svg", usize, upsampled);
#endif //DEBUG
	
	interp(usize, upsampled, isize, inout);

#ifdef DEBUG
	writePlotReIm("brute2_out.svg", isize, inout);
#endif //DEBUG
}

/**
 * @brief Comptues the powerFFT transform using FFTW for n log n performance.
 *
 * @param isize Size of input/output
 * @param in Input array, may be the same as out, length sz
 * @param out Output array, may be the same as input, length sz
 * @param alpha Fraction of full space to compute
 * @param bsz Buffer size
 * @param buffer Buffer to do computations in, may be null, in which case new
 * memory will be allocated and deallocated during processing. Note that if
 * the provided buffer is not sufficient size a new buffer will be allocated
 * and deallocated, and a warning will be produced. 4x the padded value is
 * needed, which means this value should be around 16x sz
 * @param nonfft
 */
void powerFT_brute2(size_t isize, fftw_complex* in, fftw_complex* out, double a,
		size_t* inbsz, fftw_complex** inbuffer)
{
	// there are 3 sizes: isize: the original size of the input array, usize :
	// the size of the upsampled array, and uppadsize the padded+upsampled
	// size, we want both uppadsize and usize to be odd, and we want uppadsize
	// to be the product of small primes (3,5,7)
	double approxratio = 10;
	int64_t uppadsize = round357(isize*approxratio);;
	int64_t usize;
	while((usize = (uppadsize-1)/2)%2 == 0) {
		uppadsize = round357(uppadsize+2);
	}

	cerr << "input size: " << isize << endl;
	cerr << "upsample size: " << usize << endl;
	cerr << "upsample+pad size: " << uppadsize << endl;

	size_t bsz = 0;
	fftw_complex* buffer = NULL;
	if(inbsz) 
		bsz = *inbsz;

	if(inbuffer) 
		buffer = *inbuffer;

	// check/allocate buffer
	bool allocated = false;
	if(bsz < isize+3*uppadsize || !buffer) {
		std::cerr << "WARNING! Allocating vector in fractional_ft" << std::endl;
		bsz = isize+3*uppadsize;
		buffer = fftw_alloc_complex(bsz);
		allocated = true;
	}

	fftw_complex* current = &buffer[0];

	// copy input to buffer
	for(size_t ii=0; ii<isize; ii++) {
		current[ii][0] = in[ii][0];
		current[ii][1] = in[ii][1];
	}

	powerFT_brute2_help(isize, usize, uppadsize, current, &buffer[isize], a);

	// copy current to output
	for(size_t ii=0; ii<isize; ii++) {
		out[ii][0] = current[ii][0];
		out[ii][1] = current[ii][1];
	}

	if(allocated) {
		if(inbsz && inbuffer) {
			*inbsz = bsz;
			*inbuffer = buffer;
		} else {
			fftw_free(buffer);
		}
	} 
}

/**
 * @brief Comptues the powerFFT transform using FFTW for n log n performance.
 *
 * @param isize Size of input/output
 * @param in Input array, may be the same as out, length sz
 * @param out Output array, may be the same as input, length sz
 * @param alpha Fraction of full space to compute
 * @param bsz Buffer size
 * @param buffer Buffer to do computations in, may be null, in which case new
 * memory will be allocated and deallocated during processing. Note that if
 * the provided buffer is not sufficient size a new buffer will be allocated
 * and deallocated, and a warning will be produced. 4x the padded value is
 * needed, which means this value should be around 16x sz
 * @param nonfft
 */
void powerFFT(size_t isize, fftw_complex* in, fftw_complex* out, double a,
		size_t* inbsz, fftw_complex** inbuffer)
{
	// there are 3 sizes: isize: the original size of the input array, usize :
	// the size of the upsampled array, and uppadsize the padded+upsampled
	// size, we want both uppadsize and usize to be odd, and we want uppadsize
	// to be the product of small primes (3,5,7)
	double approxratio = 10;
	int64_t uppadsize = round357(isize*approxratio);;
	int64_t usize;
	while((usize = (uppadsize-1)/2)%2 == 0) {
		uppadsize = round357(uppadsize+2);
	}

	cerr << "input size: " << isize << endl;
	cerr << "upsample size: " << usize << endl;
	cerr << "upsample+pad size: " << uppadsize << endl;

	size_t bsz = 0;
	fftw_complex* buffer = NULL;
	if(inbsz) 
		bsz = *inbsz;

	if(inbuffer) 
		buffer = *inbuffer;

	// check/allocate buffer
	bool allocated = false;
	if(bsz < isize+3*uppadsize || !buffer) {
		std::cerr << "WARNING! Allocating vector in fractional_ft" << std::endl;
		bsz = isize+3*uppadsize;
		buffer = fftw_alloc_complex(bsz);
		allocated = true;
	}

	fftw_complex* current = &buffer[0];

	// copy input to buffer
	for(size_t ii=0; ii<isize; ii++) {
		current[ii][0] = in[ii][0];
		current[ii][1] = in[ii][1];
	}

	powerFFT_help(isize, usize, uppadsize, current, &buffer[isize], a);

	// copy current to output
	for(size_t ii=0; ii<isize; ii++) {
		out[ii][0] = current[ii][0];
		out[ii][1] = current[ii][1];
	}

	if(allocated) {
		if(inbsz && inbuffer) {
			*inbsz = bsz;
			*inbuffer = buffer;
		} else {
			fftw_free(buffer);
		}
	} 
}

void powerFT_brute(size_t len, fftw_complex* in, fftw_complex* out, double a)
{
	const complex<double> I(0,1);
	const double PI = acos(-1);
	int64_t ilen = len;

	for(int64_t ii=0; ii<ilen; ii++) {
		double ff=(ii-(ilen)/2.);
		out[ii][0]=0;
		out[ii][1]=0;

		for(int64_t jj=0; jj<ilen; jj++) {
			double xx=(jj-(ilen)/2.);
			complex<double> tmp1(in[jj][0], in[jj][1]);
			complex<double> tmp2 = tmp1*std::exp(-2.*PI*I*a*xx*ff/(double)ilen);
			
			out[ii][0] += tmp2.real();
			out[ii][1] += tmp2.imag();
		}
	}
}

void pseudoPolar(shared_ptr<MRImage> in, size_t praddim)
{
	std::vector<int64_t> index(in->ndim());
	
	assert(in->dim(praddim) >= in->dim(0));
	assert(in->dim(praddim) >= in->dim(1));
	assert(in->dim(praddim) >= in->dim(2));
	size_t buffsize = 0;
	size_t worksize = 0;

	// figure out which lines to break up (non praddim's)
	size_t line[2];
	{
		size_t ll = 0;
		for(size_t dd=0; dd<3; dd++) {
			if(dd != praddim) {
				line[ll++] = dd;
				if(in->dim(dd) > buffsize)
					buffsize = in->dim(dd);
			}
		}
	}
	worksize = buffsize*16;
	
	// take the longest line for the buffer
	auto linebuf = fftw_alloc_complex((int)buffsize);
	auto fullbuf = fftw_alloc_complex((int)worksize);
	
	// process the non-praddim limnes
	for(size_t ii=0; ii<2; ii++) {

		size_t linelen = in->dim(line[ii]);
		fftw_plan rev = fftw_plan_dft_1d((int)linelen, linebuf, linebuf, 
				FFTW_BACKWARD, FFTW_MEASURE);

		ChunkIter<cdouble_t> it(in);
		it.setLineChunk(line[ii]);
		cerr << "PP: " << praddim << ", " << line[ii] << " line" << endl;
		for(it.goBegin(); !it.isEnd() ; it.nextChunk()) {
			it.index(index.size(), index.data());

			// fill from line
			for(size_t tt=0; !it.isChunkEnd(); ++it, tt++) {
				linebuf[tt][0] = (*it).real();
				linebuf[tt][1] = (*it).imag();
			}

			// un-fourier transform
			fftw_execute(rev);

			// perform powerFFT transform
			int64_t k = index[praddim]-((int64_t)in->dim(praddim)/2);
			double n = in->dim(praddim);
			double a = 2*k/n; // -3n/2 <= k <= 3n/2, -3 <= alpha <= 3
			cerr << "k: " << k << ", a: " << a << endl;
			powerFFT((int64_t)linelen, linebuf, linebuf, a, &worksize, &fullbuf);
		}

		fftw_destroy_plan(rev);
	}
	
	fftw_free(linebuf);
	fftw_free(fullbuf);
}

int testPowerFFT(size_t length, double alpha)
{
	auto line = fftw_alloc_complex(length);
	auto line_brute = fftw_alloc_complex(length);
	auto line_brute2 = fftw_alloc_complex(length);
	auto line_fft = fftw_alloc_complex(length);
	size_t worklen = length*40;
	auto workbuff= fftw_alloc_complex(worklen);
	
	// fill with a noisy square
	double sum = 0;
	for(size_t ii=0; ii<length; ii++){ 
		if(ii > 2.*length/5 && ii < length*3./5) {
			line[ii][0] = 1;
			line[ii][1] = 0;
			sum += 1;
		} else {
			line[ii][0] = 0;
			line[ii][1] = 0;
		}
	}
	for(size_t ii=0; ii<length; ii++) 
		line[ii][0] /= sum;
	
	//writePlotReIm("input.svg", length, line);

	powerFT_brute(length, line, line_brute, alpha);
	powerFT_brute2(length, line, line_brute2, alpha, &worklen, &workbuff);

	writePlotReIm("input.svg", length, line);
	powerFFT(length, line, line_fft, alpha, &worklen, &workbuff);

	writePlotReIm("powerBruteFT.svg", length, line_brute);
	writePlotReIm("powerBruteFT2.svg", length, line_brute2);
	writePlotReIm("powerFFT.svg", length, line_fft);

	for(size_t ii=0; ii<length; ii++) {
		complex<double> a(line_brute[ii][0], line_brute[ii][1]);
		complex<double> b(line[ii][0], line[ii][1]);
		
		if(abs(abs(a) - abs(b)) > 0.001) {
			cerr << "Error, absolute difference in powerFFT" << endl;
			return -1;
		}
		if(abs(arg(a) - arg(b)) > 0.1) {
			cerr << "Error, angle difference in powerFFT" << endl;
			return -1;
		}
	}

	return 0;
}

int testPseudoPolar()
{
//	// create an image
//	int64_t index[3];
//	size_t sz[] = {64, 64, 64};
//	auto in = createMRImage(sizeof(sz)/sizeof(size_t), sz, FLOAT64);
//
//	// fill with square
//	OrderIter<double> sit(in);
//	while(!sit.eof()) {
//		sit.index(3, index);
//		if(index[0] > sz[0]/4 && index[0] < sz[0]/3 && 
//				index[1] > sz[1]/5 && index[1] < sz[1]/2 && 
//				index[2] > sz[2]/3 && index[2] < 2*sz[2]/3) {
//			sit.set(1);
//		} else {
//			sit.set(0);
//		}
//		++sit;
//	}
//	
//	// test the pseudopolar transforms
//	for(size_t dd=0; dd<3; dd++) {
//		auto pp1_fft = pseudoPolar(img1, pseudo_radius);
//		auto pp1_brute = pseudoPolarBrute(img1, pseudo_radius);
//		if(closeCompare(pp1_fft, pp1_brute) != 0) {
//			cerr << "FFT and BruteForce pseudopolar differ" << endl;
//			return -1;
//		}
//	}
	
	return 0;
}

int main()
{
	// test the 'Power' Fourier Transform
	if(testPowerFFT(128, 1) != 0)
		return -1;

//	if(testPseudoPolar() != 0) 
//		return -1;
	
	return 0;
}




