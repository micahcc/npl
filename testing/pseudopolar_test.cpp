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

/**
 * @brief 
 *
 * @param in		Input Image (in normal space)
 * @param praddim	Pseudo-Radius dimension
 *
 * @return 			Pseudo-polar space image
 */
shared_ptr<MRImage> pseudoPolarBrute(shared_ptr<MRImage> in, size_t praddim)
{
	if(in->ndim() != 3) 
		return NULL;

	shared_ptr<MRImage> out = padFFT(in, praddim);
	shared_ptr<MRImage> tmp = padFFT(in, praddim);

	// interpolate along lines
	std::vector<double> index(3);
	std::vector<double> index2(3);
	LinInterp3DView interp(out);
	for(OrderIter oit(out); !oit.eof(); ++oit) {
		oit.index(index.size(), index.data());
		
		// make index into slope, then back to a flat index
		for(size_t ii=0; ii<3; ii++) {
			if(ii != praddim)
				index[ii] = 2*(index[ii] - (in->dim(ii)-1)/2.)/(in->dim(ii)-1.);
		}

		// centered radius
		double crad = index[praddim]-(in->dim(praddim)-1.)/2.;
		for(size_t ii=0; ii<3; ii++) {
			if(ii != praddim) {
				index2[ii] = crad*index[ii];
			} else { 
				// pseudo radius
				index2[ii] = index[ii];
			}
		}

		out.set(interp(index2));
	}

	return out;
}

shared_ptr<MRImage> pseudoPolar(shared_ptr<MRImage> in, size_t praddim)
{
	assert(out->dim(praddim) >= out->dim(0));
	assert(out->dim(praddim) >= out->dim(1));
	assert(out->dim(praddim) >= out->dim(2));

	// create output
	shared_ptr<MRImage> out = padFFT(in, praddim);

	// declare variables
	std::vector<int64_t> index(out->ndim()); 
	const double approxratio = 2; // how much to upsample by
	size_t buffsize = 0; // minimum buffer size needed
	
	// figure out which lines to break up (non praddim's)
	size_t line[2];
	{
		size_t ll = 0;
		for(size_t dd=0; dd<3; dd++) {
			if(dd != praddim) {
				line[ll++] = dd;
				if(out->dim(dd) > buffsize)
					buffsize = out->dim(dd);
			}
		}
	}


	buffsize *= 16;
	fftw_complex* buffer = fftw_alloc_complex(bsz);

	for(size_t dd=0; dd<3; dd++) {
		if(dd == praddim)
			continue;

		size_t isize = out->dim(dd);
		int64_t usize = round2(isize*approxratio);
		int64_t uppadsize = usize*2;
		double upratio = (double)usize/(double)isize;
		fftw_complex* current = &buffer[0];
		fftw_complex* nchirp = &buffer[isize+uppadsize];
		fftw_complex* pchirp = &buffer[isize+2*uppadsize];

		assert(buffsize >= isize+3*uppadsize);
		
		ChunkIter it(out);
		it.setLineChunk(dd);
		it.setOrder({praddim}, true); // make pseudoradius slowest
		double alpha, prevAlpha;
		for(it.goBegin(); !it.eof(); it.nextChunk()) {
			it.index(index);
			
			alpha = (2*index[praddim]-out->dim(praddim)+1)/in->dim(praddim);

			// recompute chirps if alpha changed
			if(alhpa != prevAlpha) {
				cerr << "Recomputing chirps for alpha = " << alpha << endl;
				createChirp(uppadsize, nchirp, isize, upratio, -a, false);
				createChirp(uppadsize, pchirp, isize, upratio, a, true);
			}

			for(it.goChunkBegin(); !it.eoc(); ++it) {

			}

			prevAlpha = alpha;
		}


		// copy input to buffer
		for(size_t ii=0; ii<isize; ii++) {
			current[ii][0] = in[ii][0];
			current[ii][1] = in[ii][1];
		}

		chirpzFFT(isize, usize, current, uppadsize, &buffer[isize], nchirp, pchirp);

		// copy current to output
		for(size_t ii=0; ii<isize; ii++) {
			out[ii][0] = current[ii][0];
			out[ii][1] = current[ii][1];
		}
		}
	}
	fftw_free(buffer);

	worksize = buffsize*16;
	
	// take the longest line for the buffer
	auto linebuf = fftw_alloc_complex((int)buffsize);
	auto fullbuf = fftw_alloc_complex((int)worksize);
	
	// process the non-praddim limnes
//	for(size_t ii=0; ii<2; ii++) {
//
//		size_t linelen = in->dim(line[ii]);
//		fftw_plan rev = fftw_plan_dft_1d((int)linelen, linebuf, linebuf, 
//				FFTW_BACKWARD, FFTW_MEASURE);
//
//		ChunkIter<cdouble_t> it(in);
//		it.setLineChunk(line[ii]);
//		cerr << "PP: " << praddim << ", " << line[ii] << " line" << endl;
//		for(it.goBegin(); !it.isEnd() ; it.nextChunk()) {
//			it.index(index.size(), index.data());
//
//			// fill from line
//			for(size_t tt=0; !it.isChunkEnd(); ++it, tt++) {
//				linebuf[tt][0] = (*it).real();
//				linebuf[tt][1] = (*it).imag();
//			}
//
//			// un-fourier transform
//			fftw_execute(rev);
//
//			// perform powerFFT transform
//			int64_t k = index[praddim]-((int64_t)in->dim(praddim)/2);
//			double n = in->dim(praddim);
//			double a = 2*k/n; // -3n/2 <= k <= 3n/2, -3 <= alpha <= 3
//			cerr << "k: " << k << ", a: " << a << endl;
//			powerFFT((int64_t)linelen, linebuf, linebuf, a, &worksize, &fullbuf);
//		}
//
//		fftw_destroy_plan(rev);
//	}
//	
	fftw_free(linebuf);
	fftw_free(fullbuf);

	return out;
}

int testPseudoPolar(size_t dim, double alpha)
{
	// create an image
	int64_t index[3];
	size_t sz[] = {64, 64, 64};
	auto in = createMRImage(sizeof(sz)/sizeof(size_t), sz, FLOAT64);

	// fill with square
	OrderIter<double> sit(in);
	while(!sit.eof()) {
		sit.index(3, index);
		if(index[0] > sz[0]/4 && index[0] < sz[0]/3 && 
				index[1] > sz[1]/5 && index[1] < sz[1]/2 && 
				index[2] > sz[2]/3 && index[2] < 2*sz[2]/3) {
			sit.set(1);
		} else {
			sit.set(0);
		}
		++sit;
	}
	
	// test the pseudopolar transforms
	for(size_t dd=0; dd<3; dd++) {
		auto pp1_fft = pseudoPolar(in, dd);
		auto pp1_brute = pseudoPolarBrute(img1, dd);
		if(closeCompare(pp1_fft, pp1_brute) != 0) {
			cerr << "FFT and BruteForce pseudopolar differ" << endl;
			return -1;
		}
	}
	
	return 0;
}

int main()
{
	// test the 'Power' Fourier Transform
	if(testPseudoPolar(128, 1) != 0)
		return -1;

//	if(testPseudoPolar() != 0) 
//		return -1;
	
	return 0;
}



