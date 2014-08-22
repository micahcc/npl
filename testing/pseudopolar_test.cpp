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
#include "chirpz.h"

#include "fftw3.h"


using namespace npl;
using namespace std;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::AngleAxisd;

void writeComplex(string basename, shared_ptr<const MRImage> in)
{
	auto re = dynamic_pointer_cast<MRImage>(in->copyCast(FLOAT64));
	auto im = dynamic_pointer_cast<MRImage>(in->copyCast(FLOAT64));

	OrderIter<double> rit(re);
	OrderIter<double> iit(im);
	OrderConstIter<cdouble_t> init(in);
	while(!init.eof()) {
		iit.set((*init).imag());
		rit.set((*init).real());
		++init;
		++rit;
		++iit;
	}

	re->write(basename + "_re.nii.gz");
	im->write(basename + "_im.nii.gz");
}

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

shared_ptr<MRImage> padFFT(shared_ptr<const MRImage> in)
{
	if(in->ndim() != 3) {
		throw std::invalid_argument("Error, input image should be 3D!");
	}

	std::vector<size_t> osize(3, 0);
	for(size_t ii=0; ii<3; ii++) 
		osize[ii] = round2(in->dim(ii));

	auto oimg = dynamic_pointer_cast<MRImage>(in->copyCast(osize.size(),
			osize.data(), COMPLEX128));
	
	// copy data
	std::vector<int64_t> index(in->ndim());
//#ifdef DEBUG
	in->write("prepadded.nii.gz");
	writeComplex("padded", oimg);
//#endif //DEBUG

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
			for(size_t tt=0; !it.eoc(); ++it, tt++) {
				buffer[tt][0] = (*it).real();
				buffer[tt][1] = (*it).imag();
			}

			// fourier transform
			fftw_execute(fwd);
			
			// copy/shift
			it.goChunkBegin();
			for(size_t tt=osize[dd]/2; !it.isChunkEnd(); ++it) {
				cdouble_t tmp(buffer[tt][0], buffer[tt][1]);
				it.set(tmp);
				tt=(tt+1)%osize[dd];
			}
		}

		fftw_free(buffer);
		fftw_destroy_plan(fwd);
	}

	double normf = 1;
	for(size_t dd=0; dd<oimg->ndim(); dd++)
		normf *= oimg->dim(dd);
	normf = 1./normf;
	for(OrderIter<cdouble_t> it(oimg); !it.eof(); ++it) {
		it.set(normf*it.get());
	}

	writeComplex("padded_fft", oimg);
	
	return oimg;
}

/**
 * @brief 
 *
 * @param in		Input Image (in normal space)
 * @param prdim	Pseudo-Radius dimension
 *
 * @return 			Pseudo-polar space image
 */
shared_ptr<MRImage> pseudoPolarBrute(shared_ptr<MRImage> in, size_t prdim)
{
	if(in->ndim() != 3) 
		return NULL;

	shared_ptr<MRImage> out = padFFT(in);
	shared_ptr<MRImage> tmp = padFFT(in);

	// interpolate along lines
	std::vector<double> index(3);
	std::vector<double> index2(3);
	LinInterp3DView<cdouble_t> interp(out);
	for(OrderIter<cdouble_t> oit(out); !oit.eof(); ++oit) {
		oit.index(index.size(), index.data());
		
		// make index into slope, then back to a flat index
		for(size_t ii=0; ii<3; ii++) {
			if(ii != prdim)
				index[ii] = 2*(index[ii] - (in->dim(ii)-1)/2.)/(in->dim(ii)-1.);
		}

		// centered radius
		double crad = index[prdim]-(in->dim(prdim)-1.)/2.;
		for(size_t ii=0; ii<3; ii++) {
			if(ii != prdim) {
				index2[ii] = crad*index[ii];
			} else { 
				// pseudo radius
				index2[ii] = index[ii];
			}
		}

		oit.set(interp(index2[0], index2[1], index2[2]));
	}

	return out;
}

shared_ptr<MRImage> pseudoPolar(shared_ptr<MRImage> in, size_t prdim)
{
	// create output
	shared_ptr<MRImage> out = padFFT(in);

	// declare variables
	std::vector<int64_t> index(out->ndim()); 
	const double approxratio = 2; // how much to upsample by

	// compute/initialize buffer
	size_t buffsize = [&]
	{
		size_t m = 0;
		for(size_t dd=0; dd<3; dd++) {
			if(dd != prdim) {
				if(out->dim(dd) > m)
					m = out->dim(dd);
			}
		}
		return m*16;
	}();

	fftw_complex* buffer = fftw_alloc_complex(buffsize);

	for(size_t dd=0; dd<3; dd++) {
		if(dd == prdim)
			continue;

		size_t isize = out->dim(dd);
		int64_t usize = round2(isize*approxratio);
		int64_t uppadsize = usize*2;
		double upratio = (double)usize/(double)isize;
		fftw_complex* current = &buffer[0];
		fftw_complex* nchirp = &buffer[isize+uppadsize];
		fftw_complex* pchirp = &buffer[isize+2*uppadsize];

		assert(buffsize >= isize+3*uppadsize);
		
		ChunkIter<cdouble_t> it(out);
		it.setLineChunk(dd);
		it.setOrder({prdim}, true); // make pseudoradius slowest
		double alpha, prevAlpha = NAN;
		for(it.goBegin(); !it.eof(); it.nextChunk()) {
			it.index(index);
			
			// recompute chirps if alpha changed
			alpha = 2*(index[prdim]/(double)out->dim(prdim)) - 1;
			if(alpha != prevAlpha) {
				cerr << "Recomputing chirps for alpha = " << alpha << endl;
				createChirp(uppadsize, nchirp, isize, upratio, -alpha, false);
				createChirp(uppadsize, pchirp, isize, upratio, alpha, true);
			}

			// copy from input image
			it.goChunkBegin();
			for(size_t ii=0; !it.eoc(); ii++, ++it) {
				current[ii][0] = (*it).real();
				current[ii][1] = (*it).imag();
			}
		
			// compute chirpz transform
			// TODO buffer[isize] contains an upsampled version, use that 
			chirpzFFT(isize, usize, current, uppadsize, &buffer[isize], nchirp, pchirp);
			
			// copy from buffer back to output
			it.goChunkBegin();
			for(size_t ii=0; !it.eoc(); ii++, ++it) {
				cdouble_t tmp(current[ii][0], current[ii][1]);
				it.set(tmp);
			}
			prevAlpha = alpha;
		}
	}
	fftw_free(buffer);

	return out;
}

int testPseudoPolar()
{
	// create an image
	int64_t index[3];
	size_t sz[] = {64, 64, 64};
	auto in = createMRImage(sizeof(sz)/sizeof(size_t), sz, FLOAT64);

	// fill with square
	OrderIter<double> sit(in);
	double sum = 0;
	while(!sit.eof()) {
		sit.index(3, index);
		if(index[0] > 3*sz[0]/5 && index[0] < 4*sz[0]/5 && 
				index[1] > 1*sz[1]/5 && index[1] < 2*sz[1]/5 && 
				index[2] > 1*sz[2]/5 && index[2] < 3*sz[2]/5) {
			sit.set(1);
			sum += 1;
		} else {
			sit.set(0);
		}
		++sit;
	}
	
	for(sit.goBegin(); !sit.eof(); ++sit) 
		sit.set(sit.get()/sum);
	
	// test the pseudopolar transforms
	for(size_t dd=0; dd<3; dd++) {
		cerr << "Computing With PseudoRadius = " << dd << endl;
		auto pp1_fft = pseudoPolar(in, dd);
		auto pp1_brute = pseudoPolarBrute(in, dd);

		writeComplex("fft_pp"+to_string(dd), pp1_fft);
		writeComplex("brute_pp"+to_string(dd), pp1_brute);
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
	if(testPseudoPolar() != 0)
		return -1;

//	if(testPseudoPolar() != 0) 
//		return -1;
	
	return 0;
}




