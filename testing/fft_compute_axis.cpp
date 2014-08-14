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
#include "mrimage_utils.h"
#include "ndarray_utils.h"
#include "iterators.h"
#include "accessors.h"
#include "utility.h"
#include "basic_functions.h"
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
			
			double normf = 1./osize[dd];
			for(size_t tt=0; !it.isChunkEnd(); ++it, tt++) {
				cdouble_t tmp(buffer[tt][0]*normf, buffer[tt][1]*normf);
				it.set(tmp);
			}
		}

		fftw_free(buffer);
		fftw_destroy_plan(fwd);
	}
	oimg->write("padded_fftd.nii.gz");
	
	return oimg;
}

//
void pseudoPolar(shared_ptr<MRImage> in, size_t praddim, size_t origsz)
{
	std::vector<int64_t> index(in->ndim());
	
	assert(in->dim(praddim) >= in->dim(0));
	assert(in->dim(praddim) >= in->dim(1));
	assert(in->dim(praddim) >= in->dim(2));
	size_t buffsize = 0;

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
	
	// take the longest line for the buffer
	auto linebuf = fftw_alloc_complex((int)buffsize);
	auto fullbuf = fftw_alloc_complex((int)buffsize*16);

	// process the non-praddim limnes
	for(size_t ii=0; ii<2; ii++) {
		size_t linelen = in->dim(line[ii]);
		ChunkIter<cdouble_t> it(in);
		it.setLineChunk(line[ii]);
		for(it.goBegin(); !it.isEnd() ; it.nextChunk()) {
			it.index(index.size(), index.data());

			// fill from line
			for(size_t tt=0; !it.isChunkEnd(); ++it, tt++) {
				linebuf[tt][0] = (*it).real();
				linebuf[tt][1] = (*it).imag();
			}

			// perform fractional fourier transform
			// we start at alpha = +3, since we did NOT perform the inverse FFT
			int64_t k = index[praddim]-((int64_t)in->dim(praddim))/2;
			double n = origsz;
			double a = 2*k/n; // -3n/2 <= k <= 3n/2, -3 <= alpha <= 3
			cerr << "k: " << k << ", a: " << a << endl;
			a -= 3;
			fractional_ft(linelen, linebuf, linebuf, a, buffsize*16, 
					fullbuf, false);
		}
	}
	
	fftw_free(linebuf);
	fftw_free(fullbuf);
}

Vector3d getAxis(shared_ptr<const MRImage> img1, shared_ptr<const MRImage> img2)
{
	Vector3d axis;
	
	std::vector<size_t> index(3);
	size_t pseudo_radius = 0;
	size_t pseudo_slope[2];
	
	shared_ptr<MRImage> pp1;
	shared_ptr<MRImage> pp2;
	for(size_t ii=0; ii<3; ii++) {
		size_t tmp = 0;
		for(size_t jj=0; jj<3; jj++) {
			if(jj != ii) 
				pseudo_slope[tmp++] = jj;
		}
		pseudo_radius = ii;

		cerr << "ii: " << ii << " pseudo radius: " << pseudo_radius << 
			", pseudo slope 1: " << pseudo_slope[0] << ", pseudo slope 2: " <<
			pseudo_slope[1] << endl;

		pp1 = padFFT(img1, pseudo_radius);
//		pseudoPolar(pp1, pseudo_radius, img1->dim(ii));
//		ChunkIter<cdouble_t> it1(pp1);
//		it1->setLineChunk(pseudo_radius);
//		
		pp2 = padFFT(img2, pseudo_radius);
		
		{
		std::string name = "pad1-0.nii.gz";
		name[5] = '0'+ii;
		dynamic_pointer_cast<MRImage>(pp1->copyCast(FLOAT64))->write(name);
		name = "pad2-0.nii.gz";
		name[5] = '0'+ii;
		dynamic_pointer_cast<MRImage>(pp2->copyCast(FLOAT64))->write(name);
		}
//		pseudoPolar(pp2, pseudo_radius, img2->dim(ii));
//		ChunkIter<cdouble_t> it2(pp2);
//		it2->setLineChunk(pseudo_radius);
//
//		double maxcor = 0;
//		int64_t bestang1 = -1;
//		int64_t bestang2 = -1;
//		for(; !it1.isEnd() && !it2.isEnd(); it1.nextChunk(), it2.nextChunk()) {
//			double corr = 0;
//			double mu1 = 0, mu2 = 0;
//			double sig1 = 0, sig2 = 0;
//			size_t count = 0;
//			for(; !it1.isChunkEnd() && !it2.isChunkEnd(); ++it1, ++it2) {
//				double m1 = (*it1).abs();
//				double m2 = (*it2).abs();
//				corr += m1*m2;
//				mu1 += m1;
//				mu2 += m2;
//				sig1 += m1*m1;
//				sig2 += m2*m2;
//				count++;
//			}
//			assert(it1.isChunkEnd());
//			assert(it2.isChunkEnd());
//		}
//
//		assert(it1.isEnd());
//		assert(it2.isEnd());
//
//		for(size_t ll = 0 ; ll < pp1->dim(pseudo_slope[0]); ll++) {
//			for(size_t mm = 0 ; mm < pp1->dim(pseudo_slope[1]); mm++) {
//				sig1 = 0; sig2 = 0;
//				mu1 = 0; mu2 = 0;
//				corr = 0;
//				for(size_t kk = 0 ; kk < pp1->dim(pseudo_radius); kk++) {
//					index[pseudo_slope[0]] = ll;
//					index[pseudo_slope[1]] = mm;
//					index[pseudo_radius] = kk;
//
//				}
//
//				if(corr > maxcor) {
//					bestang1 = ll;
//					bestang1 = mm;
//					maxcor = corr;
//					cerr << "New Best: " << ll << "," << mm << " -> " << corr << endl;
//				}
//			}
//		}
	}
	
	
	return axis;
}

int main()
{
	// create an image
	int64_t index[3];
	size_t sz[] = {128, 128, 128};
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

	in->write("original.nii.gz");
	cerr << "Rotating manually" << endl;
	Vector3d ax(.1, .2, .3);
	ax.normalize();
	auto out = bruteForceRotate(ax, .2, in);
	out->write("brute_rotated.nii.gz");
	cerr << "Done" << endl;

	Vector3d newax = getAxis(in, out);

	return 0;
}



