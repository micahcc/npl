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

//
void pseudoPolar(shared_ptr<MRImage> in, size_t praddim)
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

			// perform chirplet transform
			int64_t k = index[praddim]-((int64_t)in->dim(praddim)/2);
			double n = in->dim(praddim);
			double a = 2*k/n; // -3n/2 <= k <= 3n/2, -3 <= alpha <= 3
			cerr << "k: " << k << ", a: " << a << endl;
			chirplet((int64_t)linelen, linebuf, linebuf, a, buffsize*16, fullbuf);
		}

		fftw_destroy_plan(rev);
	}
	
	fftw_free(linebuf);
	fftw_free(fullbuf);
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

Vector3d getAxis(shared_ptr<const MRImage> img1, shared_ptr<const MRImage> img2)
{
	ostringstream oss;
	Vector3d axis;
	
	std::vector<int64_t> index(3);
	size_t pseudo_radius = 0;
	size_t pseudo_slope[2];
	double bestang1 = -1;
	double bestang2 = -1;
	double maxcor = 0;
	
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
		pseudoPolar(pp1, pseudo_radius);
		ChunkIter<cdouble_t> it1(pp1);
		it1.setLineChunk(pseudo_radius);
		
		pp2 = padFFT(img2, pseudo_radius);
		pseudoPolar(pp2, pseudo_radius);
		ChunkIter<cdouble_t> it2(pp2);
		it2.setLineChunk(pseudo_radius);

		oss.str("pp1-");
		oss << ii << ".nii.gz";
		writeAbs(oss.str(), pp1);
		
		oss.str("pp2-");
		oss << ii << ".nii.gz";
		writeAbs(oss.str(), pp2);

		maxcor = 0;
		for(; !it1.eof() && !it2.eof(); it1.nextChunk(), it2.nextChunk()) {
			it1.index(index);
//			cerr << "Index: " << index[pseudo_slope[0]] << "," <<
//						index[pseudo_slope[1]] << endl;
			double corr = 0;
			double sum1 = 0, sum2 = 0;
			double ssq1 = 0, ssq2= 0;
			size_t count = 0;
			for(; !it1.eoc() && !it2.eoc(); ++it1, ++it2) {
				double m1 = abs(*it1);
				double m2 = abs(*it2);
				corr += m1*m2;
				sum1 += m1;
				sum1 += m2;
				ssq1 += m1*m1;
				ssq2 += m2*m2;
				count++;
			}
			assert(it1.isChunkEnd());
			assert(it2.isChunkEnd());

			corr = (count*corr - sum1*sum2)/
				((count*ssq1-sum1*sum1)*(count*ssq2-sum2*sum2));
			if(fabs(corr) > maxcor) {
				bestang1 = (index[pseudo_slope[0]]-pp1->dim(pseudo_slope[0])/2.)/
					pp1->dim(pseudo_slope[0]);
				bestang2 = (index[pseudo_slope[1]]-pp2->dim(pseudo_slope[1])/2.)/
					pp2->dim(pseudo_slope[1]);
				maxcor = corr;
				cerr << "New Max Cor: " << corr << ", " << bestang1 << ","
					<< bestang2 << endl;
			}
			
		}

		cerr << "ii: " << ii << " pseudo radius: " << pseudo_radius << 
			", pseudo slope 1: " << pseudo_slope[0] << ", pseudo slope 2: " 
			<< pseudo_slope[1] << " best cor: " << maxcor << " at " << bestang1 
			<< ", " << bestang2 << endl;
		assert(it1.isEnd());
		assert(it2.isEnd());
	}
	
	
	return axis;
}

int main()
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

	in->write("original.nii.gz");
	cerr << "Rotating manually" << endl;
	Vector3d ax(.0, .0, .1);
	ax.normalize();
	auto out = bruteForceRotate(ax, .2, in);
	out->write("brute_rotated.nii.gz");
	cerr << "Done" << endl;

	Vector3d newax = getAxis(in, out);

	return 0;
}



