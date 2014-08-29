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
 * @file pseudopolar_test.cpp Tests the ability of FFT and Zoom based pseudo
 * polar gridded fourier transform to match a brute-force linear interpolation
 * method, for highly variable (striped) fourier domain
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

clock_t brute_time = 0;
clock_t fft_time = 0;
clock_t zoom_time = 0;

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

void writeComplexAA(string basename, shared_ptr<const MRImage> in)
{
	auto absimg = dynamic_pointer_cast<MRImage>(in->copyCast(FLOAT64));
	auto angimg = dynamic_pointer_cast<MRImage>(in->copyCast(FLOAT64));

	OrderIter<double> rit(absimg);
	OrderIter<double> iit(angimg);
	OrderConstIter<cdouble_t> init(in);
	while(!init.eof()) {
		rit.set(abs(*init));
		iit.set(arg(*init));
		++init;
		++rit;
		++iit;
	}

	absimg->write(basename + "_abs.nii.gz");
	angimg->write(basename + "_ang.nii.gz");
}

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

shared_ptr<MRImage> createTestImage(size_t sz1)
{
	// create an image
	int64_t index[3];
	size_t sz[] = {sz1, sz1, sz1};
	auto in = createMRImage(sizeof(sz)/sizeof(size_t), sz, COMPLEX128);

	// fill with a shape
	OrderIter<double> sit(in);
	double sum = 0;
	while(!sit.eof()) {
		sit.index(3, index);
		double v = index[0]>(sz[0]/2. - 10) && index[0]<(sz[0]/2. + 2) 
					&& index[1]>(sz[1]/2. + 3) && index[1]<(sz[1]/2. + 10) 
					&& index[2]>(sz[2]/2. - 1) && index[2]<(sz[2]/2. + 1);
		sit.set(v);
		sum += v;
		++sit;
	}
	
	for(sit.goBegin(); !sit.eof(); ++sit) 
		sit.set(sit.get()/sum);

	return in;
}

Vector3d getAxis(shared_ptr<const MRImage> img1, shared_ptr<const MRImage> img2)
{
	ostringstream oss;
	Vector3d axis;
	
	std::vector<int64_t> index(3);
	size_t pslope[2];
	double bestang1 = -1;
	double bestang2 = -1;
	double mineang1 = -1;
	double mineang2 = -1;
	double maxcor = 0;
	double minerr = INFINITY;
	
	for(size_t ii=0; ii<3; ii++) {
		size_t tmp = 0;
		for(size_t jj=0; jj<3; jj++) {
			if(jj != ii) 
				pslope[tmp++] = jj;
		}

		cerr << "pseudo radius: " << ii << 
			", pseudo slope 1: " << pslope[0] << ", pseudo slope 2: " <<
			pslope[1] << endl;

		auto s1_pp = dynamic_pointer_cast<MRImage>(pseudoPolar(img1, ii));
		auto s2_pp = dynamic_pointer_cast<MRImage>(pseudoPolar(img2, ii));
		
		ChunkIter<cdouble_t> it1(s1_pp);
		it1.setLineChunk(ii);
		ChunkIter<cdouble_t> it2(s2_pp);
		it2.setLineChunk(ii);

		writeComplex("s1_pp"+to_string(ii)+".nii.gz", s1_pp);
		writeComplex("s2_pp"+to_string(ii)+".nii.gz", s2_pp);

		maxcor = 0;
		for(; !it1.eof() && !it2.eof(); it1.nextChunk(), it2.nextChunk()) {
			it1.index(index);
			double corr = 0;
			double sum1 = 0, sum2 = 0;
			double ssq1 = 0, ssq2= 0;
			size_t count = 0;
			double err = 0;
			for(; !it1.eoc() && !it2.eoc(); ++it1, ++it2) {
				double m1 = abs(*it1);
				double m2 = abs(*it2);
				corr += m1*m2;
				sum1 += m1;
				sum2 += m2;
				ssq1 += m1*m1;
				ssq2 += m2*m2;
				count++;
				err += pow(abs(*it1)-abs(*it2),2);
			}
			assert(it1.isChunkEnd());
			assert(it2.isChunkEnd());

			cerr << err << endl;
			if(err < minerr) {
				minerr = err;
				mineang1 = (index[pslope[0]]-s1_pp->dim(pslope[0])/2.)/
					s1_pp->dim(pslope[0]);
				mineang2 = (index[pslope[1]]-s2_pp->dim(pslope[1])/2.)/
					s2_pp->dim(pslope[1]);
				cerr << "New Min Err: " << err << ", " << mineang1 << ","
					<< mineang2 << endl;
			}

			corr = sample_corr(count, sum1, sum2, ssq1, ssq2, corr); 
			if(fabs(corr) > maxcor) {
				bestang1 = (index[pslope[0]]-s1_pp->dim(pslope[0])/2.)/
					s1_pp->dim(pslope[0]);
				bestang2 = (index[pslope[1]]-s2_pp->dim(pslope[1])/2.)/
					s2_pp->dim(pslope[1]);
				maxcor = corr;
				cerr << "New Max Cor: " << corr << ", " << bestang1 << ","
					<< bestang2 << endl;
			}
			
		}

		cerr << "pseudo radius: " << ii << 
			", pseudo slope 1: " << pslope[0] << ", pseudo slope 2: " 
			<< pslope[1] << " best cor: " << maxcor << " at " << bestang1 
			<< ", " << bestang2 << ", best err: " << minerr << " at " <<
			mineang1 << ", " << mineang2 << endl;
		assert(it1.isEnd());
		assert(it2.isEnd());
	}
	
	
	return axis;
}

int testRotationAxis()
{
	cerr << "Creating Test Image" << endl;
	auto in = createTestImage(64);
	writeComplex("input", in);
	cerr << "Done" << endl;

	cerr << "Rotating" << endl;
	Vector3d ax(.0, .2, .1);
	ax.normalize();
	auto out = bruteForceRotate(ax, .5, in);
	out->write("brute_rotated.nii.gz");
	cerr << "Done" << endl;

	Vector3d newax = getAxis(in, out);
	cerr << "Axis: " << newax << endl;

	return 0;
}

int main()
{
	// test the 'Power' Fourier Transform
	if(testRotationAxis() != 0)
		return -1;

//	if(testPseudoPolar() != 0) 
//		return -1;
	
	return 0;
}





