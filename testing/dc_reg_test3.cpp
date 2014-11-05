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
 * @file dc_reg_test3.cpp Tests mutual information deformation computation
 *
 *****************************************************************************/

#include "mrimage.h"
#include "iterators.h"
#include "accessors.h"
#include "ndarray_utils.h"
#include "mrimage_utils.h"
#include "registration.h"
#include "byteswap.h"

#include <string>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <memory>
#include <cstring>

using namespace std;
using namespace npl;

shared_ptr<MRImage> squareImage()
{
    // create test image
	int64_t index[3];
	size_t sz[] = {32, 32, 32};
	auto in = createMRImage(sizeof(sz)/sizeof(size_t), sz, FLOAT64);

	// fill with square
	OrderIter<double> sit(in);
	while(!sit.eof()) {
		sit.index(3, index);
		if(index[0] > sz[0]/4 && index[0] < 2*sz[0]/3 &&
				index[1] > sz[1]/5 && index[1] < sz[1]/2 &&
				index[2] > sz[2]/3 && index[2] < 2*sz[2]/3) {
			sit.set(1);
		} else {
			sit.set(0);
		}
		++sit;
	}

    return in;
};

int main()
{
	double ind[3];
	double pt[3];
	double def, ddef;
	int dir = 0;
    auto origimg = squareImage();
	origimg->write("dcrt3_original.nii.gz");
	LinInterp3DView<double> orig_vw(origimg);

	BSplineView<double> b_vw;
	b_vw.m_ras = true;
	b_vw.createOverlay(origimg, 10);
	for(NDIter<double> it(b_vw.getParams()); !it.eof(); ++it) {
		it.index(3, ind);
		it.set(ind[dir]*ind[dir]);
	}
	b_vw.getParams()->write("dcrt3_init_params.nii.gz");
	b_vw.reconstruct(origimg)->write("dcrt3_init_field.nii.gz");

	// Create Distorted Image
    auto distorted = dPtrCast<MRImage>(origimg->copy());
	for(NDIter<double> it(distorted); !it.eof(); ++it) {
		it.index(3, ind);
		distorted->indexToPoint(3, ind, pt);
		b_vw.get(3, pt, dir, def, ddef);
		ind[dir] += def/distorted->spacing(dir);
		double Fm = orig_vw.get(ind[0], ind[1], ind[2]);
		double Fc = Fm*(1+ddef);
		it.set(Fc);
	}

	// create image with gaussian kernel in it
	distorted->write("dcrt3_distorted.nii.gz");

	vector<double> sigmas({5, 4, 3, 2, 1, 0.5, 0});
	auto p = infoDistCor(origimg, distorted, dir, 10, 0, 0, sigmas,
			100, 4, "MI");

	cerr << "True Params:\n" << *b_vw.getParams() << endl;
	cerr << "Est Params:\n" << *p << endl;

	p->write("dcrt3_params.nii.gz");

	// Un-distorting image
	b_vw.setArray(p);
	b_vw.reconstruct(origimg)->write("dcrt3_out_field.nii.gz");
    auto undistorted = dPtrCast<MRImage>(distorted->copy());
	LinInterp3DView<double> dist_vw(distorted);
	for(NDIter<double> it(undistorted); !it.eof(); ++it) {
		it.index(3, ind);
		undistorted->indexToPoint(3, ind, pt);
		b_vw.get(3, pt, dir, def, ddef);
		ind[dir] += def/distorted->spacing(dir);
		double Fm = dist_vw.get(ind[0], ind[1], ind[2]);
		double Fc = Fm*(1+ddef);
		it.set(Fc);
	}
	undistorted->write("dcrt3_undistorted.nii.gz");

	for(FlatIter<double> oit(origimg), eit(undistorted); !eit.eof();
			++eit, ++oit) {
		if(fabs(oit.get() - eit.get()) > 0.9) {
			cerr << "Difference in distortion-corrected!" << endl;
			return -1;
		}
	}
    return 0;
}



