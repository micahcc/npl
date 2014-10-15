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
 * @file histeq_test.cpp Tests histogram equalizer
 *
 *****************************************************************************/

#include "mrimage.h"
#include "ndarray.h"
#include "ndarray_utils.h"
#include "macros.h"
#include "iterators.h"

#include <iostream>
#include <random>

using namespace npl;
using namespace std;


double gaussGen(double x, double y, double z, double xsz, double ysz, double zsz)
{
    double v = exp(-pow(xsz/2-x,2)/9)*exp(-pow(ysz/2-y,2)/16)*exp(-pow(zsz/2-z,2)/64);
	return v;
}

shared_ptr<MRImage> gaussianImage()
{
	std::random_device rd;
	std::default_random_engine rng(rd());
	std::normal_distribution<double> randn;

    // create an image
    size_t sz[] = {64,64,64};
    int64_t index[3];
    auto in = createMRImage(sizeof(sz)/sizeof(size_t), sz, FLOAT32);

    // fill with a shape that is somewhat unique when rotated. 
    OrderIter<double> sit(in);
    double sum = 0;
    while(!sit.eof()) {
        sit.index(3, index);
        double v= gaussGen(index[0], index[1], index[2], sz[0], sz[1], sz[2]);
        sit.set(v);
        sum += v;
        ++sit;
    }

    for(sit.goBegin(); !sit.eof(); ++sit) 
        sit.set(randn(rng)/1000 + sit.get());

    return in;
}

int main()
{
    auto img = gaussianImage();
    img->write("histeq_original.nii.gz");

	auto out = dPtrCast<MRImage>(histEqualize(img));
	out->write("histeq_equalized.nii.gz");

    return 0;
}

