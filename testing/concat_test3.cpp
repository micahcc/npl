/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file concat_test2.cpp Tests concatinating images along a particular
 * dimension (z dimension in this case)
 *
 *****************************************************************************/

#include "mrimage.h"
#include "ndarray.h"
#include "ndarray_utils.h"
#include "macros.h"
#include "iterators.h"

#include <iostream>

using namespace npl;
using namespace std;

int main()
{
    vector<size_t> imgsize({3, 8, 0});

    // create 3 images, same size
    imgsize[2] = 3;
    auto img1 = createNDArray(imgsize.size(), imgsize.data(), INT32);

    imgsize[2] = 4;
    auto img2 = createNDArray(imgsize.size(), imgsize.data(), INT32);

    imgsize[2] = 5;
    auto img3 = createNDArray(imgsize.size(), imgsize.data(), INT32);

    vector<int64_t> index(imgsize.size());
    // fill first with 1000*x+100*y+z
    for(NDIter<int> it(img1); !it.eof(); ++it) {
        it.index(index);

        size_t pval = index[0]*1000+index[1]*100+index[2];
        it.set(pval);
    }

    // image 1 will have 3 added to z
    for(NDIter<int> it(img2); !it.eof(); ++it) {
        it.index(index);

        size_t pval = index[0]*1000+index[1]*100+index[2]+3;
        it.set(pval);
    }

    // image 2 will have 7 added to z
    for(NDIter<int> it(img3); !it.eof(); ++it) {
        it.index(index);

        size_t pval = index[0]*1000+index[1]*100+index[2]+7;
        it.set(pval);
    }

    // concat/elevent
    vector<ptr<NDArray>> v({img1, img2, img3});
    auto out = concat(v, 2);

    // check output
    for(NDIter<int> it(out); !it.eof(); ++it) {
        it.index(index);
        size_t pval = index[0]*1000+index[1]*100+index[2];
        if(*it != pval) {
            cerr << "Error concat failed!" << endl;
            return -1;
        }
    }

    return 0;
}


