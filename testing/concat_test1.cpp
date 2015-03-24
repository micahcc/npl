/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file concat_test1.cpp Tests concatinating images to create a higher
 * dimensional imagee
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
    vector<size_t> imgsize({7, 8, 9});

    // create 3 images, same size
    auto img1 = createNDArray(imgsize.size(), imgsize.data(), INT32);
    auto img2 = createNDArray(imgsize.size(), imgsize.data(), INT32);
    auto img3 = createNDArray(imgsize.size(), imgsize.data(), INT32);

    // fill
    size_t ii=0;
    for(FlatIter<int> it(img1); !it.eof(); ++it)
        it.set(3*(ii++));

    ii = 0;
    for(FlatIter<int> it(img2); !it.eof(); ++it)
        it.set(1+3*(ii++));

    ii = 0;
    for(FlatIter<int> it(img3); !it.eof(); ++it)
        it.set(2+3*(ii++));

    // concat/elevent
    vector<ptr<NDArray>> v({img1, img2, img3});
    auto out = concatElevate(v);

    for(size_t dd=0; dd<imgsize.size(); dd++) {
        if(imgsize[dd] != out->dim(dd)) {
            cerr << "Concat failed to create the right size output!" << endl;
            return -1;
        }
    }
    if(imgsize.size() != out->dim(imgsize.size())) {
        cerr << "Concat failed to create a higher dimensional image!" << endl;
    }

    // check output
    ii=0;
    for(FlatIter<int> it(out); !it.eof(); ++it) {
        if(*it != ii++) {
            cerr << "Error concat elevate failed!" << endl;
            return -1;
        }
    }

    return 0;
}
