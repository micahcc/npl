/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file nifti_orient_test1.cpp
 *
 *****************************************************************************/

#include <iostream>

#include "mrimage.h"
#include "nplio.h"
#include "mrimage_utils.h"
#include "iterators.h"

using namespace std;
using namespace npl;

int main()
{
	// create an image
	size_t sz[] = {2, 1, 4, 3};
    {
        auto in = createNDArray(sizeof(sz)/sizeof(size_t), sz, FLOAT64);

        NDIter<double> sit(in);
        for(int ii=0; !sit.eof(); ++sit, ++ii) {
            sit.set(ii);
        }
        in->write("test.json.gz");
    }

    {
        auto reread = readNDArray("test.json.gz");

        NDIter<double> sit(reread);
        for(int ii=0; !sit.eof(); ++sit, ++ii) {
            if(*sit != ii) {
                cerr << "Value Mismatch!" << endl;
                return -1;
            }
        }

        if(reread->type() != FLOAT64) {
            cerr << "Type Mismatch!" << endl;
            return -1;
        }

    }
	return 0;
}



