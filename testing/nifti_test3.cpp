/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file nifti_test3.cpp Reads a pre-created nifti image with pixel values
 * 1000x+100y+10z+z
 *
 *****************************************************************************/

#include <iostream>

#include "mrimage.h"
#include "nplio.h"
#include "accessors.h"

using namespace std;
using namespace npl;

int main()
{
    /* Read an image with: x*1000+y*100+z*10+t */
    auto oimage = readMRImage("../../data/test_nifti4.nii.gz");
    NDView<double> acc(oimage);

    for(int64_t tt=0; tt < oimage->dim(3); tt++) {
        for(int64_t zz=0; zz < oimage->dim(2); zz++) {
            for(int64_t yy=0; yy < oimage->dim(1); yy++) {
                for(int64_t xx=0; xx < oimage->dim(0); xx++) {
                    double pix = xx*1000+yy*100+zz*10+tt;
                    if(pix != acc.get({xx,yy,zz,tt})) {
                        cerr << "Difference between theoretical and read!" << endl;
                        return -1;
                    }
                }
            }
        }
    }

    return 0;
}

