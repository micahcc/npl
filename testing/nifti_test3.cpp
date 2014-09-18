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
    NDAccess<double> acc(oimage);

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

