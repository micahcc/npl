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
 * @file nifti_orient_test1.cpp
 *
 *****************************************************************************/

#include <iostream>

#include "mrimage.h"
#include "mrimage_utils.h"
#include "iterators.h"

using namespace std;
using namespace npl;

int main()
{
	// create an image
	size_t sz[] = {2, 1, 4, 3};
    Eigen::VectorXd neworigin(4);
    neworigin << 1.3, 75, 9, 0;
    
    Eigen::VectorXd newspacing(4);
    newspacing << 4.3, 4.7, 1.2, .3;

    Eigen::MatrixXd newdir(4,4);
    newdir << 
            -0.16000, -0.98424,  0.07533, 0,
            0.62424, -0.16000, -0.76467, 0,
            0.76467, -0.07533,  0.64000, 0, 
                  0,        0,        0, 1;

    {
        auto in = createMRImage(sizeof(sz)/sizeof(size_t), sz, FLOAT64);

        NDIter<double> sit(in);
        for(int ii=0; !sit.eof(); ++sit, ++ii) {
            sit.set(ii);
        }
        in->setOrient(neworigin, newspacing, newdir);
        in->write("test.json");
    }

    {
        auto reread = dPtrCast<MRImage>(readMRImage("test.json"));
        
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

        if(reread->getOrigin() != neworigin) {
            cerr << "Origin Mismatch" << endl;
            return -1;
        }
        
        if(reread->getSpacing() != newspacing) {
            cerr << "Spacing Mismatch" << endl;
            return -1;
        }
        
        if(reread->getDirection() != newdir) {
            cerr << "Direction Mismatch" << endl;
            return -1;
        }

    }
	return 0;
}


