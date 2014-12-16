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
 * @file txt_write3.cpp Writes an integer image
 *
 *****************************************************************************/

#include <iostream>

#include "mrimage.h"
#include "nplio.h"
#include "iterators.h"

using namespace std;
using namespace npl;

int main()
{
	// create an image
	size_t sz[] = {4, 30};
	vector<double> tmp(sz[0]*sz[1]);
	for(size_t ii=0; ii<sz[1]*sz[0]; ii++)
		tmp[ii] = rand();
    
	{
        auto in = createNDArray(sizeof(sz)/sizeof(size_t), sz, FLOAT64);
        NDIter<double> sit(in);
        for(int ii=0; !sit.eof(); ++sit, ++ii) {
            sit.set(tmp[ii]);
        }
        in->write("test_txt3.txt.gz");
    }

    {
        auto reread = dPtrCast<MRImage>(readMRImage("test_txt3.txt.gz"));
        
        NDIter<double> sit(reread);
        for(int ii=0; !sit.eof(); ++sit, ++ii) {
            if(*sit != tmp[ii]) {
                cerr << "Value Mismatch!" << endl; 
                return -1;
            }
        }

        if(reread->type() != UINT32) {
            cerr << "Type Mismatch!" << endl;
            return -1;
        }
    }
	return 0;
}




