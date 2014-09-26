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
 * @file kern_test.cpp Test speed of kernel slicer.
 *
 *****************************************************************************/

#include <cassert>
#include <iostream>
#include "iterators.h"
#include "accessors.h"
#include "ndarray.h"
#include "npltypes.h"

using namespace std;
using namespace npl;

int64_t clamp(int64_t inf, int64_t sup, int64_t v)
{
  return std::max(inf, std::min(sup, v));
}

void looper(size_t* sz, size_t* radius,
        std::function<void(int64_t*, int64_t*, int64_t*)> foo)
{
    int64_t kern[4];
    for(size_t ii=0; ii<4; ii++)
        kern[ii] = radius[ii];

    int64_t index[4];
    int64_t clampoff[4];
    int64_t offset[4];
    for(index[0] = 0; index[0] < sz[0]; ++index[0]){
        for(index[1] = 0; index[1] < sz[1]; ++index[1]){
            for(index[2] = 0; index[2] < sz[3]; ++index[2]){
                for(index[3] = 0; index[3] < sz[2]; ++index[3]){
                    for(offset[0] = -kern[0]; offset[0] <= kern[0]; ++offset[0]){
                        clampoff[0] = clamp(0, sz[0]-1, offset[0]+index[0]) - index[0];
                        for(offset[1] = -kern[1]; offset[1] <= kern[1]; ++offset[1]){
                            clampoff[1] = clamp(0, sz[1]-1, offset[1]+index[1]) - index[1];
                            for(offset[2] = -kern[2]; offset[2] <= kern[2]; ++offset[2]){
                                clampoff[2] = clamp(0, sz[2]-1, offset[2]+index[2]) - index[2];
                                for(offset[3] = -kern[3]; offset[3] <= kern[3]; ++offset[3]){
                                    clampoff[3] = clamp(0, sz[3]-1, offset[3]+index[3]) - index[3];
                                    foo(index, offset, clampoff);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

int main()
{
//    size_t sz[] = {(size_t)rand()%10,(size_t)rand()%20,
//        (size_t)rand()%30,(size_t)rand()%10};
    size_t sz[] = {10,20,30,10};
    auto arr = createNDArray(4, sz, FLOAT32);
    std::vector<int64_t> kern({1,1,1,1});
    std::vector<size_t> kernz({1,1,1,1});
    npl::KernelIter<double> it(arr);
    it.setRadius(kernz);

    Vector3DView<double> ac(arr);

    assert(it.isBegin());
    assert(!it.isEnd());

    std::vector<int64_t> index({1,1,1,0});
    it.goIndex(index);
    assert(!it.isBegin());
    assert(!it.isEnd());

    assert(!it.isEnd());

    it.goBegin();
    index.assign({3,4,5,6});
    it.goIndex(index);

    // brute force method
    size_t count = 0;
    double tmp= 0;
    clock_t c;
    for(it.goBegin(); !it.eof(); ++it) {
        for(size_t k=0; k<it.ksize(); k++) {
            tmp += sqrt(it.getK(k));
            count++;
        }
    }
    count = 0;
    c = clock();
    looper(sz, kernz.data(), [&](int64_t* ind, int64_t*, int64_t* off)
    { 
        tmp += sqrt(ac(ind[0]+off[0], ind[1]+off[1], ind[2]+off[2], ind[3]+off[3]));
        count++;
    });
    c = clock()-c;
    cerr << "Lambda Method: " << c << endl;
    cerr << count << endl;
    
    count = 0;
    c = clock();
    int64_t offset[4];
    size_t clampoff[4];
    for(index[0] = 0; index[0] < sz[0]; ++index[0]){
        for(index[1] = 0; index[1] < sz[1]; ++index[1]){
            for(index[2] = 0; index[2] < sz[2]; ++index[2]){
                for(index[3] = 0; index[3] < sz[3]; ++index[3]){
                    for(offset[0] = -kern[0]; offset[0] <= kern[0]; ++offset[0]){
                        clampoff[0] = clamp(0, sz[0]-1, offset[0]+index[0]) - index[0];
                        for(offset[1] = -kern[1]; offset[1] <= kern[1]; ++offset[1]){
                            clampoff[1] = clamp(0, sz[1]-1, offset[1]+index[1]) - index[1];
                            for(offset[2] = -kern[2]; offset[2] <= kern[2]; ++offset[2]){
                                clampoff[2] = clamp(0, sz[2]-1, offset[2]+index[2]) - index[2];
                                for(offset[3] = -kern[3]; offset[3] <= kern[3]; ++offset[3]){
                                    clampoff[3] = clamp(0, sz[3]-1, offset[3]+index[3]) - index[3];
                                    tmp += sqrt(ac(index[0]+clampoff[0], index[1]+clampoff[1], index[2]+clampoff[2], index[3]+clampoff[3]));
                                    count++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    c = clock()-c;
    cerr << "Nested Loop Method: " << c << endl;
    cerr << count << endl;
    
    c = clock();
    for(it.goBegin(); !it.eof(); ++it) {
        for(size_t k=0; k<it.ksize(); k++) {
            tmp += sqrt(it.getK(k));
            count++;
        }
    }
    c = clock()-c;
    cerr << "Kernel Slicer Time: " << c << endl;
    cerr << count << endl;

    return 0;
}
