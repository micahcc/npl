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
 * @file mmap_test Test the Memory-map wrapper
 *
 *****************************************************************************/

#include "utility.h"
#include <iostream>
 
using namespace std; 
using namespace npl; 
 
void write_double(double* buffer, size_t bsize) 
{ 
    for(size_t ii=0; ii<bsize; ii++) { 
        buffer[ii] = ii; 
    } 
} 
 
int test_double(double* buffer, size_t bsize) 
{ 
    for(size_t ii=0; ii<bsize; ii++) { 
        if(buffer[ii] != ii) 
            return -1; 
    } 
 
    return 0; 
} 
 
int main(int argc, char** argv) 
{
    string filename = ".tmp000123";
    size_t ndouble = 1024;
    if(argc == 2) {
        filename = argv[1];
    } else if(argc == 3) {
        filename = argv[1];
        ndouble = atoi(argv[2]);
    } else {
        cerr << "Using default Values (to set different mapfile use argument "
            "1, to set a different length use argument 2" << endl;
    }

    size_t mapsize = ndouble*sizeof(double);
    MemMap datamap(filename, mapsize, true);
    if(datamap.size() <= 0) {
        cerr << "Memory Map Failed!" << endl;
        return -1;
    }

    write_double((double*)datamap.data(), ndouble);
    if(test_double((double*)datamap.data(), ndouble) != 0) {
        cerr << "Error Difference in writen double 1" << endl;
        return -1;
    }

    // Open Another
    datamap.open(filename, mapsize*10, true);
    write_double((double*)datamap.data(), ndouble*10);
    if(test_double((double*)datamap.data(), ndouble) != 0) {
        cerr << "Error Difference in writen double 2" << endl;
        return -1;
    }

    // Test Close then Reopen
    datamap.close();
    datamap.open(filename, mapsize*10, false);
    if(datamap.size() <= 0) {
        cerr << "Error when re-opening!" << endl;
        return -1;
    }

    if(test_double((double*)datamap.data(), ndouble) != 0) {
        cerr << "Error Difference in re-opened" << endl;
        return -1;
    }

    // Test Close then Reopen With Wrong Size
    cerr << "IGNORE THE NEXT ERROR" << endl;
    datamap.open(filename, mapsize, false);
    cerr << "OK, STOP IGNORING" << endl;
    if(datamap.size() > 0) {
        cerr<<"Error should have gotten an error when opening with wrong size"
            " but did not!" << endl;
        return -1;
    }
    return 0;
}

