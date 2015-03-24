/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
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
    datamap.openNew(filename, mapsize*10);
    write_double((double*)datamap.data(), ndouble*10);
    if(test_double((double*)datamap.data(), ndouble) != 0) {
        cerr << "Error Difference in writen double 2" << endl;
        return -1;
    }

    // Test Close then Reopen
    datamap.close();
    datamap.openExisting(filename);
    if(datamap.size() <= 0) {
        cerr << "Error when re-opening!" << endl;
        return -1;
    }

    if(test_double((double*)datamap.data(), ndouble) != 0) {
        cerr << "Error Difference in re-opened" << endl;
        return -1;
    }

    return 0;
}

