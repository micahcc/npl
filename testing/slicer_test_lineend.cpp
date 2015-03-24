/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file slicer_test_lineend.cpp
 *
 *****************************************************************************/


#include "slicer.h"

#include <vector>
#include <list>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <cstdint>

using namespace std;
using namespace npl;

template <size_t DIM>
int test()
{
	cerr << "Not yet implemented" << endl;
	return 0;

//	// random size
//	std::vector<int64_t> index(DIM);
//	std::vector<size_t> sz(DIM, 0);
//	size_t tsize = 1;
//	for(size_t ii=0; ii<sz.size(); ii++) {
//		sz[ii] = rand()%20+1;
//		tsize *= sz[ii];
//	}
//
//	// random direction
//	size_t dir = rand()%DIM;
//
//	size_t tcount = 0;
//	Slicer slice(DIM, sz.data());
//	slice.setOrder({dir});
//	cerr << "Starting Iteration" << endl;
//	for(slice.goBegin(); !slice.eof(); ) {
//		// while not end of line
//		size_t count = 0;
//		do {
//			++slice, ++tcount, ++count;
//		} while(!slice.isLineEnd(dir));
//
//		// check to see if we are at the end of other lines, and make sure that
//		// they get reported correctly
//		slice.index(DIM, index.data());
//		for(size_t dd=0; dd<DIM; dd++) {
//			cerr << index[dd] << ",";
//			if(index[dd] == 0 && !slice.isLineBegin(dd)) {
//				cerr << "Begin of line failure!" << endl;
//				return -1;
//			}
//			if(index[dd] == sz[dd]-1 && !slice.isLineEnd(dd)) {
//				cerr << "End of line failure!" << endl;
//				return -1;
//			}
//			if((index[dd] < sz[dd]-1 && index[dd] > 0) &&
//					(slice.isLineBegin(dd) || slice.isLineEnd(dd))) {
//				cerr << "Middle of Line failure!" << endl;
//				return -1;
//			}
//		}
//		cerr << endl;
//
//		if(count != sz[dir]) {
//			cerr << "Counts differ: " << count << " vs " << sz[dir] << endl;
//			return -1;
//		}
//	}
//
//	if(tcount != tsize) {
//		cerr << "Error, didn't iterate correctly!" << endl;
//		return -1;
//	}
//
//	return 0;
}

int main()
{
	if(test<2>() != 0) {
		return -2;
	} else if(test<3>() != 0) {
		return -3;
	} else if(test<4>() != 0) {
		return -4;
	} else if(test<5>() != 0) {
		return -5;
	}

	return 0;
}

