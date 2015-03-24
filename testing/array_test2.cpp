/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file array_test2.cpp
 *
 *****************************************************************************/

#include "ndarray.h"
#include "accessors.h"
#include "iterators.h"
#include "slicer.h"
#include <iostream>
#include <list>
#include <ctime>

using namespace std;
using namespace npl;

int main()
{
	size_t ii=0;
	std::vector<size_t> sz({100,78,83});
	std::vector<size_t> order({2,1,0});
	auto testp = make_shared<NDArrayStore<3, float>>(sz);
	cerr << "Bytes: " << testp->bytes() << endl;

	for(ii = 0; ii < testp->elements(); ii++)
		(*testp)[ii] = ii;

	clock_t t;

	cerr << "Comparing Direct with Slicer" << endl;
	ii = 0;
	double total = 0;
	t = clock();
	Slicer it(testp->ndim(), testp->dim());
	it.setOrder(order);
	for(it.goBegin(); !it.isEnd(); ++it, ++ii) {
		if((*testp)[*it] != (*testp)[ii]) {
			cerr << "Values Differ" << endl;
			return -1;
		}
		if(*it != ii) {
			cerr << "Indices Differ" << endl;
			return -1;
		}
	}
	t = clock()-t;
	std::cout << "Time: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";

	cerr << "Comparing Direct with Slicer (Through Accessor)" << endl;
	NDView<double> img(testp);
	ii = 0;
	t = clock();

	it.setOrder(order);
	for(it.goBegin(); !it.isEnd(); ++it, ++ii) {
		if(img[*it] != (*testp)[ii]) {
			cerr << "Values Differ" << endl;
			return -1;
		}
		if(*it != ii) {
			cerr << "Indices Differ" << endl;
			return -1;
		}
	}
	t = clock()-t;
	std::cout << "Time: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";

	cerr << "Comparing Direct with iterator" << endl;
	ii = 0;
	t = clock();
	for(OrderIter<double> it(testp); !it.isEnd(); ++it, ++ii) {
		if(*it != (*testp)[ii]) {
			cerr << "Values Differ" << endl;
			return -1;
		}
	}
	t = clock()-t;
	std::cout << "Time: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";

	cerr << "Comparing Reordered" << endl;
	std::vector<int64_t> index(3);
	{
	OrderIter<double> it(testp);
	it.setOrder({2,1,0});
	it.goBegin();
	// forward order
	ii = 0;
	for(int64_t xx=0; xx < testp->dim(0); xx++) {
		for(int64_t yy=0; yy < testp->dim(1); yy++) {
			for(int64_t zz=0; zz < testp->dim(2); zz++, ++it, ii++) {
				if(*it != (*testp)[{xx,yy,zz}]) {
					cerr << "iter, initializer list mismatch" << endl;
					return -1;
				}

				index[0] = xx;
				index[1] = yy;
				index[2] = zz;
				if(*it != (*testp)[index]) {
					cerr << "iter, vector mismatch" << endl;
					return -1;
				}

				if(*it != (*testp)[index]) {
					cerr << "iter, array mismatch" << endl;
					return -1;
				}


				if(*it != (*testp)[index]) {
					cerr << "iter, array mismatch" << endl;
					return -1;
				}

				if(*it != (*testp)[ii]) {
					cerr << "iter, flat mismatch" << endl;
					return -1;
				}
			}
		}
	}
	}

	cerr << "Flat Speed: ";
	total = 0;
	t = clock();
	ii = 0;
	for(int64_t xx=0; xx < testp->dim(0); xx++) {
		for(int64_t yy=0; yy < testp->dim(1); yy++) {
			for(int64_t zz=0; zz < testp->dim(2); zz++, ii++) {
				total += (*testp)[ii];
			}
		}
	}
	t = clock()-t;
	std::cout << "Time: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";

	cerr << "Vector Index Speed: ";
	t = clock();
	for(int64_t xx=0; xx < testp->dim(0); xx++) {
		for(int64_t yy=0; yy < testp->dim(1); yy++) {
			for(int64_t zz=0; zz < testp->dim(2); zz++) {
				index[0] = xx;
				index[1] = yy;
				index[2] = zz;
				total += (*testp)[index];
			}
		}
	}
	t = clock()-t;
	std::cout << "Time: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";

	cerr << "Iterator Speed: ";
	t = clock();
	it.goBegin();
	for(int64_t xx=0; xx < testp->dim(0); xx++) {
		for(int64_t yy=0; yy < testp->dim(1); yy++) {
			for(int64_t zz=0; zz < testp->dim(2); zz++, ++it) {
				total += *it;
			}
		}
	}
	t = clock()-t;
	std::cout << "Time: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";

	cerr << "Initializer List Speed: ";
	t = clock();
	for(int64_t xx=0; xx < testp->dim(0); xx++) {
		for(int64_t yy=0; yy < testp->dim(1); yy++) {
			for(int64_t zz=0; zz < testp->dim(2); zz++) {
				total += (*testp)[{xx,yy,zz}];
			}
		}
	}
	t = clock()-t;
	std::cout << "Time: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";


	return 0;
}

