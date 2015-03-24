/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file array_test4.cpp
 *
 *****************************************************************************/

#include "ndarray.h"
#include "iterators.h"
#include <iostream>
#include <memory>
#include <ctime>

using namespace std;
using namespace npl;

template <typename IT, size_t D, typename P>
bool compare(std::string name, std::shared_ptr<NDArrayStore<D, P>> array)
{
	size_t ii;
	IT fit(array);
	fit.goBegin();
	for(ii=0, fit.goBegin(); !fit.eof(); ++fit, ++ii) {
		if(*fit != (*array)[ii]) {
			cerr << "Mismatch in " << name << " vs direct access" << endl;
			return -1;
		}
	}

	if(ii != array->elements()) {
		cerr << "Iterator ended at wrong element in " << name  << endl;
		return -1;
	}
	return 0;
}

int main()
{
	cerr << "Testing ND Ordered Iterator" << endl;
	{
	std::vector<size_t> sz({100,78,83});
	auto test1 = make_shared<NDArrayStore<3, float>>(sz);
	cerr << "Bytes: " << test1->bytes() << endl;

	size_t ii;
	for(ii = 0; ii < test1->elements(); ii++)
		(*test1)[ii] = rand();


	if(compare<OrderIter<float>, 3, float>("Float Modifiable Iterator", test1))
		return -1;
	if(compare<OrderIter<  int>, 3, float>("Int Modifiable Iterator", test1))
		return -1;
	if(compare<OrderConstIter<float>, 3, float>("Float Const Iterator", test1))
		return -1;
	if(compare<OrderConstIter<  int>, 3, float>("Int Const Iterator", test1))
		return -1;

	OrderIter<float> fit(test1);
	for(fit.goBegin(); !fit.eof(); ++fit)
		fit.set(rand());

	if(compare<OrderIter<float>, 3, float>("Set Float Iterator", test1))
		return -1;

	}

	{
	std::vector<size_t> sz({71,78,83,7});
	auto test1 = make_shared<NDArrayStore<4, uint8_t>>(sz);
	cerr << "Bytes: " << test1->bytes() << endl;

	OrderIter<float> fit(test1);
	for(fit.goBegin(); !fit.eof(); ++fit)
		fit.set(rand());

	if(compare<OrderIter<float>, 4, uint8_t>("Float Modifiable Iterator", test1))
		return -1;
	if(compare<OrderIter<  int>, 4, uint8_t>("Int Modifiable Iterator", test1))
		return -1;
	if(compare<OrderConstIter<float>, 4, uint8_t>("Float Const Iterator", test1))
		return -1;
	if(compare<OrderConstIter<  int>, 4, uint8_t>("Int Const Iterator", test1))
		return -1;

	}
	return 0;
}

