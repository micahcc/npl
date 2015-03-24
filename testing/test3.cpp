/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file test3.cpp
 *
 *****************************************************************************/

#include <iostream>

#include "mrimage.h"

using namespace std;
using namespace npl;

int main()
{

	MRImageStore<3, double> testimage({10, 23, 39});

	MRImage* testbase = &testimage;

	testimage.dbl({0,0,0}, 0);
	cerr << testimage.dbl({0,0,0}) << endl;
	cerr << testbase->dbl({0,0,0}) << endl;

	testimage.dbl({0,0,0}, 10);
	cerr << testimage.dbl({0,0,0}) << endl;
	cerr << testbase->dbl({0,0,0}) << endl;

}
