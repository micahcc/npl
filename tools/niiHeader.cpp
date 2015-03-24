/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file niiHeader.cpp
 *
 *****************************************************************************/

#include <version.h>
#include <string>
#include <iostream>

#include "mrimage.h"
#include "nplio.h"

using std::endl;
using std::cerr;

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
	if(argc <= 1) {
		return -1;
	}

	npl::readMRImage(argv[1], true, true);
}

