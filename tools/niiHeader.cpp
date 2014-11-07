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

