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
 * @file applyDeform.cpp Tool to apply a deformation field to another image. 
 * Not yet functional
 *
 *****************************************************************************/

#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include "mrimage.h"
#include "mrimage_utils.h"
#include "kernel_slicer.h"
#include "kdtree.h"
#include "iterators.h"
#include "accessors.h"

std::string example = 
"{\n"
" \"type\": \"double\""
" \"size\" : [3,2,9],\n"
" \"values\": [0,1,2,3,4,5,6,7,8,9,\n"
"          11,12,13,14,15,16,17,18,19,20\n"
"          21,22,23,24,25,26,27,28,29,30\n"
"          31,32,33,34,35,36],\n"
" \"spacing\" : [1,3,5],\n"
" \"direction\": [[1,0,0],[0,1,0],[0,0,1],\n"
" \"origin\": [3.2, 32.1, 1]\n"
"}";

int main(int argc, char** argv)
{
	try {
	/*
	 * Command Line
	 */

	TCLAP::CmdLine cmd("Creates a Nifti Image from a text description file. "
            "The text description file should be a json file with at minimum "
            "size and values keys.", ' ', __version__ );

	TCLAP::ValueArg<string> a_in("i", "input", "Input json description.",
			false, "", "*.json", cmd);
	TCLAP::ValueArg<string> a_out("o", "output", "Output image.",
			false, "", "*.nii.gz", cmd);
	TCLAP::SwitchArg a_example("E", "example", "print an example file and exit", 
            cmd);

    if(a_example.isSet()) {
        cout << example << endl;
        return 0;
    }

    if(!a_in.isSet()) {
        throw TCLAP::ArgException("Need to provide an input image descrption file.");
        return -1;
    }
    if(!a_out.isSet()) {
        throw TCLAP::ArgException("Need to provide an output image file.");
        return -1;
    }

    // read json

	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
}

