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

#include <iostream>
#include <cstdio>
#include <iomanip>
#include <string>
#include <fstream>

#include <set>
#include <vector>
#include <list>
#include <cmath>
#include <regex>

#include <tclap/CmdLine.h>

#include "version.h"
#include "mrimage.h"
#include "nplio.h"
#include "iterators.h"

using namespace npl;
using namespace std;


template <typename T>
ptr<MRImage> copyHelp(ptr<MRImage> in, string re_str, const vector<string>& lookup)
{
    // figure out size
    size_t odim = 0;
    regex re(re_str);
    for(size_t ii=0; ii<lookup.size(); ii++) {
        if(regex_match(lookup[ii], re)) {
            cerr << "Keeping " << ii << endl;
            odim++;
        } else {
            cerr << "Removign " << ii << endl;
        }
    }

    vector<size_t> osize(4);
    for(size_t dd=0; dd<3; dd++) {
        osize[dd] = in->dim(dd);
    }
    osize[3] = odim;
 
    auto out = dPtrCast<MRImage>(createMRImage(4, osize.data(), in->type()));
    Vector3DConstIter<T> iit(in);
    Vector3DIter<T> oit(out);
    for(iit.goBegin(), oit.goBegin(); !iit.eof() && !oit.eof(); ++iit, ++oit) {
        size_t time_out = 0;
        for(size_t time_in=0; time_in<lookup.size(); time_in++) {
            // copy lines conditionally
            if(regex_match(lookup[time_in], re)) 
                oit.set(time_out++, iit[time_in]);
        }
    }
    assert(iit.eof() && oit.eof());
    return out;
}

int main(int argc, char* argv[])
{
	try {
	/* 
	 * Command Line 
	 */
	TCLAP::CmdLine cmd("This program takes an 4D volume, a list file with "
            "values coresponding to each volume and a regular expression. "
            "Volumes (time-points) whose values match the regular expression "
            "will be kept, all others will be removed",
            ' ', __version__ );

	// arguments
	TCLAP::ValueArg<std::string> a_input("i","in","Input 4D Image",true,"",
			"4D Image", cmd);
	
	TCLAP::ValueArg<std::string> a_lookup("l","lookup",
			"Lookup file. There should be 1 value (whitespace separated) for "
            "each volume.", true,"","*.txt", cmd);
	
	TCLAP::ValueArg<std::string> a_regex("r","regex", "Regular expression to "
            "match values in the lookup file.", true, ".*", "regex", cmd);

	// parse arguments
	cmd.parse(argc, argv);

    auto img = dPtrCast<MRImage>(readMRImage(a_input.getValue()));
    
    // read lookup
    vector<string> lookup;
    ifstream ifs(a_lookup.getValue());
    string v;
    ifs >> v;
    while(ifs.good()) {
        lookup.push_back(v);
        ifs >> v;
    }

    // perform
	if(lookup.size() != img->tlen()) {
        cerr << "Error, number of volumes does not match number of values in "
            "lookup (" << a_lookup.getValue() << endl;
        return -1;
    }

    switch(img->type()) {
        case UINT8:
            copyHelp<uint8_t>(img, a_regex.getValue(), lookup);
            break;
        case INT16:
            copyHelp<int16_t>(img, a_regex.getValue(), lookup);
            break;
        case INT32:
            copyHelp<int32_t>(img, a_regex.getValue(), lookup);
            break;
        case FLOAT32:
            copyHelp<float>(img, a_regex.getValue(), lookup);
            break;
        case COMPLEX64:
            copyHelp<cfloat_t>(img, a_regex.getValue(), lookup);
            break;
        case FLOAT64:
            copyHelp<double>(img, a_regex.getValue(), lookup);
            break;
        case RGB24:
            copyHelp<rgb_t>(img, a_regex.getValue(), lookup);
            break;
        case INT8:
            copyHelp<int8_t>(img, a_regex.getValue(), lookup);
            break;
        case UINT16:
            copyHelp<uint16_t>(img, a_regex.getValue(), lookup);
            break;
        case UINT32:
            copyHelp<uint32_t>(img, a_regex.getValue(), lookup);
            break;
        case INT64:
            copyHelp<int64_t>(img, a_regex.getValue(), lookup);
            break;
        case UINT64:
            copyHelp<uint64_t>(img, a_regex.getValue(), lookup);
            break;
        case FLOAT128:
            copyHelp<long double>(img, a_regex.getValue(), lookup);
            break;
        case COMPLEX128:
            copyHelp<cdouble_t>(img, a_regex.getValue(), lookup);
            break;
        case COMPLEX256:
            copyHelp<cquad_t>(img, a_regex.getValue(), lookup);
            break;
        case RGBA32:
            copyHelp<rgba_t>(img, a_regex.getValue(), lookup);
            break;
        default:
            return -1;
    }

	// done, catch all argument errors
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
    return 0;
}

