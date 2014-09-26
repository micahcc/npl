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
#include "ndarray.h"
#include "nplio.h"
#include "iterators.h"
#include "utility.h"

using namespace npl;
using namespace std;

string regexErrorDescription(std::regex_error& e)
{
    std::ostringstream oss;
    switch(e.code()) {
        case std::regex_constants::error_collate:
            oss << "The expression contained an invalid collating "
                "element name." << endl;
            break;
        case std::regex_constants::error_ctype:
            oss << "The expression contained an invalid character class"
                " name." << endl;
            break;
        case std::regex_constants::error_escape:
            oss << "The expression contained an invalid escaped "
                "character, or a trailing escape." << endl;
            break;
        case std::regex_constants::error_backref:
            oss << "The expression contained an invalid back reference."
                << endl;
            break;
        case std::regex_constants::error_brack:
            oss << "The expression contained mismatched brackets ([ and "
                "])." << endl;
            break;
        case std::regex_constants::error_paren:
            oss << "The expression contained mismatched parentheses (( "
                "and ))." << endl;
            break;
        case std::regex_constants::error_brace:
            oss << "The expression contained mismatched braces ({ and })."
                << endl;
            break;
        case std::regex_constants::error_badbrace:
            oss << "The expression contained an invalid range between "
                "braces ({ and })." << endl;
            break;
        case std::regex_constants::error_range:
            oss << "The expression contained an invalid character range."
                << endl;
            break;
        case std::regex_constants::error_space:
            oss << "There was insufficient memory to convert the "
                "expression into a finite state machine." << endl;
            break;
        case std::regex_constants::error_badrepeat:
            oss << "The expression contained a repeat specifier (one of "
                "*?+{) that was not preceded by a valid regular "
                "expression." << endl;
            break;
        case std::regex_constants::error_complexity:
            oss << "The complexity of an attempted match against a "
                "regular expression exceeded a pre-set level." << endl;
            break;
        case std::regex_constants::error_stack:
            oss << "There was insufficient memory to determine whether "
                "the regular expression could match the specified "
                "character sequence." << endl;
            break;
        default:
            oss << "Unknown regex error occurred" << endl;
    }
    return oss.str();
}

template <typename T>
ptr<NDArray> copyHelp(ptr<const NDArray> in, const vector<bool>& keepers)
{
    // figure out size
    size_t odim = 0;
    for(size_t ii=0; ii<keepers.size(); ii++) {
        if(keepers[ii])
            odim++;
    }

    vector<size_t> osize(4);
    for(size_t dd=0; dd<3; dd++) {
        osize[dd] = in->dim(dd);
    }
    osize[3] = odim;
 
    auto out = in->copyCast(4, osize.data(), in->type());
    Vector3DConstIter<T> iit(in);
    Vector3DIter<T> oit(out);
    for(iit.goBegin(), oit.goBegin(); !iit.eof() && !oit.eof(); ++iit, ++oit) {
        size_t time_out = 0;
        for(size_t time_in=0; time_in<lookup.size(); time_in++) {
            // copy lines conditionally
            if(keepers[time_in]) 
                oit.set(time_out++, iit[time_in]);
        }
    }
    assert(iit.eof() && oit.eof());
    return out;
}

int filter(string infile, string ofile, const vector<bool>& keepers)
{
    size_t nkeepers = 0;
    for(size_t ii=0; ii<keepers.size(); ii++) 
        nkeepers += keepers[ii];

    char delim = ' ';
    auto incsv = readStrCSV(infile, delim);
    if(incsv.size() == 0) {
        cerr << "Error in input file!" << infile << endl;
        return -1;
    }

    size_t nrow = incsv.size();
    size_t ncol = incsv[0].size();
    for(size_t rr=0; rr<incsv.size(); rr++) {
        if(ncol != incsv[rr].size()) {
            cerr << "Error, mismatch in row width (# columns) in !" 
                << infile << endl;
            return -1;
        }
    }

    vector<vector<string>> out;

    if(ncol == keepers.size() && nrow == keepers.size()) {
        cout << "Square Input, assuming you want to extract both rows "
            "and columns." << endl;
        out.resize(nkeepers);
        size_t irow = 0; 
        size_t orow = 0; 
        size_t icol = 0; 
        size_t ocol = 0; 

        for(irow=0, orow=0; irow < nrow && orow < nkeepers; ++irow) {
            if(keepers[irow]) {
                out[orow].resize(nkeepers);
                for(icol=0, ocol=0; icol<ncol && ocol<nkeepers; ++icol) {
                    if(keepers[icol]) {
                        out[orow][ocol] = incsv[irow][icol];
                        ++ocol;
                    }
                }
                orow++;
            }
        }
    } else if(ncol != keepers.size() && nrow != keepers.size()) {
        cerr << "Error, neither number of rows nor columns "
            "matched the number of values in " << infile << endl;
        return -1;
    } else if(ncol == keepers.size()) {
        out.resize(nrow);
        size_t icol = 0; // cols
        size_t ocol = 0; // cols

        for(size_t rr=0; rr<nrow; rr++) {
            out[rr].resize(nkeepers);
            for(icol=0, ocol=0; icol<ncol && ocol<nkeepers; ++icol) {
                if(keepers[icol]) {
                    out[rr][ocol] = incsv[rr][icol];
                    ++ocol;
                }
            }
        }
    } else if(nrow == keepers.size()) {
        out.resize(nkeepers);
        size_t irow = 0; // rows
        size_t orow = 0; // rows

        for(irow=0, orow=0; irow < nrow && orow < nkeepers; ++irow) {
            if(keepers[irow]) {
                out[orow].assign(incsv[irow].begin(), incsv[irow].end());
                ++orow;
            }
        }
    }

    // write
    ofstream ofs(ofile.c_str());
    for(size_t rr=0; rr<out.size(); rr++) {
        for(size_t cc=0; cc<out[rr].size(); cc++) {
            if(cc) ofs << delim;
            ofs << out[rr][cc];
        }
        ofs << endl;
    }
    
    return 0;
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
	TCLAP::ValueArg<std::string> a_out("o","out","Output 4D Image, only the "
			"selected volumes will be kept.",true,"", "4D Image", cmd);
	
	TCLAP::ValueArg<std::string> a_lookup("l","lookup",
			"Lookup file. There should be 1 value (whitespace separated) for "
            "each volume.", true,"","*.txt", cmd);

	TCLAP::MultiArg<std::string> a_imatch("","imat",
            "Input of other text-matrix files to filter identically with 4D"
            " image.", false,"*.txt", cmd);
	
    TCLAP::MultiArg<std::string> a_omatch("","omat",
			"Output of filter text matrices, in same order of --imat args.",
            false,"*.txt", cmd);
	
	TCLAP::ValueArg<std::string> a_regex("r","regex", "Regular expression to "
            "match values in the lookup file.", false, ".*", "regex", cmd);

	TCLAP::ValueArg a_auto("a","auto", "If the lookup values are numberic "
            "then this will break up the file into multiple files. Provide a"
            "threshold for splitting here. ", false, 500, "maxdist", cmd);

    TCLAP::SwitchArg a_auto_keepmin("K","keep-min", "Keep the volume for the "
            "minimum group in every group. IE if the is a 'B=0' image then "
            "it will always be aded to all ouptuts when auto-splitting",
            true, cmd);

	// parse arguments
	cmd.parse(argc, argv);

    if(a_imatch.getValue().size() != a_omatch.getValue().size()) {
        cerr << "Error: Must provide same number of --imat and --omat "
            "arguments" << endl;
        return -1;
    }
	
    vector<int> keepers;
    if(a_auto.isSet()) { 
        // read lookup
        vector<double> lookup;
        ifstream ifs(a_lookup.getValue());
        string v;
        ifs >> v;
        while(ifs.good()) {
            lookup.push_back(v);
            ifs >> v;
        }
        
        double scatter = 0;
    } else {
        if(!a_regex.isSet()){
            cerr << "If not doing splitting auomaticlly, you need to provide a"
                " regular expression to selected the desired group of "
                "time-points" << endl;
            return -1;
        }

        regex re;
        try {//std::regex::egrep
            re.assign(a_regex.getValue());
        } catch(regex_error& e) {
            cerr << "Error in regular expression: '" << a_regex.getValue() << "'\n";
            cerr << regexErrorDescription(e) << "\nQutting" << endl;
            return -1;
        }

        // read lookup
        vector<string> lookup;
        ifstream ifs(a_lookup.getValue());
        string v;
        ifs >> v;
        while(ifs.good()) {
            lookup.push_back(v);
            ifs >> v;
        }

        // figure out size
        cerr << "Regex: " << a_regex.getValue() << endl;
        keepers.resize(lookup.size(), 0);
        size_t nkeepers = 0;
        for(size_t ii=0; ii<lookup.size(); ii++) {
            if(regex_match(lookup[ii], re)) {
                cerr << left << setw(10) << lookup[ii] << " Matches, keeping" << endl;
                keepers[ii] = 1;
                nkeepers++;
            } else {
                cerr << left << setw(10) << lookup[ii] << " Does not match, Removing" << endl;
                keepers[ii] = 0;
            }
        }

    }

    // perform matched arrays first
    for(size_t ii=0; ii<a_imatch.getValue().size(); ii++) {
        if(filter(a_imatch.getValue()[ii], a_omatch.getValue()[ii], keepers) != 0)
            return -1;
    }
    
    // perform
    auto img = dPtrCast<NDArray>(readMRImage(a_input.getValue()));
	if(keepers.size() != img->tlen()) {
        cerr << "Error, number of volumes does not match number of values in "
            "lookup" << endl;
        return -1;
    }

    if(a_auto.isSet()) {
        ptr<MRImage> out;
        switch(img->type()) {
            case UINT8:
                out = copyHelp<uint8_t>(img, keepers);
                break;
            case INT16:
                out = copyHelp<int16_t>(img,keepers);
                break;
            case INT32:
                out = copyHelp<int32_t>(img,keepers);
                break;
            case FLOAT32:
                out = copyHelp<float>(img,keepers);
                break;
            case COMPLEX64:
                out = copyHelp<cfloat_t>(img,keepers);
                break;
            case FLOAT64:
                out = copyHelp<double>(img,keepers);
                break;
            case RGB24:
                out = copyHelp<rgb_t>(img,keepers);
                break;
            case INT8:
                out = copyHelp<int8_t>(img,keepers);
                break;
            case UINT16:
                out = copyHelp<uint16_t>(img,keepers);
                break;
            case UINT32:
                out = copyHelp<uint32_t>(img,keepers);
                break;
            case INT64:
                out = copyHelp<int64_t>(img,keepers);
                break;
            case UINT64:
                out = copyHelp<uint64_t>(img,keepers);
                break;
            case FLOAT128:
                out = copyHelp<long double>(img,keepers);
                break;
            case COMPLEX128:
                out = copyHelp<cdouble_t>(img,keepers);
                break;
            case COMPLEX256:
                out = copyHelp<cquad_t>(img,keepers);
                break;
            case RGBA32:
                out = copyHelp<rgba_t>(img,keepers);
                break;
            default:
                return -1;
        }

	img->write(a_out.getValue());

	// done, catch all argument errors
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }
    return 0;
}

