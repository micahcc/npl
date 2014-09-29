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

#include <unordered_map>
#include <version.h>
#include <tclap/CmdLine.h>
#include <string>
#include <stdexcept>

#include "mrimage.h"
#include "nplio.h"
#include "mrimage_utils.h"
#include "kdtree.h"
#include "iterators.h"
#include "accessors.h"
#include "mathexpression.h"

using namespace npl;
using namespace std;

void usage(int status)
{
    cerr << "Usage: nplMath --out <image> [options] [-a <image>] [-b <image>] ... \"<equation>\"" << endl;
    cerr << "\tMath is performed in the space of the first image on the "
        "command line which is not necessarily -a. All other images are "
        "lanczos resampled to that space (unless --nn/--lin are provided). "
        "Pixel type is by default the same as this image, but it can be set "
        "with --short/--double/--float. Any single character can follow a - to "
        "create a variable for use in the equation. "
        "Acceptable operations in the equation are:\n ";
    cerr << setw(10) << "sin" << "sine trig function" << endl;
    cerr << setw(10) << "cos" << "cosine trig function" << endl;
    cerr << setw(10) << "tan" << "tangent trig function" << endl;
    cerr << setw(10) << "log" << "natural log" << endl;
    cerr << setw(10) << "^" << "power function" << endl;
    cerr << setw(10) << "*" << "multiplication" << endl;
    cerr << setw(10) << "/" << "division" << endl;
    cerr << setw(10) << "+" << "addition" << endl;
    cerr << setw(10) << "-" << "subtraction" << endl;
    cerr << setw(10) << "==" << "equal-to" << endl;
    cerr << "TODO\n" << endl;
    cerr << setw(10) << "abs" << "absoute value" << endl;
    cerr << setw(10) << "round" << "round values to nearest int" << endl;
    cerr << setw(10) << "floor" << "floor value to next int below number" << endl;
    cerr << setw(10) << "ceil" << "ceil next int above number" << endl;
    cerr << setw(10) << "<" << "binarize" << endl;
    cerr << setw(10) << ">" << "binarize" << endl;
    cerr << "Options:\n"<<endl;
    cerr << setw(10) << "--nn" << "Nearest neighbor resampling" << endl;
    cerr << setw(10) << "--lin" << "Linear resampling" << endl;
    cerr << setw(10) << "--short" << "Use short int out for type" << endl;
    cerr << setw(10) << "--int" << "Use int for out type" << endl;
    cerr << setw(10) << "--float" << "Use float for out type" << endl;
    cerr << setw(10) << "--double" << "Use double for out type" << endl;
    exit(status);
}

int main(int argc, char** argv)
{
    if(argc == 1) {
        usage(0);
        return 0;
    }

    string outname = "";
    enum RESAMPMETHOD { LINEAR, NEAREST, LANCZOS };
    RESAMPMETHOD resampler = LANCZOS;
    PixelT type = UNKNOWN_TYPE;

    string equation;
    map<char, shared_ptr<NDConstView<double>>> imgargs;
    char refimg = 0;
    unordered_map<char, string> args;
    for(int ii=1; ii<argc; ii++) {
        if(argv[ii][0] == '-' && argv[ii][1] == '-') {
            if(!strcmp(&argv[ii][2], "out")) {
                if(ii+1 >= argc) {
                    cerr << "Must provide an argument to --out";
                    usage(-1);
                }
                outname = argv[ii+1];
                ii++;
            } else if(!strcmp(&argv[ii][2], "nn")) {
                resampler = NEAREST;
            } else if(!strcmp(&argv[ii][2], "lin")) {
                resampler = LINEAR;
            } else if(!strcmp(&argv[ii][2], "short")) {
                type = INT16;
            } else if(!strcmp(&argv[ii][2], "int")) {
                type = INT32;
            } else if(!strcmp(&argv[ii][2], "double")) {
                type = FLOAT64;
            } else if(!strcmp(&argv[ii][2], "float")) {
                type = FLOAT32;
            }
        } else if(argv[ii][0] == '-') {
            if(!isalpha(argv[ii][1]) || ii+1 >= argc) 
                usage(-1);

            // set image for variable
            args[argv[ii][1]] = argv[ii+1];

            // set reference image
            if(!refimg) 
                refimg = argv[ii][1];

            ii++;
        } else {
            equation = argv[ii];
        }
    }

    if(args.size() < 1) {
        cerr << "Error, need at least 1 image" << endl;
        usage(-1);
    }

    // parse math
    MathExpression expr(equation);

    // load reference image/create output
    ptr<MRImage> out;
    {
        auto ptr = dPtrCast<MRImage>(readMRImage(args[refimg]));
        imgargs[refimg].setArray(ptr);
        if(type != UNKNOWN_TYPE) {
            out = ptr->createAnother();
        } else {
            out = ptr->createAnother(type);
        }
    }

    // load/resample non-reference imagesimages
    for(auto it=args.begin(); it!=args.end(); ++it) {
        // don't reload the reference
        if(imgargs.count(it->first) == 0) {
            // read image
            auto ptr = dPtrCast<MRImage>(readMRImage(it->second));

            //check dimensions
            if(ptr->ndim() != out->ndim()) {
                cerr << "Input image dimensions must match!" << endl;
                usage(-1);
            }

            // resample, create Viewer
            imgargs[it->first].setArray(ptr);
        }
    }

    // perform math
    vector<int64_t> ind(out->ndim());
    for(NDIter<double> it(out); !it.eof(); ++it) {
        // get index
        it.index(ind);
        
        // set all the values
        for(auto eit=imgargs.begin(); eit!=imgargs.end(); ++eit) {
            double v = (eit->second)[ind];
            expr.setarg(eit->first, v);
        }

        // execute
        it.set(expr.exec());
    }

    // write out image
    out->wrtie(outname);
}
///**
// * @Brief Resamples this image into the same sapce as the input and returns
// * the result. This is not modified.
// *
// * @param in image to resample 
// * @param ref reference image to sample to
// * @param otype output type
// * @param method LANCZOS or LINEAR or NEAREST for lanczos resampling,
// * linear resampling or nearest neighbor. 
// *
// * @return this image resampled into the space of ref
// */
//template <typename IT>
//ptr<MRImage> resample(ptr<MRImage> input, ptr<MRImage> refe, PixelT otype,
//        SampleT method) const
//{
//    ptr<MRImage> out = dPtrCast<MRImage>(ref->createAnother(otype));
//
//    // perform resampling
//    if(method == LANCZOS) {
//        for(NDIter<IT> it(out); !it.eof(); ++it) {
//            // convert point
//            //
//        }
//    } else if(method == LINEAR) { 
//
//    } else if(method == NEAREST) {
//
//    } else {
//        throw INVALID_ARGUMENT("Unknown resampling method requested: "+
//                to_str(method));
//    }
//}
//
