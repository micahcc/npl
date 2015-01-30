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
        "create a variable for use in the equation. ";
    cerr << "Options:\n"<<endl;
    cerr << '\t' << setw(10) << left << "--nn"     << "Nearest neighbor resampling" << endl;
    cerr << '\t' << setw(10) << left << "--lin"    << "Linear resampling" << endl;
    cerr << '\t' << setw(10) << left << "--short"  << "Use short int out for type" << endl;
    cerr << '\t' << setw(10) << left << "--int"    << "Use int for out type" << endl;
    cerr << '\t' << setw(10) << left << "--float"  << "Use float for out type" << endl;
    cerr << '\t' << setw(10) << left << "--double" << "Use double for out type" << endl;
    cerr << "\nAcceptable operations in the equation are:\n";
    listops();
    exit(status);
}

int main(int argc, char** argv)
{
	cerr << "Version: " << __version__ << endl;
    if(argc == 1) {
        usage(0);
        return 0;
    }

    string outname = "";
    enum RESAMPMETHOD { LINEAR, NEAREST, LANCZOS };
    RESAMPMETHOD resampler = LANCZOS;
    PixelT type = UNKNOWN_TYPE;

    string equation;
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
			if(!equation.empty()) {
				cerr << "Two equations provided: \"" << equation << "\" and \""
					<< argv[ii] << endl;
				usage(-1);
			}
            equation = argv[ii];
        }
    }

    if(args.size() < 1) {
        cerr << "Error, need at least 1 image" << endl;
        usage(-1);
    }
	cerr << "Equation: " << endl;
    // parse math
    MathExpression expr(equation);
    expr.randomTest();

    // load reference image/create output
    ptr<MRImage> out;
	{
	auto ptr = dPtrCast<MRImage>(readMRImage(args[refimg]));
	out = dPtrCast<MRImage>(ptr->createAnother(FLOAT64));
	type = ptr->type();
	}

	// TODO only resample lowest matching dimensions, then
	// iterate over the highest dimensions slowest so that
	// we can deal with 4D+ images.
	// load all images and create interpolated version
	map<char, NDConstView<double>> imgargs;
	for(auto fit = args.begin(); fit != args.end(); ++fit) {
		auto img = readMRImage(fit->second);

		if(img->matchingOrient(out, false, true)) {
			imgargs[fit->first] = NDConstView<double>(img);
		} else {
			vector<int64_t> ind(out->ndim());
			vector<double> point(out->ndim());
			auto tmp = dPtrCast<MRImage>(out->createAnother());

			switch(resampler) {
				case NEAREST: {
					cerr << "NN-Interpolating " << fit->second << endl;
					NNInterpNDView<double> interp(img);
					interp.m_ras = true;
					for(NDIter<double> it(tmp); !it.eof(); ++it) {
						// get index, point
						it.index(ind);
						tmp->indexToPoint(ind.size(), ind.data(), point.data());
						it.set(interp.get(point));
					}
				} break;
				case LINEAR: {
					cerr << "Linear-Interpolating " << fit->second << endl;
					LinInterpNDView<double> interp(img);
					interp.m_ras = true;
					for(NDIter<double> it(tmp); !it.eof(); ++it) {
						// get index, point
						it.index(ind);
						tmp->indexToPoint(ind.size(), ind.data(), point.data());
						it.set(interp.get(point));
					}
				} break;
				case LANCZOS: {
					cerr << "Lanczos-Interpolating " << fit->second << endl;
					LanczosInterpNDView<double> interp(img);
					interp.m_ras = true;
					for(NDIter<double> it(tmp); !it.eof(); ++it) {
						// get index, point
						it.index(ind);
						tmp->indexToPoint(ind.size(), ind.data(), point.data());
						it.set(interp.get(point));
					}
				}
			}
			imgargs[fit->first] = NDConstView<double>(tmp);
		}
	}

	// do math
	vector<int64_t> ind(out->ndim());
	for(NDIter<double> it(out); !it.eof(); ++it) {
		it.index(ind);
		for(auto eit=imgargs.begin(); eit!=imgargs.end(); ++eit) {
			// get value at the set point
			double v = eit->second.get(ind);
			expr.setarg(eit->first, v);
		}
		it.set(expr.exec());
	}

    // write out image
    if(!outname.empty()) {
		out->copyCast(type)->write(outname);
	}
}

