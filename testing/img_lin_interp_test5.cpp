/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file img_lin_interp_test.cpp Tests the linear interpolator and compares
 * the returned result with a known analytical function
 *
 *****************************************************************************/

#include <iostream>

#include "mrimage.h"
#include "mrimage_utils.h"
#include "accessors.h"

using namespace std;
using namespace npl;

int main()
{
	/* Create an image with: x+y*100+z*10000*/
	std::vector<size_t> sz({9, 4, 5, 7, 3});
	std::vector<int64_t> index(5, 0);
	std::vector<double> cindex(5, 0);
	shared_ptr<MRImage> testimg = createMRImage(sz, FLOAT64);
	LinInterpNDView<double> interp(testimg);
	NDView<double> view(testimg);

	/* Create an image with: x+y*10+z*100+t*1000 */
	double val = 0;
	for(index[0] = 0; index[0] < sz[0] ; index[0]++) {
		for(index[1] = 0; index[1] < sz[1] ; index[1]++) {
			for(index[2] = 0; index[2] < sz[2] ; index[2]++) {
				for(index[3] = 0; index[3] < sz[3] ; index[3]++) {
					for(index[4] = 0; index[4] < sz[4] ; index[4]++) {
						val = index[0]+index[1]*10 + index[2]*100
							+ index[3]*1000+index[4]*10000;
						view.set(index, val);
					}
				}
			}
		}
	}

	for(index[0] = 0; index[0] < sz[0] ; index[0]++) {
		for(index[1] = 0; index[1] < sz[1] ; index[1]++) {
			for(index[2] = 0; index[2] < sz[2] ; index[2]++) {
				val = index[0]+index[1]*10 + index[2]*100;
				double s = interp(index[0], index[1], index[2]);

				if(fabs(val - s) > 0.00000000001) {
					std::cerr << "On-grid point value mismatch" << endl;
					return -1;
				}
			}
		}
	}

	for(cindex[0] = 0; cindex[0] < sz[0] ; cindex[0]++) {
		for(cindex[1] = 0; cindex[1] < sz[1] ; cindex[1]++) {
			for(cindex[2] = 0; cindex[2] < sz[2] ; cindex[2]++) {
				val = cindex[0]+cindex[1]*10 + cindex[2]*100;
				double s = interp(cindex[0], cindex[1], cindex[2]);

				if(fabs(val - s) > 0.00000000001) {
					std::cerr << "On-grid point (summed double) value "
						"mismatch" << endl;
					return -1;
				}
			}
		}
	}

	for(cindex[0] = .5; cindex[0] < sz[0]-.5 ; cindex[0]++) {
		for(cindex[1] = 0; cindex[1] < sz[1]; cindex[1]++) {
			for(cindex[2] = 0; cindex[2] < sz[2] ; cindex[2]++) {
				val = cindex[0]+cindex[1]*10 + cindex[2]*100;
				double s = interp(cindex[0], cindex[1], cindex[2]);

				if(fabs(val - s) > 0.00000000001) {
					std::cerr << "X Off-grid point (summed double) value "
						"mismatch" << endl << "Calcu: " << val
						<< " versus Estimate: " << s << endl << " at "
						<< cindex[0] << ", " << cindex[1] << ", "
						<< cindex[2] << ", " << cindex[3] << endl;

					return -1;
				}
			}
		}
	}

	for(cindex[0] = .5; cindex[0] < sz[0]-.5 ; cindex[0]++) {
		for(cindex[1] = .5; cindex[1] < sz[1]-.5 ; cindex[1]++) {
			for(cindex[2] = .5; cindex[2] < sz[2]-.5 ; cindex[2]++) {
				val = cindex[0]+cindex[1]*10 + cindex[2]*100;
				double s = interp(cindex[0], cindex[1], cindex[2]);

				if(fabs(val - s) > 0.00000000001) {
					std::cerr << "Off-grid point (summed double) value mismatch"
						<< endl << "Calcu: " << val
						<< " versus Estimate: " << s << endl << " at "
						<< cindex[0] << ", " << cindex[1] << ", "
						<< cindex[2] << ", " << cindex[3] << endl;

					return -1;
				}
			}
		}
	}

	return 0;
}



