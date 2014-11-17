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
 * @file img_lin_interp_test.cpp Tests the 3D linear interpolator and compares
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
	LinInterp3DView<double> interp(testimg);
	NDView<double> view(testimg);
	Pixel3DView<double> pview(testimg);
	Vector3DView<double> tview(testimg);

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

	for(index[0] = 0; index[0] < sz[0] ; index[0]++) {
		for(index[1] = 0; index[1] < sz[1] ; index[1]++) {
			for(index[2] = 0; index[2] < sz[2] ; index[2]++) {
				val = index[0]+index[1]*10 + index[2]*100;
				if(pview.get(index[0],index[1],index[2]) != val) {
					std::cerr << "Error in pixel view" << std::endl;
					return -1;
				}
			}
		}
	}

	int64_t tt = 0;
	for(index[0] = 0; index[0] < sz[0] ; index[0]++) {
		for(index[1] = 0; index[1] < sz[1] ; index[1]++) {
			for(index[2] = 0; index[2] < sz[2] ; index[2]++) {
				tt = 0;
				for(index[3] = 0; index[3] < sz[3] ; index[3]++) {
					for(index[4] = 0; index[4] < sz[4] ; index[4]++, ++tt) {
						val = index[0]+index[1]*10 + index[2]*100
							+ index[3]*1000+index[4]*10000;
						double s = tview.get(index[0],index[1],index[2], tt);
						if(s != val) {
							std::cerr << "Error in vector view" << std::endl;
							std::cerr << index[0] << "," << index[1] << ","
								<< index[2] << "," << index[3] << ","
								<< index[4] << ":" << val << endl;
							std::cerr << tt << ":" << s << endl;
							return -1;
						}
					}
				}
			}
		}
	}

	return 0;
}


