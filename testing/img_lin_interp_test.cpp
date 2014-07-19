/*******************************************************************************
This file is part of Neural Program Library (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neural Program Library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The Neural Programs and Libraries are distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the Neural Programs Library.  If not, see <http://www.gnu.org/licenses/>.
*******************************************************************************/

#include <iostream>

#include "mrimage.h"
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
	NDAccess<double> view(testimg);
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
						view.set(val, index);
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


