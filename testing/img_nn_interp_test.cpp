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

using namespace std;
using namespace npl;

int main(int argc, char** argv)
{
	/* Create an image with: x+y*100+z*10000*/
	BoundaryConditionT bound = CONSTZERO;
	bool outside = false;
	std::vector<size_t> sz({9, 4, 5, 8});
	std::vector<int64_t> index(4, 0);
	std::vector<double> cindex(4, 0);
	shared_ptr<MRImage> testimg = createMRImage(sz, FLOAT64);

	/* Create an image with: x+y*10+z*100+t*1000 */
	double val = 0;
	for(index[0] = 0; index[0] < sz[0] ; index[0]++) {
		for(index[1] = 0; index[1] < sz[1] ; index[1]++) {
			for(index[2] = 0; index[2] < sz[2] ; index[2]++) {
				for(index[3] = 0; index[3] < sz[3] ; index[3]++) {
					val = index[0]+index[1]*10 + index[2]*100 + index[3]*1000;
					testimg->set_dbl(index.size(), index.data(), val);
				}
			}
		}
	}
	
	for(index[0] = 0; index[0] < sz[0] ; index[0]++) {
		for(index[1] = 0; index[1] < sz[1] ; index[1]++) {
			for(index[2] = 0; index[2] < sz[2] ; index[2]++) {
				for(index[3] = 0; index[3] < sz[3] ; index[3]++) {
					val = index[0]+index[1]*10 + index[2]*100 + index[3]*1000;
					for(size_t ii=0; ii<index.size(); ii++)
						cindex[ii] = index[ii];
					double s = testimg->nnSampleInd(cindex, bound, outside);
					if(outside) {
						std::cerr << "Should not be outside" << endl;
						return -1;
					}

					if(fabs(val - s) > 0.00000000001) {
						std::cerr << "On-grid point value mismatch" << endl;
						return -1;
					}
				}
			}
		}
	}
	
	for(cindex[0] = 0; cindex[0] < sz[0] ; cindex[0]++) {
		for(cindex[1] = 0; cindex[1] < sz[1] ; cindex[1]++) {
			for(cindex[2] = 0; cindex[2] < sz[2] ; cindex[2]++) {
				for(cindex[3] = 0; cindex[3] < sz[3] ; cindex[3]++) {
					val = cindex[0]+cindex[1]*10 + cindex[2]*100 + cindex[3]*1000;
					double s = testimg->nnSampleInd(cindex, bound, outside);
					if(outside) {
						std::cerr << "Should not be outside" << endl;
						return -1;
					}

					if(fabs(val - s) > 0.00000000001) {
						std::cerr << "On-grid point (summed double) value mismatch" << endl;
						return -1;
					}
				}
			}
		}
	}
	
	for(cindex[0] = .45; cindex[0] < sz[0]-.5 ; cindex[0]++) {
		for(cindex[1] = .45; cindex[1] < sz[1]-.5 ; cindex[1]++) {
			for(cindex[2] = .45; cindex[2] < sz[2]-.5 ; cindex[2]++) {
				for(cindex[3] = .45; cindex[3] < sz[3]-.5 ; cindex[3]++) {
					val = round(cindex[0])+round(cindex[1])*10 +
						round(cindex[2])*100 + round(cindex[3])*1000;
					double s = testimg->nnSampleInd(cindex, bound, outside);
					if(outside) {
						std::cerr << "Should not be outside" << endl;
						return -1;
					}

					if(fabs(val - s) > 0.00000000001) {
						std::cerr << "Off-grid point (.45) value mismatch" << endl;
						std::cerr << "Calcu: " << val << " versus Estimate: " << s << endl;
						std::cerr << " at " << cindex[0] << ", " << cindex[1]
									<< ", " << cindex[2] << ", " << cindex[3]
									<< endl;

						for(size_t ii=0 ; ii < 4; ii++) 
							cindex[ii] = round(cindex[ii]);
						std::cerr << testimg->nnSampleInd(cindex, bound, outside) << endl;
						
						for(size_t ii=0 ; ii < 4; ii++) 
							cindex[ii]++;
						std::cerr << testimg->nnSampleInd(cindex, bound, outside) << endl;
						return -1;
					}
				}
			}
		}
	}
	
	for(cindex[0] = .50; cindex[0] < sz[0]-.5 ; cindex[0]++) {
		for(cindex[1] = .50; cindex[1] < sz[1]-.5 ; cindex[1]++) {
			for(cindex[2] = .50; cindex[2] < sz[2]-.5 ; cindex[2]++) {
				for(cindex[3] = .50; cindex[3] < sz[3]-.5 ; cindex[3]++) {
					val = round(cindex[0])+round(cindex[1])*10 +
						round(cindex[2])*100 + round(cindex[3])*1000;
					double s = testimg->nnSampleInd(cindex, bound, outside);
					if(outside) {
						std::cerr << "Should not be outside" << endl;
						return -1;
					}

					if(fabs(val - s) > 0.00000000001) {
						std::cerr << "Off-grid point (.5) value mismatch" << endl;
						std::cerr << "Calcu: " << val << " versus Estimate: " << s << endl;
						std::cerr << " at " << cindex[0] << ", " << cindex[1]
									<< ", " << cindex[2] << ", " << cindex[3]
									<< endl;

						std::cerr << "Lower Corner: " ;
						for(size_t ii=0 ; ii < 4; ii++) 
							cindex[ii] = round(cindex[ii]);
						std::cerr << testimg->linSampleInd(cindex, bound, outside) << endl;
						
						std::cerr << "Upper Corner: " ;
						for(size_t ii=0 ; ii < 4; ii++) 
							cindex[ii]++;
						std::cerr << testimg->linSampleInd(cindex, bound, outside) << endl;
						return -1;
					}
				}
			}
		}
	}
	
	return 0;
}



