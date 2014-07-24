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
#include "mrimage_utils.h"

using namespace std;
using namespace npl;

ostream& operator<<(ostream& os, const std::vector<int64_t>& vec);
ostream& operator<<(ostream& os, const std::vector<size_t>& vec);
ostream& operator<<(ostream& os, const std::vector<double>& vec);

int main()
{
	/* Read the Image */
	shared_ptr<MRImage> img = readMRImage("../../testing/test_nifti2.nii.gz", true);
	if(!img) {
		std::cerr << "Failed to open image!" << std::endl;
		return -1;
	}
	
	cerr << "Affine: " << endl << img->affine() << endl;
	cerr << "Inverse Affine: " << endl << img->iaffine() << endl;


	std::vector<std::vector<double>> correct({
			{1.3, 	75, 9, 0},
			{-28.94, 	71.7735, 195.537, 0},
			{-2080.07, 	-1997.7, -364.267, 0},
			{-2110.31, 	-2000.93, -177.73, 0},
			{-165.027, 	246.345, -15, 0},
			{-195.267, 	243.119, 171.537, 0},
			{-2246.4, 	-1826.35, -388.267, 0},
			{-2276.64, 	-1829.58, -201.73, 0},
			{1.3, 	75, 9, 537},
			{-28.94, 	71.7735, 195.537, 537},
			{-2080.07, 	-1997.7, -364.267, 537},
			{-2110.31, 	-2000.93, -177.73, 537},
			{-165.027, 	246.345, -15, 537},
			{-195.267, 	243.119, 171.537, 537},
			{-2246.4, 	-1826.35, -388.267, 537},
			{-2276.64, 	-1829.58, -201.73, 537}});
	
	std::vector<int64_t> index(img->ndim(), 0);
	std::vector<double> cindex(img->ndim(), 0);
	std::vector<double> ras(img->ndim(), 0);
	size_t DIM = img->ndim();

	std::cerr << "Corners: " << endl;
    for(int32_t i = 0 ; i < (1<<DIM) ; i++) {
        for(uint32_t j = 0 ; j < DIM ; j++) {
            index[j] = ((bool)(i&(1<<j)))*(img->dim(j)-1);
        }
		img->indexToPoint(img->ndim(), index.data(), ras.data());
		std::cerr << "Mine: " << index << " -> " << ras << endl;
		std::cerr << "Prev: " << index << " -> " << correct[i] << endl;
		
		for(size_t dd=0; dd<DIM; dd++) {
			if(fabs(ras[dd]-correct[i][dd]) > 0.01) {
				std::cerr << "Difference! " << endl;
				return -1;
			}
		}

		img->pointToIndex(img->ndim(), ras.data(), cindex.data());
		std::cerr << "Back to index: " << cindex << endl;
		img->indexToPoint(img->ndim(), cindex.data(), ras.data());
		std::cerr << "Back to Point: " << ras << endl;
		img->pointToIndex(img->ndim(), ras.data(), cindex.data());
		std::cerr << "Back to Index: " << cindex << endl;
		
		for(size_t dd=0; dd<DIM; dd++) {
			if(fabs(index[dd]-cindex[dd]) > 0.00001) {
				std::cerr << "Difference in cindex/index! " << endl;
				return -1;
			}
		}
    }
	return 0;
}

ostream& operator<<(ostream& os, const std::vector<int64_t>& vec)
{
	os << "[ ";
	for(auto v : vec) {
		os << std::setw(4) << v << ",";
	}
	os << " ]";
	return os;
}

ostream& operator<<(ostream& os, const std::vector<size_t>& vec)
{
	os << "[ ";
	for(auto v : vec) {
		os << std::setw(4) << v << ",";
	}
	os << " ]";
	return os;
}

ostream& operator<<(ostream& os, const std::vector<double>& vec)
{
	os << "[";
	for(auto v : vec) {
		os << std::setw(7) << std::setprecision(2) << std::fixed << v << ",";
	}
	os << "] ";
	return os;
}


