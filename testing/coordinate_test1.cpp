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
 * @file coordinate_test1.cpp
 *
 *****************************************************************************/

#include <iostream>
#include "mrimage.h"
#include "nplio.h"

using namespace std;
using namespace npl;


ostream& operator<<(ostream& os, const std::vector<int64_t>& vec);
ostream& operator<<(ostream& os, const std::vector<size_t>& vec);
ostream& operator<<(ostream& os, const std::vector<double>& vec);

int main()
{
	/* Read the Image */
	shared_ptr<MRImage> img = readMRImage("../../data/test_nifti1.nii.gz", true);
	if(!img) {
		std::cerr << "Failed to open image!" << std::endl;
		return -1;
	}
	
	std::vector<std::vector<double>> correct({
			{1.3, 		75, 		9, 	0},
			{-42.044, 	244.106, 	216.15, 0},
			{-290.133, 	27.624, 	-13.3041, 0},
			{-333.477, 	196.73, 	193.846, 0},
			{3.10783,	56.6478, 	24.36, 0},
			{-40.2362, 	225.754, 	231.51, 0},
			{-288.325, 	9.27184, 	2.0559, 0},
			{-331.669, 	178.378, 	209.206, 0},
			{1.3, 		75, 		9, 53.7},
			{-42.044, 	244.106, 	216.15, 53.7},
			{-290.133, 	27.624, 	-13.3041, 53.7},
			{-333.477, 	196.73, 	193.846, 53.7},
			{3.10783, 	56.6478, 	24.36, 53.7},
			{-40.2362, 	225.754, 	231.51, 53.7},
			{-288.325, 	9.27184, 	2.0559, 53.7},
			{-331.669, 	178.378, 	209.206, 53.7}});
	
	std::vector<int64_t> index(img->ndim(), 0);
	std::vector<double> cindex(img->ndim(), 0);
	std::vector<double> ras(img->ndim(), 0);;
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
			if(fabs(ras[dd]-correct[i][dd]) > 0.001) {
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


